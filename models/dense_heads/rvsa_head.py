"""RVSA Detection Head for Rotated Object Detection.

End-to-end rotated object detection head using VSA Transformer.
Outputs rotated bounding boxes (cx, cy, w, h, theta) via Hungarian matching.

Box convention
--------------
Rotated boxes are (cx, cy, w, h, theta) with theta in radians (le90,
``[-pi/2, pi/2)``). Internally the network predicts a *normalized* 5-dim box in
which all five components are sigmoided to ``[0, 1]``; the angle is mapped with
``theta_norm = (theta + pi/2) / pi`` so that prediction and target share the
same scale. Absolute boxes are only reconstructed (for the rotated IoU loss and
inference) via the helpers in :mod:`models.layers.rotated_match`.
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob, constant_init
from mmcv.runner import force_fp32

from mmdet.core import multi_apply, reduce_mean
from mmdet.models.builder import HEADS
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models.dense_heads.detr_head import DETRHead

from ..layers.rotated_match import rbboxes_to_norm, norm_to_theta

PI = math.pi


@HEADS.register_module()
class RVSAHead(DETRHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 num_query=300,
                 num_reg_fcs=2,
                 transformer=None,
                 sync_cls_avg_factor=False,
                 positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=2.0),
                 loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                 loss_iou=dict(
                     type='RotatedIoULoss', mode='linear', loss_weight=2.0),
                 train_cfg=dict(
                     assigner=dict(
                         type='RotatedHungarianAssigner',
                         cls_cost=dict(type='FocalLossCost', weight=2.0),
                         reg_cost=dict(type='RotatedL1Cost', weight=5.0),
                         iou_cost=dict(type='RotatedIoUCost', weight=2.0))),
                 test_cfg=dict(max_per_img=300),
                 init_cfg=None,
                 **kwargs):
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            num_query=num_query,
            num_reg_fcs=num_reg_fcs,
            transformer=transformer,
            sync_cls_avg_factor=sync_cls_avg_factor,
            positional_encoding=positional_encoding,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_iou=loss_iou,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            **kwargs,
        )

    def _init_layers(self):
        fc_cls = Linear(self.embed_dims, self.cls_out_channels)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, 5))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

        num_pred = (self.transformer.decoder.num_layers
                    if hasattr(self.transformer, 'decoder') else 6)
        self.cls_branches = _get_clones(fc_cls, num_pred)
        self.reg_branches = _get_clones(reg_branch, num_pred)
        self.query_embedding = nn.Embedding(self.num_query, self.embed_dims * 2)

    def init_weights(self):
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m.bias, bias_init)
        for m in self.reg_branches:
            constant_init(m[-1], 0, bias=0)
        # zero the w/h regression bias so the initial box is a unit square.
        nn.init.constant_(self.reg_branches[0][-1].bias.data[2:4], 0.0)

    def forward(self, mlvl_feats, img_metas):
        batch_size = mlvl_feats[0].size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        img_masks = mlvl_feats[0].new_ones((batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w = img_metas[img_id]['img_shape'][:2]
            img_masks[img_id, :img_h, :img_w] = 0

        mlvl_masks = []
        mlvl_pos_embeds = []
        for feat in mlvl_feats:
            mlvl_masks.append(
                F.interpolate(img_masks[None], size=feat.shape[-2:])
                .to(torch.bool).squeeze(0))
            mlvl_pos_embeds.append(self.positional_encoding(mlvl_masks[-1]))

        query_embeds = self.query_embedding.weight
        hs, init_reference, inter_references, _, _ = self.transformer(
            mlvl_feats, mlvl_masks, query_embeds, mlvl_pos_embeds,
            reg_branches=None, cls_branches=None)

        # The VSATransformer/decoder operate with batch_first=True, hence
        # hs is [num_layers, bs, num_query, embed_dims] and the reference
        # points are [bs, num_query, 2]. No permutation is needed.
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])
            tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        all_cls_scores = torch.stack(outputs_classes)
        all_bbox_preds = torch.stack(outputs_coords)
        return all_cls_scores, all_bbox_preds, None, None

    @force_fp32(apply_to=('all_cls_scores', 'all_bbox_preds'))
    def loss(self, all_cls_scores, all_bbox_preds, enc_cls_scores,
             enc_bbox_preds, gt_bboxes_list, gt_labels_list, img_metas,
             gt_bboxes_ignore=None):
        assert gt_bboxes_ignore is None
        num_dec_layers = len(all_cls_scores)
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]

        losses_cls, losses_bbox, losses_iou = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list, img_metas_list)

        loss_dict = dict()
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_iou'] = losses_iou[-1]
        for i, (lc, lb, li) in enumerate(
                zip(losses_cls[:-1], losses_bbox[:-1], losses_iou[:-1])):
            loss_dict[f'd{i}.loss_cls'] = lc
            loss_dict[f'd{i}.loss_bbox'] = lb
            loss_dict[f'd{i}.loss_iou'] = li
        return loss_dict

    def loss_single(self, cls_scores, bbox_preds, gt_bboxes_list,
                    gt_labels_list, img_metas, gt_bboxes_ignore_list=None):
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(
            cls_scores_list, bbox_preds_list, gt_bboxes_list, gt_labels_list,
            img_metas, gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # bbox_preds / targets are normalized 5-dim (cx, cy, w, h, theta_norm)
        bbox_preds = bbox_preds.reshape(-1, 5)
        bbox_targets = bbox_targets.reshape(-1, 5)

        # regression L1 loss on normalized boxes
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)

        # rotated IoU loss on positive samples (decode to absolute boxes)
        loss_iou = self._rotated_iou_loss(
            bbox_preds, bbox_targets, bbox_weights, img_metas, num_imgs,
            num_total_pos)
        return loss_cls, loss_bbox, loss_iou

    def _rotated_iou_loss(self, bbox_preds, bbox_targets, bbox_weights,
                          img_metas, num_imgs, num_total_pos):
        pos_mask = bbox_weights.sum(-1) > 0
        if not pos_mask.any():
            return bbox_preds.sum() * 0.0

        num_query = bbox_preds.size(0) // num_imgs
        factors = []
        for img_meta in img_metas:
            img_h, img_w = img_meta['img_shape'][:2]
            f = bbox_preds.new_tensor([img_w, img_h, img_w, img_h])
            factors.append(f.unsqueeze(0).expand(num_query, -1))
        factors = torch.cat(factors, 0)

        def decode(boxes):
            abs_boxes = boxes.new_empty(boxes.shape)
            abs_boxes[:, :4] = boxes[:, :4] * factors
            abs_boxes[:, 4] = norm_to_theta(boxes[:, 4])
            return abs_boxes

        pred_abs = decode(bbox_preds)[pos_mask]
        target_abs = decode(bbox_targets)[pos_mask]
        return self.loss_iou(
            pred_abs, target_abs, avg_factor=num_total_pos)

    def _get_target_single(self, cls_score, bbox_pred, gt_bboxes, gt_labels,
                           img_meta, gt_bboxes_ignore=None):
        num_bboxes = bbox_pred.size(0)
        assign_result = self.assigner.assign(
            bbox_pred, cls_score, gt_bboxes, gt_labels, img_meta,
            gt_bboxes_ignore)
        sampling_result = self.sampler.sample(assign_result, bbox_pred, gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        labels = gt_bboxes.new_full(
            (num_bboxes,), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        bbox_targets = torch.zeros_like(bbox_pred)
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        img_h, img_w = img_meta['img_shape'][:2]

        pos_gt_bboxes = sampling_result.pos_gt_bboxes
        if pos_gt_bboxes.size(-1) == 4:
            pad = torch.zeros(
                len(pos_gt_bboxes), 5, device=pos_gt_bboxes.device)
            pad[:, :4] = pos_gt_bboxes
            pos_gt_bboxes = pad
        bbox_targets[pos_inds] = rbboxes_to_norm(pos_gt_bboxes, img_h, img_w)
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)

    @force_fp32(apply_to=('all_cls_scores', 'all_bbox_preds'))
    def get_bboxes(self, all_cls_scores, all_bbox_preds, enc_cls_scores,
                   enc_bbox_preds, img_metas, rescale=False):
        cls_scores = all_cls_scores[-1]
        bbox_preds = all_bbox_preds[-1]
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(
                cls_score, bbox_pred, img_shape, scale_factor, rescale)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self, cls_score, bbox_pred, img_shape, scale_factor,
                           rescale=False):
        max_per_img = self.test_cfg.get('max_per_img', self.num_query)
        if self.loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
            scores, indexes = cls_score.view(-1).topk(max_per_img)
            det_labels = indexes % self.num_classes
            bbox_index = indexes // self.num_classes
            bbox_pred = bbox_pred[bbox_index]
        else:
            scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(max_per_img)
            bbox_pred = bbox_pred[bbox_index]
            det_labels = det_labels[bbox_index]

        img_h, img_w = img_shape[0], img_shape[1]
        det_bboxes = bbox_pred.new_empty(bbox_pred.shape)
        det_bboxes[:, 0] = bbox_pred[:, 0] * img_w
        det_bboxes[:, 1] = bbox_pred[:, 1] * img_h
        det_bboxes[:, 2] = bbox_pred[:, 2] * img_w
        det_bboxes[:, 3] = bbox_pred[:, 3] * img_h
        det_bboxes[:, 4] = norm_to_theta(bbox_pred[:, 4])
        det_bboxes[:, 0].clamp_(min=0, max=img_w)
        det_bboxes[:, 1].clamp_(min=0, max=img_h)
        det_bboxes[:, 2].clamp_(min=1, max=img_w * 2)
        det_bboxes[:, 3].clamp_(min=1, max=img_h * 2)
        det_bboxes[:, 4].clamp_(min=-PI / 2, max=PI / 2)
        if rescale:
            det_bboxes[:, :4] = det_bboxes[:, :4] / det_bboxes[:, :4].new_tensor(
                [scale_factor[0], scale_factor[1], scale_factor[0],
                 scale_factor[1]])
        det_bboxes = torch.cat((det_bboxes, scores.unsqueeze(1)), -1)
        return det_bboxes, det_labels
