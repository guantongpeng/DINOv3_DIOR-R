"""Pure-PyTorch OrientedRPNHead (ports mmrotate OrientedRPNHead test path)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import box_ops


def gen_base_anchors(base_size, scales, ratios, center_offset=0.5):
    h = w = float(base_size)
    x_center = center_offset * w
    y_center = center_offset * h
    h_ratios = torch.sqrt(torch.tensor(ratios, dtype=torch.float32))
    w_ratios = 1.0 / h_ratios
    ws = (w * w_ratios[:, None] * torch.tensor(scales, dtype=torch.float32)[None, :]).view(-1)
    hs = (h * h_ratios[:, None] * torch.tensor(scales, dtype=torch.float32)[None, :]).view(-1)
    base = torch.stack([x_center - 0.5 * ws, y_center - 0.5 * hs,
                        x_center + 0.5 * ws, y_center + 0.5 * hs], dim=-1)
    return base


def grid_anchors(base_anchors, feat_h, feat_w, stride, device):
    shift_x = torch.arange(feat_w, device=device).float() * stride
    shift_y = torch.arange(feat_h, device=device).float() * stride
    # mmdet _meshgrid (row_major): y outer, x inner
    shift_xx = shift_x.repeat(feat_h)
    shift_yy = shift_y.view(-1, 1).repeat(1, feat_w).view(-1)
    shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
    all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
    return all_anchors.view(-1, 4)


class OrientedRPNHead(nn.Module):
    def __init__(self, in_channels=256, feat_channels=256, num_anchors=3,
                 scales=(8.0,), ratios=(0.5, 1.0, 2.0), strides=(4, 8, 16, 32, 64)):
        super().__init__()
        self.rpn_conv = nn.Conv2d(in_channels, feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(feat_channels, num_anchors, 1)
        self.rpn_reg = nn.Conv2d(feat_channels, num_anchors * 6, 1)
        self.strides = list(strides)
        self.scales = list(scales)
        self.ratios = list(ratios)
        self.num_anchors = num_anchors
        self.base_anchors = [gen_base_anchors(s, self.scales, self.ratios) for s in self.strides]

    def forward(self, feats):
        cls_scores, bbox_preds = [], []
        for x in feats:
            x = F.relu(self.rpn_conv(x))
            cls_scores.append(self.rpn_cls(x))
            bbox_preds.append(self.rpn_reg(x))
        return cls_scores, bbox_preds

    @torch.no_grad()
    def get_bboxes_single(self, cls_scores, bbox_preds, img_shape, nms_pre=2000,
                          nms_iou=0.8, max_per_img=2000, min_bbox_size=0, device='cuda'):
        mlvl_scores, mlvl_bbox_preds, mlvl_anchors, level_ids = [], [], [], []
        for idx, _ in enumerate(cls_scores):
            rpn_cls = cls_scores[idx]              # (A,H,W)
            rpn_bbox = bbox_preds[idx]             # (A*6,H,W)
            rpn_cls = rpn_cls.permute(1, 2, 0).reshape(-1)
            scores = rpn_cls.sigmoid()
            rpn_bbox = rpn_bbox.permute(1, 2, 0).reshape(-1, 6)
            feat_h, feat_w = cls_scores[idx].shape[-2], cls_scores[idx].shape[-1]
            anchors = grid_anchors(self.base_anchors[idx].to(device), feat_h, feat_w,
                                   self.strides[idx], device)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                ranked, rank = scores.sort(descending=True)
                topk = rank[:nms_pre]
                scores, rpn_bbox, anchors = ranked[:nms_pre], rpn_bbox[topk], anchors[topk]
            mlvl_scores.append(scores)
            mlvl_bbox_preds.append(rpn_bbox)
            mlvl_anchors.append(anchors)
            level_ids.append(scores.new_full((scores.size(0),), idx, dtype=torch.long))

        scores = torch.cat(mlvl_scores)
        anchors = torch.cat(mlvl_anchors)
        rpn_bbox = torch.cat(mlvl_bbox_preds)
        proposals = box_ops.midpoint_offset_decode(
            anchors, rpn_bbox, stds=(1, 1, 1, 1, 0.5, 0.5))
        ids = torch.cat(level_ids)
        if min_bbox_size > 0:
            valid = (proposals[:, 2] >= min_bbox_size) & (proposals[:, 3] >= min_bbox_size)
            proposals, scores, ids = proposals[valid], scores[valid], ids[valid]
        hproposals = box_ops.obb2xyxy_le90(proposals)
        keep = box_ops.batched_nms(hproposals, scores, ids, nms_iou)
        dets = torch.cat([proposals, scores[:, None]], dim=1)[keep]
        return dets[:max_per_img]

    @torch.no_grad()
    def simple_test_rpn(self, feats, img_metas):
        cls_scores, bbox_preds = self(feats)
        proposals = []
        for i, meta in enumerate(img_metas):
            cs = [cls_scores[l][i] for l in range(len(cls_scores))]
            bp = [bbox_preds[l][i] for l in range(len(bbox_preds))]
            dets = self.get_bboxes_single(
                cs, bp, meta['img_shape'], device=cls_scores[0].device)
            proposals.append(dets[:, :5])   # drop score column -> (n,5)
        return proposals
