"""RVSA Detector for Rotated Object Detection.

Wraps Backbone + Neck + RVSAHead in an end-to-end detector.
"""

import numpy as np
import torch
from mmdet.models.builder import DETECTORS
from mmrotate.core import multiclass_nms_rotated, rbbox2result
from mmrotate.models.detectors.base import RotatedBaseDetector


@DETECTORS.register_module()
class RVSADetector(RotatedBaseDetector):
    """RVSA: Rotated Varied-Size Attention Network.

    End-to-end rotated object detector using DINOv3 backbone, FPN neck,
    and RVSA detection head with VSA Transformer.

    Args:
        backbone (dict): Backbone config.
        neck (dict): Neck config.
        bbox_head (dict): RVSAHead config.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if pretrained:
            backbone.pretrained = pretrained
        from mmrotate.models.builder import build_backbone
        self.backbone = build_backbone(backbone)

        if neck is not None:
            from mmrotate.models.builder import build_neck
            self.neck = build_neck(neck)
        else:
            self.neck = None

        if bbox_head is not None:
            from mmrotate.models.builder import build_head
            # Only inject detector-level train/test cfg when provided, so that
            # configs that place train_cfg/test_cfg inside bbox_head are not
            # clobbered with None.
            if train_cfg is not None:
                bbox_head.update(train_cfg=train_cfg)
            if test_cfg is not None:
                bbox_head.update(test_cfg=test_cfg)
            self.bbox_head = build_head(bbox_head)
        else:
            self.bbox_head = None

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      **kwargs):
        # NOTE the batched image size information is needed by the RVSA head to
        # build attention masks (mirrors BaseDetector.forward_train).
        batch_input_shape = tuple(img.size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        x = self.extract_feat(img)
        outs = self.bbox_head(x, img_metas)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.bbox_head.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_metas, rescale=False, **kwargs):
        batch_input_shape = tuple(img.size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        x = self.extract_feat(img)
        outs = self.bbox_head(x, img_metas)
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        bbox_results = [
            rbbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, imgs, img_metas, rescale=False, **kwargs):
        """Test with test-time augmentation.

        Runs the head on every augmentation, then merges the rotated
        detections across augmentations with a per-class rotated NMS. Returns
        the same list-per-image / list-per-class structure as
        :meth:`simple_test`, with numpy arrays (mmrotate eval format).
        """
        num_classes = self.bbox_head.num_classes
        nms_cfg = dict(type='nms_rotated', iou_thr=0.1)
        # Per-augmentation results: list (n_aug) of list (n_imgs) of
        # list (n_classes) of (n, 6) np.ndarray.
        aug_results = [self.simple_test(img, meta, rescale=rescale)
                       for img, meta in zip(imgs, img_metas)]
        num_imgs = len(aug_results[0])
        merged = []
        for img_idx in range(num_imgs):
            merged_img = []
            for cls in range(num_classes):
                cls_boxes = [aug_results[a][img_idx][cls]
                             for a in range(len(aug_results))]
                cls_boxes = (np.concatenate(cls_boxes, axis=0)
                             if any(b.shape[0] for b in cls_boxes)
                             else np.zeros((0, 6), dtype=np.float32))
                if cls_boxes.shape[0] > 0:
                    bboxes = torch.from_numpy(
                        np.ascontiguousarray(cls_boxes[:, :5])).float()
                    scores = torch.from_numpy(
                        cls_boxes[:, 5:6]).float()
                    det_bboxes, _ = multiclass_nms_rotated(
                        bboxes, scores, 0.0, nms_cfg, -1)
                    cls_boxes = det_bboxes.cpu().numpy()
                merged_img.append(cls_boxes)
            merged.append(merged_img)
        return merged
