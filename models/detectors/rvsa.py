"""RVSA Detector for Rotated Object Detection.

Wraps Backbone + Neck + RVSAHead in an end-to-end detector.
"""

import torch
from mmdet.models.builder import DETECTORS
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
        x = self.extract_feat(img)
        outs = self.bbox_head(x, img_metas)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.bbox_head.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_metas, **kwargs):
        x = self.extract_feat(img)
        outs = self.bbox_head(x, img_metas)
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=kwargs.get('rescale', False))
        bbox_results = [
            bbox_result2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, imgs, img_metas, **kwargs):
        return self.simple_test(imgs[0], img_metas[0], **kwargs)


def bbox_result2result(bboxes, labels, num_classes):
    if bboxes.shape[0] == 0:
        return [torch.zeros((0, 6), dtype=torch.float32, device=bboxes.device)
                for _ in range(num_classes)]
    result = []
    for i in range(num_classes):
        mask = labels == i
        cls_bboxes = bboxes[mask]
        if cls_bboxes.size(0) > 0:
            result.append(torch.cat([cls_bboxes[:, :5], cls_bboxes[:, 5:6]], dim=-1))
        else:
            result.append(torch.zeros((0, 6), dtype=torch.float32, device=bboxes.device))
    return result
