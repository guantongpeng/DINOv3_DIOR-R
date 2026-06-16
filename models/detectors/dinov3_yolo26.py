# Copyright (c) OpenMMLab. All rights reserved.
"""DINOv3 + YOLO26 Rotated Object Detector.

This detector combines:
    - DINOv3 ViT backbone (pretrained self-supervised features)
    - SimpleFPN neck (multi-scale feature pyramid for ViT features)
    - YOLO26RotatedHead (anchor-free rotated detection with NMS-free inference)

It is designed for oriented object detection on remote sensing datasets
like DIOR-R, using a single-stage architecture with end-to-end detection.
"""

import warnings
from typing import Dict, List, Optional, Tuple

import torch

from mmrotate.core import rbbox2result
from mmrotate.models.builder import ROTATED_DETECTORS, build_backbone, build_head, build_neck
from mmrotate.models.detectors.base import RotatedBaseDetector


@ROTATED_DETECTORS.register_module()
class DINOv3YOLO26(RotatedBaseDetector):
    """DINOv3 + YOLO26 single-stage rotated object detector.

    Architecture:
        Image → DINOv3 ViT Backbone → SimpleFPN Neck → YOLO26RotatedHead → Detections

    Key features:
        - DINOv3 provides powerful self-supervised features
        - SimpleFPN builds multi-scale pyramid from ViT features
        - YOLO26Head provides NMS-free end-to-end rotated detection
        - Dual-head training (O2M + O2O) with progressive loss shifting
        - Anchor-free design, no manual anchor tuning needed

    Args:
        backbone (dict): DINOv3 ViT backbone config.
        neck (dict): SimpleFPN neck config.
        bbox_head (dict): YOLO26RotatedHead config.
        train_cfg (dict, optional): Training configuration.
        test_cfg (dict, optional): Testing configuration.
        pretrained (str, optional): Deprecated. Use init_cfg.
        init_cfg (dict, optional): Initialization config.
    """

    def __init__(
        self,
        backbone: dict,
        neck: Optional[dict] = None,
        bbox_head: Optional[dict] = None,
        train_cfg: Optional[dict] = None,
        test_cfg: Optional[dict] = None,
        pretrained: Optional[str] = None,
        init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg=init_cfg)

        if pretrained:
            warnings.warn(
                'DeprecationWarning: pretrained is deprecated, '
                'please use "init_cfg" instead'
            )
            backbone.pretrained = pretrained

        # Build backbone (DINOv3 ViT)
        self.backbone = build_backbone(backbone)

        # Build neck (SimpleFPN)
        if neck is not None:
            self.neck = build_neck(neck)

        # Build head (YOLO26RotatedHead)
        if bbox_head is not None:
            bbox_head.update(train_cfg=train_cfg)
            bbox_head.update(test_cfg=test_cfg)
            self.bbox_head = build_head(bbox_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @property
    def with_neck(self) -> bool:
        """Check if the detector has a neck module."""
        return hasattr(self, 'neck') and self.neck is not None

    def extract_feat(self, img: torch.Tensor) -> List[torch.Tensor]:
        """Extract features from backbone and neck.

        Args:
            img: Input images (B, C, H, W).

        Returns:
            List of multi-scale feature maps.
        """
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img: torch.Tensor) -> tuple:
        """Forward pass for FLOPs computation.

        Args:
            img: Input images.

        Returns:
            Tuple of head outputs.
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(
        self,
        img: torch.Tensor,
        img_metas: List[dict],
        gt_bboxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        gt_bboxes_ignore: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass.

        Args:
            img: Input images (B, C, H, W).
            img_metas: Image meta information.
            gt_bboxes: Ground truth rotated boxes (x, y, w, h, a).
            gt_labels: Ground truth class labels.
            gt_bboxes_ignore: Boxes to ignore.

        Returns:
            Dict of loss components.
        """
        super().forward_train(img, img_metas)

        # Extract features through backbone + neck
        x = self.extract_feat(img)

        # Forward through head and compute losses
        losses = self.bbox_head.forward_train(
            x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore
        )

        return losses

    def simple_test(
        self,
        img: torch.Tensor,
        img_metas: List[dict],
        rescale: bool = False,
    ) -> List[list]:
        """Test without augmentation.

        Args:
            img: Input images (B, C, H, W).
            img_metas: Image meta information.
            rescale: Whether to rescale results to original image size.

        Returns:
            BBox results per image and class.
        """
        # Extract features
        x = self.extract_feat(img)

        # Forward through head
        outs = self.bbox_head(x)

        # Get final bounding boxes
        bbox_list = self.bbox_head.get_bboxes(
            *outs,
            img_metas=img_metas,
            rescale=rescale,
        )

        # Format results
        bbox_results = [
            rbbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]

        return bbox_results

    def aug_test(
        self,
        imgs: List[torch.Tensor],
        img_metas: List[List[dict]],
        rescale: bool = False,
    ) -> List[list]:
        """Test with test-time augmentation.

        Args:
            imgs: List of image tensors for different augmentations.
            img_metas: Meta information for each augmentation.
            rescale: Whether to rescale results.

        Returns:
            BBox results per image and class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), (
            f'{self.bbox_head.__class__.__name__} '
            'does not support test-time augmentation'
        )

        feats = self.extract_feats(imgs)
        results_list = self.bbox_head.aug_test(feats, img_metas, rescale=rescale)
        bbox_results = [
            rbbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def set_o2o_weight(self, weight: float):
        """Set the O2O (one-to-one) head loss weight for progressive training.

        Progressive loss schedule:
            - Early epochs: weight = 0 (only O2M head)
            - Mid epochs: weight linearly increases (O2M + O2O)
            - Late epochs: weight = 1.0 (equal O2M and O2O)

        Args:
            weight: O2O loss weight in [0, 1].
        """
        if hasattr(self.bbox_head, 'o2o_weight'):
            self.bbox_head.o2o_weight = weight
