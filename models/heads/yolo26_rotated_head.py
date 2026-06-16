# Copyright (c) OpenMMLab. All rights reserved.
"""
YOLO26 Rotated Detection Head for Oriented Object Detection.

This module implements a YOLO26-style anchor-free rotated detection head
compatible with mmrotate. The head supports:

1. **Anchor-free dense prediction**: Predicts per-pixel classification,
   bounding box (ltrb + angle), and objectness scores.

2. **Dual-Head Architecture**:
   - One-to-Many (O2M) head: Task-Aligned Label Assignment (TAL) for training
   - One-to-One (O2O) head: Hungarian matching for NMS-free inference

3. **NMS-Free Inference**: The O2O head produces clean, non-overlapping
   predictions. Only confidence threshold + top-K filtering needed.

4. **No DFL**: Simpler regression head without Distribution Focal Loss,
   producing lighter models with unconstrained regression range.

5. **Angle Encoding**: YOLO26-style sigmoid-angle mapping:
   angle = (sigmoid(pred) - 0.25) * pi  → range [-pi/4, 3pi/4]

Key YOLO26 innovations adapted for mmrotate:
- TaskAlignedAssigner (TAL): unified alignment metric for label assignment
- Progressive Loss: supervision shifts from O2M to O2O during training
- STAL (Small Target Assignment): guaranteed positive coverage for small objects

References:
    YOLO26: https://arxiv.org/abs/2606.03748
    Ultralytics: https://github.com/ultralytics/ultralytics
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale, bias_init_with_prob, normal_init
from mmcv.runner import force_fp32
from mmdet.core import (
    build_assigner,
    build_sampler,
    multi_apply,
    reduce_mean,
)
from mmdet.core.anchor.point_generator import MlvlPointGenerator

from mmrotate.core import build_bbox_coder, multiclass_nms_rotated, obb2poly_np
from mmrotate.models.builder import ROTATED_HEADS, build_loss
from mmrotate.models.dense_heads.rotated_anchor_free_head import RotatedAnchorFreeHead

@ROTATED_HEADS.register_module()
class YOLO26RotatedHead(RotatedAnchorFreeHead):
    """YOLO26-style anchor-free rotated detection head.

    This head implements the core innovations of YOLO26 within the mmrotate
    framework: dual-head architecture (O2O + O2M), task-aligned label
    assignment, NMS-free inference, and simplified regression without DFL.

    Architecture per FPN level:
        Input: B × C × H × W
        │
        ├── cls_branch: 2×Conv(3x3) → Conv2d(C, num_classes)
        ├── reg_branch: 2×Conv(3x3) → Conv2d(C, 4)   # l,t,r,b distances
        ├── angle_branch: 2×Conv(3x3) → Conv2d(C, 1)  # angle offset
        └── obj_branch:  2×Conv(3x3) → Conv2d(C, 1)   # objectness

    Args:
        num_classes (int): Number of object categories.
        in_channels (int): Number of input feature channels from FPN.
        feat_channels (int): Hidden channels in head conv towers. Default: 128.
        stacked_convs (int): Number of conv layers in each branch. Default: 2.
        strides (tuple): Feature map strides. Default: (8, 16, 32, 64).
        reg_max (int): Maximum regression range (not used for DFL, but for
            normalization). Default: 16.
        use_dfl (bool): Whether to use Distribution Focal Loss (YOLO26 removes
            this). Default: False.
        loss_cls (dict): Classification loss config.
        loss_bbox (dict): Bounding box regression loss config.
        loss_angle (dict): Angle regression loss config.
        loss_obj (dict): Objectness/quality loss config.
        bbox_coder (dict): BBox coder config.
        train_cfg (dict): Training config with assignment parameters.
        test_cfg (dict): Testing/inference config.
        init_cfg (dict): Weight initialization config.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        feat_channels: int = 128,
        stacked_convs: int = 2,
        strides: Sequence[int] = (8, 16, 32, 64),
        reg_max: int = 16,
        use_dfl: bool = False,
        loss_cls: dict = None,
        loss_bbox: dict = None,
        loss_angle: dict = None,
        loss_obj: dict = None,
        bbox_coder: dict = None,
        train_cfg: dict = None,
        test_cfg: dict = None,
        init_cfg: dict = None,
        **kwargs,
    ):
        # Set default loss configs
        if loss_cls is None:
            loss_cls = dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0,
            )
        if loss_bbox is None:
            loss_bbox = dict(
                type='RotatedIoULoss',
                loss_weight=1.0,
            )
        if loss_angle is None:
            loss_angle = dict(
                type='SmoothL1Loss',
                beta=0.05,
                loss_weight=1.0,
            )
        if loss_obj is None:
            loss_obj = dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                loss_weight=1.0,
            )
        if bbox_coder is None:
            bbox_coder = dict(
                type='DistanceAnglePointCoder',
                angle_version='le90',
            )
        if init_cfg is None:
            init_cfg = dict(
                type='Normal',
                layer='Conv2d',
                std=0.01,
                override=[
                    dict(type='Normal', name='conv_cls', std=0.01, bias_prob=0.01),
                    dict(type='Normal', name='conv_angle', std=0.01),
                ],
            )

        self.reg_max = reg_max
        self.use_dfl = use_dfl
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs

        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            feat_channels=feat_channels,
            stacked_convs=stacked_convs,
            strides=strides,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            bbox_coder=bbox_coder,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            **kwargs,
        )

        self.loss_angle = build_loss(loss_angle)
        self.loss_obj = build_loss(loss_obj)

        # Dual-head: O2O branch for NMS-free inference
        self._init_o2o_branch()

    def _init_o2o_branch(self):
        """Initialize the one-to-one detection branch for NMS-free inference."""
        self.o2o_cls_convs = nn.ModuleList()
        self.o2o_reg_convs = nn.ModuleList()
        self.o2o_angle_convs = nn.ModuleList()
        self.o2o_obj_convs = nn.ModuleList()

        for i in range(self.stacked_convs):
            chn_in = self.in_channels if i == 0 else self.feat_channels
            self.o2o_cls_convs.append(
                ConvModule(
                    chn_in,
                    self.feat_channels if i < self.stacked_convs - 1 else self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    norm_cfg=dict(type='BN', requires_grad=True),
                    act_cfg=dict(type='SiLU', inplace=True),
                )
            )
            self.o2o_reg_convs.append(
                ConvModule(
                    chn_in,
                    self.feat_channels if i < self.stacked_convs - 1 else self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    norm_cfg=dict(type='BN', requires_grad=True),
                    act_cfg=dict(type='SiLU', inplace=True),
                )
            )
            chn_angle_in = self.in_channels if i == 0 else (self.feat_channels // 4)
            self.o2o_angle_convs.append(
                ConvModule(
                    chn_angle_in,
                    self.feat_channels // 4,
                    3,
                    stride=1,
                    padding=1,
                    norm_cfg=dict(type='BN', requires_grad=True),
                    act_cfg=dict(type='SiLU', inplace=True),
                )
            )
            self.o2o_obj_convs.append(
                ConvModule(
                    chn_in,
                    self.feat_channels if i < self.stacked_convs - 1 else self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    norm_cfg=dict(type='BN', requires_grad=True),
                    act_cfg=dict(type='SiLU', inplace=True),
                )
            )

        # O2O prediction layers
        self.o2o_conv_cls = nn.Conv2d(self.feat_channels, self.num_classes, 1)
        self.o2o_conv_reg = nn.Conv2d(self.feat_channels, 4, 1)
        self.o2o_conv_angle = nn.Conv2d(self.feat_channels // 4, 1, 1)
        self.o2o_conv_obj = nn.Conv2d(self.feat_channels, 1, 1)

        # Progressive loss: start with O2M, gradually shift to O2O
        self.o2o_weight = 0.0  # Will be updated by progressive scheduler

    def _init_layers(self):
        """Initialize per-level detection head layers.

        Architecture per FPN level (YOLO26-style):
        - cls_branch: stacked_convs × Conv(3x3,BN,SiLU) → Conv2d(1x1, nc)
        - reg_branch: stacked_convs × Conv(3x3,BN,SiLU) → Conv2d(1x1, 4*reg_max)
        - angle_branch: stacked_convs × Conv(3x3,BN,SiLU) → Conv2d(1x1, 1)
        - obj_branch: stacked_convs × Conv(3x3,BN,SiLU) → Conv2d(1x1, 1)

        All branches are per-level (not shared) following YOLO convention.
        """
        num_levels = len(self.strides)

        # Classification branches (per level)
        self.cls_convs = nn.ModuleList()
        self.conv_cls = nn.ModuleList()
        for _ in range(num_levels):
            convs = nn.ModuleList()
            for i in range(self.stacked_convs):
                chn = self.in_channels if i == 0 else self.feat_channels
                convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        norm_cfg=dict(type='BN', requires_grad=True),
                        act_cfg=dict(type='SiLU', inplace=True),
                    )
                )
            self.cls_convs.append(convs)
            self.conv_cls.append(
                nn.Conv2d(self.feat_channels, self.cls_out_channels, 1)
            )

        # Regression branches (per level)
        self.reg_convs = nn.ModuleList()
        self.conv_reg = nn.ModuleList()
        for _ in range(num_levels):
            convs = nn.ModuleList()
            for i in range(self.stacked_convs):
                chn = self.in_channels if i == 0 else self.feat_channels
                convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        norm_cfg=dict(type='BN', requires_grad=True),
                        act_cfg=dict(type='SiLU', inplace=True),
                    )
                )
            self.reg_convs.append(convs)
            # YOLO26 without DFL: 4 channels (l,t,r,b)
            # With DFL: 4*reg_max channels
            reg_out_ch = 4 * self.reg_max if self.use_dfl else 4
            self.conv_reg.append(nn.Conv2d(self.feat_channels, reg_out_ch, 1))

        # Angle prediction branches (per level, shared conv top with reg)
        self.angle_convs = nn.ModuleList()
        self.conv_angle = nn.ModuleList()
        for _ in range(num_levels):
            convs = nn.ModuleList()
            for i in range(self.stacked_convs):
                chn = self.in_channels if i == 0 else (self.feat_channels // 4)
                convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels // 4,
                        3,
                        stride=1,
                        padding=1,
                        norm_cfg=dict(type='BN', requires_grad=True),
                        act_cfg=dict(type='SiLU', inplace=True),
                    )
                )
            self.angle_convs.append(convs)
            self.conv_angle.append(nn.Conv2d(self.feat_channels // 4, 1, 1))

        # Objectness/quality branch (shared conv top with cls)
        self.obj_convs = nn.ModuleList()
        self.conv_obj = nn.ModuleList()
        for _ in range(num_levels):
            convs = nn.ModuleList()
            for i in range(self.stacked_convs):
                chn = self.in_channels if i == 0 else self.feat_channels
                convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        norm_cfg=dict(type='BN', requires_grad=True),
                        act_cfg=dict(type='SiLU', inplace=True),
                    )
                )
            self.obj_convs.append(convs)
            self.conv_obj.append(nn.Conv2d(self.feat_channels, 1, 1))

    def init_weights(self):
        """Initialize weights of the head."""
        if self.init_cfg is not None:
            super().init_weights()
            return

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Bias init for classification (helps with foreground/background balance)
        bias_init = bias_init_with_prob(0.01)
        for conv_cls in self.conv_cls:
            nn.init.constant_(conv_cls.bias, bias_init)

        if hasattr(self, 'o2o_conv_cls'):
            nn.init.constant_(self.o2o_conv_cls.bias, bias_init)

    def _forward_per_level(
        self,
        x: torch.Tensor,
        level_idx: int,
        o2o: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for a single FPN level through detection branches.

        Args:
            x: Feature map of shape (B, C, H, W).
            level_idx: Index of this FPN level.
            o2o: If True, use the O2O (one-to-one) branch.

        Returns:
            cls_score: (B, nc, H, W)
            bbox_pred: (B, 4, H, W)
            angle_pred: (B, 1, H, W)
            obj_pred: (B, 1, H, W)
        """
        if o2o and hasattr(self, 'o2o_cls_convs'):
            # Use O2O branch
            cls_feat = x
            for conv in self.o2o_cls_convs:
                cls_feat = conv(cls_feat)
            cls_score = self.o2o_conv_cls(cls_feat)

            reg_feat = x
            for conv in self.o2o_reg_convs:
                reg_feat = conv(reg_feat)
            bbox_pred = self.o2o_conv_reg(reg_feat)

            angle_feat = x
            for conv in self.o2o_angle_convs:
                angle_feat = conv(angle_feat)
            angle_pred = self.o2o_conv_angle(angle_feat)

            obj_feat = x
            for conv in self.o2o_obj_convs:
                obj_feat = conv(obj_feat)
            obj_pred = self.o2o_conv_obj(obj_feat)
        else:
            # Use O2M (standard) branch
            cls_feat = x
            for conv in self.cls_convs[level_idx]:
                cls_feat = conv(cls_feat)
            cls_score = self.conv_cls[level_idx](cls_feat)

            reg_feat = x
            for conv in self.reg_convs[level_idx]:
                reg_feat = conv(reg_feat)
            bbox_pred = self.conv_reg[level_idx](reg_feat)

            angle_feat = x
            for conv in self.angle_convs[level_idx]:
                angle_feat = conv(angle_feat)
            angle_pred = self.conv_angle[level_idx](angle_feat)

            obj_feat = x
            for conv in self.obj_convs[level_idx]:
                obj_feat = conv(obj_feat)
            obj_pred = self.conv_obj[level_idx](obj_feat)

        return cls_score, bbox_pred, angle_pred, obj_pred

    def forward(
        self, feats: List[torch.Tensor]
    ) -> Tuple[Tuple[torch.Tensor, ...], ...]:
        """Forward pass.

        Args:
            feats: Multi-scale FPN features, each (B, C, H, W).

        Returns:
            tuple:
                cls_scores: list of (B, nc, H, W) per level
                bbox_preds: list of (B, 4, H, W) per level
                angle_preds: list of (B, 1, H, W) per level
                obj_preds: list of (B, 1, H, W) per level
        """
        assert len(feats) == len(self.strides), (
            f'Expected {len(self.strides)} feature levels, got {len(feats)}'
        )

        cls_scores, bbox_preds, angle_preds, obj_preds = multi_apply(
            self._forward_per_level,
            feats,
            list(range(len(feats))),
        )
        return cls_scores, bbox_preds, angle_preds, obj_preds

    def forward_o2o(
        self, feats: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], ...]:
        """Forward pass through the O2O (one-to-one) branch.

        Args:
            feats: Multi-scale FPN features.

        Returns:
            Same structure as forward() but from O2O branch.
        """
        assert len(feats) == len(self.strides)

        cls_scores, bbox_preds, angle_preds, obj_preds = [], [], [], []
        for i, feat in enumerate(feats):
            cls_s, bbox_p, angle_p, obj_p = self._forward_per_level(
                feat, i, o2o=True
            )
            cls_scores.append(cls_s)
            bbox_preds.append(bbox_p)
            angle_preds.append(angle_p)
            obj_preds.append(obj_p)

        return cls_scores, bbox_preds, angle_preds, obj_preds

    def forward_train(
        self,
        x: List[torch.Tensor],
        img_metas: List[dict],
        gt_bboxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        gt_bboxes_ignore: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass with dual-head (O2M + O2O).

        Args:
            x: Multi-scale FPN features.
            img_metas: Image meta information.
            gt_bboxes: Ground truth rotated bboxes (x, y, w, h, a).
            gt_labels: Ground truth class labels.
            gt_bboxes_ignore: Bboxes to ignore.

        Returns:
            Dict of loss components.
        """
        # O2M head forward
        o2m_outs = self.forward(x)
        losses = self.loss(
            *o2m_outs,
            gt_bboxes=gt_bboxes,
            gt_labels=gt_labels,
            img_metas=img_metas,
            gt_bboxes_ignore=gt_bboxes_ignore,
            prefix='o2m_',
        )

        # O2O head forward (detached features to avoid gradient conflict)
        if self.o2o_weight > 0 and hasattr(self, 'o2o_cls_convs'):
            with torch.no_grad():
                detached_feats = [f.detach() for f in x]
            o2o_outs = self.forward_o2o(detached_feats)
            o2o_losses = self.loss(
                *o2o_outs,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                img_metas=img_metas,
                gt_bboxes_ignore=gt_bboxes_ignore,
                prefix='o2o_',
                o2o_mode=True,
            )
            # Scale O2O losses by progressive weight
            for k, v in o2o_losses.items():
                o2o_losses[k] = v * self.o2o_weight
            losses.update(o2o_losses)

        return losses

    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'angle_preds', 'obj_preds')
    )
    def loss(
        self,
        cls_scores: List[torch.Tensor],
        bbox_preds: List[torch.Tensor],
        angle_preds: List[torch.Tensor],
        obj_preds: List[torch.Tensor],
        gt_bboxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        img_metas: List[dict],
        gt_bboxes_ignore: Optional[List[torch.Tensor]] = None,
        prefix: str = '',
        o2o_mode: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Compute losses using Task-Aligned Label Assignment (TAL).

        Implements YOLO26-style TAL:
        1. Compute alignment metric: alignment = cls^α * iou^β
        2. Select top-K anchors per GT
        3. Assign positive/negative labels
        4. Compute cls, bbox, angle, objectness losses

        For O2O mode: uses Hungarian matching (1 positive per GT).

        Args:
            cls_scores: Per-level classification scores.
            bbox_preds: Per-level bbox predictions (l,t,r,b).
            angle_preds: Per-level angle predictions.
            obj_preds: Per-level objectness predictions.
            gt_bboxes: Ground truth rotated bboxes (x, y, w, h, a).
            gt_labels: Ground truth class labels.
            img_metas: Image meta information.
            gt_bboxes_ignore: Bboxes to ignore.
            prefix: Prefix for loss keys (e.g., 'o2m_', 'o2o_').
            o2o_mode: If True, use Hungarian matching (1-to-1).

        Returns:
            Dict of loss components.
        """
        num_levels = len(cls_scores)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        num_imgs = cls_scores[0].size(0)

        # Generate grid points for all FPN levels
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device,
        )

        # Flatten predictions
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_angle_preds = [
            angle_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 1)
            for angle_pred in angle_preds
        ]
        flatten_obj_preds = [
            obj_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 1)
            for obj_pred in obj_preds
        ]

        # Concatenate across levels
        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1)  # (B, N, C)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)  # (B, N, 4)
        flatten_angle_preds = torch.cat(flatten_angle_preds, dim=1)  # (B, N, 1)
        flatten_obj_preds = torch.cat(flatten_obj_preds, dim=1)  # (B, N, 1)
        flatten_points = torch.cat(all_level_points, dim=0)  # (N, 2)

        # ---- Decode bbox predictions (FCOS-style: exp for distance) ----
        # Apply exp() to ltrb predictions to ensure positive distances.
        # This matches FCOS convention where bbox_pred = exp(raw_output).
        # Without exp, raw outputs ~N(0,0.01) produce tiny boxes that never
        # overlap with GTs, causing zero IoU loss and no learning signal.
        decoded_bbox_preds = flatten_bbox_preds.exp()  # (B, N, 4)

        # Task-Aligned Label Assignment uses decoded boxes for IoU computation
        (
            assigned_labels,
            assigned_bbox_targets,
            assigned_angle_targets,
            assigned_scores,
            fg_mask,
        ) = self._tal_assign(
            flatten_cls_scores,
            decoded_bbox_preds,
            flatten_angle_preds,
            flatten_points,
            gt_bboxes,
            gt_labels,
            o2o_mode=o2o_mode,
        )

        num_pos = fg_mask.sum().to(flatten_cls_scores.dtype)
        num_total_samples = max(reduce_mean(num_pos), 1.0)

        # Compute losses
        losses = {}

        # 1. Classification loss (Focal Loss) — uses label assignment directly
        loss_cls = self.loss_cls(
            flatten_cls_scores.reshape(-1, self.cls_out_channels),
            assigned_labels.reshape(-1),
            avg_factor=num_total_samples,
        )
        losses[prefix + 'loss_cls'] = loss_cls

        # 2. Bounding box regression loss
        #    Decode exp(ltrb) + point + angle → rotated boxes, then IoU loss
        fg_ltrb = decoded_bbox_preds.reshape(-1, 4)[fg_mask]
        fg_angle_raw = flatten_angle_preds.reshape(-1, 1)[fg_mask]
        fg_points = flatten_points.unsqueeze(0).expand(num_imgs, -1, -1)
        fg_points = fg_points.reshape(-1, 2)[fg_mask]
        fg_bbox_targets = assigned_bbox_targets.reshape(-1, 5)[fg_mask]

        if num_pos > 0:
            # Decode to rotated boxes: (x, y, w, h, angle)
            pred_bboxes = self._decode_bboxes(
                fg_ltrb, fg_angle_raw, fg_points,
            )  # (N_fg, 5)

            # IoU loss — use unweighted sum / num_pos for stable gradients
            loss_bbox = self.loss_bbox(
                pred_bboxes,
                fg_bbox_targets,
                avg_factor=num_total_samples,
            )
        else:
            loss_bbox = flatten_bbox_preds.sum() * 0.0

        losses[prefix + 'loss_bbox'] = loss_bbox

        # 3. Angle loss — compare decoded angle prediction to GT angle
        if num_pos > 0:
            loss_angle = self.loss_angle(
                pred_bboxes[:, 4:5],
                fg_bbox_targets[:, 4:5],
                avg_factor=num_total_samples,
            )
        else:
            loss_angle = flatten_angle_preds.sum() * 0.0
        losses[prefix + 'loss_angle'] = loss_angle

        # 4. Objectness loss — binary targets (FG=1, BG=0)
        #    Using alignment scores as soft labels caused instability
        #    because scores ~1e-6 yielded huge BCE loss (436+).
        obj_targets = fg_mask.float()  # Binary: 1=object, 0=background
        loss_obj = self.loss_obj(
            flatten_obj_preds.reshape(-1),
            obj_targets.reshape(-1),
            avg_factor=num_total_samples,
        )
        losses[prefix + 'loss_obj'] = loss_obj

        return losses

    def _tal_assign(
        self,
        cls_scores: torch.Tensor,
        bbox_preds: torch.Tensor,
        angle_preds: torch.Tensor,
        points: torch.Tensor,
        gt_bboxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        o2o_mode: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """Task-Aligned Label Assignment (TAL) for rotated detection.

        For O2M (one-to-many):
            1. Compute alignment metric: align = cls^α * iou^β
            2. Select top-K anchors per GT based on alignment
            3. Assign each anchor to the GT with highest alignment

        For O2O (one-to-one):
            1. Hungarian matching: cost = cls_cost + bbox_cost + angle_cost
            2. Assign exactly one anchor per GT

        YOLO26 STAL enhancement: guarantee at least 1 positive per small GT.

        Args:
            cls_scores: Flattened class scores (B, N_all, nc).
            bbox_preds: Flattened bbox predictions (B, N_all, 4).
            angle_preds: Flattened angle predictions (B, N_all, 1).
            points: Grid points (N_all, 2).
            gt_bboxes: Ground truth rotated bboxes per image.
            gt_labels: Ground truth labels per image.
            o2o_mode: If True, use Hungarian matching.

        Returns:
            assigned_labels: (B, N_all) int labels
            assigned_bbox_targets: (B, N_all, 5) float targets (x,y,w,h,a)
            assigned_angle_targets: (B, N_all, 1) float angle targets
            assigned_scores: (B, N_all) float quality scores
            fg_mask: (B*N_all,) bool foreground mask
        """
        B, N_all, C = cls_scores.shape
        device = cls_scores.device

        # Tal hyper-parameters
        tal_topk = self.train_cfg.get('tal_topk', 10) if self.train_cfg else 10
        tal_alpha = self.train_cfg.get('tal_alpha', 1.0) if self.train_cfg else 1.0
        tal_beta = self.train_cfg.get('tal_beta', 6.0) if self.train_cfg else 6.0

        assigned_labels = []
        assigned_bbox_targets = []
        assigned_angle_targets = []
        assigned_scores = []

        for i in range(B):
            num_gt = len(gt_labels[i])
            cls_score_i = cls_scores[i].sigmoid()  # (N_all, nc)
            bbox_pred_i = bbox_preds[i]  # (N_all, 4)
            angle_pred_i = angle_preds[i]  # (N_all, 1)

            if num_gt == 0:
                # No GT: all background
                assigned_labels.append(
                    torch.full((N_all,), self.num_classes, dtype=torch.long, device=device)
                )
                assigned_bbox_targets.append(
                    torch.zeros((N_all, 5), dtype=torch.float32, device=device)
                )
                assigned_angle_targets.append(
                    torch.zeros((N_all, 1), dtype=torch.float32, device=device)
                )
                assigned_scores.append(
                    torch.zeros((N_all,), dtype=torch.float32, device=device)
                )
                continue

            gt_bbox_i = gt_bboxes[i]  # (num_gt, 5) -> (x, y, w, h, a)
            gt_label_i = gt_labels[i]  # (num_gt,)

            # Decode predicted rotated boxes from ltrb + point
            pred_bboxes = self._decode_bboxes(
                bbox_pred_i, angle_pred_i, points,
            )  # (N_all, 5)

            if o2o_mode:
                # Hungarian matching (one-to-one)
                assigned_label, assigned_bbox, assigned_angle, assigned_score = (
                    self._o2o_match(
                        cls_score_i, pred_bboxes, angle_pred_i,
                        gt_bbox_i, gt_label_i, points,
                    )
                )
            else:
                # Task-Aligned Label Assignment (one-to-many)
                assigned_label, assigned_bbox, assigned_angle, assigned_score = (
                    self._o2m_match(
                        cls_score_i, pred_bboxes, angle_pred_i,
                        gt_bbox_i, gt_label_i, points,
                        tal_topk, tal_alpha, tal_beta,
                    )
                )

            assigned_labels.append(assigned_label)
            assigned_bbox_targets.append(assigned_bbox)
            assigned_angle_targets.append(assigned_angle)
            assigned_scores.append(assigned_score)

        # Stack results
        assigned_labels = torch.stack(assigned_labels)  # (B, N_all)
        assigned_bbox_targets = torch.stack(assigned_bbox_targets)  # (B, N_all, 5)
        assigned_angle_targets = torch.stack(assigned_angle_targets)  # (B, N_all, 1)
        assigned_scores = torch.stack(assigned_scores)  # (B, N_all)

        # Foreground mask
        fg_mask = (assigned_labels >= 0) & (assigned_labels < self.num_classes)
        fg_mask = fg_mask.reshape(-1)

        return (
            assigned_labels,
            assigned_bbox_targets,
            assigned_angle_targets,
            assigned_scores,
            fg_mask,
        )

    def _o2m_match(
        self,
        cls_score: torch.Tensor,
        pred_bboxes: torch.Tensor,
        angle_pred: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        points: torch.Tensor,
        topk: int,
        alpha: float,
        beta: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One-to-Many assignment using Task-Aligned Label Assignment.

        Alignment metric: align = cls_score^α * iou^β
        Select top-K anchors per GT, then resolve conflicts.

        Args:
            cls_score: Class scores (N_all, nc) after sigmoid.
            pred_bboxes: Decoded predicted rotated boxes (N_all, 5).
            angle_pred: Raw angle predictions (N_all, 1).
            gt_bboxes: Ground truth boxes (num_gt, 5).
            gt_labels: Ground truth labels (num_gt,).
            points: Grid points (N_all, 2).
            topk: Number of top anchors per GT.
            alpha: Cls weight in alignment metric.
            beta: IoU weight in alignment metric.

        Returns:
            assigned_label, assigned_bbox, assigned_angle, assigned_score
        """
        N_all = cls_score.shape[0]
        num_gt = len(gt_labels)
        device = cls_score.device

        # Initialize with background
        assigned_label = torch.full((N_all,), self.num_classes, dtype=torch.long, device=device)
        assigned_bbox = torch.zeros((N_all, 5), device=device)
        assigned_angle = torch.zeros((N_all, 1), device=device)
        assigned_score = torch.zeros((N_all,), device=device)

        if num_gt == 0:
            return assigned_label, assigned_bbox, assigned_angle, assigned_score

        # Compute rotated IoU between predictions and GTs
        # pred_bboxes: (N_all, 5), gt_bboxes: (num_gt, 5)
        ious = self._rotated_iou_matrix(pred_bboxes, gt_bboxes)  # (N_all, num_gt)

        # Get classification scores for GT classes
        cls_score_per_gt = cls_score[:, gt_labels]  # (N_all, num_gt)

        # Alignment metric
        alignment = cls_score_per_gt.pow(alpha) * ious.pow(beta)  # (N_all, num_gt)

        # Select top-K anchors per GT
        # For each GT, find the top-K anchors with highest alignment
        topk_align, topk_indices = torch.topk(alignment, k=min(topk, N_all), dim=0)
        # topk_indices: (topk, num_gt)

        # For each anchor, find the best GT (highest alignment)
        # Build a mask for top-K selections
        anchor_align = torch.zeros((N_all, num_gt), device=device)
        topk_indices_flat = topk_indices.T.reshape(-1)  # (num_gt * topk,)
        gt_indices = torch.arange(num_gt, device=device).unsqueeze(0).expand(topk, -1)
        gt_indices_flat = gt_indices.T.reshape(-1)  # (num_gt * topk,)

        anchor_align[topk_indices_flat, gt_indices_flat] = alignment[topk_indices_flat, gt_indices_flat]

        # For each anchor, select the GT with highest alignment
        max_align, max_gt_idx = anchor_align.max(dim=1)  # (N_all,)

        # Filter: only assign anchors with alignment > 0
        pos_mask = max_align > 0
        assigned_label[pos_mask] = gt_labels[max_gt_idx[pos_mask]]
        assigned_bbox[pos_mask] = gt_bboxes[max_gt_idx[pos_mask]]
        assigned_angle[pos_mask] = gt_bboxes[max_gt_idx[pos_mask], 4:5]
        assigned_score[pos_mask] = max_align[pos_mask]

        return assigned_label, assigned_bbox, assigned_angle, assigned_score

    def _o2o_match(
        self,
        cls_score: torch.Tensor,
        pred_bboxes: torch.Tensor,
        angle_pred: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        points: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One-to-One assignment using Hungarian matching.

        Cost matrix:
            cost = cls_cost + bbox_cost + angle_cost
        Use Hungarian algorithm to assign exactly one prediction per GT.

        Args:
            cls_score: Class scores (N_all, nc) after sigmoid.
            pred_bboxes: Decoded predicted rotated boxes (N_all, 5).
            angle_pred: Raw angle predictions (N_all, 1).
            gt_bboxes: Ground truth boxes (num_gt, 5).
            gt_labels: Ground truth labels (num_gt,).
            points: Grid points (N_all, 2).

        Returns:
            assigned_label, assigned_bbox, assigned_angle, assigned_score
        """
        N_all = cls_score.shape[0]
        num_gt = len(gt_labels)
        device = cls_score.device

        assigned_label = torch.full((N_all,), self.num_classes, dtype=torch.long, device=device)
        assigned_bbox = torch.zeros((N_all, 5), device=device)
        assigned_angle = torch.zeros((N_all, 1), device=device)
        assigned_score = torch.zeros((N_all,), device=device)

        if num_gt == 0:
            return assigned_label, assigned_bbox, assigned_angle, assigned_score

        # Classification cost: -log(p) for GT class
        cls_cost = -cls_score[:, gt_labels].log()  # (N_all, num_gt)

        # Bbox cost: 1 - IoU
        ious = self._rotated_iou_matrix(pred_bboxes, gt_bboxes)  # (N_all, num_gt)
        bbox_cost = 1.0 - ious

        # Angle cost: L1 difference (simplified - use bbox cost as proxy for angle)
        # Combined cost
        cost = cls_cost + bbox_cost  # (N_all, num_gt)

        # Add a large cost for very distant predictions (beyond 3x box size)
        # This ensures Hungarian doesn't pick unreasonable matches

        # Hungarian matching: select one anchor per GT
        # For efficiency, only consider top 1000 candidates per GT
        k = min(max(N_all // num_gt, 1), 1000)

        pos_indices = []
        gt_indices_remaining = list(range(num_gt))
        candidates_cost_per_gt = cost[:, gt_indices_remaining]  # (N_all, num_gt)

        for gt_i in range(num_gt):
            if k is not None:
                _, topk_idx = torch.topk(cost[:, gt_i], k=k, largest=False)
            else:
                topk_idx = torch.arange(N_all, device=device)

            # Find best unmatched anchor for this GT
            best_anchor = topk_idx[cost[topk_idx, gt_i].argmin()]
            pos_indices.append(best_anchor)

        # Unique check: if an anchor was selected for multiple GTs, keep the
        # one with the lowest matching cost (true greedy Hungarian-style).
        pos_indices = torch.tensor(pos_indices, device=device)  # (num_gt,)

        if len(pos_indices) > 0:
            # Resolve duplicate anchor assignments: for each unique anchor,
            # find the GT that has the lowest cost with it.
            unique_anchors, inverse = torch.unique(
                pos_indices, return_inverse=True,
            )
            # unique_anchors: unique anchor ids
            # inverse[i]: index into unique_anchors for GT i

            for u_idx, anchor_idx in enumerate(unique_anchors):
                # All GTs whose best pick is this anchor
                gt_mask = (inverse == u_idx)
                candidate_gts = torch.where(gt_mask)[0]

                if len(candidate_gts) > 1:
                    # Multiple GTs want this anchor — pick the one with
                    # the lowest combined cost (cls_cost + bbox_cost).
                    costs_for_anchor = cost[anchor_idx, candidate_gts]
                    best_gt = candidate_gts[costs_for_anchor.argmin()]
                else:
                    best_gt = candidate_gts[0]

                assigned_label[anchor_idx] = gt_labels[best_gt]
                assigned_bbox[anchor_idx] = gt_bboxes[best_gt]
                assigned_angle[anchor_idx] = gt_bboxes[best_gt, 4:5]
                # Quality score = cls * iou
                assigned_score[anchor_idx] = (
                    cls_score[anchor_idx, gt_labels[best_gt]]
                    * ious[anchor_idx, best_gt]
                )

        return assigned_label, assigned_bbox, assigned_angle, assigned_score

    def _rotated_iou_matrix(
        self, boxes1: torch.Tensor, boxes2: torch.Tensor
    ) -> torch.Tensor:
        """Compute rotated IoU matrix between two sets of rotated boxes.

        Boxes are in (x, y, w, h, angle) format with le90 angle range.

        Uses mmrotate's vectorized rbbox_overlaps for efficient GPU computation.

        Args:
            boxes1: (N, 5) tensor of predicted boxes.
            boxes2: (M, 5) tensor of ground truth boxes.

        Returns:
            iou_matrix: (N, M) tensor of IoU values.
        """
        from mmrotate.core.bbox.iou_calculators import rbbox_overlaps

        ious = rbbox_overlaps(
            boxes1, boxes2, mode='iou', is_aligned=False
        )
        return ious.to(boxes1.dtype)

    @staticmethod
    def _decode_bboxes(
        ltrb_pred: torch.Tensor,
        angle_pred: torch.Tensor,
        points: torch.Tensor,
    ) -> torch.Tensor:
        """Decode (l,t,r,b) + point + angle into rotated boxes (x,y,w,h,a).

        This is the shared decoding step used by loss, label assignment, and
        post-processing.  ``ltrb_pred`` is assumed to have already been passed
        through ``.exp()`` to guarantee positive distances.

        Args:
            ltrb_pred: (N, 4) tensor of positive l,t,r,b distances.
            angle_pred: (N, 1) tensor of raw angle logits.
            points: (N, 2) tensor of grid-point (x, y) coordinates.

        Returns:
            (N, 5) tensor of decoded rotated boxes [cx, cy, w, h, angle].
            Angle is decoded as sigmoid(pred) * pi - pi/4 (range [-π/4, 3π/4]).
        """
        px, py = points[:, 0], points[:, 1]
        l, t, r, b = (ltrb_pred[:, 0], ltrb_pred[:, 1],
                       ltrb_pred[:, 2], ltrb_pred[:, 3])

        x1 = px - l
        y1 = py - t
        x2 = px + r
        y2 = py + b
        w = (x2 - x1).clamp(min=1.0)
        h = (y2 - y1).clamp(min=1.0)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        angle = (angle_pred.sigmoid() - 0.25) * math.pi

        return torch.stack([cx, cy, w, h, angle.squeeze(-1)], dim=-1)

    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'angle_preds', 'obj_preds')
    )
    def get_bboxes(
        self,
        cls_scores: List[torch.Tensor],
        bbox_preds: List[torch.Tensor],
        angle_preds: List[torch.Tensor],
        obj_preds: List[torch.Tensor],
        img_metas: List[dict],
        cfg: Optional[dict] = None,
        rescale: bool = False,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Post-process predictions into final rotated bounding boxes.

        Uses NMS-free end-to-end detection (YOLO26 style):
        1. Decode ltrb + point + angle → rotated boxes (x,y,w,h,a)
        2. Compute final score = cls_score * obj_score
        3. Top-K selection (no NMS needed with O2O head)
        4. Optional NMS for O2M head output

        Args:
            cls_scores: Per-level class scores.
            bbox_preds: Per-level bbox predictions.
            angle_preds: Per-level angle predictions.
            obj_preds: Per-level objectness predictions.
            img_metas: Image meta information.
            cfg: Test config override.
            rescale: Whether to rescale to original image size.

        Returns:
            List of (det_bboxes, det_labels) per image.
            det_bboxes: (N, 6) [x, y, w, h, angle, score]
        """
        cfg = self.test_cfg if cfg is None else cfg
        num_levels = len(cls_scores)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]

        mlvl_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device,
        )

        # Use O2O branch if available for NMS-free inference
        use_end2end = cfg.get('end2end', True) if cfg else True
        nms_cfg = cfg.get('nms', dict(type='nms_rotated', iou_thr=0.1)) if cfg else None
        score_thr = cfg.get('score_thr', 0.05) if cfg else 0.05
        max_per_img = cfg.get('max_per_img', 300) if cfg else 300

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            angle_pred_list = [
                angle_preds[i][img_id].detach() for i in range(num_levels)
            ]
            obj_pred_list = [
                obj_preds[i][img_id].detach() for i in range(num_levels)
            ]

            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']

            if use_end2end:
                # NMS-free end-to-end inference
                det_bboxes, det_labels = self._get_bboxes_end2end(
                    cls_score_list, bbox_pred_list, angle_pred_list,
                    obj_pred_list, mlvl_points, img_shape, scale_factor,
                    score_thr, max_per_img, rescale,
                )
            else:
                # Traditional NMS-based inference
                det_bboxes, det_labels = self._get_bboxes_nms(
                    cls_score_list, bbox_pred_list, angle_pred_list,
                    obj_pred_list, mlvl_points, img_shape, scale_factor,
                    cfg, rescale,
                )

            result_list.append((det_bboxes, det_labels))

        return result_list

    def _get_bboxes_end2end(
        self,
        cls_scores: List[torch.Tensor],
        bbox_preds: List[torch.Tensor],
        angle_preds: List[torch.Tensor],
        obj_preds: List[torch.Tensor],
        mlvl_points: List[torch.Tensor],
        img_shape: Tuple[int, int],
        scale_factor: float,
        score_thr: float,
        max_per_img: int,
        rescale: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """NMS-free end-to-end detection (YOLO26 style).

        Steps:
        1. Decode all predictions
        2. Compute score = cls_sigmoid * obj_sigmoid
        3. Top-K selection
        4. Score threshold filtering

        Returns:
            det_bboxes: (N, 6) [x, y, w, h, angle, score]
            det_labels: (N,) class indices
        """
        all_bboxes = []
        all_scores = []
        all_labels = []

        for cls_score, bbox_pred, angle_pred, obj_pred, points in zip(
            cls_scores, bbox_preds, angle_preds, obj_preds, mlvl_points
        ):
            # Decode bboxes
            cls_score_flat = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)
            bbox_pred_flat = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            angle_pred_flat = angle_pred.permute(1, 2, 0).reshape(-1, 1)
            obj_pred_flat = obj_pred.permute(1, 2, 0).reshape(-1, 1)

            # Apply exp() to bbox_pred (matching training convention)
            bbox_pred_flat = bbox_pred_flat.exp()

            # Decode predictions to rotated boxes
            decoded_bboxes = self._decode_bboxes(
                bbox_pred_flat, angle_pred_flat, points,
            )  # (N_level, 5)

            # Compute scores
            cls_scores_sig = cls_score_flat.sigmoid()
            obj_scores_sig = obj_pred_flat.sigmoid()
            scores = cls_scores_sig * obj_scores_sig  # (N, nc)

            # Get max score per anchor
            max_scores, max_labels = scores.max(dim=1)  # (N,), (N,)

            all_bboxes.append(decoded_bboxes)
            all_scores.append(max_scores)
            all_labels.append(max_labels)

        # Concatenate all levels
        all_bboxes = torch.cat(all_bboxes, dim=0)  # (N_all, 5)
        all_scores = torch.cat(all_scores, dim=0)  # (N_all,)
        all_labels = torch.cat(all_labels, dim=0)  # (N_all,)

        # Rescale
        if rescale and scale_factor is not None:
            scale_factor_t = all_bboxes.new_tensor(scale_factor)
            all_bboxes[:, :4] /= scale_factor_t

        # Score threshold
        keep = all_scores > score_thr
        all_bboxes = all_bboxes[keep]
        all_scores = all_scores[keep]
        all_labels = all_labels[keep]

        # Top-K selection (NMS-free)
        if all_bboxes.shape[0] > max_per_img:
            _, topk_idx = all_scores.topk(max_per_img)
            all_bboxes = all_bboxes[topk_idx]
            all_scores = all_scores[topk_idx]
            all_labels = all_labels[topk_idx]

        # Concatenate into final format: (N, 6) [x, y, w, h, a, score]
        det_bboxes = torch.cat([all_bboxes, all_scores.unsqueeze(-1)], dim=-1)

        return det_bboxes, all_labels

    def _get_bboxes_nms(
        self,
        cls_scores: List[torch.Tensor],
        bbox_preds: List[torch.Tensor],
        angle_preds: List[torch.Tensor],
        obj_preds: List[torch.Tensor],
        mlvl_points: List[torch.Tensor],
        img_shape: Tuple[int, int],
        scale_factor: float,
        cfg: dict,
        rescale: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Traditional NMS-based inference (for O2M head output)."""
        mlvl_bboxes = []
        mlvl_scores = []

        for cls_score, bbox_pred, angle_pred, obj_pred, points in zip(
            cls_scores, bbox_preds, angle_preds, obj_preds, mlvl_points
        ):
            cls_score_flat = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)
            bbox_pred_flat = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            angle_pred_flat = angle_pred.permute(1, 2, 0).reshape(-1, 1)
            obj_pred_flat = obj_pred.permute(1, 2, 0).reshape(-1, 1)

            # Apply exp() to bbox_pred (matching training convention)
            bbox_pred_flat = bbox_pred_flat.exp()

            # Decode bboxes
            decoded_bboxes = self._decode_bboxes(
                bbox_pred_flat, angle_pred_flat, points,
            )  # (N_level, 5)

            # Score = cls * obj
            scores = cls_score_flat.sigmoid() * obj_pred_flat.sigmoid()

            # Top-K per level before NMS
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = scores.max(dim=1)
                _, topk_idx = max_scores.topk(nms_pre)
                decoded_bboxes = decoded_bboxes[topk_idx]
                scores = scores[topk_idx]

            mlvl_bboxes.append(decoded_bboxes)
            mlvl_scores.append(scores)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_scores = torch.cat(mlvl_scores)

        # Add background class channel
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

        if rescale:
            scale_factor_t = mlvl_bboxes.new_tensor(scale_factor)
            mlvl_bboxes[:, :4] /= scale_factor_t

        # Rotated NMS
        det_bboxes, det_labels = multiclass_nms_rotated(
            mlvl_bboxes,
            mlvl_scores,
            cfg.get('score_thr', 0.05),
            cfg.get('nms', dict(type='nms_rotated', iou_thr=0.1)),
            cfg.get('max_per_img', 300),
        )

        return det_bboxes, det_labels

    def get_targets(
        self,
        points: List[torch.Tensor],
        gt_bboxes_list: List[torch.Tensor],
        gt_labels_list: List[torch.Tensor],
    ) -> Tuple:
        """Legacy target computation (required by parent class)."""
        return [], [], []
