# Copyright (c) OpenMMLab. All rights reserved.
"""ViTDet Feature Pyramid Network for ViT backbones.

This module creates multi-scale feature maps from same-resolution ViT features
by upsampling early features and downsampling later features. Supports both
single-scale (num_ins=1, standard ViTDet) and multi-scale (num_ins=4, DINOv3)
input modes.

Unlike standard FPN which expects backbone features at decreasing resolutions
(e.g., 1/4, 1/8, 1/16, 1/32), ViT backbones output features at a single
spatial resolution (typically 1/16 for patch_size=16).

ViTDetFPN builds a proper feature pyramid:
    - Single input mode (num_ins=1):
        P0: upsample(feat)     → stride S/2
        P1: pass-through(feat) → stride S
        P2: downsample(feat)   → stride 2S
        P3: downsample ×2      → stride 4S

    - Multi input mode (num_ins=4):
        P0: upsample(feat[0])    → stride S/2
        P1: pass-through(feat[1])→ stride S
        P2: downsample(feat[2])  → stride 2S
        P3: downsample ×2(feat[3])→ stride 4S

Reference:
    ViTDet: https://arxiv.org/abs/2203.16527
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_norm_layer, xavier_init
from mmcv.runner import BaseModule, auto_fp16

from mmdet.models.builder import NECKS


@NECKS.register_module()
class ViTDetFPN(BaseModule):
    """ViTDet-style FPN for ViT-based backbones.

    Takes backbone features (either single or multiple same-resolution features)
    and creates a multi-scale feature pyramid through deconvolution (upsampling)
    and strided convolution (downsampling).

    Args:
        in_channels (int): Number of input channels from backbone. Default: 256.
        out_channels (int): Number of output channels per level. Default: 256.
        num_outs (int): Number of output feature levels. Default: 4.
        num_ins (int): Number of input features from backbone. Default: 4.
            Set to 1 for standard ViTDet (builds pyramid from single scale),
            Set to 4 for DINOv3 (4 intermediate block outputs).
        start_level (int): Index of the first output level. Default: 0.
        add_extra_convs (bool | str): Whether to add extra conv layers.
        conv_cfg (dict): Config for convolution layers.
        norm_cfg (dict): Config for normalization layers.
        act_cfg (dict): Config for activation layers.
        init_cfg (dict): Initialization config.
    """

    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 256,
        num_outs: int = 4,
        num_ins: int = 4,
        start_level: int = 0,
        add_extra_convs: Union[bool, str] = False,
        conv_cfg: Optional[dict] = None,
        norm_cfg: Optional[dict] = None,
        act_cfg: Optional[dict] = None,
        init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg=init_cfg)

        if norm_cfg is None:
            norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
        if act_cfg is None:
            act_cfg = dict(type='GELU')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_outs = num_outs
        self.num_ins = num_ins
        self.start_level = start_level
        self.add_extra_convs = add_extra_convs

        # Lateral convolutions: 1x1 conv to unify each input's channels
        self.lateral_convs = nn.ModuleList()
        for i in range(self.num_ins):
            l_conv = ConvModule(
                in_channels, out_channels, kernel_size=1,
                norm_cfg=norm_cfg, act_cfg=act_cfg, inplace=False,
            )
            self.lateral_convs.append(l_conv)

        # FPN output convolutions: 3x3 conv on each output level
        # We produce up to num_outs levels, some may be extra
        num_base = min(self.num_ins, num_outs)
        self.fpn_convs = nn.ModuleList()
        for i in range(num_base):
            fpn_conv = ConvModule(
                out_channels, out_channels, kernel_size=3,
                stride=1, padding=1,
                norm_cfg=norm_cfg, act_cfg=act_cfg, inplace=False,
            )
            self.fpn_convs.append(fpn_conv)

        # Upsample module for P0 (stride S/2)
        self.upsample_p0 = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels,
                               kernel_size=2, stride=2, bias=False),
            build_norm_layer(norm_cfg, out_channels)[1],
        )

        # Downsample modules
        self.downsamples = nn.ModuleList()
        for i in range(min(num_outs, self.num_ins + 1)):
            ds = ConvModule(
                out_channels, out_channels, kernel_size=3,
                stride=2, padding=1,
                norm_cfg=norm_cfg, act_cfg=act_cfg, inplace=False,
            )
            self.downsamples.append(ds)

        # Extra downsampling for additional output levels
        extra_levels = max(0, num_outs - num_base)
        self.extra_downsamples = nn.ModuleList()
        for i in range(extra_levels):
            extra_conv = ConvModule(
                out_channels, out_channels, kernel_size=3,
                stride=2, padding=1,
                norm_cfg=norm_cfg, act_cfg=act_cfg, inplace=False,
            )
            self.extra_downsamples.append(extra_conv)

    def init_weights(self):
        """Initialize weights."""
        if self.init_cfg is not None:
            super().init_weights()
            return

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
            elif isinstance(m, nn.ConvTranspose2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        """Forward pass.

        Builds a multi-scale feature pyramid from ViT backbone features.

        Supports both single-scale (1 feature) and multi-scale (4 features) inputs.
        When num_ins=1, all pyramid levels are built from the same feature.
        When num_ins=4, each level uses a different intermediate block output.

        Args:
            inputs (list[Tensor]): Feature maps from ViT backbone.
                num_ins tensors, each (B, C, H, W) at the same resolution.

        Returns:
            tuple[Tensor]: Multi-scale feature maps at different strides.
        """
        # Validate input count
        if len(inputs) != self.num_ins:
            raise AssertionError(
                f'ViTDetFPN expects {self.num_ins} input features, '
                f'got {len(inputs)}. '
                f'Check backbone out_indices or set neck num_ins={len(inputs)}.'
            )

        # Step 1: Lateral convolutions (1x1 channel unification)
        laterals = [
            self.lateral_convs[i](inputs[i])
            for i in range(self.num_ins)
        ]

        # Step 2: Build multi-scale outputs
        outs = []

        if self.num_ins >= 1:
            # P0: Upsample from first lateral feature (stride S/2)
            p0 = self.upsample_p0(laterals[0])
            p0 = self.fpn_convs[0](p0) if len(self.fpn_convs) > 0 else p0
            outs.append(p0)

        if self.num_ins >= 2:
            # P1: Pass-through from second lateral feature (stride S)
            p1 = self.fpn_convs[1](laterals[1]) if len(self.fpn_convs) > 1 else laterals[1]
            outs.append(p1)
        elif self.num_ins == 1:
            # Single input: build P1 from the same feature
            p1 = self.downsamples[0](laterals[0]) if len(self.downsamples) > 0 else laterals[0]
            if len(self.fpn_convs) > 0:
                p1 = self.fpn_convs[0](p1) if len(self.fpn_convs) == 1 else self.fpn_convs[min(1, len(self.fpn_convs)-1)](p1)
            outs.append(p1)

        if self.num_ins >= 3:
            # P2: Downsample from third lateral feature (stride 2S)
            p2 = self.downsamples[0](laterals[2]) if len(self.downsamples) > 0 else laterals[2]
            p2 = self.fpn_convs[2](p2) if len(self.fpn_convs) > 2 else p2
            outs.append(p2)
        elif self.num_ins <= 2 and len(outs) < 3:
            src = laterals[-1] if self.num_ins >= 1 else inputs[0]
            p2 = self.downsamples[min(0, len(self.downsamples)-1)](src)
            outs.append(p2)

        if self.num_ins >= 4:
            # P3: Double downsample from fourth lateral feature (stride 4S)
            p3 = laterals[3]
            for ds_idx in range(min(2, len(self.downsamples))):
                p3 = self.downsamples[min(ds_idx, len(self.downsamples)-1)](p3)
            p3 = self.fpn_convs[3](p3) if len(self.fpn_convs) > 3 else p3
            outs.append(p3)
        elif self.num_ins < 4 and len(outs) < 4:
            src = laterals[-1] if self.num_ins >= 1 else inputs[0]
            p3 = src
            for ds_idx in range(min(3, len(self.downsamples))):
                p3 = self.downsamples[min(ds_idx, len(self.downsamples)-1)](p3)
            outs.append(p3)

        # Trim to num_outs if we generated more than needed
        outs = outs[:self.num_outs]

        # Pad to num_outs if we generated fewer
        while len(outs) < self.num_outs:
            extra_source = laterals[-1] if self.num_ins >= 1 else inputs[-1]
            if self.add_extra_convs == 'on_input':
                extra_feat = extra_source
            else:
                extra_feat = outs[-1]
            if len(self.extra_downsamples) > 0:
                extra_feat = self.extra_downsamples[len(outs) - min(self.num_ins, self.num_outs)](extra_feat)
            else:
                extra_feat = F.max_pool2d(extra_feat, 1, 2, 0)
            outs.append(extra_feat)

        # Step 3: Add extra output levels via further downsampling
        extra_start = max(0, self.num_outs - min(self.num_ins, self.num_outs))
        for i in range(extra_start):
            if i < len(self.extra_downsamples):
                if self.add_extra_convs == 'on_input':
                    extra_source = laterals[-1]
                else:
                    extra_source = outs[-1]
                extra_feat = self.extra_downsamples[i](extra_source)
                if len(outs) < self.num_outs:
                    outs.append(extra_feat)
                    extra_source = extra_feat

        return tuple(outs[:self.num_outs])
