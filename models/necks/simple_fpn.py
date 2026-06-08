# Copyright (c) OpenMMLab. All rights reserved.
"""Simple Feature Pyramid Network for ViT backbones.

This module creates multi-scale feature maps from same-resolution ViT features
by upsampling early features and downsampling later features.

Unlike standard FPN which expects backbone features at decreasing resolutions
(e.g., 1/4, 1/8, 1/16, 1/32), ViT backbones output features at a single
spatial resolution (typically 1/16 for patch_size=16).

SimpleFPN builds a proper feature pyramid by:
    - Level 0 (stride S/2): upsample early feature
    - Level 1 (stride S):   pass-through mid feature
    - Level 2 (stride 2S):  downsample later feature
    - Level 3 (stride 4S):  further downsample

Reference:
    ViTDet: https://arxiv.org/abs/2203.16527
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_norm_layer, xavier_init
from mmcv.runner import BaseModule, auto_fp16

from mmdet.models.builder import NECKS


@NECKS.register_module()
class SimpleFPN(BaseModule):
    """Simple FPN for ViT-based backbones.

    Takes backbone features at the same spatial resolution and creates
    a multi-scale feature pyramid through deconvolution (upsampling)
    and strided convolution (downsampling).

    Architecture (for 4 input features at stride S):
        f0, f1, f2, f3 → all same resolution (H×W)

        After SimpleFPN:
        P0: upsample(f0)         → stride S/2 (2H × 2W)
        P1: pass-through(f1)     → stride S   (H × W)
        P2: downsample(f2)       → stride 2S  (H/2 × W/2)
        P3: downsample(f3 again) → stride 4S  (H/4 × W/4)

    Args:
        in_channels (int): Number of input channels from backbone.
            Default: 256.
        out_channels (int): Number of output channels per level.
            Default: 256.
        num_outs (int): Number of output feature levels.
            Default: 4.
        start_level (int): Index of the first output level. Default: 0.
        add_extra_convs (bool | str): Whether to add extra conv layers
            to produce more levels. Options: False (no extra),
            'on_input' (on original features), 'on_output' (on FPN output).
            Default: False.
        conv_cfg (dict): Config dict for convolution layers.
            Default: None (use nn.Conv2d).
        norm_cfg (dict): Config dict for normalization layers.
            Default: dict(type='GN', num_groups=32).
        act_cfg (dict): Config dict for activation layers.
            Default: dict(type='GELU').
        init_cfg (dict, optional): Initialization config.
            Default: None.

    Example:
        >>> neck = SimpleFPN(in_channels=256, out_channels=256, num_outs=4)
        >>> # Input: 4 features at stride 16
        >>> feats = [torch.randn(2, 256, 50, 50) for _ in range(4)]
        >>> outs = neck(feats)
        >>> for o in outs:
        ...     print(o.shape)
        torch.Size([2, 256, 100, 100])  # stride 8  (upsampled)
        torch.Size([2, 256, 50, 50])    # stride 16 (pass-through)
        torch.Size([2, 256, 25, 25])    # stride 32 (downsampled)
        torch.Size([2, 256, 13, 13])    # stride 64 (downsampled)
    """

    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 256,
        num_outs: int = 4,
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
        self.start_level = start_level
        self.add_extra_convs = add_extra_convs

        # Number of input features (expected 4 from ViT backbone)
        self.num_ins = 4

        # Lateral convolutions: 1x1 conv to unify each input's channels
        self.lateral_convs = nn.ModuleList()
        for i in range(self.num_ins):
            l_conv = ConvModule(
                in_channels, out_channels, kernel_size=1,
                norm_cfg=norm_cfg, act_cfg=act_cfg, inplace=False,
            )
            self.lateral_convs.append(l_conv)

        # FPN output convolutions: 3x3 conv on each output level
        self.fpn_convs = nn.ModuleList()
        for i in range(self.num_ins):
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

        # Downsample modules for P2 and P3
        self.downsample_p2 = ConvModule(
            out_channels, out_channels, kernel_size=3,
            stride=2, padding=1,
            norm_cfg=norm_cfg, act_cfg=act_cfg, inplace=False,
        )
        self.downsample_p3 = ConvModule(
            out_channels, out_channels, kernel_size=3,
            stride=2, padding=1,
            norm_cfg=norm_cfg, act_cfg=act_cfg, inplace=False,
        )

        # Extra downsampling for additional output levels beyond 4
        extra_levels = num_outs - self.num_ins
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

        Builds a multi-scale feature pyramid from same-resolution inputs.

        Args:
            inputs (list[Tensor]): Feature maps from the ViT backbone.
                4 tensors, each (B, C, H, W) at the same resolution.

        Returns:
            tuple[Tensor]: Multi-scale feature maps at different strides.
                P0: upsampled (2× resolution)
                P1: pass-through (1× resolution)
                P2: downsampled (1/2× resolution)
                P3: further downsampled (1/4× resolution)
        """
        assert len(inputs) == self.num_ins, (
            f'SimpleFPN expects {self.num_ins} input features, '
            f'got {len(inputs)}'
        )

        # Step 1: Lateral convolutions (1x1 channel unification)
        laterals = [
            self.lateral_convs[i](inputs[i])
            for i in range(self.num_ins)
        ]

        # Step 2: Build multi-scale outputs
        outs = []

        # P0: Upsample from f0 (stride S/2)
        p0 = self.upsample_p0(laterals[0])
        p0 = self.fpn_convs[0](p0)
        outs.append(p0)

        # P1: Pass-through from f1 (stride S)
        p1 = self.fpn_convs[1](laterals[1])
        outs.append(p1)

        # P2: Downsample from f2 (stride 2S)
        p2 = self.downsample_p2(laterals[2])
        p2 = self.fpn_convs[2](p2)
        outs.append(p2)

        # P3: Downsample from f3 → then downsample again (stride 4S)
        p3 = self.downsample_p3(self.downsample_p2(laterals[3]))
        p3 = self.fpn_convs[3](p3)
        outs.append(p3)

        # Step 3: Add extra output levels via further downsampling
        if len(self.extra_downsamples) > 0:
            if self.add_extra_convs == 'on_input':
                extra_source = laterals[-1]
            else:
                extra_source = outs[-1]

            for extra_conv in self.extra_downsamples:
                if self.add_extra_convs == 'on_input':
                    extra_feat = extra_conv(self.lateral_convs[-1](extra_source))
                else:
                    extra_feat = extra_conv(extra_source)
                outs.append(extra_feat)
                extra_source = extra_feat

        return tuple(outs)
