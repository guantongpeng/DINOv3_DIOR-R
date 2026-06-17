"""ViTDetFPN — Strong Feature Pyramid Neck for ViT Backbones.

Unlike SimpleFPN (which generates scales independently), ViTDetFPN builds
a proper FPN with progressive upsampling, cross-scale top-down fusion with
channel attention, and outputs features at strides [4, 8, 16, 32].

Key improvements over SimpleFPN:
    1. Stride-4 output (256×256 at 1024×1024) for small object detection
    2. Progressive 2× upsampling (bilinear + conv) instead of single deconv
    3. Proper FPN-style top-down pathway with lateral skip connections
    4. Squeeze-and-Excitation channel attention after each fusion
    5. Pre-norm layer to adapt ViT's LayerNorm features to GroupNorm space

Reference:
    ViTDet (Li et al., ECCV 2022): Exploring Plain ViT Backbones for Detection
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_norm_layer, xavier_init
from mmcv.runner import BaseModule, auto_fp16

from mmdet.models.builder import NECKS


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention block."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


class UpsampleBlock(nn.Module):
    """Upsample block: bilinear 2× up + 3×3 conv + GN + GELU."""

    def __init__(self, in_channels: int, out_channels: int, norm_cfg: dict, act_cfg: dict):
        super().__init__()
        self.conv = ConvModule(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1,
            norm_cfg=norm_cfg, act_cfg=act_cfg, inplace=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return self.conv(x)


@NECKS.register_module()
class ViTDetFPN(BaseModule):
    """Feature Pyramid Network designed for ViT backbones.

    Takes 4 ViT backbone features at the same spatial resolution (stride 16)
    and builds a proper multi-scale pyramid through progressive upsampling,
    lateral connections, and top-down cross-scale fusion.

    Architecture::

        ViT features (all stride-16)
        f0(block 3)   f1(block 5)   f2(block 7)   f3(block 11)
            |              |              |              |
        lateral0      lateral1      lateral2      lateral3
            |              |              |              |
            |              |              |          downsample (stride-2)
            |              |              |              |
            |              |          upsample 2× <── P2_out (stride 16)
            |              |              |              |
            |          upsample 2× <── SE + fusion      |
            |              |              |              |
        upsample 2× <── SE + fusion      |              |
            |              |              |              |
        SE + fusion       |              |              |
            |              |              |              |
        P0_out (s4)   P1_out (s8)    P2_out (s16)   P3_out (s32)

    Output strides: [4, 8, 16, 32]
        - P0: 256×256 (for 1024×1024 input) — fine detail for small objects
        - P1: 128×128 — medium objects
        - P2: 64×64 — large objects
        - P3: 32×32 — very large objects

    Args:
        in_channels (int): Number of input channels from backbone. Default: 256.
        out_channels (int): Number of output channels per level. Default: 256.
        num_outs (int): Number of output feature levels. Default: 4.
        start_level (int): Index of the first output level. Default: 0.
        add_extra_convs (bool | str): Add extra conv layers for more outputs.
            Default: False.
        se_reduction (int): Reduction ratio for SE attention. Default: 16.
        norm_cfg (dict): Config for normalization layers.
            Default: dict(type='GN', num_groups=32).
        act_cfg (dict): Config for activation layers.
            Default: dict(type='GELU').
        init_cfg (dict, optional): Weight initialization config.
    """

    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 256,
        num_outs: int = 4,
        start_level: int = 0,
        add_extra_convs: Union[bool, str] = False,
        se_reduction: int = 16,
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
        self.num_ins = 4

        # ViT output → GN adaptation layer (pre-norm)
        # ViT features come from LayerNorm space; this adapts them to the
        # GroupNorm-based neck with an explicit activation.
        self.pre_norms = nn.ModuleList()
        for _ in range(self.num_ins):
            self.pre_norms.append(
                nn.Sequential(
                    build_norm_layer(norm_cfg, in_channels)[1],
                    nn.GELU(),
                )
            )

        # Lateral convolutions: 1×1 conv to unify channels
        self.lateral_convs = nn.ModuleList()
        for _ in range(self.num_ins):
            l_conv = ConvModule(
                in_channels, out_channels, kernel_size=1,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False,
            )
            self.lateral_convs.append(l_conv)

        # Downsample: stride-2 3×3 conv (for deepest → P3)
        self.downsample = ConvModule(
            out_channels, out_channels, kernel_size=3, stride=2, padding=1,
            norm_cfg=norm_cfg, act_cfg=act_cfg, inplace=False,
        )

        # Progressive upsampling blocks (deep → shallow)
        # up_2→1: upsample from P2 level by 2× to build P1
        # up_1→0: upsample from P1 level by 2× to build P0
        self.upsample_2_1 = UpsampleBlock(out_channels, out_channels, norm_cfg, act_cfg)
        self.upsample_1_0 = UpsampleBlock(out_channels, out_channels, norm_cfg, act_cfg)

        # SE channel attention after each fusion
        self.se_blocks = nn.ModuleList()
        for _ in range(self.num_ins - 1):
            self.se_blocks.append(SEBlock(out_channels, se_reduction))

        # FPN output smoothing convolutions (3×3)
        self.fpn_convs = nn.ModuleList()
        for _ in range(self.num_ins):
            fpn_conv = ConvModule(
                out_channels, out_channels, kernel_size=3,
                stride=1, padding=1,
                norm_cfg=norm_cfg, act_cfg=act_cfg, inplace=False,
            )
            self.fpn_convs.append(fpn_conv)

        # Extra output levels
        extra_levels = num_outs - self.num_ins
        self.extra_downsamples = nn.ModuleList()
        for _ in range(extra_levels):
            self.extra_downsamples.append(
                ConvModule(
                    out_channels, out_channels, kernel_size=3,
                    stride=2, padding=1,
                    norm_cfg=norm_cfg, act_cfg=act_cfg, inplace=False,
                )
            )

    def init_weights(self):
        """Initialize weights."""
        if self.init_cfg is not None:
            super().init_weights()
            return
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        """Forward pass — build multi-scale pyramid from ViT features.

        Args:
            inputs: 4 feature maps from ViT backbone.
                    Each (B, C, H, W), all at the same spatial resolution.

        Returns:
            tuple[Tensor]: Multi-scale feature maps.
                P0: stride 4 (4H × 4W)
                P1: stride 8 (2H × 2W)
                P2: stride 16 (H × W)
                P3: stride 32 (H/2 × W/2)
        """
        assert len(inputs) == self.num_ins

        # Step 1: Pre-norm adaptation (LN → GN space + GELU)
        adapted = [self.pre_norms[i](inputs[i]) for i in range(self.num_ins)]

        # Step 2: Lateral convolutions
        laterals = [self.lateral_convs[i](adapted[i]) for i in range(self.num_ins)]

        # Step 3: Generate raw scale features (before top-down fusion)
        # P3 (stride 32): downsample deepest feature f3 (block 11)
        p3_raw = self.downsample(laterals[3])

        # P2 (stride 16): pass-through f2 (block 7)
        p2_raw = laterals[2]

        # P1 (stride 8): upsample f1 (block 5) by 2×
        p1_raw = self.upsample_2_1(laterals[1])

        # P0 (stride 4): upsample f0 (block 3) by 2× twice (= 4×)
        p0_raw = self.upsample_1_0(self.upsample_2_1(laterals[0]))

        # Step 4: Top-down cross-scale fusion with SE attention
        # P3 → P2: upsample P3 by 2×, fuse with P2 lateral
        p2_up = F.interpolate(p3_raw, size=p2_raw.shape[-2:], mode='bilinear', align_corners=False)
        p2_fused = p2_raw + p2_up
        p2_fused = self.se_blocks[2](p2_fused) if len(self.se_blocks) > 2 else p2_fused

        # P2 → P1: upsample fused P2 by 2×, fuse with P1 lateral
        p1_up = F.interpolate(p2_fused, size=p1_raw.shape[-2:], mode='bilinear', align_corners=False)
        p1_fused = p1_raw + p1_up
        p1_fused = self.se_blocks[1](p1_fused) if len(self.se_blocks) > 1 else p1_fused

        # P1 → P0: upsample fused P1 by 2×, fuse with P0 lateral
        p0_up = F.interpolate(p1_fused, size=p0_raw.shape[-2:], mode='bilinear', align_corners=False)
        p0_fused = p0_raw + p0_up
        p0_fused = self.se_blocks[0](p0_fused) if len(self.se_blocks) > 0 else p0_fused

        # Step 5: Apply output smoothing convolutions (3×3)
        p0 = self.fpn_convs[0](p0_fused)
        p1 = self.fpn_convs[1](p1_fused)
        p2 = self.fpn_convs[2](p2_fused)
        p3 = self.fpn_convs[3](p3_raw)

        outs = [p0, p1, p2, p3]

        # Step 6: Add extra output levels
        extra_source = outs[-1]
        for extra_conv in self.extra_downsamples:
            extra_feat = extra_conv(extra_source)
            outs.append(extra_feat)
            extra_source = extra_feat

        return tuple(outs)
