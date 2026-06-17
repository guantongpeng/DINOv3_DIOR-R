"""ViTDet-style Simple Feature Pyramid for plain ViT backbones (DINOv3).

This neck consumes a SINGLE feature map — the last ViT block output at
stride = patch_size (16) — and synthesizes the whole detection pyramid
[stride 4, 8, 16, 32] with transposed convolutions (upsampling) and strided
convolutions (downsampling), each followed by channels-last LayerNorm + GELU.

This is the recipe proven by the ViTDet paper and used by the DINOv2/DINOv3
detection fine-tuning setups. It differs from the project's `SimpleFPN` /
`ViTDetFPN`, which both feed 4 different transformer-block outputs in as if
they were 4 scales — that injects shallow, noisy block-3 features into the
fine stride-4 level. Using only the richest (last) feature and letting the
deconv stems learn the upsampling is cleaner and stronger.

Reference:
    ViTDet: Exploring Plain Vision Backbones for Object Detection
        Li et al., ECCV 2022 — https://arxiv.org/abs/2203.16527
    Detectron2 `SimpleFeaturePyramidNetwork`.
"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer, xavier_init
from mmcv.runner import BaseModule, auto_fp16

from mmdet.models.builder import NECKS


class LayerNorm2d(nn.Module):
    """Channels-last LayerNorm for (B, C, H, W) tensors.

    Normalizes over the channel dimension at every spatial location — the same
    operation as a ViT LayerNorm applied to feature maps.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(dim=1, keepdim=True)
        s = (x - u).pow(2).mean(dim=1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


@NECKS.register_module()
class SimpleFeaturePyramid(BaseModule):
    """Single-input ViTDet Simple Feature Pyramid.

    Args:
        in_channels (int | list[int]): Channels of the single backbone feature.
        out_channels (int): Channels of every output level. Default: 256.
        num_outs (int): Number of output levels. Default: 4.
        in_stride (int): Stride of the input feature (= ViT patch_size). With
            ``num_outs=4`` and ``in_stride=16`` the output strides are
            ``[4, 8, 16, 32]``. Default: 16.
        out_strides (list[int], optional): Explicit output strides. If given,
            overrides the automatic log2-contiguous layout derived from
            ``in_stride``/``num_outs``.
        act_cfg (dict): Activation config. Default: ``dict(type='GELU')``.
        init_cfg (dict, optional): Initialization config.

    Note:
        ``start_level`` / ``add_extra_convs`` / ``norm_cfg`` are accepted for
        config drop-in compatibility with mmdet FPN-style configs but are not
        used — the pyramid always starts from the single input level and uses
        internal LayerNorm2d (matching ViTDet).
    """

    def __init__(
        self,
        in_channels: Union[int, List[int]] = 1024,
        out_channels: int = 256,
        num_outs: int = 4,
        in_stride: int = 16,
        out_strides: Optional[List[int]] = None,
        start_level: int = 0,
        add_extra_convs: Union[bool, str] = False,
        norm_cfg: Optional[dict] = None,
        act_cfg: Optional[dict] = None,
        init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg=init_cfg)

        if isinstance(in_channels, (list, tuple)):
            assert len(in_channels) == 1, (
                'SimpleFeaturePyramid takes a SINGLE backbone feature; got '
                f'{len(in_channels)} inputs. Set the backbone to output only '
                'the last ViT block (e.g. layers_to_use=[n_blocks-1], '
                'out_indices=(0,)).')
            in_channels = in_channels[0]

        if act_cfg is None:
            act_cfg = dict(type='GELU')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_outs = num_outs
        self.num_ins = 1
        self.in_stride = in_stride

        # Derive output strides: log2-contiguous, centered on the input stride.
        if out_strides is None:
            n_below = num_outs // 2
            n_above = num_outs - 1 - n_below
            out_strides = []
            for i in range(n_below, 0, -1):
                out_strides.append(in_stride // (2 ** i))
            out_strides.append(in_stride)
            for i in range(1, n_above + 1):
                out_strides.append(in_stride * (2 ** i))
        assert len(out_strides) == num_outs, (
            f'out_strides ({out_strides}) length != num_outs ({num_outs})')
        self.out_strides = out_strides

        # Build one stem per output level from the single input feature.
        self.stages = nn.ModuleList()
        for stride in out_strides:
            scale = in_stride / stride
            layers: List[nn.Module] = []
            if scale > 1.0:
                n_up = int(round(math.log2(scale)))
                for _ in range(n_up):
                    layers += [
                        nn.ConvTranspose2d(
                            in_channels, in_channels,
                            kernel_size=2, stride=2, bias=False),
                        LayerNorm2d(in_channels),
                        build_activation_layer(act_cfg),
                    ]
            elif scale < 1.0:
                n_down = int(round(math.log2(1.0 / scale)))
                for _ in range(n_down):
                    layers += [
                        nn.Conv2d(
                            in_channels, in_channels,
                            kernel_size=2, stride=2, bias=False),
                        LayerNorm2d(in_channels),
                        build_activation_layer(act_cfg),
                    ]
            # Final 1x1 projection to out_channels + norm (no activation).
            layers += [
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, bias=False),
                LayerNorm2d(out_channels),
            ]
            self.stages.append(nn.Sequential(*layers))

    def init_weights(self):
        """Initialize conv / deconv weights with Xavier."""
        if self.init_cfg is not None:
            super().init_weights()
            return
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        """Build the multi-scale pyramid from a single backbone feature.

        Args:
            inputs (list[Tensor]): A single feature map (list of length 1),
                each (B, C, H, W) at ``in_stride``.

        Returns:
            tuple[Tensor]: ``num_outs`` feature maps at ``out_strides``.
        """
        assert len(inputs) == self.num_ins, (
            f'SimpleFeaturePyramid expects {self.num_ins} input, '
            f'got {len(inputs)}')
        feat = inputs[0]
        outs = [stage(feat) for stage in self.stages]
        return tuple(outs)
