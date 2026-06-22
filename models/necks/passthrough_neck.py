"""Passthrough neck for the ViT-Adapter backbone.

The ViT-Adapter backbone already produces the full 4-level pyramid
(strides 4/8/16/32, 256 channels). This neck is an identity wrapper so the
standard OrientedRCNN `backbone -> neck -> head` dataflow stays valid without
an extra FPN that would (re)process an already-good pyramid.

Usage in config:
    neck=dict(type='PassthroughNeck')
"""

import torch
from mmcv.runner import BaseModule

from mmdet.models.builder import NECKS


@NECKS.register_module()
class PassthroughNeck(BaseModule):
    """Identity neck: returns the backbone's multi-level features unchanged."""

    def __init__(self, init_cfg=None):
        super().__init__(init_cfg=init_cfg)

    def init_weights(self):
        if self.init_cfg is not None:
            super().init_weights()
        return

    def forward(self, inputs):
        # inputs: tuple of feature maps from the backbone.
        if isinstance(inputs, torch.Tensor):
            return (inputs,)
        return tuple(inputs)
