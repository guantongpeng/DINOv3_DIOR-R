"""Import DINOv3 backbone from the official dinov3 repository and register
it with MMDetection's model registry.

This replaces the custom ViTDinoV3 wrapper that uses timm with Meta's
official DINOv3 implementation.

Usage in config:
    custom_imports = dict(imports=['models.backbones.dinov3_wrapper'])
    backbone=dict(type='DinoVisionTransformerBackbone', ...)
"""

import sys
import os

_DINOV3_PATH = '/mnt/ht2-nas2/00-model/guantp/dino/dinov3'
if _DINOV3_PATH not in sys.path:
    sys.path.insert(0, _DINOV3_PATH)

from dinov3.eval.detection.mm_backbone import (
    DinoVisionTransformerBackbone,
)

# Register with mmdet's MODELS registry (which mmrotate uses as ROTATED_BACKBONES)
# Note: register_mm_backbone() uses mmengine.registry.MODELS which is a
# different registry from mmdet.models.builder.MODELS in mmdet v2.x.
# We need to register with the correct registry for mmrotate compatibility.
from mmdet.models.builder import MODELS
MODELS.register_module(
    name='DinoVisionTransformerBackbone',
    module=DinoVisionTransformerBackbone,
    force=True,
)

__all__ = ['DinoVisionTransformerBackbone']
