"""DINOv3 ViT multi-level backbone for MMDetection/mmrotate.

This module defines and registers ``DinoVisionTransformerBackbone`` locally.
It no longer imports ``dinov3.eval.detection.mm_backbone`` (which only exists in
some dinov3 forks). Instead the ViT is built and loaded the same way as
``vit-adapter/dinov3_backbone.py``: via ``dinov3.models.build_model_for_eval``
from an OmegaConf student config, then multi-level features are extracted with
``get_intermediate_layers``.

Usage in config:
    custom_imports = dict(imports=['models.backbones.dinov3_wrapper'])
    backbone=dict(type='DinoVisionTransformerBackbone', ...)
"""

import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

_log = logging.getLogger("dinov3")

# Canonical dinov3 source root (all dinov3 imports use this path only).
# Override with the DINOV3_SRC env var if needed.
_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DINOV3_PATH = os.environ.get(
    'DINOV3_SRC', os.path.join(_PROJ, 'third_party', 'dinov3'))
if _DINOV3_PATH not in sys.path:
    sys.path.insert(0, _DINOV3_PATH)

from mmdet.models.builder import MODELS  # noqa: E402


# model_name -> (build arch, embed_dim, n_blocks, patch_size)
_MODEL_SPECS: Dict[str, Dict] = {
    "dinov3_vits16": {"arch": "vit_small", "embed_dim": 384, "n_blocks": 12, "patch_size": 16},
    "dinov3_vits16plus": {"arch": "vit_small", "embed_dim": 384, "n_blocks": 12, "patch_size": 16},
    "dinov3_vitb16": {"arch": "vit_base", "embed_dim": 768, "n_blocks": 12, "patch_size": 16},
    "dinov3_vitl16": {"arch": "vit_large", "embed_dim": 1024, "n_blocks": 24, "patch_size": 16},
    "dinov3_vitl16plus": {"arch": "vit_large", "embed_dim": 1024, "n_blocks": 24, "patch_size": 16},
    "dinov3_vith16plus": {"arch": "vit_huge2", "embed_dim": 1280, "n_blocks": 32, "patch_size": 16},
    "dinov3_vit7b16": {"arch": "vit_7b", "embed_dim": 4096, "n_blocks": 40, "patch_size": 16},
}

# LVD1689M checkpoint config (matches the dinov3 hub backbones: 4 storage tokens,
# mask_k_bias, layernormbf16, RoPE base=100/rescale=2). Built the same way as
# vit-adapter/dinov3_backbone.py.
_LVD1689M_CFG = dict(
    pos_embed_rope_base=100.0,
    pos_embed_rope_min_period=None,
    pos_embed_rope_max_period=None,
    pos_embed_rope_normalize_coords="separate",
    pos_embed_rope_rescale_coords=2,
    pos_embed_rope_dtype="fp32",
    norm_layer="layernormbf16",
    n_storage_tokens=4,
    mask_k_bias=True,
)


def _get_default_layers_to_use(n_blocks: int, num_levels: int = 4) -> List[int]:
    """Last block of each quarter (existing convention)."""
    return [m * n_blocks // num_levels - 1 for m in range(1, num_levels + 1)]


@MODELS.register_module(name="DinoVisionTransformerBackbone", force=True)
class DinoVisionTransformerBackbone(nn.Module):
    """MMDetection-compatible DINOv3 ViT backbone (built via build_model_for_eval).

    Extracts patch-token features from selected intermediate transformer blocks,
    returning multi-level feature maps at stride=patch_size. MMDetection's FPN
    (or SimpleFeaturePyramid) neck builds the multi-scale pyramid from these.

    The ViT is constructed and weight-loaded through the official eval recipe
    (``dinov3.models.build_model_for_eval``), exactly like
    ``vit-adapter/dinov3_backbone.py``. Weights are loaded at construction from
    ``init_cfg.checkpoint``; ``init_weights()`` is a no-op.

    Args:
        model_name: dinov3 hub backbone name (e.g. 'dinov3_vitb16').
        pretrained: accepted for API compatibility; loading is driven by
            ``init_cfg.checkpoint`` (kept False in configs to skip hub download).
        layers_to_use: ViT block indices whose outputs to return.
        out_indices: which of the extracted layers to actually output.
        use_layernorm: apply LayerNorm2D to each extracted output.
        out_channels: if set, concatenate features and project to this dim via
            1x1 conv (single-level output, for DETR-style heads).
        frozen_stages: -1 = freeze all backbone params; 0 = train all.
        init_cfg: mmdet init config whose ``checkpoint`` is the pretrained ViT
            .pth (LVD1689M or SAT493M).
        vit_cfg_overrides: extra student-config overrides merged onto the
            LVD1689M defaults used to build the ViT.
    """

    def __init__(
        self,
        model_name: str = "dinov3_vitb16",
        pretrained: bool = False,
        layers_to_use: Optional[List[int]] = None,
        out_indices: Tuple[int, ...] = (0, 1, 2, 3),
        use_layernorm: bool = True,
        out_channels: Optional[int] = None,
        frozen_stages: int = 0,
        init_cfg: Optional[dict] = None,
        vit_cfg_overrides: Optional[dict] = None,
    ):
        super().__init__()

        if model_name not in _MODEL_SPECS:
            raise ValueError(
                f"Unknown model_name: {model_name}. "
                f"Available: {list(_MODEL_SPECS.keys())}"
            )
        spec = _MODEL_SPECS[model_name]
        self.model_name = model_name
        self.embed_dim: int = spec["embed_dim"]
        self.n_blocks: int = spec["n_blocks"]
        self.patch_size: int = spec["patch_size"]
        self.pretrained = pretrained
        self.frozen_stages = frozen_stages
        self.init_cfg = init_cfg

        if layers_to_use is None:
            layers_to_use = _get_default_layers_to_use(self.n_blocks)
        self.layers_to_use = list(layers_to_use)
        self.out_indices = tuple(out_indices)

        # Resolve the pretrained ViT checkpoint from init_cfg (mmdet flow).
        checkpoint = None
        if init_cfg and isinstance(init_cfg, dict):
            checkpoint = init_cfg.get("checkpoint", None)

        self.backbone = self._build_vit(
            spec["arch"], spec["patch_size"], checkpoint, vit_cfg_overrides)

        # Optional per-level LayerNorm2D.
        if use_layernorm:
            from dinov3.eval.detection.models.utils import LayerNorm2D
            self.layer_norms = nn.ModuleList(
                [LayerNorm2D(self.embed_dim) for _ in self.layers_to_use])
        else:
            self.layer_norms = None

        # Optional single-level projection (DETR-style); unused by FPN configs.
        self.out_proj = None
        if out_channels is not None:
            concat_dim = self.embed_dim * len(self.layers_to_use)
            self.out_proj = nn.Conv2d(concat_dim, out_channels, kernel_size=1)
            self._out_channels_val = [out_channels]
        else:
            self._out_channels_val = [self.embed_dim] * len(self.out_indices)

        if frozen_stages == -1:
            self._freeze_backbone()

    def _build_vit(self, arch, patch_size, checkpoint, overrides):
        """Build + load the ViT via the official eval recipe (build_model_for_eval)."""
        from omegaconf import OmegaConf
        from dinov3.models import build_model_for_eval

        student = dict(
            arch=arch,
            patch_size=patch_size,
            pos_embed_rope_base=None,
            pos_embed_rope_min_period=4,
            pos_embed_rope_max_period=50,
            pos_embed_rope_normalize_coords="separate",
            pos_embed_rope_shift_coords=None,
            pos_embed_rope_jitter_coords=None,
            pos_embed_rope_rescale_coords=None,
            qkv_bias=True,
            layerscale=1e-5,
            norm_layer="layernorm",
            ffn_layer="mlp",
            ffn_bias=True,
            proj_bias=True,
            n_storage_tokens=0,
            mask_k_bias=False,
            untie_cls_and_patch_norms=False,
            untie_global_and_local_cls_norm=False,
            fp8_enabled=False,
        )
        student.update(_LVD1689M_CFG)
        # SAT493M checkpoints untie the global/local cls norm (adds an eval-unused
        # local_cls_norm module). Match it so the checkpoint loads cleanly.
        if checkpoint and "sat493m" in os.path.basename(checkpoint).lower():
            student["untie_global_and_local_cls_norm"] = True
        if overrides:
            student.update(overrides)

        cfg = OmegaConf.create({
            "student": student,
            "crops": {"global_crops_size": 224},
        })
        vit = build_model_for_eval(cfg, pretrained_weights=checkpoint)
        _log.info(
            f"[DinoVisionTransformerBackbone] built {arch} (embed_dim={vit.embed_dim}) "
            f"from checkpoint={checkpoint}")
        return vit

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def init_weights(self):
        """No-op: ViT weights are loaded at construction via build_model_for_eval.

        Kept for the mmdet init flow (and tools that call init_weights()).
        """
        return

    @property
    def out_channels(self) -> List[int]:
        return self._out_channels_val

    @property
    def strides(self) -> List[int]:
        if self.out_proj is not None:
            return [self.patch_size]
        return [self.patch_size] * len(self.out_indices)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen_stages == -1:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False
        return self

    def forward(self, x: torch.Tensor):
        """Forward: tuple of (B, embed_dim, H//patch, W//patch) per output level.

        get_intermediate_layers handles patch embedding, prepending CLS + storage
        tokens, running the transformer blocks, then stripping CLS/register tokens,
        applying norm, and reshaping to spatial feature maps.
        """
        features = self.backbone.get_intermediate_layers(
            x,
            n=self.layers_to_use,
            reshape=True,
            norm=True,
            return_class_token=False,
            return_extra_tokens=False,
        )

        if self.layer_norms is not None:
            features = tuple(
                ln(f).contiguous() for ln, f in zip(self.layer_norms, features))

        if self.out_proj is not None:
            concat = torch.cat(features, dim=1)
            return (self.out_proj(concat),)

        return tuple(features[i] for i in self.out_indices)


__all__ = ['DinoVisionTransformerBackbone']
