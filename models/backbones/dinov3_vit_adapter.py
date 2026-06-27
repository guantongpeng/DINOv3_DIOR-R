"""DINOv3 ViT-Adapter backbone for mmdetection/mmrotate oriented detection.

This is a port of ``vit-adapter/dinov3_backbone.py`` (the mmsegmentation
wrapper): instead of re-implementing the ViT-Adapter recipe, it wraps the
OFFICIAL ``DINOv3_Adapter`` from the dinov3 package and swaps its
MSDeformAttn for an mmcv-backed core. This keeps behaviour identical to the
upstream segmentation adapter while staying portable (mmcv CUDA op when
available) and registering cleanly with mmdet's registry.

Design (mirrors vit-adapter/dinov3_backbone.py):
  - The ViT is built via ``dinov3.models.build_model_for_eval`` (official eval
    recipe; loads the LVD1689M checkpoint, runs the frozen backbone under
    bfloat16 autocast + no_grad exactly like upstream).
  - ``DINOv3_Adapter`` produces a 4-level pyramid {"1".."4"} at strides
    [4, 8, 16, 32], each at ``embed_dim`` channels.
  - Every MSDeformAttn in the adapter is replaced with ``_MmcvMSDeformAttn``
    (mmcv MultiScaleDeformableAttention), so no custom CUDA op is required to
    build the adapter.
  - A per-level 1x1 conv projects embed_dim -> ``out_channels`` (256) so the
    pyramid feeds RPN/RoI via ``PassthroughNeck`` directly.

All dinov3 imports resolve to a single canonical source tree
(``DINOV3_SRC``), defaulting to ``.../mm_dino/third_party/dinov3``.

Registered as a BACKBONE; pair with ``PassthroughNeck``.
"""

import logging
import os
import sys
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention

_log = logging.getLogger("dinov3")

# ---------------------------------------------------------------------------
# Canonical dinov3 source root. All dinov3 imports use this path only.
# Override with the DINOV3_SRC env var if needed; otherwise the in-repo tree
# at .../mm_dino/third_party/dinov3 is used.
# ---------------------------------------------------------------------------
_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DINOV3_SRC = os.environ.get(
    "DINOV3_SRC",
    os.path.join(_PROJ, "third_party", "dinov3"),
)
if _DINOV3_SRC and _DINOV3_SRC not in sys.path:
    sys.path.insert(0, _DINOV3_SRC)
    _log.info(f"[DINOv3ViTAdapter] dinov3 source -> {_DINOV3_SRC}")

from mmdet.models.builder import BACKBONES  # noqa: E402


class _MmcvMSDeformAttn(nn.Module):
    """Drop-in replacement for DINOv3's MSDeformAttn backed by mmcv's op.

    Matches DINOv3's MSDeformAttn __init__ and forward signatures exactly so it
    can be swapped in-place inside DINOv3_Adapter without touching adapter code.

    `ratio` is accepted for API compatibility but not used -- mmcv's MSDA always
    operates at full d_model. This differs from DINOv3's default ratio=0.5 but
    avoids a double value-projection bug. Requires starting training fresh (no
    --resume from checkpoints saved with the original MSDeformAttn).

    mmcv adds an internal residual; we suppress it (identity=zeros) because
    Extractor.forward() adds the residual externally.
    """

    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, ratio=1.0):
        super().__init__()
        self.attn = MultiScaleDeformableAttention(
            embed_dims=d_model,
            num_levels=n_levels,
            num_heads=n_heads,
            num_points=n_points,
            batch_first=True,
        )

    def init_weights(self):
        self.attn.init_weights()

    def forward(self, query, reference_points, input_flatten,
                input_spatial_shapes, input_level_start_index,
                input_padding_mask=None):
        return self.attn(
            query=query,
            value=input_flatten,
            identity=torch.zeros_like(query),
            query_pos=None,
            key_padding_mask=input_padding_mask,
            reference_points=reference_points,
            spatial_shapes=input_spatial_shapes,
            level_start_index=input_level_start_index,
        )


# Imported lazily so the module imports cleanly even before the dinov3 package
# is on the path at *import* time (it is inserted above, but keep it lazy to
# avoid hard-failing tools that only need the class definition).


@BACKBONES.register_module()
class DINOv3ViTAdapter(nn.Module):
    """DINOv3 ViT + official DINOv3_Adapter, wrapped as an mmdet backbone.

    Produces a 4-level pyramid at strides [4, 8, 16, 32] with ``out_channels``
    (default 256), ready to feed RPN/RoI via ``PassthroughNeck``.

    Args:
        model_name: dinov3 hub backbone name (e.g. 'dinov3_vitb16'). Mapped to
            the corresponding build arch and LVD1689M checkpoint config.
        interaction_indexes: ViT block indices whose tokens each interaction
            block fuses with. Defaults per arch (e.g. [2,5,8,11] for ViT-B).
        out_channels: per-level output channels (RPN expects 256).
        freeze_vit: if True, keep the ViT frozen (only the adapter trains).
            Note: the official adapter runs the ViT under no_grad regardless, so
            this mainly controls requires_grad for end-to-end fine-tuning setups.
        pretrain_size: pos-embed interpolation size (forwarded to DINOv3_Adapter).
        conv_inplane / n_points / deform_num_heads / drop_path_rate /
            cffn_ratio / deform_ratio / use_extra_extractor / with_cp: forwarded
            to the official DINOv3_Adapter.
        bf16_vit: accepted for API compatibility; the official adapter already
            runs the frozen ViT under bfloat16 autocast (upstream eval recipe).
        vit_cfg_overrides: extra student-config overrides merged onto the
            LVD1689M defaults used to build the ViT.
        init_cfg: mmdet init config whose ``checkpoint`` is the pretrained ViT
            .pth (or DCP dir). The weights are loaded at construction via
            build_model_for_eval; init_weights() is a no-op.
    """

    # model_name -> (build arch, embed_dim, depth, patch_size)
    _MODEL_SPECS = {
        "dinov3_vits16": {"arch": "vit_small", "embed_dim": 384, "depth": 12, "patch_size": 16},
        "dinov3_vits16plus": {"arch": "vit_small", "embed_dim": 384, "depth": 12, "patch_size": 16},
        "dinov3_vitb16": {"arch": "vit_base", "embed_dim": 768, "depth": 12, "patch_size": 16},
        "dinov3_vitl16": {"arch": "vit_large", "embed_dim": 1024, "depth": 24, "patch_size": 16},
        "dinov3_vitl16plus": {"arch": "vit_large", "embed_dim": 1024, "depth": 24, "patch_size": 16},
        "dinov3_vith16plus": {"arch": "vit_huge2", "embed_dim": 1280, "depth": 32, "patch_size": 16},
        "dinov3_vit7b16": {"arch": "vit_7b", "embed_dim": 4096, "depth": 40, "patch_size": 16},
    }

    # Default interaction block indices per arch (evenly spaced).
    _DEFAULT_INTERACTION_INDEXES = {
        "vit_small": [2, 5, 8, 11],
        "vit_base": [2, 5, 8, 11],
        "vit_large": [5, 11, 17, 23],
        "vit_huge2": [7, 15, 23, 31],
        "vit_7b": [9, 19, 29, 39],
    }

    # LVD1689M checkpoint config (matches the dinov3 hub backbones: 4 storage
    # tokens, mask_k_bias, layernormbf16, RoPE base=100/rescale=2). These are
    # merged onto the base student cfg below, like vit-adapter/dinov3_mmrotate0.
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

    def __init__(
        self,
        model_name: str = "dinov3_vitb16",
        interaction_indexes: Optional[List[int]] = None,
        out_channels: Optional[int] = None,
        freeze_vit: bool = True,
        pretrain_size: int = 512,
        conv_inplane: int = 64,
        n_points: int = 4,
        deform_num_heads: int = 16,
        drop_path_rate: float = 0.3,
        cffn_ratio: float = 0.25,
        deform_ratio: float = 0.5,
        use_extra_extractor: bool = True,
        with_cp: bool = True,
        bf16_vit: bool = True,
        vit_cfg_overrides: Optional[dict] = None,
        init_cfg: Optional[dict] = None,
    ):
        super().__init__()
        from omegaconf import OmegaConf
        from dinov3.models import build_model_for_eval
        from dinov3.eval.segmentation.models.backbone.dinov3_adapter import DINOv3_Adapter

        if model_name not in self._MODEL_SPECS:
            raise ValueError(
                f"Unknown model_name: {model_name}. "
                f"Available: {list(self._MODEL_SPECS.keys())}"
            )
        spec = self._MODEL_SPECS[model_name]
        arch = spec["arch"]
        patch_size = spec["patch_size"]
        self.model_name = model_name
        self.embed_dim = spec["embed_dim"]
        self.patch_size = patch_size
        self.freeze_vit = freeze_vit
        self.bf16_vit = bf16_vit
        self.init_cfg = init_cfg

        # Resolve the pretrained ViT checkpoint from init_cfg (mmdet flow).
        checkpoint = None
        if init_cfg and isinstance(init_cfg, dict):
            checkpoint = init_cfg.get("checkpoint", None)

        # ---- Build the ViT via the official eval recipe ---------------------
        student_cfg = dict(
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
        # Apply LVD1689M defaults, then any caller overrides (last wins).
        student_cfg.update(self._LVD1689M_CFG)
        if vit_cfg_overrides:
            student_cfg.update(vit_cfg_overrides)

        cfg = OmegaConf.create({
            "student": student_cfg,
            "crops": {"global_crops_size": 224},
        })

        vit = build_model_for_eval(cfg, pretrained_weights=checkpoint)
        self.embed_dim = vit.embed_dim

        if interaction_indexes is None:
            interaction_indexes = self._DEFAULT_INTERACTION_INDEXES.get(
                arch, [2, 5, 8, 11]
            )
        self.interaction_indexes = interaction_indexes

        # ---- Official DINOv3_Adapter ----------------------------------------
        self.adapter = DINOv3_Adapter(
            vit,
            interaction_indexes=interaction_indexes,
            pretrain_size=pretrain_size,
            conv_inplane=conv_inplane,
            n_points=n_points,
            deform_num_heads=deform_num_heads,
            drop_path_rate=drop_path_rate,
            cffn_ratio=cffn_ratio,
            deform_ratio=deform_ratio,
            use_extra_extractor=use_extra_extractor,
            with_cp=with_cp,  # checkpointing in the extractors
        )

        # Swap every MSDeformAttn for the mmcv-backed wrapper (after the adapter
        # __init__ which inits the original modules).
        self._replace_msda_with_mmcv()

        # The official adapter freezes the backbone in its __init__; flip grads
        # back on when the user wants end-to-end fine-tuning.
        self.adapter.finetune_vit = not freeze_vit
        if not freeze_vit:
            self.adapter.backbone.requires_grad_(True)
        # Adapter-specific params are always trainable; backbone respects freeze.
        for name, param in self.adapter.named_parameters():
            if freeze_vit and "backbone" in name:
                continue
            param.requires_grad = True

        # ---- Per-level projection embed_dim -> out_channels -----------------
        # When out_channels is None, skip the projection and return the raw
        # embed_dim features (e.g. for a downstream FPN that fuses the full-width
        # adapter pyramid).
        if out_channels is not None:
            self._proj_out_channels = [out_channels] * 4
            self._out_strides = [4, 8, 16, 32]
            self.out_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(self.embed_dim, out_channels, kernel_size=1, bias=False),
                    nn.GroupNorm(32, out_channels),
                )
                for _ in range(4)
            ])
            self._init_projection()
        else:
            self._proj_out_channels = None
            self._out_strides = [4, 8, 16, 32]
            self.out_proj = None

    def _replace_msda_with_mmcv(self):
        """Swap all DINOv3 MSDeformAttn modules with _MmcvMSDeformAttn."""
        from dinov3.eval.segmentation.models.utils.ms_deform_attn import MSDeformAttn

        for parent in self.adapter.modules():
            for name, child in list(parent.named_children()):
                if isinstance(child, MSDeformAttn):
                    replacement = _MmcvMSDeformAttn(
                        d_model=child.d_model,
                        n_levels=child.n_levels,
                        n_heads=child.n_heads,
                        n_points=child.n_points,
                        ratio=child.ratio,
                    )
                    replacement.init_weights()
                    setattr(parent, name, replacement)

    def _init_projection(self):
        for m in self.out_proj.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def init_weights(self):
        """No-op: ViT weights are loaded at construction via build_model_for_eval.

        Kept for the mmdet init flow (and tools that call init_weights()).
        """
        return

    # ----- mmdet interface --------------------------------------------------
    @property
    def out_channels(self):
        if self._proj_out_channels is not None:
            return self._proj_out_channels
        return [self.embed_dim] * 4

    @property
    def strides(self):
        return self._out_strides

    def train(self, mode=True):
        super().train(mode)
        # Keep the frozen ViT in eval mode (disables dropout/stochastic-depth).
        if self.freeze_vit:
            self.adapter.backbone.eval()
        return self

    # ----- forward ----------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # DINOv3_Adapter returns {"1": f1, "2": f2, "3": f3, "4": f4}
        # strides: f1=4, f2=8, f3=16, f4=32, each at embed_dim channels.
        out = self.adapter(x)
        if self.out_proj is not None:
            feats = tuple(
                proj(out[key]).contiguous()
                for proj, key in zip(self.out_proj, ("1", "2", "3", "4"))
            )
        else:
            feats = tuple(
                out[key].contiguous() for key in ("1", "2", "3", "4")
            )
        return feats
