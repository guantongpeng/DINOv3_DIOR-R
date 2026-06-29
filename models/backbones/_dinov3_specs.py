"""Shared DINOv3 model specifications for the backbone wrappers.

Both the plain ViT wrapper (``dinov3_wrapper.py``) and the ViT-Adapter backbone
(``dinov3_vit_adapter.py``) need the same model_name -> arch mapping and the
same LVD1689M checkpoint config. Keeping them here avoids drift between the two.
"""

# model_name -> build arch / embed_dim / depth (n_blocks) / patch_size.
MODEL_SPECS = {
    "dinov3_vits16": {"arch": "vit_small", "embed_dim": 384, "depth": 12, "patch_size": 16},
    "dinov3_vits16plus": {"arch": "vit_small", "embed_dim": 384, "depth": 12, "patch_size": 16},
    "dinov3_vitb16": {"arch": "vit_base", "embed_dim": 768, "depth": 12, "patch_size": 16},
    "dinov3_vitl16": {"arch": "vit_large", "embed_dim": 1024, "depth": 24, "patch_size": 16},
    "dinov3_vitl16plus": {"arch": "vit_large", "embed_dim": 1024, "depth": 24, "patch_size": 16},
    "dinov3_vith16plus": {"arch": "vit_huge2", "embed_dim": 1280, "depth": 32, "patch_size": 16},
    "dinov3_vit7b16": {"arch": "vit_7b", "embed_dim": 4096, "depth": 40, "patch_size": 16},
}

# LVD1689M checkpoint config (matches the dinov3 hub backbones: 4 storage
# tokens, mask_k_bias, layernormbf16, RoPE base=100/rescale=2).
LVD1689M_CFG = dict(
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
