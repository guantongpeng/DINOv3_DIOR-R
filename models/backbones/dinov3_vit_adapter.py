"""DINOv3 ViT-Adapter backbone for MMDetection/mmrotate oriented detection.

Replaces SimpleFeaturePyramid with the ViT-Adapter recipe
(Chen et al., "Vision Transformer Adapter for Dense Predictions", ICLR 2023),
ported from the official DINOv3 segmentation adapter
(`dinov3/eval/segmentation/models/backbone/dinov3_adapter.py`) and adapted for
the MMDetection Oriented R-CNN two-stage detector.

Design:
  - The frozen DINOv3 ViT produces patch tokens from `len(interaction_indexes)`
    selected blocks (multi-layer interaction).
  - A learnable Spatial Prior Module (light CNN stem on the raw image) produces
    4-level queries (c1..c4 at strides 4/8/16/32).
  - Interaction blocks fuse the spatial priors with each ViT layer's tokens via
    Multi-Scale Deformable Attention.
  - Final per-level 1x1 conv projects embed_dim -> out_channels (256) so the
    pyramid can feed RPN/RoI directly.

Key differences vs. the segmentation original (for robustness/compatibility):
  * MSDeformAttn is implemented in PURE PYTORCH (grid_sample) so it works
    WITHOUT compiling the CUDA extension — autograd handles backward.
  * `nn.SyncBatchNorm` -> `nn.GroupNorm` (works on single-GPU and multi-GPU
    without DDP process-group constraints).
  * The frozen ViT runs under `torch.autocast(bfloat16)` + `no_grad()` to match
    the official eval recipe and cut activation memory.
  * CLS / register tokens are stripped by `get_intermediate_layers(
    return_extra_tokens=False)` — no hardcoded `[:, 5:]` slicing.

This module is registered as a BACKBONE; pair it with `PassthroughNeck`
(`models/necks/passthrough_neck.py`) so the standard OrientedRCNN
backbone -> neck -> head dataflow is unchanged.
"""

import math
import os
import sys
from functools import partial
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

# Make the official dinov3 package importable (hub backbones / ViT weights).
_DINOV3_PATH = '/mnt/ht2-nas2/00-model/guantp/dino/dinov3'
if _DINOV3_PATH not in sys.path and os.path.isdir(_DINOV3_PATH):
    sys.path.insert(0, _DINOV3_PATH)

from mmdet.models.builder import BACKBONES


# ---------------------------------------------------------------------------
# Multi-Scale Deformable Attention (pure pytorch, autograd-friendly)
# ---------------------------------------------------------------------------
def ms_deform_attn_core_pytorch(
    value,
    value_spatial_shapes,
    sampling_locations,
    attention_weights,
):
    """Pure pytorch MS-DeformAttn core (forward + backward via autograd).

    Same math as the official implementation; no compiled CUDA op required.
    """
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_ * M_, 1, Lq_, L_ * P_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1)
    output = output.view(N_, M_ * D_, Lq_)
    return output.transpose(1, 2).contiguous()


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=1, n_heads=8, n_points=4, ratio=1.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.im2col_step = 64
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.ratio = ratio

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, int(d_model * ratio))
        self.output_proj = nn.Linear(int(d_model * ratio), d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        nn.init.constant_(self.attention_weights.weight.data, 0.0)
        nn.init.constant_(self.attention_weights.bias.data, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.0)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes,
                input_level_start_index, input_padding_mask=None):
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, int(self.ratio * self.d_model) // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(
            N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError("reference_points last dim must be 2 or 4")

        output = ms_deform_attn_core_pytorch(
            value, input_spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)
        return output


# ---------------------------------------------------------------------------
# Geometry helpers (reference points / deform inputs)
# ---------------------------------------------------------------------------
def get_reference_points(spatial_shapes, device):
    reference_points_list = []
    for (H_, W_) in spatial_shapes:
        H_i, W_i = int(H_), int(W_)
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_i - 0.5, H_i, dtype=torch.float32, device=device),
            torch.linspace(0.5, W_i - 0.5, W_i, dtype=torch.float32, device=device),
            indexing="ij",
        )
        ref_y = ref_y.reshape(-1)[None] / H_i
        ref_x = ref_x.reshape(-1)[None] / W_i
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None]
    return reference_points


def deform_inputs(x, patch_size):
    """Build deform inputs for the adapter interactions.

    Returns deform_inputs2 = [reference_points, spatial_shapes, level_start_index]
    where reference_points are normalized coords over the multiscale prior grid
    (strides 8/16/32) and spatial_shapes describe the single patch-level feature.
    """
    bs, c, h, w = x.shape
    # reference points over the multiscale prior grid (strides 8, 16, 32) — as a
    # python list of (H, W) tuples for reference-point generation.
    ref_shapes = [(h // 8, w // 8), (h // 16, w // 16), (h // 32, w // 32)]
    reference_points = get_reference_points(ref_shapes, x.device)

    # the feature the queries attend to is the single patch-level map (stride 16)
    spatial_shapes = torch.as_tensor(
        [(h // patch_size, w // patch_size)], dtype=torch.long, device=x.device)
    level_start_index = torch.cat(
        (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    return reference_points, spatial_shapes, level_start_index


# ---------------------------------------------------------------------------
# FFN with depthwise conv (operates on the multiscale-prior token sequence)
# ---------------------------------------------------------------------------
class DWConv(nn.Module):
    """Depthwise conv over the concatenated c2|c3|c4 prior tokens.

    Assumes the 3-level layout with token counts 16n|4n|n (n = H_c*W_c/4),
    i.e. input H,W divisible by 32.
    """

    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        n = N // 21
        x1 = x[:, 0:16 * n, :].transpose(1, 2).view(B, C, H * 2, W * 2).contiguous()
        x2 = x[:, 16 * n:20 * n, :].transpose(1, 2).view(B, C, H, W).contiguous()
        x3 = x[:, 20 * n:, :].transpose(1, 2).view(B, C, H // 2, W // 2).contiguous()
        x1 = self.dwconv(x1).flatten(2).transpose(1, 2)
        x2 = self.dwconv(x2).flatten(2).transpose(1, 2)
        x3 = self.dwconv(x3).flatten(2).transpose(1, 2)
        return torch.cat([x1, x2, x3], dim=1)


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ---------------------------------------------------------------------------
# Extractor + Interaction block
# ---------------------------------------------------------------------------
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x * random_tensor / keep_prob


class Extractor(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 with_cffn=True, cffn_ratio=0.25, drop=0.0, drop_path=0.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), with_cp=False):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(
            d_model=dim, n_levels=n_levels, n_heads=num_heads, n_points=n_points, ratio=deform_ratio)
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        if with_cffn:
            self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index, H, W):
        def _inner_forward(query, feat):
            attn = self.attn(
                self.query_norm(query), reference_points, self.feat_norm(feat),
                spatial_shapes, level_start_index, None)
            query = query + attn
            if self.with_cffn:
                query = query + self.drop_path(self.ffn(self.ffn_norm(query), H, W))
            return query

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)
        return query


class InteractionBlock(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), drop=0.0, drop_path=0.0,
                 with_cffn=True, cffn_ratio=0.25, deform_ratio=1.0,
                 extra_extractor=False, with_cp=False):
        super().__init__()
        self.extractor = Extractor(
            dim=dim, n_levels=1, num_heads=num_heads, n_points=n_points, norm_layer=norm_layer,
            deform_ratio=deform_ratio, with_cffn=with_cffn, cffn_ratio=cffn_ratio,
            drop=drop, drop_path=drop_path, with_cp=with_cp)
        if extra_extractor:
            self.extra_extractors = nn.Sequential(*[
                Extractor(
                    dim=dim, num_heads=num_heads, n_points=n_points, norm_layer=norm_layer,
                    with_cffn=with_cffn, cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                    drop=drop, drop_path=drop_path, with_cp=with_cp)
                for _ in range(2)
            ])
        else:
            self.extra_extractors = None

    def forward(self, x, c, reference_points, spatial_shapes, level_start_index, H_c, W_c):
        c = self.extractor(
            query=c, reference_points=reference_points, feat=x,
            spatial_shapes=spatial_shapes, level_start_index=level_start_index, H=H_c, W=W_c)
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(
                    query=c, reference_points=reference_points, feat=x,
                    spatial_shapes=spatial_shapes, level_start_index=level_start_index, H=H_c, W=W_c)
        return x, c


# ---------------------------------------------------------------------------
# Spatial Prior Module (light CNN stem on the raw image)
# ---------------------------------------------------------------------------
def _norm(num_channels, num_groups=32):
    # GroupNorm works on single- and multi-GPU without DDP constraints.
    g = num_groups if num_channels % num_groups == 0 else 1
    return nn.GroupNorm(g, num_channels)


class SpatialPriorModule(nn.Module):
    def __init__(self, inplanes=64, embed_dim=768, with_cp=False):
        super().__init__()
        self.with_cp = with_cp
        self.stem = nn.Sequential(
            nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            _norm(inplanes), nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            _norm(inplanes), nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            _norm(inplanes), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            _norm(2 * inplanes), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            _norm(4 * inplanes), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            _norm(4 * inplanes), nn.ReLU(inplace=True))
        self.fc1 = nn.Conv2d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc4 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        def _inner_forward(x):
            c1 = self.stem(x)
            c2 = self.conv2(c1)
            c3 = self.conv3(c2)
            c4 = self.conv4(c3)
            c1 = self.fc1(c1)
            c2 = self.fc2(c2)
            c3 = self.fc3(c3)
            c4 = self.fc4(c4)
            bs = c1.shape[0]
            # c1 stays spatial (stride 4); c2/c3/c4 flattened to token sequences.
            c2 = c2.view(bs, c2.shape[1], -1).transpose(1, 2)
            c3 = c3.view(bs, c3.shape[1], -1).transpose(1, 2)
            c4 = c4.view(bs, c4.shape[1], -1).transpose(1, 2)
            return c1, c2, c3, c4

        if self.with_cp and x.requires_grad:
            return cp.checkpoint(_inner_forward, x)
        return _inner_forward(x)


# ---------------------------------------------------------------------------
# The ViT-Adapter backbone
# ---------------------------------------------------------------------------
@BACKBONES.register_module()
class DINOv3ViTAdapter(nn.Module):
    """ViT-Adapter backbone: frozen DINOv3 ViT + deformable spatial-prior fusion.

    Produces a 4-level pyramid at strides [4, 8, 16, 32] with `out_channels`
    (default 256), ready to feed RPN/RoI via `PassthroughNeck`.

    Args:
        model_name: hub backbone name (e.g. 'dinov3_vitb16').
        interaction_indexes: ViT block indices whose tokens each interaction
            block fuses with. For ViT-B/16 use [2, 5, 8, 11] (4 interactions).
        out_channels: per-level output channels (RPN expects 256).
        freeze_vit: if True, the ViT is frozen (Stage-1). Set False for Stage-2
            end-to-end fine-tuning (use a low lr_mult on 'backbone.backbone').
        pretrain_size: pos-embed interpolation size of the pretrained ViT.
        with_cp: gradient checkpointing in extractors (saves memory).
        bf16_vit: run the frozen ViT under bfloat16 autocast (official recipe).
        init_cfg: mmdet-style init config with a checkpoint path.
    """

    _MODEL_SPECS = {
        "dinov3_vits16": {"embed_dim": 384, "n_blocks": 12, "patch_size": 16},
        "dinov3_vits16plus": {"embed_dim": 384, "n_blocks": 12, "patch_size": 16},
        "dinov3_vitb16": {"embed_dim": 768, "n_blocks": 12, "patch_size": 16},
        "dinov3_vitl16": {"embed_dim": 1024, "n_blocks": 24, "patch_size": 16},
        "dinov3_vitl16plus": {"embed_dim": 1024, "n_blocks": 24, "patch_size": 16},
        "dinov3_vith16plus": {"embed_dim": 1280, "n_blocks": 32, "patch_size": 16},
        "dinov3_vit7b16": {"embed_dim": 4096, "n_blocks": 40, "patch_size": 16},
    }

    def __init__(
        self,
        model_name: str = "dinov3_vitb16",
        interaction_indexes: Optional[List[int]] = None,
        out_channels: int = 256,
        freeze_vit: bool = True,
        pretrain_size: int = 512,
        conv_inplane: int = 64,
        n_points: int = 4,
        deform_num_heads: int = 16,
        drop_path_rate: float = 0.3,
        with_cffn: bool = True,
        cffn_ratio: float = 0.25,
        deform_ratio: float = 0.5,
        use_extra_extractor: bool = True,
        with_cp: bool = True,
        bf16_vit: bool = True,
        init_cfg: Optional[dict] = None,
    ):
        super().__init__()
        if model_name not in self._MODEL_SPECS:
            raise ValueError(f"Unknown model_name: {model_name}. "
                             f"Available: {list(self._MODEL_SPECS.keys())}")
        spec = self._MODEL_SPECS[model_name]
        self.model_name = model_name
        self.embed_dim = spec["embed_dim"]
        self.n_blocks = spec["n_blocks"]
        self.patch_size = spec["patch_size"]
        self.init_cfg = init_cfg
        self.bf16_vit = bf16_vit

        if interaction_indexes is None:
            interaction_indexes = [2, 5, 8, 11]
        self.interaction_indexes = interaction_indexes

        # Build + freeze the ViT (weights loaded in init_weights).
        self._build_vit()
        self.freeze_vit = freeze_vit
        self._apply_freeze(freeze_vit)

        self.pretrain_size = (pretrain_size, pretrain_size)
        embed_dim = self.embed_dim
        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=embed_dim, with_cp=False)
        self.interactions = nn.Sequential(*[
            InteractionBlock(
                dim=embed_dim, num_heads=deform_num_heads, n_points=n_points,
                drop_path=drop_path_rate, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                with_cffn=with_cffn, cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                extra_extractor=((i == len(interaction_indexes) - 1) and use_extra_extractor),
                with_cp=with_cp)
            for i in range(len(interaction_indexes))
        ])
        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.norm1 = _norm(embed_dim)
        self.norm2 = _norm(embed_dim)
        self.norm3 = _norm(embed_dim)
        self.norm4 = _norm(embed_dim)

        # Per-level projection to RPN channels.
        self.out_proj = nn.ModuleList([
            nn.Sequential(nn.Conv2d(embed_dim, out_channels, kernel_size=1, bias=False),
                          _norm(out_channels))
            for _ in range(4)])
        self._out_channels = [out_channels] * 4
        self._out_strides = [4, 8, 16, 32]

        self._init_weights_local()

    # ----- ViT build / freeze / load ----------------------------------------
    def _build_vit(self):
        import dinov3.hub.backbones as hub_backbones
        backbone_fn = getattr(hub_backbones, self.model_name, None)
        if backbone_fn is None:
            raise ValueError(f"Hub function '{self.model_name}' not found in dinov3.hub.backbones")
        # pretrained=False: we load weights from init_cfg checkpoint in init_weights.
        self.backbone = backbone_fn(pretrained=False)

    def _apply_freeze(self, freeze_vit: bool):
        for p in self.backbone.parameters():
            p.requires_grad_(not freeze_vit)

    def _init_weights_local(self):
        # Only init the adapter's own modules — leave self.backbone (ViT) alone;
        # its pretrained weights are loaded in init_weights().
        own = [self.up, self.spm, self.interactions,
               self.norm1, self.norm2, self.norm3, self.norm4]
        own += list(self.out_proj)
        for module in own:
            self._apply_init(module)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        nn.init.normal_(self.level_embed, std=0.02)

    @staticmethod
    def _apply_init(module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()

    def init_weights(self):
        """Load pretrained ViT weights from the init_cfg checkpoint (mmdet flow)."""
        if self.init_cfg is None:
            return
        checkpoint = self.init_cfg.get("checkpoint", None)
        if checkpoint is None or not isinstance(checkpoint, str):
            return
        state_dict = torch.load(checkpoint, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        # Strip possible wrapper prefixes, keep only the ViT weights.
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()
                      if k.startswith("backbone.")} or state_dict
        missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
        if missing:
            import logging
            logging.getLogger("dinov3").warning(
                f"[DINOv3ViTAdapter] missing keys when loading ViT: {missing[:5]} ...")
        if unexpected:
            import logging
            logging.getLogger("dinov3").info(
                f"[DINOv3ViTAdapter] unexpected keys ignored: {unexpected[:5]} ...")

    # ----- mmdet interface --------------------------------------------------
    @property
    def out_channels(self):
        return self._out_channels

    @property
    def strides(self):
        return self._out_strides

    def train(self, mode=True):
        super().train(mode)
        # Keep the ViT in eval mode when frozen (disables dropout/stochastic-depth).
        if self.freeze_vit:
            self.backbone.eval()
        return self

    # ----- forward ----------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        bs = x.shape[0]
        H, W = x.shape[2], x.shape[3]
        patch_size = self.patch_size
        assert H % 32 == 0 and W % 32 == 0, (
            "ViT-Adapter requires input H,W divisible by 32 (for the stride-32 "
            f"prior level); got H={H}, W={W}. Pad to a multiple of 32.")
        H_c, W_c = H // 16, W // 16
        H_toks, W_toks = H // patch_size, W // patch_size

        reference_points, spatial_shapes, level_start_index = deform_inputs(x, patch_size)

        # Spatial prior queries from the raw image (strides 4/8/16/32).
        c1, c2, c3, c4 = self.spm(x)
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        c = torch.cat([c2, c3, c4], dim=1)

        # Frozen ViT multi-layer patch tokens (clean patch tokens, sequences).
        # Run under bf16 autocast for the heavy frozen backbone (official recipe),
        # then upcast to fp32 for the trainable interaction (matches the official
        # ModelWithIntermediateLayers.float() pattern).
        ctx = torch.autocast("cuda", torch.bfloat16) if (self.bf16_vit and x.is_cuda) else _nullctx()
        with ctx:
            with torch.set_grad_enabled(not self.freeze_vit):
                all_layers = self.backbone.get_intermediate_layers(
                    x, n=self.interaction_indexes, reshape=False, norm=True,
                    return_class_token=False, return_extra_tokens=False)
        all_layers = tuple(t.float() for t in all_layers)

        outs = []
        for i, layer in enumerate(self.interactions):
            x_tokens = all_layers[i]  # (B, H_toks*W_toks, C)
            _, c = layer(
                x_tokens, c, reference_points, spatial_shapes, level_start_index, H_c, W_c)
            outs.append(x_tokens.transpose(1, 2).view(bs, self.embed_dim, H_toks, W_toks).contiguous())

        # Split the prior query sequence back into c2/c3/c4 and reshape.
        n2, n3 = c2.shape[1], c3.shape[1]
        c2 = c[:, 0:n2, :].transpose(1, 2).view(bs, self.embed_dim, H_c * 2, W_c * 2).contiguous()
        c3 = c[:, n2:n2 + n3, :].transpose(1, 2).view(bs, self.embed_dim, H_c, W_c).contiguous()
        c4 = c[:, n2 + n3:, :].transpose(1, 2).view(bs, self.embed_dim, H_c // 2, W_c // 2).contiguous()
        c1 = self.up(c2) + c1  # c1 is spatial (stride 4) from SPM

        # Fuse in the ViT-layer features at the matching resolutions.
        x1, x2, x3, x4 = outs
        x1 = F.interpolate(x1, size=(4 * H_c, 4 * W_c), mode="bilinear", align_corners=False)
        x2 = F.interpolate(x2, size=(2 * H_c, 2 * W_c), mode="bilinear", align_corners=False)
        x3 = F.interpolate(x3, size=(1 * H_c, 1 * W_c), mode="bilinear", align_corners=False)
        x4 = F.interpolate(x4, size=(H_c // 2, W_c // 2), mode="bilinear", align_corners=False)
        c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Norm + project to out_channels.
        feats = [self.norm1(c1), self.norm2(c2), self.norm3(c3), self.norm4(c4)]
        feats = [proj(f).contiguous() for proj, f in zip(self.out_proj, feats)]
        return tuple(feats)


class _nullctx:
    """No-op context manager (used when bf16 autocast is disabled)."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False
