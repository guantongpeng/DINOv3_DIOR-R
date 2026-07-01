"""Pure-PyTorch port of the DINOv3 ViT-L/16 backbone (no dinov3/mm* deps).

Faithful port of ``third_party/dinov3`` used by the trained checkpoint:
  * PatchEmbed (Conv2d 16x16, flatten=False)
  * cls_token + 4 storage_tokens + mask_token
  * RoPE positional encoding (base=100, separate normalisation, le90-style)
  * LinearKMaskedBias qkv (mask_k_bias=True zeroes the K bias third)
  * LayerScale (gamma), LayerNorm eps=1e-5 ('layernormbf16')
  * get_intermediate_layers(...) returning (patch_tokens, cls_token) per index

Parameter/buffer names match the checkpoint keys exactly so the trained weights
load with ``load_state_dict``.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=16, in_chans=3, embed_dim=1024):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                 # B C H W
        H, W = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # B HW C
        return x.reshape(-1, H, W, self.embed_dim)  # B H W C (flatten_embedding=False)


class RopePositionEmbedding(nn.Module):
    """RoPE sin/cos. periods = base^(2*i/(D_head/2)), base=100."""

    def __init__(self, embed_dim, num_heads, base=100.0, dtype=torch.float32, device=None):
        super().__init__()
        self.D_head = embed_dim // num_heads
        self.base = base
        self.dtype = dtype
        periods = base ** (2 * torch.arange(self.D_head // 4, dtype=dtype) / (self.D_head // 2))
        self.register_buffer('periods', periods, persistent=True)

    def forward(self, H, W):
        dd = dict(device=self.periods.device, dtype=self.dtype)
        coords_h = torch.arange(0.5, H, **dd) / H
        coords_w = torch.arange(0.5, W, **dd) / W
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'), dim=-1)  # H W 2
        coords = coords.flatten(0, 1)            # HW 2
        coords = 2.0 * coords - 1.0
        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]  # HW 2 D//4
        angles = angles.flatten(1, 2).tile(2)    # HW D
        return torch.sin(angles), torch.cos(angles)


class LinearKMaskedBias(nn.Linear):
    """nn.Linear whose bias middle third (the K part) is masked to zero."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        assert out_features % 3 == 0
        mask = torch.ones(out_features)
        o = out_features
        mask[o // 3: 2 * o // 3] = 0
        self.register_buffer('bias_mask', mask)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias * self.bias_mask.to(self.bias.dtype)
                        if self.bias is not None else None)


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=16, qkv_bias=True, proj_bias=True, mask_k_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        linear = LinearKMaskedBias if mask_k_bias else nn.Linear
        self.qkv = linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def _rope_apply(self, x, sin, cos):
        return x * cos + self._rotate_half(x) * sin

    def _apply_rope(self, q, k, rope):
        sin, cos = rope
        rope_dtype = sin.dtype
        q = q.to(rope_dtype)
        k = k.to(rope_dtype)
        N = q.shape[-2]
        prefix = N - sin.shape[-2]
        qp, q = q[:, :, :prefix], self._rope_apply(q[:, :, prefix:], sin, cos)
        q = torch.cat([qp, q], dim=-2)
        kp, k = k[:, :, :prefix], self._rope_apply(k[:, :, prefix:], sin, cos)
        k = torch.cat([kp, k], dim=-2)
        return q, k

    def forward(self, x, rope=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]   # B h N d
        if rope is not None:
            q, k = self._apply_rope(q, k, rope)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, act_layer=nn.GELU):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.empty(dim))

    def forward(self, x):
        return x * self.gamma


class SelfAttentionBlock(nn.Module):
    """Pre-norm block: x += ls1(attn(norm1(x))); x += ls2(mlp(norm2(x)))."""

    def __init__(self, dim, num_heads, ffn_ratio=4.0, qkv_bias=True, proj_bias=True,
                 ffn_bias=True, init_values=1e-5, norm_eps=1e-5, mask_k_bias=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn = SelfAttention(dim, num_heads, qkv_bias, proj_bias, mask_k_bias)
        self.ls1 = LayerScale(dim, init_values)
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps)
        self.mlp = Mlp(dim, int(dim * ffn_ratio)) if ffn_bias is not None else Mlp(dim, int(dim * ffn_ratio))
        self.ls2 = LayerScale(dim, init_values)

    def forward(self, x, rope=None):
        x = x + self.ls1(self.attn(self.norm1(x), rope=rope))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class DinoVisionTransformer(nn.Module):
    def __init__(self, embed_dim=1024, depth=24, num_heads=16, patch_size=16,
                 n_storage_tokens=4, rope_base=100.0, norm_eps=1e-5,
                 layerscale_init=1e-5, mask_k_bias=True, rope_dtype=torch.float32):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.n_blocks = depth
        self.num_heads = num_heads
        self.n_storage_tokens = n_storage_tokens

        self.patch_embed = PatchEmbed(patch_size, 3, embed_dim)
        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        self.storage_tokens = nn.Parameter(torch.empty(1, n_storage_tokens, embed_dim))
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim))
        self.rope_embed = RopePositionEmbedding(embed_dim, num_heads, base=rope_base, dtype=rope_dtype)
        self.blocks = nn.ModuleList([
            SelfAttentionBlock(embed_dim, num_heads, ffn_ratio=4.0, init_values=layerscale_init,
                               norm_eps=norm_eps, mask_k_bias=mask_k_bias)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=norm_eps)

    def _prepare_tokens(self, x):
        x = self.patch_embed(x)             # B H W C
        B, H, W, _ = x.shape
        x = x.flatten(1, 2)                 # B HW C
        cls = self.cls_token + 0 * self.mask_token
        x = torch.cat([cls.expand(B, -1, -1), self.storage_tokens.expand(B, -1, -1), x], dim=1)
        return x, (H, W)

    def get_intermediate_layers(self, x, n, norm=True):
        """Run all blocks; collect outputs at indices `n`; return
        [(patch_tokens, cls_token), ...] with patch_tokens = out[:, 5:]."""
        x, (H, W) = self._prepare_tokens(x)
        blocks_to_take = n if not isinstance(n, int) else range(self.n_blocks - n, self.n_blocks)
        outputs = []
        rope = self.rope_embed(H, W)
        for i, blk in enumerate(self.blocks):
            x = blk(x, rope=rope)
            if i in blocks_to_take:
                outputs.append(x)
        if norm:
            outputs = [self.norm(o) for o in outputs]
        return [(o[:, self.n_storage_tokens + 1:], o[:, 0]) for o in outputs]

    def forward(self, x, interaction_indexes):
        return self.get_intermediate_layers(x, interaction_indexes)
