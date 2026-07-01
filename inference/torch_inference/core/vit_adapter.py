"""Pure-PyTorch port of the official DINOv3 ViT-Adapter (no dinov3/mm* deps).

Ported from ``third_party/dinov3/.../dinov3_adapter.py`` + the mmcv multi-scale
deformable attention that the trained checkpoint actually uses (the backbone
wrapper swaps the adapter's MSDeformAttn for mmcv's). All parameter names match
the checkpoint so weights load directly. SyncBatchNorm -> BatchNorm2d (identical
in eval mode with the saved running stats).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dinov3_vit import DinoVisionTransformer


# ---------------------------------------------------------------------------
# deformable attention (pure-torch port of mmcv MultiScaleDeformableAttention)
# ---------------------------------------------------------------------------
def multi_scale_deformable_attn_pytorch(value, value_spatial_shapes,
                                        sampling_locations, attention_weights):
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, _, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([int(H_ * W_) for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        value_l = value_list[level].flatten(2).transpose(1, 2).reshape(
            bs * num_heads, embed_dims, int(H_), int(W_))
        sampling_grid_l = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampling_value_list.append(F.grid_sample(value_l, sampling_grid_l, mode='bilinear',
                                                  padding_mode='zeros', align_corners=False))
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights
              ).sum(-1).view(bs, num_heads * embed_dims, num_queries)
    return output.transpose(1, 2).contiguous()


class _MSDA(nn.Module):
    """mmcv MultiScaleDeformableAttention (the trained checkpoint's structure)."""

    def __init__(self, embed_dims=1024, num_levels=1, num_heads=16, num_points=4):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

    def forward(self, query, reference_points, value, spatial_shapes, level_start_index,
                padding_mask=None):
        bs, num_query, _ = query.shape
        _, num_value, _ = value.shape
        value = self.value_proj(value)
        if padding_mask is not None:
            value = value.masked_fill(padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points)
        # reference_points: (bs, num_query, num_levels, 2)
        offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        sampling_locations = (reference_points[:, :, None, :, None, :]
                              + sampling_offsets / offset_normalizer[None, None, None, :, None, :])
        output = multi_scale_deformable_attn_pytorch(value, spatial_shapes, sampling_locations,
                                                      attention_weights)
        return self.output_proj(output)


class _MSDAWrapper(nn.Module):
    """Mirrors the project's _MmcvMSDeformAttn (suppresses mmcv residual)."""

    def __init__(self, d_model, n_levels, n_heads, n_points):
        super().__init__()
        self.attn = _MSDA(d_model, n_levels, n_heads, n_points)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes,
                input_level_start_index, input_padding_mask=None):
        return self.attn(query=query, value=input_flatten, reference_points=reference_points,
                         spatial_shapes=input_spatial_shapes,
                         level_start_index=input_level_start_index,
                         padding_mask=input_padding_mask)


# ---------------------------------------------------------------------------
# adapter building blocks (verbatim from dinov3_adapter.py)
# ---------------------------------------------------------------------------
def get_reference_points(spatial_shapes, device):
    reference_points_list = []
    for (H_, W_) in spatial_shapes:
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        reference_points_list.append(torch.stack((ref_x, ref_y), -1))
    reference_points = torch.cat(reference_points_list, 1)[:, :, None]
    return reference_points


def deform_inputs(x, patch_size):
    bs, c, h, w = x.shape
    spatial_shapes = torch.as_tensor([(h // 8, w // 8), (h // 16, w // 16), (h // 32, w // 32)],
                                     dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // patch_size, w // patch_size)], x.device)
    deform_inputs1 = [reference_points, spatial_shapes, level_start_index]

    spatial_shapes = torch.as_tensor([(h // patch_size, w // patch_size)], dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // 8, w // 8), (h // 16, w // 16), (h // 32, w // 32)], x.device)
    deform_inputs2 = [reference_points, spatial_shapes, level_start_index]
    return deform_inputs1, deform_inputs2


class DWConv(nn.Module):
    def __init__(self, dim=1024):
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
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Extractor(nn.Module):
    def __init__(self, dim, num_heads=16, n_points=4, n_levels=1, with_cffn=True,
                 cffn_ratio=0.25, drop_path=0.0):
        super().__init__()
        norm = lambda d: nn.LayerNorm(d, eps=1e-6)
        self.query_norm = norm(dim)
        self.feat_norm = norm(dim)
        self.attn = _MSDAWrapper(dim, n_levels, num_heads, n_points)
        self.with_cffn = with_cffn
        if with_cffn:
            self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * cffn_ratio))
            self.ffn_norm = norm(dim)

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index, H, W):
        attn = self.attn(self.query_norm(query), reference_points, self.feat_norm(feat),
                         spatial_shapes, level_start_index, None)
        query = query + attn
        if self.with_cffn:
            query = query + self.ffn(self.ffn_norm(query), H, W)
        return query


class InteractionBlockWithCls(nn.Module):
    def __init__(self, dim, num_heads=16, n_points=4, drop_path=0.3, with_cffn=True,
                 cffn_ratio=0.25, deform_ratio=0.5, extra_extractor=False):
        super().__init__()
        self.extractor = Extractor(dim, num_heads, n_points, 1, with_cffn, cffn_ratio, drop_path)
        if extra_extractor:
            self.extra_extractors = nn.Sequential(
                Extractor(dim, num_heads, n_points, 1, with_cffn, cffn_ratio, drop_path),
                Extractor(dim, num_heads, n_points, 1, with_cffn, cffn_ratio, drop_path))
        else:
            self.extra_extractors = None

    def forward(self, x, c, cls, deform_inputs1, deform_inputs2, H_c, W_c, H_toks, W_toks):
        c = self.extractor(query=c, reference_points=deform_inputs2[0], feat=x,
                           spatial_shapes=deform_inputs2[1], level_start_index=deform_inputs2[2],
                           H=H_c, W=W_c)
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(query=c, reference_points=deform_inputs2[0], feat=x,
                              spatial_shapes=deform_inputs2[1], level_start_index=deform_inputs2[2],
                              H=H_c, W=W_c)
        return x, c, cls


class SpatialPriorModule(nn.Module):
    def __init__(self, inplanes=64, embed_dim=1024):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, inplanes, 3, 2, 1, bias=False), nn.BatchNorm2d(inplanes), nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, 3, 1, 1, bias=False), nn.BatchNorm2d(inplanes), nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, 3, 1, 1, bias=False), nn.BatchNorm2d(inplanes), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.conv2 = nn.Sequential(nn.Conv2d(inplanes, 2 * inplanes, 3, 2, 1, bias=False),
                                   nn.BatchNorm2d(2 * inplanes), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(2 * inplanes, 4 * inplanes, 3, 2, 1, bias=False),
                                   nn.BatchNorm2d(4 * inplanes), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(4 * inplanes, 4 * inplanes, 3, 2, 1, bias=False),
                                   nn.BatchNorm2d(4 * inplanes), nn.ReLU(inplace=True))
        self.fc1 = nn.Conv2d(inplanes, embed_dim, 1, 1, 0, bias=True)
        self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, 1, 1, 0, bias=True)
        self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, 1, 1, 0, bias=True)
        self.fc4 = nn.Conv2d(4 * inplanes, embed_dim, 1, 1, 0, bias=True)

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c1, c2, c3, c4 = self.fc1(c1), self.fc2(c2), self.fc3(c3), self.fc4(c4)
        bs, dim = c1.shape[0], c1.shape[1]
        c2 = c2.view(bs, dim, -1).transpose(1, 2)
        c3 = c3.view(bs, dim, -1).transpose(1, 2)
        c4 = c4.view(bs, dim, -1).transpose(1, 2)
        return c1, c2, c3, c4


class DINOv3_Adapter(nn.Module):
    def __init__(self, backbone, interaction_indexes=(5, 11, 17, 23), pretrain_size=512,
                 conv_inplane=64, n_points=4, deform_num_heads=16, drop_path_rate=0.3,
                 cffn_ratio=0.25, deform_ratio=0.5, use_extra_extractor=True):
        super().__init__()
        self.backbone = backbone
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.interaction_indexes = list(interaction_indexes)
        embed_dim = backbone.embed_dim
        self.patch_size = backbone.patch_size

        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.spm = SpatialPriorModule(conv_inplane, embed_dim)
        self.interactions = nn.Sequential(*[
            InteractionBlockWithCls(
                embed_dim, deform_num_heads, n_points, drop_path_rate, True, cffn_ratio,
                deform_ratio, extra_extractor=(i == len(self.interaction_indexes) - 1) and use_extra_extractor)
            for i in range(len(self.interaction_indexes))])
        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.norm1 = nn.BatchNorm2d(embed_dim)
        self.norm2 = nn.BatchNorm2d(embed_dim)
        self.norm3 = nn.BatchNorm2d(embed_dim)
        self.norm4 = nn.BatchNorm2d(embed_dim)

    def _add_level_embed(self, c2, c3, c4):
        return c2 + self.level_embed[0], c3 + self.level_embed[1], c4 + self.level_embed[2]

    def forward(self, x):
        deform_inputs1, deform_inputs2 = deform_inputs(x, self.patch_size)
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        H_c, W_c = x.shape[2] // 16, x.shape[3] // 16
        H_toks, W_toks = x.shape[2] // self.patch_size, x.shape[3] // self.patch_size
        bs = x.shape[0]
        dim = self.backbone.embed_dim

        all_layers = self.backbone.get_intermediate_layers(x, n=self.interaction_indexes)

        outs = []
        for i, layer in enumerate(self.interactions):
            x_tok, cls = all_layers[i]
            _, c, _ = layer(x_tok, c, cls, deform_inputs1, deform_inputs2, H_c, W_c, H_toks, W_toks)
            outs.append(x_tok.transpose(1, 2).view(bs, dim, H_toks, W_toks).contiguous())

        n2 = c2.size(1)
        n3 = c3.size(1)
        c2 = c[:, 0:n2, :].transpose(1, 2).view(bs, dim, H_c * 2, W_c * 2).contiguous()
        c3 = c[:, n2:n2 + n3, :].transpose(1, 2).view(bs, dim, H_c, W_c).contiguous()
        c4 = c[:, n2 + n3:, :].transpose(1, 2).view(bs, dim, H_c // 2, W_c // 2).contiguous()
        c1 = self.up(c2) + c1

        x1, x2, x3, x4 = outs
        x1 = F.interpolate(x1, size=(4 * H_c, 4 * W_c), mode='bilinear', align_corners=False)
        x2 = F.interpolate(x2, size=(2 * H_c, 2 * W_c), mode='bilinear', align_corners=False)
        x3 = F.interpolate(x3, size=(H_c, W_c), mode='bilinear', align_corners=False)
        x4 = F.interpolate(x4, size=(H_c // 2, W_c // 2), mode='bilinear', align_corners=False)
        c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        return {'1': self.norm1(c1), '2': self.norm2(c2), '3': self.norm3(c3), '4': self.norm4(c4)}


class DINOv3ViTAdapter(nn.Module):
    """Wraps DINOv3_Adapter as a 4-level backbone (strides 4,8,16,32, 1024-ch)."""

    def __init__(self, model_name='dinov3_vitl16', interaction_indexes=(5, 11, 17, 23),
                 pretrain_size=512, conv_inplane=64, n_points=4, deform_num_heads=16,
                 drop_path_rate=0.3, cffn_ratio=0.25, deform_ratio=0.5,
                 use_extra_extractor=True):
        super().__init__()
        embed_dim, depth, patch_size = 1024, 24, 16
        vit = DinoVisionTransformer(embed_dim=embed_dim, depth=depth, num_heads=16,
                                    patch_size=patch_size, n_storage_tokens=4, rope_base=100.0)
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.adapter = DINOv3_Adapter(
            vit, interaction_indexes, pretrain_size, conv_inplane, n_points, deform_num_heads,
            drop_path_rate, cffn_ratio, deform_ratio, use_extra_extractor)

    @property
    def out_channels(self):
        return [self.embed_dim] * 4

    def forward(self, x):
        out = self.adapter(x)
        return tuple(out[k].contiguous() for k in ('1', '2', '3', '4'))
