"""VSA Transformer Encoder and Decoder for RVSA. Uses batch_first=True."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, build_feedforward_network
from mmcv.runner import BaseModule, ModuleList
from mmdet.models.utils.builder import TRANSFORMER


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


@TRANSFORMER.register_module()
class VSATransformer(BaseModule):

    def __init__(self, encoder=None, decoder=None, num_feature_levels=4,
                 embed_dims=256, two_stage_num_proposals=300, **kwargs):
        super().__init__(**kwargs)
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        if encoder is not None:
            encoder = {k: v for k, v in encoder.items() if k != 'type'}
            self.encoder = VSAEncoder(**encoder)
        else:
            self.encoder = None
        if decoder is not None:
            decoder = {k: v for k, v in decoder.items() if k != 'type'}
            self.decoder = VSADecoder(**decoder)
        else:
            self.decoder = None
        self.level_embeds = nn.Parameter(torch.Tensor(num_feature_levels, embed_dims))
        self.reference_points = nn.Linear(embed_dims, 2)
        self.two_stage_num_proposals = two_stage_num_proposals
        self._init_params()

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.level_embeds)

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            H_, W_ = int(H_), int(W_)
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
                indexing='ij',
            )
            # Standard Deformable DETR normalization; keep everything on-device
            # (no .item() sync) by scaling with the per-image valid ratios.
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def get_valid_ratio(self, mask):
        B, H, W = mask.shape
        valid_H = (~mask[:, :, 0]).sum(dim=1).float()
        valid_W = (~mask[:, 0, :]).sum(dim=1).float()
        return torch.stack([valid_W / W, valid_H / H], -1)

    def forward(self, mlvl_feats, mlvl_masks, query_embed, mlvl_pos_embeds,
                reg_branches=None, cls_branches=None, **kwargs):
        B = mlvl_feats[0].size(0)

        spatial_shapes = torch.as_tensor(
            [(f.shape[2], f.shape[3]) for f in mlvl_feats],
            dtype=torch.long, device=mlvl_feats[0].device)
        level_start_index = torch.cat((
            spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in mlvl_masks], 1)

        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        for lvl, feat in enumerate(mlvl_feats):
            B_, C, H, W = feat.shape
            feat_flatten.append(feat.view(B_, C, -1))
            mask_flatten.append(mlvl_masks[lvl].view(B_, -1))
            lvl_pos_embed_flatten.append(mlvl_pos_embeds[lvl].view(B_, C, -1))
        feat_flatten = torch.cat(feat_flatten, 2).permute(0, 2, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 2).permute(0, 2, 1)

        memory = feat_flatten
        if self.encoder is not None:
            ref_pts = self.get_reference_points(spatial_shapes, valid_ratios, feat_flatten.device)
            memory = self.encoder(
                query=feat_flatten, query_pos=lvl_pos_embed_flatten,
                spatial_shapes=spatial_shapes, level_start_index=level_start_index,
                valid_ratios=valid_ratios, reference_points=ref_pts,
                key_padding_mask=mask_flatten)

        init_reference_out = None
        inter_references_out = None
        hs = None

        if self.decoder is not None and query_embed is not None:
            _, _, C = memory.shape
            query_embed_tgt, tgt = torch.split(query_embed, C, dim=1)
            query_embed_tgt = query_embed_tgt.unsqueeze(0).expand(B, -1, -1)
            tgt = tgt.unsqueeze(0).expand(B, -1, -1)
            reference_points = self.reference_points(query_embed_tgt).sigmoid()
            init_reference_out = reference_points
            hs, inter_references_out = self.decoder(
                query=tgt, key=memory, value=memory,
                query_pos=query_embed_tgt, reference_points=reference_points,
                spatial_shapes=spatial_shapes, level_start_index=level_start_index,
                valid_ratios=valid_ratios, key_padding_mask=mask_flatten,
                reg_branches=reg_branches)
        return hs, init_reference_out, inter_references_out, None, None


class VSAEncoder(BaseModule):

    def __init__(self, num_layers=6, embed_dims=256, feedforward_channels=1024,
                 num_heads=8, num_levels=4, num_points=4, dilation_rates=None,
                 ffn_cfg=None, norm_cfg=None, **kwargs):
        super().__init__(**kwargs)
        if norm_cfg is None:
            norm_cfg = dict(type='LN')
        if ffn_cfg is None:
            ffn_cfg = dict(type='FFN', embed_dims=embed_dims,
                           feedforward_channels=feedforward_channels,
                           num_fcs=2, ffn_drop=0.0,
                           act_cfg=dict(type='ReLU', inplace=True))
        if dilation_rates is None:
            dilation_rates = [float(i + 1) for i in range(num_heads)]
        self.layers = ModuleList()
        for _ in range(num_layers):
            self.layers.append(VSAEncoderLayer(
                embed_dims=embed_dims, num_heads=num_heads,
                num_levels=num_levels, num_points=num_points,
                dilation_rates=dilation_rates,
                feedforward_channels=feedforward_channels,
                ffn_cfg=ffn_cfg, norm_cfg=norm_cfg))
        self.norms = ModuleList(
            [build_norm_layer(norm_cfg, embed_dims)[1] for _ in range(num_layers)])

    def forward(self, query, query_pos=None, spatial_shapes=None,
                level_start_index=None, valid_ratios=None,
                reference_points=None, key_padding_mask=None, **kwargs):
        output = query
        for i, layer in enumerate(self.layers):
            output = layer(output, query_pos=query_pos,
                           spatial_shapes=spatial_shapes,
                           level_start_index=level_start_index,
                           valid_ratios=valid_ratios,
                           reference_points=reference_points,
                           key_padding_mask=key_padding_mask)
            output = self.norms[i](output)
        return output


class VSAEncoderLayer(BaseModule):

    def __init__(self, embed_dims=256, num_heads=8, num_levels=4, num_points=4,
                 dilation_rates=None, feedforward_channels=1024,
                 ffn_cfg=None, norm_cfg=None, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        if norm_cfg is None:
            norm_cfg = dict(type='LN')
        if ffn_cfg is None:
            ffn_cfg = dict(type='FFN', embed_dims=embed_dims,
                           feedforward_channels=feedforward_channels,
                           num_fcs=2, ffn_drop=0.0,
                           act_cfg=dict(type='ReLU', inplace=True))
        from .vsa_attention import VariedSizeAttention
        self.self_attn = VariedSizeAttention(
            embed_dims=embed_dims, num_heads=num_heads,
            num_levels=num_levels, num_points=num_points,
            dilation_rates=dilation_rates, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.ffn = build_feedforward_network(ffn_cfg)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]

    def forward(self, query, query_pos=None, spatial_shapes=None,
                level_start_index=None, valid_ratios=None,
                reference_points=None, key_padding_mask=None, **kwargs):
        src = query
        query2 = self.self_attn(
            query=query, key=query, value=query, query_pos=query_pos,
            reference_points=reference_points, spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            key_padding_mask=key_padding_mask)
        query = src + self.dropout1(query2)
        query = self.norm1(query)
        query2 = self.ffn(query)
        query = query + self.dropout2(query2)
        query = self.norm2(query)
        return query


class VSADecoder(BaseModule):

    def __init__(self, num_layers=6, embed_dims=256, feedforward_channels=1024,
                 num_heads=8, num_levels=4, num_points=4, dilation_rates=None,
                 return_intermediate=True, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.layers = ModuleList()
        for _ in range(num_layers):
            self.layers.append(VSADecoderLayer(
                embed_dims=embed_dims, num_heads=num_heads,
                num_levels=num_levels, num_points=num_points,
                dilation_rates=dilation_rates,
                feedforward_channels=feedforward_channels))

    def forward(self, query, key=None, value=None, query_pos=None,
                reference_points=None, spatial_shapes=None,
                level_start_index=None, valid_ratios=None,
                key_padding_mask=None, reg_branches=None, **kwargs):
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            ref_pts_input = reference_points[:, :, None] * valid_ratios[:, None]
            output = layer(output, key, value, query_pos=query_pos,
                           reference_points=ref_pts_input,
                           spatial_shapes=spatial_shapes,
                           level_start_index=level_start_index,
                           key_padding_mask=key_padding_mask)
            if reg_branches is not None:
                tmp = reg_branches[lid](output)
                new_ref = tmp[..., :2] + inverse_sigmoid(reference_points)
                new_ref = new_ref.sigmoid()
                reference_points = new_ref.detach()
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)
        return output, reference_points


class VSADecoderLayer(BaseModule):

    def __init__(self, embed_dims=256, num_heads=8, num_levels=4,
                 num_points=4, dilation_rates=None,
                 feedforward_channels=1024, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.self_attn = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dims)
        from .vsa_attention import VariedSizeAttention
        self.cross_attn = VariedSizeAttention(
            embed_dims=embed_dims, num_heads=num_heads,
            num_levels=num_levels, num_points=num_points,
            dilation_rates=dilation_rates, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.ffn = FFN(embed_dims, feedforward_channels, num_fcs=2,
                       act_cfg=dict(type='ReLU', inplace=True),
                       dropout=dropout, add_residual=False)
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(embed_dims)

    def forward(self, query, key=None, value=None, query_pos=None,
                reference_points=None, spatial_shapes=None,
                level_start_index=None, key_padding_mask=None, **kwargs):
        q = k = self._with_pos_embed(query, query_pos)
        query2 = self.self_attn(q, k, value=query)[0]
        query = query + self.dropout1(query2)
        query = self.norm1(query)
        query2 = self.cross_attn(
            query=self._with_pos_embed(query, query_pos),
            key=key, value=value, reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            key_padding_mask=key_padding_mask)
        query = query + self.dropout2(query2)
        query = self.norm2(query)
        query2 = self.ffn(query)
        query = query + self.dropout3(query2)
        query = self.norm3(query)
        return query

    @staticmethod
    def _with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos
