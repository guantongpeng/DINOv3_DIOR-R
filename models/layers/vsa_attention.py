"""Varied-Size Attention (VSA) for RVSA.

Extends MultiScaleDeformableAttention with per-head offset dilation.
"""

import torch
import torch.nn as nn
from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention, MultiScaleDeformableAttnFunction


@ATTENTION.register_module()
class VariedSizeAttention(MultiScaleDeformableAttention):

    def __init__(self, embed_dims=256, num_heads=8, num_levels=4,
                 num_points=4, dilation_rates=None, dropout=0.0, **kwargs):
        if dilation_rates is None:
            dilation_rates = [float(i + 1) for i in range(num_heads)]
        assert len(dilation_rates) == num_heads
        self.dilation_rates = dilation_rates
        super().__init__(
            embed_dims=embed_dims, num_heads=num_heads, num_levels=num_levels,
            num_points=num_points, dropout=dropout, batch_first=True, **kwargs)

    def forward(self, query, key=None, value=None, query_pos=None,
                key_padding_mask=None, reference_points=None,
                spatial_shapes=None, level_start_index=None, **kwargs):
        # NOTE: no internal residual here; the caller (encoder/decoder layer)
        # is responsible for adding the identity connection, matching the
        # standard Deformable DETR transformer layer.
        if value is None:
            value = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)

        # VSA: per-head dilation
        dilation = query.new_tensor(self.dilation_rates).view(1, 1, -1, 1, 1, 1)
        sampling_offsets = sampling_offsets * dilation

        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1).float()
            sampling_locations = (reference_points[:, :, None, :, None, :]
                                  + sampling_offsets
                                  / offset_normalizer[None, None, None, :, None, :])
        else:
            raise NotImplementedError

        output = MultiScaleDeformableAttnFunction.apply(
            value, spatial_shapes, level_start_index,
            sampling_locations.contiguous(), attention_weights, self.im2col_step)

        output = self.output_proj(output)
        if not self.batch_first:
            output = output.permute(1, 0, 2)
        return self.dropout(output)
