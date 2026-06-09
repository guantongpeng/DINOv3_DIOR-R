# Copyright (c) OpenMMLab. All rights reserved.
"""DINOv3 Vision Transformer backbone for mmrotate/mmdet.

This module wraps the DINOv3 ViT models from timm as a backbone compatible
with mmdetection/mmrotate detection frameworks.

DINOv3 is a self-supervised vision transformer pretrained by Meta AI that
produces high-quality visual features suitable for downstream dense prediction
tasks like object detection.

Supported models:
    - vit_small_patch16_dinov3 (embed_dim=384, depth=12)
    - vit_base_patch16_dinov3  (embed_dim=768, depth=12)
    - vit_large_patch16_dinov3 (embed_dim=1024, depth=24)
    - vit_huge_plus_patch16_dinov3 (embed_dim=1280, depth=32)

Reference:
    DINOv3: https://github.com/facebookresearch/dinov3
"""

import math
import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import build_norm_layer
from mmcv.runner import BaseModule, _load_checkpoint
from mmcv.utils import to_2tuple
from torch.nn.init import trunc_normal_

from mmdet.models.builder import BACKBONES
from mmdet.utils import get_root_logger

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


@BACKBONES.register_module()
class ViTDinoV3(BaseModule):
    """DINOv3 Vision Transformer backbone for mmrotate/mmdet.

    This backbone wraps the DINOv3 ViT models from the `timm` library and
    provides multi-scale feature outputs suitable for FPN-based detection
    frameworks like Oriented R-CNN.

    The ViT backbone processes images through patch embedding and multiple
    transformer blocks. Intermediate features are extracted from specified
    block indices in a **single forward pass**, then reshaped from
    (N, L, C) to (N, C, H, W) format with prefix tokens (cls_token,
    register tokens) removed.

    DINOv3 Architecture Notes:
        - Uses Rotary Position Embedding (RoPE) in each attention block.
        - Token sequence: [cls_token(1), reg_tokens(4), patches(H*W)]
        - No absolute position embedding (pos_embed=None), uses RoPE instead.
        - `dynamic_img_size=True` by default for flexible input sizes.

    Args:
        model_name (str): Name of the timm DINOv3 model.
            Supported: 'vit_small_patch16_dinov3',
            'vit_base_patch16_dinov3', 'vit_large_patch16_dinov3',
            'vit_huge_plus_patch16_dinov3'.
            Default: 'vit_base_patch16_dinov3'.
        pretrained (bool): Whether to load pretrained DINOv3 weights
            from timm's hub (HuggingFace). Default: True. If
            ``checkpoint_path`` is provided, this is automatically
            set to False.
        checkpoint_path (str, optional): Path to a locally downloaded
            .pth checkpoint file (e.g. from Meta's official DINOv3
            release). When provided, the timm model is created with
            ``pretrained=False`` and the local weights are loaded.
            Supports both full checkpoints (with 'state_dict' key)
            and raw state dicts. Default: None.
        out_indices (Sequence[int]): Indices of transformer blocks to
            output features from. For a 12-block ViT, [3, 5, 7, 11]
            extracts from blocks 4, 6, 8, and 12. Default: (3, 5, 7, 11).
        out_channels (int): Number of output channels per feature level.
            Default: 256.
        frozen_stages (int): Number of stages (transformer blocks) to freeze.
            -1 = no freezing, 0 = freeze patch_embed only,
            1+ = freeze patch_embed + first N blocks. Default: -1.
        with_cp (bool): Use activation checkpointing for transformer blocks.
            Default: False.
        norm_cfg (dict): Config dict for output projection normalization.
            Default: dict(type='LN', eps=1e-6).
        init_cfg (dict, optional): Config for weight initialization.
            Default: None.
        img_size (int | tuple): Input image size for position embedding.
            Default: 1024.
        drop_path_rate (float): Stochastic depth rate. Default: 0.0.

    Example:
        >>> backbone = ViTDinoV3(
        ...     model_name='vit_base_patch16_dinov3',
        ...     pretrained=True,
        ...     out_indices=(3, 5, 7, 11),
        ...     frozen_stages=8,
        ... )
        >>> x = torch.randn(2, 3, 800, 800)
        >>> feats = backbone(x)
        >>> for f in feats:
        ...     print(f.shape)
        torch.Size([2, 256, 50, 50])
        torch.Size([2, 256, 50, 50])
        torch.Size([2, 256, 50, 50])
        torch.Size([2, 256, 50, 50])
    """

    def __init__(
        self,
        model_name: str = 'vit_base_patch16_dinov3',
        pretrained: bool = True,
        checkpoint_path: Optional[str] = None,
        out_indices: Sequence[int] = (3, 5, 7, 11),
        out_channels: int = 256,
        frozen_stages: int = -1,
        with_cp: bool = False,
        norm_cfg: dict = None,
        init_cfg: Optional[dict] = None,
        img_size: Union[int, Tuple[int, int]] = 1024,
        drop_path_rate: float = 0.0,
    ):
        if not HAS_TIMM:
            raise ImportError(
                'timm is required for ViTDinoV3 backbone. '
                'Install with: pip install timm'
            )

        super().__init__(init_cfg=init_cfg)

        if norm_cfg is None:
            norm_cfg = dict(type='LN', eps=1e-6)

        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.out_indices = list(out_indices)
        self.out_channels = out_channels
        self.frozen_stages = frozen_stages
        self.with_cp = with_cp
        self.img_size = to_2tuple(img_size)

        # If a local checkpoint is provided, skip timm hub download
        if checkpoint_path is not None:
            pretrained = False

        self.pretrained_flag = pretrained

        # Create timm model
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            img_size=img_size if not pretrained else None,
            drop_path_rate=drop_path_rate,
        )

        # Load local checkpoint if provided
        if checkpoint_path is not None:
            self._load_local_checkpoint(checkpoint_path)

        # Extract model configuration
        self.embed_dim = self.vit.embed_dim
        self.patch_size = self.vit.patch_embed.patch_size[0]
        self.depth = len(self.vit.blocks)
        self.num_prefix_tokens = getattr(self.vit, 'num_prefix_tokens', 1)

        # Validate out_indices
        validated_indices = []
        for idx in self.out_indices:
            if idx < 0:
                idx = self.depth + idx
            if idx < 0 or idx >= self.depth:
                raise ValueError(
                    f'out_indices entry {idx} out of range for model with '
                    f'{self.depth} blocks. Valid range: [0, {self.depth - 1}]'
                )
            validated_indices.append(idx)
        self.out_indices = validated_indices

        # Sorted unique output indices for efficient single-pass extraction
        self.sorted_out_indices = sorted(set(self.out_indices))

        # Create output projection layers (1x1 conv + GroupNorm)
        # Use GroupNorm instead of LayerNorm since features are in NCHW format
        self.output_projections = nn.ModuleList()
        for _ in range(len(self.out_indices)):
            proj = nn.Sequential(
                nn.Conv2d(self.embed_dim, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=32, num_channels=out_channels),
            )
            self.output_projections.append(proj)

        # Store strides (all ViT features are at stride=patch_size)
        self.feat_strides = [self.patch_size] * len(self.out_indices)
        self.num_features = tuple([out_channels] * len(self.out_indices))

    def _load_local_checkpoint(self, checkpoint_path: str):
        """Load pretrained weights from a local .pth checkpoint file.

        Supports several common DINOv3 checkpoint formats:
            - Full training checkpoint with ``'state_dict'`` / ``'model'`` key
            - Raw state dict (keys map directly to timm model)
            - Meta official DINOv3 checkpoint (may have ``'teacher'`` key)

        The loader strips DDP ``module.`` prefix and logs any keys that are
        missing or unexpected.

        Args:
            checkpoint_path (str): Path to the local .pth file.
        """
        logger = get_root_logger()
        logger.info(f'Loading DINOv3 checkpoint from local path: {checkpoint_path}')

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # ---- Resolve state dict from various checkpoint formats ----
        if isinstance(checkpoint, dict):
            # Full training checkpoint (common pattern: {'state_dict': ..., ...})
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                logger.info('Extracted "state_dict" from checkpoint.')
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
                logger.info('Extracted "model" from checkpoint.')
            elif 'teacher' in checkpoint and isinstance(checkpoint['teacher'], dict):
                # Meta DINOv3 student-teacher format — use teacher weights
                state_dict = checkpoint['teacher']
                logger.info('Extracted "teacher" from checkpoint (Meta DINOv3 format).')
            else:
                # Assume the dict itself is the state dict
                state_dict = checkpoint
                logger.info('Using checkpoint dict directly as state dict.')
        else:
            state_dict = checkpoint

        # ---- Strip DDP "module." prefix if present ----
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {
                k[len('module.'):]: v for k, v in state_dict.items()
            }
            logger.info('Stripped "module." prefix from checkpoint keys.')

        # ---- Load into timm model ----
        model_state = self.vit.state_dict()
        missing, unexpected = [], []

        for k, v in state_dict.items():
            if k in model_state:
                if v.shape == model_state[k].shape:
                    model_state[k] = v
                else:
                    logger.warning(
                        f'Shape mismatch for "{k}": '
                        f'checkpoint {v.shape} vs model {model_state[k].shape}. '
                        f'Skipping.'
                    )
            else:
                unexpected.append(k)

        missing = [k for k in model_state.keys() if k not in state_dict]

        self.vit.load_state_dict(model_state, strict=False)

        # ---- Summarize load ----
        loaded_keys = len(model_state) - len(missing)
        logger.info(
            f'DINOv3 checkpoint loaded: {loaded_keys}/{len(model_state)} keys '
            f'matched, {len(missing)} missing, {len(unexpected)} unexpected.'
        )

        if missing:
            logger.debug(f'Missing keys: {missing}')
        if unexpected:
            logger.debug(f'Unexpected keys: {unexpected}')

    def _freeze_stages(self):
        """Freeze specified stages of the backbone."""
        if self.frozen_stages >= 0:
            # Freeze patch embedding
            self.vit.patch_embed.eval()
            for param in self.vit.patch_embed.parameters():
                param.requires_grad = False

            # Freeze cls_token and reg_token
            for token_name in ['cls_token', 'reg_token']:
                token = getattr(self.vit, token_name, None)
                if token is not None:
                    token.requires_grad = False

        # Freeze specified transformer blocks
        for i in range(min(self.frozen_stages, self.depth)):
            block = self.vit.blocks[i]
            block.eval()
            for param in block.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True):
        """Set training mode, handling frozen stages."""
        super().train(mode)
        self._freeze_stages()
        return self

    def init_weights(self):
        """Initialize weights.

        Since we load pretrained DINOv3 weights via timm, we only need to
        initialize the output projection layers.
        """
        logger = get_root_logger()

        if self.init_cfg is not None:
            super().init_weights()
            return

        # Initialize output projection layers
        for m in self.output_projections.modules():
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                nn.init.normal_(m.weight, 0, math.sqrt(2.0 / fan_out))
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        if self.checkpoint_path is not None:
            logger.info(
                f'DINOv3 backbone loaded with local checkpoint weights: '
                f'{self.checkpoint_path}'
            )
        elif self.pretrained_flag:
            logger.info(
                f'DINOv3 backbone loaded with pretrained weights: '
                f'{self.model_name}'
            )
        else:
            logger.warning(
                f'DINOv3 backbone initialized from scratch: '
                f'{self.model_name}'
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Forward pass - extracts multi-scale features in a single pass.

        Follows the timm DINOv3 ViT forward flow:
        1. Patch embedding → (B, H, W, C)
        2. _pos_embed → (B, L, C) tokens + RoPE frequencies
        3. norm_pre
        4. Iterate through transformer blocks (each block receives rope= kwarg)
        5. Remove prefix tokens (cls_token + reg_tokens), reshape to spatial
        6. Apply output projections (1x1 conv + LN)

        Args:
            x (Tensor): Input images of shape (B, C, H, W).

        Returns:
            tuple[Tensor]: Feature maps from each output index,
            each of shape (B, out_channels, H/patch_size, W/patch_size).
        """
        B, C, H, W = x.shape
        h_grid = H // self.patch_size
        w_grid = W // self.patch_size

        # Step 1: Patch embedding
        # timm's Eva patch_embed returns (B, H, W, C)
        x = self.vit.patch_embed(x)

        # Step 2: Position embedding
        # _pos_embed handles:
        #   - Flattening spatial dims: (B, H, W, C) -> (B, H*W, C)
        #   - Adding cls_token and reg_tokens (prefix tokens)
        #   - Adding absolute pos_embed if available (DINOv3: not used)
        #   - Generating RoPE frequency tensors
        #   - Applies pos_drop
        # Returns: (tokens (B, L, C), rot_pos_embed)
        #   L = num_prefix_tokens + H*W
        x, rot_pos_embed = self.vit._pos_embed(x)

        # Step 3: Pre-norm
        if hasattr(self.vit, 'norm_pre') and self.vit.norm_pre is not None:
            x = self.vit.norm_pre(x)

        # Step 4: Run transformer blocks, capturing intermediate outputs
        rope_mixed = getattr(self.vit, 'rope_mixed', False)
        capture_indices = set(self.sorted_out_indices)
        max_capture_idx = max(capture_indices)
        intermediate_features = {}

        for i, blk in enumerate(self.vit.blocks):
            # Determine RoPE for this specific block
            if rope_mixed and isinstance(rot_pos_embed, (list, tuple)):
                block_rope = rot_pos_embed[i]
            else:
                block_rope = rot_pos_embed

            # Forward through block (rope= is DINOv3-specific kwarg)
            if (self.with_cp and x.requires_grad
                    and not torch.jit.is_scripting()):
                x = cp.checkpoint(blk, x, block_rope, None, use_reentrant=False)
            else:
                x = blk(x, rope=block_rope)

            # Capture intermediate feature
            if i in capture_indices:
                intermediate_features[i] = x

            # Early exit optimization
            if i >= max_capture_idx:
                break

        # Step 5: Extract spatial features from tokens
        # Token layout: [prefix_tokens (num_prefix_tokens), patch_tokens (H*W)]
        # We need to strip prefix tokens and reshape patch tokens to 2D
        outs = []
        for i, out_idx in enumerate(self.out_indices):
            tokens = intermediate_features[out_idx]

            # Strip prefix tokens (cls_token + reg_tokens)
            # DINOv3: 1 cls_token + 4 reg_tokens = 5 prefix tokens
            patch_tokens = tokens[:, self.num_prefix_tokens:, :]

            # Reshape from (B, H*W, C) to (B, C, H, W)
            feat_map = patch_tokens.transpose(1, 2).reshape(
                B, self.embed_dim, h_grid, w_grid
            ).contiguous()

            # Apply output projection (1x1 conv + LayerNorm)
            feat_map = self.output_projections[i](feat_map)

            outs.append(feat_map)

        return tuple(outs)

    def forward_dummy(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Forward pass for computing FLOPs."""
        return self.forward(x)
