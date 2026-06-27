# =============================================================================
# STAGE 1: Oriented R-CNN + DINOv3 ViT-Adapter — FROZEN ViT (DIOR-R)
# =============================================================================
# Freeze the pretrained DINOv3 ViT; train only the adapter (SPM + deformable
# interactions + projection), RPN and RoI head. This adapts the strong frozen
# features to oriented detection without over-fitting / destroying them on the
# small DIOR-R train set.
#
# After stage 1 finishes, run stage 2
# (oriented_rcnn_dinov3_vitb_adapter_stage2_trainval_dior.py) which loads this
# checkpoint, unfreezes the ViT, and fine-tunes end-to-end at a low backbone LR.
#
# Usage:
#   bash scripts/orcnn_vitb_adapter_trainval.sh          # runs stage1 then stage2
#   # or stage 1 only:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_GPUS=4 STAGE=1 bash scripts/orcnn_vitb_adapter_trainval.sh
# =============================================================================

_base_ = ['_oriented_rcnn_dinov3_vitb_adapter_base_trainval_dior.py']

# Frozen ViT (already the base default; stated explicitly for clarity).
model = dict(backbone=dict(freeze_vit=True))

# ----------------------------- Optimization --------------------------------
# Backbone (ViT) is frozen -> only adapter + RPN + RoI are trained.
# lr is calibrated for the large-batch recipe used by
# orcnn_vitb_adapter_trainval.sh: effective batch 192 (8 GPUs x samples_per_gpu=24).
# If you change GPU count or samples_per_gpu, re-scale lr ~linearly (e.g. halve for
# batch ~96). Watch the first ~2 epochs: grad_norm spikes / loss -> NaN means lr too high.
optimizer = dict(
    type='AdamW',
    lr=2e-4,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            # Frozen ViT params have requires_grad=False, so this lr_mult is moot
            # for stage 1; kept so the same paramwise layout carries into stage 2.
            'backbone.backbone': dict(lr_mult=0.1),
        },
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
    ),
)

optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))

# warmup tuned to batch 192 (~62 iters/epoch): 100 iters ~= 1.6 epochs.
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)

runner = dict(type='EpochBasedRunner', max_epochs=120)
