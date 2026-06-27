# =============================================================================
# STAGE 2: RVSA + DINOv3 ViT-L/16 + ViT-Adapter -- END-TO-END (DIOR-R, TRAINVAL)
# =============================================================================
# Loads the stage-1 checkpoint (frozen-ViT, trained adapter + FPN + head),
# unfreezes the ViT, and fine-tunes the whole network end-to-end at a low
# backbone LR. Adapted from the DINOv3 two-stage recipe:
#     stage 1: freeze backbone, train detection stack at lr 4e-4
#     stage 2: unfreeze, lr 2e-4 with backbone lr_mult 0.1 (ViT @ 2e-5)
#
# Point load_from at the stage-1 best/latest checkpoint, e.g.:
#   scripts/rvsa_vitl_adapter_trainval.sh does this automatically, or manually:
#   python tools/train.py <this_config> \
#       --cfg-options load_from=work_dirs/.../stage1/best_mAP_epoch_XX.pth
#
# NOTE: RVSA has no RoI regression FC, so there is no RegZeroInitHook to drop
# (unlike the Oriented R-CNN adapter stage 2). Only EMA is kept here.
# =============================================================================

_base_ = ['_rvsa_dinov3_vitl_adapter_base_trainval_dior.py']

# Unfreeze the ViT for end-to-end fine-tuning.
model = dict(backbone=dict(freeze_vit=False))

# Only keep EMA in stage 2 (RVSA base has no RoI-specific hooks to remove).
# momentum lowered to 0.9998 for the short schedule: 0.9999's half-life would
# barely update the EMA over one training pass.
custom_hooks = [
    dict(type='EMAHook', momentum=0.9998, priority='ABOVE_NORMAL'),
]

# ----------------------------- Optimization --------------------------------
# End-to-end: head/adapter at 2e-4, ViT backbone at 0.1x = 2e-5. Stage 2 head lr
# is 0.5x stage 1 (4e-4), keeping the two-stage ratio of the DINOv3 recipe
# (stage2 = 0.5x stage1). Re-scale if you change batch size.
optimizer = dict(
    type='AdamW',
    lr=2e-4,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'backbone.backbone': dict(lr_mult=0.1),   # the DINOv3 ViT @ 2e-5
        },
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
    ),
)

optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))

# warmup tuned to batch ~128 (~92 iters/epoch on the trainval pool).
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=92,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)

runner = dict(type='EpochBasedRunner', max_epochs=24)

# Weights from stage 1. Pass via --cfg-options load_from=<path> (the two-stage
# script does this). Leave None here so running the config standalone is safe.
load_from = None
