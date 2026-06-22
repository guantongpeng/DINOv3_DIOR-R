# =============================================================================
# STAGE 2: Oriented R-CNN + DINOv3 ViT-Adapter — END-TO-END fine-tune (DIOR-R)
# =============================================================================
# Loads the stage-1 checkpoint (frozen-ViT, trained adapter + head), unfreezes
# the ViT, and fine-tunes the whole network end-to-end at a low backbone LR.
# This is the official DINOv3 two-stage recipe:
#     stage 1: freeze backbone, train detection stack at lr 1e-4
#     stage 2: unfreeze, lr 5e-5 with backbone lr_mult 0.1 (ViT @ 5e-6)
#
# Point load_from at the stage-1 best/latest checkpoint, e.g.:
#   tools/dist_train_adapter_twostage.sh does this automatically, or manually:
#   python tools/train.py <this_config> \
#       --cfg-options load_from=work_dirs/.../stage1/best_mAP_epoch_XX.pth
#
# NOTE: RegZeroInitHook is intentionally REMOVED here so we do not re-zero the
# regression head that stage 1 already learned.
# =============================================================================

_base_ = ['_oriented_rcnn_dinov3_vitb_adapter_base_dior.py']

# Unfreeze the ViT for end-to-end fine-tuning.
model = dict(backbone=dict(freeze_vit=False))

# Only keep EMA in stage 2 (drop RegZeroInitHook — do not re-zero trained head).
custom_hooks = [
    dict(type='EMAHook', momentum=0.9999, priority='ABOVE_NORMAL'),
]

# ----------------------------- Optimization --------------------------------
# End-to-end: head/adapter at 5e-5, ViT backbone at 0.1x = 5e-6.
optimizer = dict(
    type='AdamW',
    lr=5e-5,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'backbone.backbone': dict(lr_mult=0.1),   # the DINOv3 ViT @ 5e-6
        },
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
    ),
)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)

runner = dict(type='EpochBasedRunner', max_epochs=40)

# Weights from stage 1. Pass via --cfg-options load_from=<path> (the two-stage
# script does this). Leave None here so running the config standalone is safe.
load_from = None
