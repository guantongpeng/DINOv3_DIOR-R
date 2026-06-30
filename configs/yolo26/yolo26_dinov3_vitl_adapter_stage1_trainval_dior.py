# =============================================================================
# STAGE 1: YOLO26 head + DINOv3 ViT-L/16 + ViT-Adapter — FROZEN ViT (DIOR-R)
# =============================================================================
# Freeze the pretrained DINOv3 ViT; train only the adapter (SPM + deformable
# interactions + projection), FPN and the YOLO26 O2M head. This adapts the
# strong frozen features to oriented detection without over-fitting / destroying
# them on the small DIOR-R train set.
#
# The O2O (one-to-one) head is NOT trained here: o2o_weight stays at its 0.0
# default (no ProgressiveLossHook in stage 1), so only the O2M (Task-Aligned)
# head receives gradients. The O2O head is ramped in during stage 2.
#
# After stage 1 finishes, run stage 2
# (yolo26_dinov3_vitl_adapter_stage2_trainval_dior.py) which loads this
# checkpoint, unfreezes the ViT, and fine-tunes end-to-end at a low backbone LR
# while ramping the O2O loss.
#
# Usage:
#   bash scripts/yolo26_vitl_adapter_trainval.sh          # runs stage1 then stage2
#   # or stage 1 only:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_GPUS=4 STAGE=1 bash scripts/yolo26_vitl_adapter_trainval.sh
# =============================================================================

_base_ = ['_yolo26_dinov3_vitl_adapter_base_trainval_dior.py']

# Frozen ViT (already the base default; stated explicitly for clarity).
model = dict(backbone=dict(freeze_vit=True))

# ----------------------------- Optimization --------------------------------
# Backbone (ViT) is frozen -> only adapter + FPN + O2M head are trained.
# lr is calibrated for the ViT-L recipe used by yolo26_vitl_adapter_trainval.sh:
# effective batch 128 (8 GPUs x samples_per_gpu=16, the script default). The
# detection stack (adapter + FPN + YOLO26 head) trains at 4e-4, matching the
# proven-good rotated-fcos / oriented-rcnn ViT-Adapter recipes on DIOR-R. If you
# change GPU count or samples_per_gpu, re-scale lr ~linearly (e.g. batch 64 ->
# lr ~2e-4). Watch the first ~2 epochs: grad_norm spikes / loss -> NaN means lr
# too high.
optimizer = dict(
    type='AdamW',
    lr=4e-4,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            # Frozen ViT params have requires_grad=False, so this lr_mult is moot
            # for stage 1; kept so the same paramwise layout carries into stage 2.
            # The DINOv3 ViT lives at model.backbone.adapter.backbone.* (the
            # adapter wraps the ViT under self.adapter.backbone).
            'backbone.adapter.backbone': dict(lr_mult=0.1),
        },
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
    ),
)

optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))

# warmup tuned to batch 128 (~92 iters/epoch on the trainval pool): 92 iters ~= 1 epoch.
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=92,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)

runner = dict(type='EpochBasedRunner', max_epochs=36)
