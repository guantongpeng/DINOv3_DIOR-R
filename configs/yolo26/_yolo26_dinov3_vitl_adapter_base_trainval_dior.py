# =============================================================================
# BASE config: YOLO26 head + DINOv3 ViT-L/16 + ViT-Adapter (DIOR-R)
# =============================================================================
# This is a SHARED base. Do not train it directly — use the stage1/stage2
# configs that inherit from it.
#
# Same detector idea as yolo26_dinov3_fpn_train_dior.py (anchor-free YOLO26
# rotated head), but on the ViT-L/16 + ViT-Adapter backbone used by the
# rotated-fcos / oriented-rcnn ViT-L recipes, so the detectors are directly
# comparable. Data uses the TRAINVAL split: train on train+val, eval/select on
# the held-out test split.
#
# What this recipe implements (mirrors the rotated-fcos ViT-Adapter base):
#   1. BACKBONE  -> ViT-Adapter (deformable attention + multi-layer ViT fusion).
#                   The frozen ViT runs under bfloat16 autocast (official DINOv3
#                   eval recipe). Outputs the FULL embed_dim (1024-d) pyramid for
#                   the FPN to fuse.
#   2. NECK      -> FPN (lateral + top-down fusion, +P5 stride-64). 5 output
#                   levels @ strides [4,8,16,32,64], matching the YOLO26 head.
#   3. HEAD      -> YOLO26RotatedHead (anchor-free, dual-head O2M+O2O,
#                   NMS-free capable). strides=[4,8,16,32,64] match the FPN.
#   4. SIZES     -> all train/test scales + pad are multiples of 32 (ViT-Adapter
#                   needs a stride-32 prior level).
#   5. PRECISION -> bf16 for the heavy frozen ViT (inside the backbone); the
#                   trainable head/adapter train in fp32 (fp16 AMP disabled).
#   6. FREEZE    -> two-stage: see stage1 (frozen ViT) / stage2 (end-to-end).
#
# The optimizer / lr_config / runner live in the stage configs, NOT here.
#
# Backbone freeze note:
#   DINOv3ViTAdapter.freeze_vit:
#     True  = keep the ViT frozen (STAGE-1 default); only the adapter + FPN + head
#             train (the ViT still runs under no_grad via the official adapter).
#     False = unfreeze the ViT for end-to-end fine-tuning (STAGE 2).
#   The inner DINOv3 ViT lives at model.backbone.adapter.backbone.* (the adapter
#   wraps the ViT under self.adapter.backbone), so paramwise_cfg targets the
#   substring 'backbone.adapter.backbone'.
# =============================================================================

_base_ = []

custom_imports = dict(
    imports=[
        'models.backbones.dinov3_vit_adapter',  # registers DINOv3ViTAdapter
        'models.datasets.dior',
        'models.heads.yolo26_rotated_head',
        'models.detectors.dinov3_yolo26',
        'models.pipelines.albu_metadata',
        'models.hooks',
    ],
    allow_failed_imports=False,
)

model = dict(
    type='DINOv3YOLO26',

    # -------------------- Backbone: DINOv3 ViT-Adapter -----------------------
    # Frozen ViT-L/16 (1024-d, 24 blocks) + Spatial Prior Module + 4 deformable
    # InteractionBlocks fusing ViT layers [5,11,17,23]. Outputs a 4-level pyramid
    # @ 1024 channels (full embed_dim), strides [4,8,16,32], fed into FPN.
    # bf16_vit runs the frozen ViT in bfloat16.
    # freeze_vit=True is the STAGE-1 default; stage 2 overrides to False.
    backbone=dict(
        type='DINOv3ViTAdapter',
        model_name='dinov3_vitl16',
        interaction_indexes=[5, 11, 17, 23],
        freeze_vit=True,            # stage-1 default (frozen); stage 2 -> False
        pretrain_size=512,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=16,
        drop_path_rate=0.3,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        use_extra_extractor=True,
        with_cp=True,               # gradient checkpointing in extractors
        bf16_vit=True,              # bfloat16 autocast for the frozen ViT
        init_cfg=dict(
            checkpoint='data/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth',
        ),
    ),

    # -------------------- Neck: FPN on full embed_dim pyramid ----------------
    # 4 adapter levels @ 1024 -> FPN fuses them to 5 levels @ 256:
    # strides [4, 8, 16, 32, 64] (P5 added via on_output extra conv).
    neck=dict(
        type='FPN',
        in_channels=[1024, 1024, 1024, 1024],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_output',   # add P5 (stride 64) from the last level
        num_outs=5,                    # -> strides [4, 8, 16, 32, 64]
        relu_before_extra_convs=True,
    ),

    # -------------------- YOLO26 Rotated Head (anchor-free, dual-head) -------
    # O2M head: Task-Aligned Label Assignment (TAL). O2O head: Hungarian
    # matching for NMS-free inference. Angle encoded sigmoid -> [-pi/2, pi/2]
    # (le90). No DFL (lighter regression head). strides match the FPN levels.
    # The O2O head's loss weight is driven by ProgressiveLossHook (NOT by
    # train_cfg); o2o_weight defaults to 0.0, so stage 1 (no hook) trains O2M only.
    bbox_head=dict(
        type='YOLO26RotatedHead',
        num_classes=20,             # DIOR-R has 20 classes
        in_channels=256,            # matches FPN out_channels
        feat_channels=128,
        stacked_convs=2,
        strides=[4, 8, 16, 32, 64], # must match the FPN output strides
        reg_max=16,
        use_dfl=False,              # YOLO26: DFL removed for a lighter head

        loss_cls=dict(
            type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25,
            loss_weight=1.0,
        ),
        loss_bbox=dict(type='RotatedIoULoss', loss_weight=2.5),
        loss_angle=dict(type='SmoothL1Loss', beta=0.05, loss_weight=1.0),
        loss_obj=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0,
        ),
    ),

    # -------------------------- Training Config --------------------------
    # tal_* are read by the head's Task-Aligned assigner. The O2O ramp schedule
    # lives in ProgressiveLossHook (stage 2), not here.
    train_cfg=dict(
        tal_topk=13,
        tal_alpha=1.0,
        tal_beta=2.0,
    ),

    # -------------------------- Testing Config --------------------------
    # NMS path (end2end=False): the O2M head emits many overlapping boxes per
    # object, so rotated NMS de-duplicates them. (NMS-free O2O inference needs
    # a fully-trained O2O head + end2end=True.)
    test_cfg=dict(
        end2end=False,
        score_thr=0.05,
        max_per_img=300,
        nms=dict(type='nms_rotated', iou_thr=0.1),
        nms_pre=2000,
    ),
)

# ========================== Dataset Configuration ==========================
dataset_type = 'DIORDataset'
data_root = 'data/DIOR-R/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

# All sizes are multiples of 32 (ViT-Adapter requires a stride-32 prior level).
image_size = (800, 800)
train_scales = [(672, 672), (800, 800), (928, 928)]

# -------------------------- Training Pipeline --------------------------
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=train_scales, multiscale_mode='value'),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.5, 0.5],
        direction=['horizontal', 'vertical'],
        version='le90',
    ),
    dict(
        type='PolyRandomRotate',
        rotate_ratio=0.5,
        angles_range=180,
        auto_bound=False,
        version='le90',
    ),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18,
    ),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),          # must be a multiple of 32
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels'],
        meta_keys=(
            'filename', 'ori_filename', 'ori_shape', 'img_shape',
            'pad_shape', 'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg',
        ),
    ),
]

# -------------------------- Testing Pipeline --------------------------
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=image_size,
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=(
                    'filename', 'ori_filename', 'ori_shape', 'img_shape',
                    'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                    'img_norm_cfg',
                ),
            ),
        ],
    ),
]

data = dict(
    samples_per_gpu=4,           # ViT-L/16 + Adapter is memory-heavy; lower if OOM
    workers_per_gpu=4,
    # TRAINVAL split: train on the FULL train+val pool (~11.7k images), use the
    # held-out test split for both model selection (val) and final eval (test).
    # val == test here, so every eval runs over the full test set; control
    # frequency with evaluation.interval.
    train=dict(
        type=dataset_type,
        ann_file=[data_root + 'train/labelTxt/', data_root + 'val/labelTxt/'],
        img_prefix=[data_root + 'train/images/', data_root + 'val/images/'],
        pipeline=train_pipeline,
        version='le90',
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'test/labelTxt/',
        img_prefix=data_root + 'test/images/',
        pipeline=test_pipeline,
        version='le90',
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test/labelTxt/',
        img_prefix=data_root + 'test/images/',
        pipeline=test_pipeline,
        version='le90',
    ),
)

# ========================== Evaluation Configuration ==========================
evaluation = dict(
    interval=3,             # test set is large; tune me
    metric='mAP',
    save_best='mAP@0.50',
    rule='greater',
    gpu_collect=True,
)

# ========================== Runtime ==========================
checkpoint_config = dict(interval=5, max_keep_ckpts=3)

log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook')])

# EMA only here. The ProgressiveLossHook (O2O ramp) is added in the stage-2
# config — stage 1 trains O2M-only (o2o_weight stays at its 0.0 default).
# momentum 0.9998 (not 0.9999): the two-stage schedules are short; 0.9999's long
# half-life would make the EMA nearly inert over one training pass.
custom_hooks = [
    dict(type='EMAHook', momentum=0.9998, priority='ABOVE_NORMAL'),
]

# bf16 replaces fp16: the frozen ViT runs under bfloat16 autocast inside the
# adapter backbone (see bf16_vit=True). Disable the global fp16 AMP hook so the
# trainable head/adapter train in stable fp32.
fp16 = None

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

device = 'cuda'
gpu_ids = range(1)

opencv_num_threads = 0
mp_start_method = 'spawn'
