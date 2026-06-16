# =============================================================================
# YOLO26 Detection Head with DINOv3 Backbone for DIOR-R Dataset
# =============================================================================
# This configuration uses YOLO26's anchor-free rotated detection head with
# the DINOv3 (ViT-Base) backbone and SimpleFPN neck for oriented object
# detection on the DIOR-R remote sensing dataset.
#
# Model Architecture:
#   Backbone:  ViT-Base DINOv3 (pretrained, frozen first 8 blocks)
#   Neck:      SimpleFPN (builds multi-scale pyramid from ViT features)
#   Head:      YOLO26RotatedHead (anchor-free, dual-head, NMS-free)
#
# Key YOLO26 Features:
#   - Anchor-free dense prediction (no manual anchor tuning)
#   - Dual-head: O2M (Task-Aligned) + O2O (Hungarian matching)
#   - NMS-free end-to-end inference via O2O head
#   - Angle encoding: sigmoid → [-π/4, 3π/4]
#   - Progressive Loss: shifts supervision from O2M to O2O
#   - No DFL: lighter regression head
#
# Dataset: DIOR-R (20 classes, oriented bounding boxes)
# =============================================================================

_base_ = []

# ========================== Model Configuration ==========================

# Custom imports for DINOv3 backbone, SimpleFPN, and YOLO26 head
custom_imports = dict(
    imports=[
        'models.backbones.dinov3_wrapper',
        'models.necks.simple_fpn',
        'models.necks.vitdet_fpn',
        'models.datasets.dior',
        'models.heads.yolo26_rotated_head',
        'models.detectors.dinov3_yolo26',
        'models.hooks',
    ],
    allow_failed_imports=False,
)

model = dict(
    type='DINOv3YOLO26',

    # -------------------------- Backbone: DINOv3 ViT-B --------------------------
    # DINOv3 ViT-Base with patch_size=16, embed_dim=768, depth=12
    # Extract features from blocks [3, 5, 7, 11] for multi-scale representation
    # Freeze first 8 transformer blocks to preserve pretrained features
    backbone=dict(
        type='DinoVisionTransformerBackbone',
        model_name='dinov3_vitb16',
        pretrained=False,
        layers_to_use=[3, 5, 8, 11],
        out_indices=(0, 1, 2, 3),
        use_layernorm=True,
        frozen_stages=0,
        init_cfg=dict(
            checkpoint='/mnt/ht2-nas2/00-model/guantp/dino/mm_dino/data/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
        ),
    ),

    # -------------------------- Neck: ViTDetFPN --------------------------
    # Converts same-resolution ViT features into multi-scale pyramid
    # Input:  4 features at stride 16 (50x50 for 800x800 images)
    # Output: 4 features at strides [8, 16, 32, 64]
    neck=dict(
        type='SimpleFPN',
        in_channels=768,
        out_channels=256,
        num_outs=4,
        start_level=0,
        add_extra_convs=False,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        act_cfg=dict(type='GELU'),
    ),

    # -------------------------- YOLO26 Rotated Head --------------------------
    # Anchor-free dual-head for rotated detection
    # O2M head: Task-Aligned Label Assignment (TAL)
    # O2O head: Hungarian matching for NMS-free inference
    bbox_head=dict(
        type='YOLO26RotatedHead',
        num_classes=20,  # DIOR-R has 20 classes
        in_channels=256,
        feat_channels=128,
        stacked_convs=2,
        strides=[8, 16, 32, 64],
        reg_max=16,
        use_dfl=False,  # YOLO26: DFL removed for lighter head

        # Classification loss (Focal Loss with sigmoid)
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0,
        ),

        # Bounding box regression loss (RotatedIoULoss for rotated boxes)
        loss_bbox=dict(
            type='RotatedIoULoss',
            loss_weight=2.5,
        ),

        # Angle regression loss
        loss_angle=dict(
            type='SmoothL1Loss',
            beta=0.05,
            loss_weight=1.0,
        ),

        # Objectness/quality loss
        loss_obj=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
        ),
    ),

    # -------------------------- Training Config --------------------------
    train_cfg=dict(
        # Task-Aligned Label Assignment parameters
        tal_topk=13,        # Top-K anchors per GT for O2M (YOLO standard: 13)
        tal_alpha=1.0,      # Class weight in alignment metric
        tal_beta=2.0,       # IoU weight in alignment metric (reduced from 6.0
                            # for stable initial training; higher values
                            # overly suppress low-IoU anchors)

        # Progressive loss schedule (O2O weight increase)
        # Epoch:  0-12   → o2o_weight = 0 (O2M only)
        # Epoch: 12-30   → o2o_weight: 0 → 1.0 (ramp up)
        # Epoch: 30-36   → o2o_weight = 1.0 (O2M + O2O)
        progressive_loss=dict(
            start_epoch=60,
            end_epoch=150,
        ),
    ),

    # -------------------------- Testing Config --------------------------
    test_cfg=dict(
        # End-to-end NMS-free inference
        end2end=True,
        score_thr=0.05,
        max_per_img=300,
        # NMS config (fallback for O2M mode)
        nms=dict(type='nms_rotated', iou_thr=0.1),
        nms_pre=2000,
    ),
)

# ========================== Dataset Configuration ==========================

# DIOR-R dataset with 20 remote sensing object categories
dataset_type = 'DIORDataset'
data_root = 'data/DIOR-R/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

# Image size for DIOR-R
image_size = (800, 800)

# -------------------------- Training Pipeline --------------------------
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=image_size),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version='le90',
    ),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels'],
        meta_keys=(
            'filename', 'ori_filename', 'ori_shape', 'img_shape',
            'pad_shape', 'scale_factor', 'flip', 'flip_direction',
            'img_norm_cfg',
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
                    'filename', 'ori_filename', 'ori_shape',
                    'img_shape', 'pad_shape', 'scale_factor',
                    'flip', 'flip_direction', 'img_norm_cfg',
                ),
            ),
        ],
    ),
]

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train/labelTxt/',
        img_prefix=data_root + 'train/images/',
        pipeline=train_pipeline,
        version='le90',
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val/labelTxt/',
        img_prefix=data_root + 'val/images/',
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
    interval=3,  # Evaluate every N epochs
    metric='mAP',
    save_best='auto',
    rule='greater',
    gpu_collect=True,
)

# ========================== Optimization Configuration ==========================
# Optimizer: AdamW with layer-wise learning rate decay
# Lower lr for pretrained backbone, higher lr for randomly initialized head
optimizer = dict(
    type='AdamW',
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),  # Lower lr for pretrained backbone
        },
    ),
)

optimizer_config = dict(
    grad_clip=dict(max_norm=35, norm_type=2),
)

# Learning rate schedule: Cosine annealing with linear warmup
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,  # Longer warmup for YOLO-style head
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)

runner = dict(type='EpochBasedRunner', max_epochs=200)

# ========================== Runtime Configuration ==========================
checkpoint_config = dict(interval=5, max_keep_ckpts=3)

# Logging
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
    ],
)

# Mixed precision training
fp16 = dict(loss_scale=512.0)

# Distributed training
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# Device configuration
device = 'cuda'
gpu_ids = range(1)

# OpenCV config
opencv_num_threads = 0
mp_start_method = 'spawn'

# Auto-scale learning rate based on batch size
# auto_scale_lr = dict(base_batch_size=16)

# ========================== Custom Hooks ==========================
# Progressive loss hook: shifts supervision from O2M to O2O
custom_hooks = [
    dict(
        type='ProgressiveLossHook',
        start_epoch=60,
        end_epoch=150,
        priority='LOW',
    ),
]
