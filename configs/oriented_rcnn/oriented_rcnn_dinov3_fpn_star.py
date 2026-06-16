# =============================================================================
# Oriented R-CNN with DINOv3 Backbone for Star-1021+Extend3 Dataset
# =============================================================================
# This configuration fine-tunes DINOv3 (ViT-Base) as the backbone for
# Oriented R-CNN on the Star-1021+Extend3 remote sensing dataset.
#
# Model Architecture:
#   Backbone: ViT-Base DINOv3 (pretrained, frozen first 8 blocks)
#   Neck: SimpleFPN (builds multi-scale pyramid from ViT features)
#   RPN: OrientedRPNHead
#   ROI Head: OrientedStandardRoIHead with OrientedBBoxHead
#
# Dataset: Star-1021+Extend3 (25 classes, oriented bounding boxes)
# =============================================================================

_base_ = []

# ========================== Model Configuration ==========================

# Custom imports for DINOv3 backbone and SimpleFPN
custom_imports = dict(
    imports=[
        'models.backbones.vit_dinov3',
        'models.necks.simple_fpn',
        'models.datasets.star',
    ],
    allow_failed_imports=False,
)

# -------------------------- Backbone: DINOv3 ViT-B --------------------------
# DINOv3 ViT-Base with patch_size=16, embed_dim=768, depth=12
# Extract features from blocks [3, 5, 7, 11] for multi-scale representation
# Freeze first 8 transformer blocks to preserve pretrained features
model = dict(
    type='OrientedRCNN',
    backbone=dict(
        type='ViTDinoV3',
        model_name='vit_base_patch16_dinov3',
        pretrained=False,  # Set False when using local checkpoint
        checkpoint_path='/mnt/ht2-nas2/EO_test/weights/Dinov3_pretrained/DINOv3 ViT LVD-1689M/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
        out_indices=(3, 5, 7, 11),
        out_channels=256,
        frozen_stages=-1,
        with_cp=False,
        norm_cfg=dict(type='LN', eps=1e-6),
        img_size=1024,
        init_cfg=None,
    ),

    # -------------------------- Neck: SimpleFPN --------------------------
    # Converts same-resolution ViT features into multi-scale pyramid
    # Input:  4 features at stride 16 (50x50 for 800x800 images)
    # Output: 4 features at strides [8, 16, 32, 64]
    neck=dict(
        type='SimpleFPN',
        in_channels=256,
        out_channels=256,
        num_outs=4,
        start_level=0,
        add_extra_convs=False,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        act_cfg=dict(type='GELU'),
    ),

    # -------------------------- RPN Head --------------------------
    # Oriented RPN generates rotated region proposals
    rpn_head=dict(
        type='OrientedRPNHead',
        in_channels=256,
        feat_channels=256,
        version='le90',
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64],
        ),
        bbox_coder=dict(
            type='MidpointOffsetCoder',
            angle_range='le90',
            target_means=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
        ),
        loss_bbox=dict(
            type='SmoothL1Loss',
            beta=1.0 / 9.0,
            loss_weight=1.0,
        ),
    ),

    # -------------------------- ROI Head --------------------------
    roi_head=dict(
        type='OrientedStandardRoIHead',
        bbox_roi_extractor=dict(
            type='RotatedSingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlignRotated',
                out_size=7,
                sample_num=2,
                clockwise=True,
            ),
            out_channels=256,
            featmap_strides=[8, 16, 32, 64],
        ),
        bbox_head=dict(
            type='RotatedShared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=25,  # Star-1021+Extend3 has 25 classes
            bbox_coder=dict(
                type='DeltaXYWHAOBBoxCoder',
                angle_range='le90',
                target_means=[0.0, 0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2, 0.1],
            ),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0,
            ),
            loss_bbox=dict(
                type='SmoothL1Loss',
                beta=1.0,
                loss_weight=1.0,
            ),
        ),
    ),

    # -------------------------- Training Config --------------------------
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1,
                gpu_assign_thr=200,
            ),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False,
            ),
            allowed_border=0,
            pos_weight=-1,
            debug=False,
        ),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.8),
            min_bbox_size=0,
        ),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1,
                iou_calculator=dict(type='RBboxOverlaps2D'),
                gpu_assign_thr=200,
            ),
            sampler=dict(
                type='RRandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True,
            ),
            pos_weight=-1,
            debug=False,
        ),
    ),

    # -------------------------- Testing Config --------------------------
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.8),
            min_bbox_size=0,
        ),
        rcnn=dict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms', iou_thr=0.1),
            max_per_img=2000,
        ),
    ),
)

# ========================== Dataset Configuration ==========================

# Star-1021+Extend3 dataset with 25 remote sensing object categories
dataset_type = 'StarDataset'
data_root = '/mnt/ht2-nas2/00-model/guantp/dino/mm_dino/data/star-1021_1016+extend3/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

# Image size (resize to 800x800)
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
    samples_per_gpu=16,
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
    interval=5,  # Evaluate every N epochs
    metric='mAP_coco',  # mAP@50:95 + mAP@0.50, mAP@0.55, ..., mAP@0.95
    # Alternative metrics:
    #   'mAP'       - single IoU threshold (default 0.5)
    #   'mAP_multi' - mAP@0.50 + mAP@0.75
    save_best='mAP@50:95',
    rule='greater',
    gpu_collect=True,  # Use GPU all_gather instead of file-based collect (avoids NFS race conditions)
)

# ========================== Optimization Configuration ==========================
# Optimizer: AdamW with layer-wise learning rate decay
# Lower lr for pretrained backbone, higher lr for randomly initialized heads
optimizer = dict(
    type='AdamW',
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={},
    ),
)

optimizer_config = dict(
    grad_clip=dict(max_norm=35, norm_type=2),
)

# Learning rate schedule: Cosine annealing with linear warmup
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=150,
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

# ========================== Custom Hooks ==========================
custom_hooks = []