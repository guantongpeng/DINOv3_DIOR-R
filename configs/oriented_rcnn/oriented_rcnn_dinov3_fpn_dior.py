# =============================================================================
# Oriented R-CNN with DINOv3 Backbone for DIOR-R Dataset (v2 — optimized)
# =============================================================================
# Backbone: Meta official DINOv3 ViT-L/16 (imported from dinov3 repo)
# Neck: ViTDetFPN (proper FPN for ViT features)
# Head: Oriented R-CNN (rotated detection)
#
# v2 improvements (2026-06-15):
#   1. RCNN NMS iou_thr: 0.1 → 0.5 (fixed overly aggressive suppression)
#   2. RoI feat size: 7×7 → 14×14 (better small object detection)
#   3. Label Smoothing: 0.1 (reduces overfitting)
#   4. Class-balanced weights: rare classes (dam, trainstation) weighted higher
#   5. Stronger augmentation: Albu (noise, blur, color jitter, CLAHE)
#      + more aggressive PhotoMetricDistortion
#
# Baseline config:
#   1. img_size=800 (DIOR images are ~800×800)
#   2. 300-epoch training with cosine annealing
#   3. EMA for smoother convergence
#   4. Multi-scale training [600, 800, 1000]
#   5. Frozen stages=0: all backbone params trained
# =============================================================================

_base_ = []

# ========================== Model Configuration ==========================

custom_imports = dict(
    imports=[
        'models.backbones.dinov3_wrapper',
        'models.necks.vitdet_fpn',
        'models.datasets.dior',
        'models.pipelines.albu_metadata',
    ],
    allow_failed_imports=False,
)

model = dict(
    type='OrientedRCNN',
    # -------------------------- Backbone: DINOv3 ViT-B/16 ------------------------
    # Official Meta DINOv3 ViT-Base imported from the dinov3 repository.
    # Outputs 4 feature maps at embed_dim=768, stride=16.
    # frozen_stages=0 means ALL backbone params are trained (critical for mAP!).
    backbone=dict(
        type='DinoVisionTransformerBackbone',
        model_name='dinov3_vitl16',
        pretrained=False,
        layers_to_use=[5, 11, 17, 23],
        out_indices=(0, 1, 2, 3),
        use_layernorm=True,
        frozen_stages=0,
        init_cfg=dict(
            checkpoint='/mnt/ht2-nas2/00-model/guantp/dino/mm_dino/data/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth',
        ),
    ),

    # -------------------------- Neck: ViTDetFPN ----------------------------------
    # in_channels=1024 matches ViT-L embed_dim.
    neck=dict(
        type='ViTDetFPN',
        in_channels=1024,
        out_channels=256,
        num_outs=4,
        start_level=0,
        add_extra_convs=False,
        se_reduction=16,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        act_cfg=dict(type='GELU'),
    ),

    # -------------------------- RPN Head -----------------------------------------
    rpn_head=dict(
        type='OrientedRPNHead',
        in_channels=256,
        feat_channels=256,
        version='le90',
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32],
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

    # -------------------------- ROI Head -----------------------------------------
    # roi_feat_size=14 for better small object detection.
    # class_weight uses inverse-sqrt frequency: rare classes (dam, trainstation)
    # get higher weight, common classes (ship, vehicle) get lower weight.
    roi_head=dict(
        type='OrientedStandardRoIHead',
        bbox_roi_extractor=dict(
            type='RotatedSingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlignRotated',
                out_size=14,
                sample_num=2,
                clockwise=True,
            ),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],
        ),
        bbox_head=dict(
            type='RotatedShared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=14,
            num_classes=20,
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
                label_smoothing=0.1,
                class_weight=[
                    1.00,  # background
                    0.25,  # airplane        (8212 gt)
                    0.88,  # airport         (666  gt) — rare
                    0.39,  # baseballfield   (3434 gt)
                    0.49,  # basketballcourt (2146 gt)
                    0.44,  # bridge          (2584 gt)
                    0.70,  # chimney         (1031 gt)
                    0.97,  # dam             (538  gt) — rarest
                    0.69,  # ESA             (1085 gt)
                    0.86,  # ETS             (688  gt) — rare
                    0.94,  # golffield       (575  gt) — rare
                    0.52,  # groundtrackfield(1885 gt)
                    0.41,  # harbor          (3102 gt)
                    0.54,  # overpass        (1778 gt)
                    0.12,  # ship            (35183 gt) — most common
                    0.87,  # stadium         (672  gt) — rare
                    0.15,  # storagetank     (23361 gt)
                    0.26,  # tenniscourt     (7343 gt)
                    1.00,  # trainstation    (509  gt) — rarest
                    0.14,  # vehicle         (26601 gt)
                    0.41,  # windmill        (2998 gt)
                ],
            ),
            loss_bbox=dict(
                type='SmoothL1Loss',
                beta=1.0,
                loss_weight=1.0,
            ),
        ),
    ),

    # -------------------------- Training Config ----------------------------------
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

    # -------------------------- Testing Config -----------------------------------
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
            nms=dict(type='nms', iou_thr=0.5),
            max_per_img=2000,
        ),
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

# DIOR images are ~800×800 — no need for 1024 upscale
image_size = (800, 800)

# Multi-scale training scales (centered around 800)
train_scales = [(600, 600), (800, 800), (1000, 1000)]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RResize',
        img_scale=train_scales,
        multiscale_mode='value',
    ),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version='le90',
    ),
    dict(
        type='AlbuMetadata',
        transforms=[
            dict(type='GaussNoise', var_limit=(10.0, 50.0), p=0.3),
            dict(type='MotionBlur', blur_limit=(3, 7), p=0.2),
            dict(type='RandomBrightnessContrast',
                 brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            dict(type='HueSaturationValue',
                 hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
        ],
        keymap=dict(img='image'),
    ),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=48,
        contrast_range=(0.4, 1.6),
        saturation_range=(0.4, 1.6),
        hue_delta=24,
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

# Test pipeline (single-scale, no flip for safe evaluation)
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
    samples_per_gpu=4,
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

evaluation = dict(
    interval=3,
    metric='mAP',
    save_best='mAP@0.50',
    rule='greater',
    gpu_collect=True,
)

# ========================== Optimization ==========================

optimizer = dict(
    type='AdamW',
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'backbone.backbone': dict(lr_mult=0.25),
            'backbone.layer_norms': dict(lr_mult=1.0),
        },
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
    ),
)

optimizer_config = dict(
    grad_clip=dict(max_norm=35, norm_type=2),
)

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)

runner = dict(type='EpochBasedRunner', max_epochs=300)

# ========================== Runtime ==========================

checkpoint_config = dict(interval=5, max_keep_ckpts=3)

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
    ],
)

# Exponential Moving Average for smoother convergence (~+0.5-1.0 mAP)
custom_hooks = [
    dict(type='EMAHook', momentum=0.999, priority='ABOVE_NORMAL'),
]

fp16 = dict(loss_scale=512.0)

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

device = 'cuda'
gpu_ids = range(1)

opencv_num_threads = 0
mp_start_method = 'spawn'
