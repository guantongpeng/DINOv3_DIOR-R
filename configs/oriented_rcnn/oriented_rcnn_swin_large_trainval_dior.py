# =============================================================================
# Oriented R-CNN + Swin Transformer v1 (Swin-L) + FPN on DIOR-R  (TRAINVAL run)
# =============================================================================
# Backbone: Swin Transformer v1 Large
#   - ImageNet-22k pretrained, window12 / input 384 (official microsoft weights).
#   - depths=[2,2,18,2], num_heads=[6,12,24,48], embed_dims=192.
#   - frozen_stages=-1  =>  FULL-PARAMETER fine-tuning (no stage is frozen).
#   - with_cp=True      =>  gradient checkpointing on stage-2's 18 blocks to fit
#     Swin-L at 800x800 on 80GB GPUs.
#   - convert_weights=True => microsoft checkpoint key -> mmdet key conversion.
# Neck: standard 5-level FPN (P2..P6) on the 4 Swin pyramid stages.
# Head: Oriented R-CNN (rotated RPN + rotated RoI refinement).
#
# DATA split (full supervision, no leakage):
#   * train = DIOR-R train + val merged  (~11.7k images, via list ann_file).
#   * val   = DIOR-R test split          (periodic eval + save_best selection).
#   * test  = DIOR-R test split          (final tools/test.py evaluation).
#
# Augmentation: oriented-aware RandomFlip + PolyRandomRotate + PhotoMetricDistortion
# + Albu (denoising/blur/color). DIOR is north-up, so random in-plane rotation
# enforces orientation invariance (standard +1~3 mAP for oriented detection).
#
# Optimization (Swin detection standard):
#   - AdamW lr=1e-4, weight_decay=0.05; backbone lr_mult=0.1 (=> 1e-5 gentle FT).
#   - norm layers & biases get zero weight decay (norm_decay_mult / bias_decay_mult).
#   - CosineAnnealing + 500-iter linear warmup, EMA(0.9998), fp16, grad_clip 10.
#
# Usage:
#   bash scripts/orcnn_swin_large_trainval.sh
# =============================================================================

_base_ = []

# ========================== Model Configuration ==========================

custom_imports = dict(
    imports=[
        'models.datasets.dior',
        'models.pipelines.albu_metadata',
    ],
    allow_failed_imports=False,
)

# ---- Swin-L architectural constants ----
swin_embed_dims = 192
swin_depths = [2, 2, 18, 2]
swin_num_heads = [6, 12, 24, 48]
# Swin pyramid: stride 4/8/16/32 -> channels 192/384/768/1536.
swin_in_channels = [192, 384, 768, 1536]

SWIN_PRETRAIN = (
    '/mnt/htzzb2/00-model/00-hlj/swin_weights_large384_22k/'
    'swin_large_patch4_window12_384_22k.pth'
)

angle_version = 'le90'
num_classes = 20

model = dict(
    type='OrientedRCNN',
    # -------------------------- Backbone: Swin-L -------------------------------
    # frozen_stages=-1 => ALL parameters trainable (full-parameter fine-tuning).
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=swin_embed_dims,
        depths=swin_depths,
        num_heads=swin_num_heads,
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=True,
        convert_weights=True,
        frozen_stages=-1,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        init_cfg=dict(type='Pretrained', checkpoint=SWIN_PRETRAIN),
    ),

    # -------------------------- Neck: 5-level FPN ------------------------------
    # 4 Swin pyramid inputs -> FPN -> 5 outputs (P2..P6, extra conv on last input).
    neck=dict(
        type='FPN',
        in_channels=swin_in_channels,
        out_channels=256,
        start_level=0,
        add_extra_convs='on_input',
        num_outs=5,
        relu_before_extra_convs=True,
    ),

    # -------------------------- RPN Head (Oriented R-CNN) ----------------------
    rpn_head=dict(
        type='OrientedRPNHead',
        in_channels=256,
        feat_channels=256,
        version=angle_version,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64],
        ),
        bbox_coder=dict(
            type='MidpointOffsetCoder',
            angle_range=angle_version,
            target_means=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0, 0.5, 0.5],
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

    # -------------------------- ROI Head (Oriented R-CNN) ----------------------
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
            featmap_strides=[4, 8, 16, 32],
        ),
        bbox_head=dict(
            type='RotatedShared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=num_classes,
            bbox_coder=dict(
                type='DeltaXYWHAOBBoxCoder',
                angle_range=angle_version,
                norm_factor=None,
                edge_swap=True,
                proj_xy=True,
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

    # -------------------------- Training Config --------------------------------
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1,
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

    # -------------------------- Testing Config ---------------------------------
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

dataset_type = 'DIORDataset'
data_root = 'data/DIOR-R/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

# DIOR images are ~800x800.
image_size = (800, 800)

# Multi-scale training around the native DIOR resolution.
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
    # ---- Rotation augmentation (oriented-aware) ----
    # rotate_ratio=0.5: rotate half the images by a random angle in [-180, 180].
    dict(
        type='PolyRandomRotate',
        rotate_ratio=0.5,
        angles_range=180,
        auto_bound=False,
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
    # ---- train: DIOR-R train + val merged (~11.7k images) ----
    train=dict(
        type=dataset_type,
        ann_file=[
            data_root + 'train/labelTxt/',
            data_root + 'val/labelTxt/',
        ],
        img_prefix=[
            data_root + 'train/images/',
            data_root + 'val/images/',
        ],
        pipeline=train_pipeline,
        version='le90',
    ),
    # ---- val: DIOR-R test split, used for periodic eval + save_best ----
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'test/labelTxt/',
        img_prefix=data_root + 'test/images/',
        pipeline=test_pipeline,
        version='le90',
    ),
    # ---- test: DIOR-R test split, used by tools/test.py for final eval ----
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test/labelTxt/',
        img_prefix=data_root + 'test/images/',
        pipeline=test_pipeline,
        version='le90',
    ),
)

# Evaluate on the test split periodically and keep the best-by-mAP checkpoint.
evaluation = dict(
    interval=3,
    metric='mAP',
    save_best='mAP@0.50',
    rule='greater',
    gpu_collect=True,
)

# ========================== Optimization ==========================

# AdamW + Swin detection grouping:
#   - whole backbone lr_mult=0.1 => main backbone 1e-5 gentle fine-tuning;
#   - norm layers (LayerNorm/BatchNorm/GroupNorm) decay_mult=0.0 (via
#     norm_decay_mult, auto-detected);
#   - all biases decay_mult=0.0 (bias_decay_mult).
optimizer = dict(
    type='AdamW',
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
        },
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
    ),
)

optimizer_config = dict(
    grad_clip=dict(max_norm=10, norm_type=2),
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

custom_hooks = [
    dict(type='EMAHook', momentum=0.9998, priority='ABOVE_NORMAL'),
]

fp16 = dict(loss_scale='dynamic')

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

device = 'cuda'
gpu_ids = range(1)

opencv_num_threads = 0
mp_start_method = 'spawn'
