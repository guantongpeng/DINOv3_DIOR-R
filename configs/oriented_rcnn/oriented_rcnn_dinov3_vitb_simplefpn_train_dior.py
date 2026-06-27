# =============================================================================
# Oriented R-CNN + DINOv3 ViT-B/16 for DIOR-R  (v3 — ViTDet SimpleFPN + clean aug)
# =============================================================================
# Key changes vs the original ViTDetFPN config (addressing val=0.72 / test=0.60):
#
#   1. NECK -> standard ViTDet SimpleFeaturePyramid (single last-block feature
#      + deconv pyramid). The old neck fed 4 transformer-block outputs in as if
#      they were 4 scales; shallow block-3 features upscaled 4x into the stride-4
#      level were noisy. Using only the richest (last) feature is the proven
#      ViTDet recipe (Li et al., ECCV 2022).
#      -> Backbone now outputs ONLY the last block (layers_to_use=[11]).
#
#   2. AUGMENTATION: removed the double color jitter (Albu + PhotoMetricDistortion)
#      and the diagonal flip. Kept multi-scale + a single, moderate photometric
#      distortion + horizontal/vertical flips. The previous setup destroyed
#      pretrained ViT features / RS spectral content and corrupted angle labels.
#
#   3. CLASS WEIGHTS removed: the old hand-tuned weights down-weighted the
#      majority classes to 0.12-0.15, which tanks their AP and thus the
#      macro-mAP (each class = 1/20 of the mean). Uniform CE is the clean baseline.
#
#   4. SCHEDULE: 300 -> 120 epochs (the small set over-fits the trainval
#      distribution long before 300ep; EMA + cosine still converge well).
#      Eval every 2 epochs for finer model selection.
#
#   5. batch 16 -> 8 (SimpleFeaturePyramid upsamples in 768ch, heavier than the
#      old 256ch neck). Scale up if GPU memory allows.
#
# Data strategy: val is kept as a held-out set (same trainval pool as train) for
# model selection. NOTE: because val ~ train distribution, val mAP tracks
# in-distribution fit and will NOT predict test-set generalization — use it for
# relative comparisons only. For the final number, retrain on the FULL trainval
# (merge train+val) and evaluate on the official test split.
# =============================================================================

_base_ = []

# ========================== Model Configuration ==========================

custom_imports = dict(
    imports=[
        'models.backbones.dinov3_wrapper',
        'models.necks.simple_feature_pyramid',
        'models.datasets.dior',
        'models.pipelines.albu_metadata',
    ],
    allow_failed_imports=False,
)

model = dict(
    type='OrientedRCNN',
    # -------------------------- Backbone: DINOv3 ViT-B/16 -----------------------
    # Output ONLY the last block -> single stride-16 feature for SimpleFPN.
    # use_layernorm=False: get_intermediate_layers(norm=True) already applies the
    # ViT final norm; SimpleFeaturePyramid has its own LayerNorm2d stems.
    backbone=dict(
        type='DinoVisionTransformerBackbone',
        model_name='dinov3_vitb16',
        pretrained=False,
        layers_to_use=[11],
        out_indices=(0,),
        use_layernorm=False,
        frozen_stages=0,
        init_cfg=dict(
            checkpoint='data/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
        ),
    ),

    # -------------------------- Neck: ViTDet SimpleFeaturePyramid --------------
    # Single stride-16 input -> [stride 4, 8, 16, 32] via deconv/conv + LN + GELU.
    neck=dict(
        type='SimpleFeaturePyramid',
        in_channels=768,
        out_channels=256,
        num_outs=4,
        in_stride=16,
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

    # -------------------------- ROI Head -----------------------------------------
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
            num_classes=20,
            bbox_coder=dict(
                type='DeltaXYWHAOBBoxCoder',
                angle_range='le90',
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
            nms=dict(iou_thr=0.1),
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

image_size = (800, 800)

# Mild multi-scale (kept — it is a generalization aid, unlike the color jitter).
train_scales = [(700, 700), (800, 800), (900, 900)]

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
    samples_per_gpu=8,
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
    grad_clip=dict(max_norm=10, norm_type=2),
)

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)

runner = dict(type='EpochBasedRunner', max_epochs=120)

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
