# =============================================================================
# BASE config: Oriented R-CNN + DINOv3 ViT-B/16 + ViT-Adapter (DIOR-R)
# =============================================================================
# This is a SHARED base. Do not train it directly — use the stage1/stage2
# configs that inherit from it.
#
# What this recipe implements (vs. the old SimpleFPN config):
#   1. BACKBONE  -> ViT-Adapter (deformable attention + multi-layer ViT fusion),
#                   replacing SimpleFeaturePyramid. The frozen ViT runs under
#                   bfloat16 autocast (official DINOv3 eval recipe).
#   2. NECK      -> PassthroughNeck (the adapter already emits the pyramid).
#   3. SIZES     -> all train/test scales + pad are multiples of 32 (ViT-Adapter
#                   needs a stride-32 prior level).
#   4. PRECISION -> bf16 for the heavy frozen ViT (inside the backbone); the
#                   trainable head/adapter train in fp32 (fp16 AMP disabled).
#   5. INIT      -> RegZeroInitHook zeroes the RoI regression FC (DETR-style
#                   stable box init) before training starts.
#   6. FREEZE    -> two-stage: see stage1 (frozen ViT) / stage2 (end-to-end).
# =============================================================================

_base_ = []

custom_imports = dict(
    imports=[
        'models.backbones.dinov3_vit_adapter',  # registers DINOv3ViTAdapter
        'models.necks.passthrough_neck',         # registers PassthroughNeck
        'models.hooks',                          # registers RegZeroInitHook
        'models.datasets.dior',
        'models.pipelines.albu_metadata',
    ],
    allow_failed_imports=False,
)

model = dict(
    type='OrientedRCNN',

    # -------------------- Backbone: DINOv3 ViT-Adapter -----------------------
    # Frozen ViT-B/16 + Spatial Prior Module + 4 deformable InteractionBlocks
    # fusing ViT layers [2,5,8,11]. Outputs a 4-level pyramid @ 256 channels,
    # strides [4,8,16,32]. bf16_vit runs the frozen ViT in bfloat16.
    backbone=dict(
        type='DINOv3ViTAdapter',
        model_name='dinov3_vitb16',
        interaction_indexes=[2, 5, 8, 11],
        out_channels=256,
        freeze_vit=True,            # stage1 default; stage2 overrides to False
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
            checkpoint='/mnt/ht2-nas2/00-model/guantp/dino/mm_dino/data/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
        ),
    ),

    # -------------------- Neck: identity passthrough -------------------------
    neck=dict(type='PassthroughNeck'),

    # -------------------- RPN Head -------------------------------------------
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
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
    ),

    # -------------------- ROI Head -------------------------------------------
    roi_head=dict(
        type='OrientedStandardRoIHead',
        bbox_roi_extractor=dict(
            type='RotatedSingleRoIExtractor',
            roi_layer=dict(type='RoIAlignRotated', out_size=14, sample_num=2, clockwise=True),
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
            ),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
        ),
    ),

    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner', pos_iou_thr=0.7, neg_iou_thr=0.3,
                min_pos_iou=0.3, match_low_quality=True, ignore_iof_thr=-1,
                gpu_assign_thr=200,
            ),
            sampler=dict(
                type='RandomSampler', num=256, pos_fraction=0.5,
                neg_pos_ub=-1, add_gt_as_proposals=False,
            ),
            allowed_border=0, pos_weight=-1, debug=False,
        ),
        rpn_proposal=dict(
            nms_pre=2000, max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.8), min_bbox_size=0,
        ),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner', pos_iou_thr=0.5, neg_iou_thr=0.5,
                min_pos_iou=0.5, match_low_quality=False, ignore_iof_thr=-1,
                iou_calculator=dict(type='RBboxOverlaps2D'), gpu_assign_thr=200,
            ),
            sampler=dict(
                type='RRandomSampler', num=512, pos_fraction=0.25,
                neg_pos_ub=-1, add_gt_as_proposals=True,
            ),
            pos_weight=-1, debug=False,
        ),
    ),

    test_cfg=dict(
        rpn=dict(
            nms_pre=2000, max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.8), min_bbox_size=0,
        ),
        rcnn=dict(
            nms_pre=2000, min_bbox_size=0, score_thr=0.05,
            nms=dict(type='nms', iou_thr=0.5), max_per_img=2000,
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

# All sizes are multiples of 32 (ViT-Adapter requires a stride-32 prior level).
image_size = (800, 800)
train_scales = [(672, 672), (800, 800), (928, 928)]

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
                    'pad_shape', 'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg',
                ),
            ),
        ],
    ),
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    # TRAINVAL split: train on the FULL train+val pool (~11.7k images), use the
    # held-out test split for both model selection (val) and final eval (test).
    # Matches the project's trainval recipe — val == test here, so every eval
    # runs over the full test set; control frequency with evaluation.interval.
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

evaluation = dict(
    interval=3,
    metric='mAP',
    save_best='mAP@0.50',
    rule='greater',
    gpu_collect=True,
)

# ========================== Runtime ==========================
checkpoint_config = dict(interval=5, max_keep_ckpts=3)

log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook')])

custom_hooks = [
    dict(type='RegZeroInitHook', zero_weight=True, zero_bias=True, priority='HIGHEST'),
    dict(type='EMAHook', momentum=0.9999, priority='ABOVE_NORMAL'),
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
