# =============================================================================
# BASE config: Rotated FCOS + DINOv3 ViT-L/16 + ViT-Adapter (DIOR-R)
# =============================================================================
# This is a SHARED base. Do not train it directly — use the stage1/stage2
# configs that inherit from it.
#
# This recipe swaps the two-stage Oriented R-CNN head for an anchor-free
# RotatedFCOS head on the SAME DINOv3 ViT-L + ViT-Adapter backbone + FPN neck,
# so the two detectors are directly comparable.
#
# What this recipe implements (mirrors the Oriented R-CNN ViT-Adapter base):
#   1. BACKBONE  -> ViT-Adapter (deformable attention + multi-layer ViT fusion).
#                   The frozen ViT runs under bfloat16 autocast (official DINOv3
#                   eval recipe). Outputs the FULL embed_dim (1024-d) pyramid for
#                   the FPN to fuse.
#   2. NECK      -> FPN (lateral + top-down fusion, +P5 stride-64). 5 output
#                   levels @ strides [4,8,16,32,64], matching the FCOS head.
#   3. HEAD      -> RotatedFCOSHead (anchor-free, per-pixel l/t/r/b + angle +
#                   centerness). separate_angle=True: GIoU loss on the decoded
#                   horizontal box + L1 angle loss (mmrotate canonical recipe).
#                   strides=[4,8,16,32,64] match the FPN levels.
#   4. SIZES     -> all train/test scales + pad are multiples of 32 (ViT-Adapter
#                   needs a stride-32 prior level).
#   5. PRECISION -> bf16 for the heavy frozen ViT (inside the backbone); the
#                   trainable head/adapter train in fp32 (fp16 AMP disabled).
#   6. FREEZE    -> two-stage: see stage1 (frozen ViT) / stage2 (end-to-end).
# =============================================================================

_base_ = []

custom_imports = dict(
    imports=[
        'models.backbones.dinov3_vit_adapter',  # registers DINOv3ViTAdapter
        'models.datasets.dior',
        'models.pipelines.albu_metadata',
    ],
    allow_failed_imports=False,
)

model = dict(
    type='RotatedFCOS',

    # -------------------- Backbone: DINOv3 ViT-Adapter -----------------------
    # Frozen ViT-L/16 (1024-d, 24 blocks) + Spatial Prior Module + 4 deformable
    # InteractionBlocks fusing ViT layers [5,11,17,23]. Outputs a 4-level pyramid
    # @ 1024 channels (full embed_dim), strides [4,8,16,32], fed into FPN.
    # bf16_vit runs the frozen ViT in bfloat16.
    backbone=dict(
        type='DINOv3ViTAdapter',
        model_name='dinov3_vitl16',
        interaction_indexes=[5, 11, 17, 23],
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
            checkpoint='data/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth',
        ),
    ),

    # -------------------- Neck: FPN on full embed_dim pyramid ----------------
    neck=dict(
        type='FPN',
        in_channels=[1024, 1024, 1024, 1024],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_output',   # add P5 (stride 64) from the last level
        num_outs=5,                    # -> strides [4, 8, 16, 32, 64]
        relu_before_extra_convs=True,
    ),

    # -------------------- FCOS Head (anchor-free, rotated) -------------------
    # separate_angle=True: GIoULoss on the decoded horizontal box (h_bbox_coder)
    # + L1Loss on the angle branch. Inference decodes 5-d (l,t,r,b,angle) ->
    # rotated box via bbox_coder=DistanceAnglePointCoder (le90). This is the
    # canonical mmrotate rotated_fcos recipe.
    bbox_head=dict(
        type='RotatedFCOSHead',
        num_classes=20,
        in_channels=256,
        feat_channels=256,
        stacked_convs=4,
        strides=[4, 8, 16, 32, 64],          # must match the FPN output strides
        regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512), (512, 1e8)),
        center_sampling=True,
        center_sample_radius=1.5,
        norm_on_bbox=True,
        centerness_on_reg=True,
        separate_angle=True,
        scale_angle=True,
        bbox_coder=dict(type='DistanceAnglePointCoder', angle_version='le90'),
        h_bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0),
        loss_angle=dict(type='L1Loss', loss_weight=0.2),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
    ),

    # Anchor-free head: no assigner/sampler/anchor train_cfg (must be None).
    train_cfg=None,
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(iou_thr=0.1),      # rotated NMS (multiclass_nms_rotated)
        max_per_img=2000,
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
    # 0.9998 (not 0.9999): the stage schedules are short (~7.4k iters); 0.9999's
    # ~6.9k-iter half-life would make the EMA nearly inert over one training pass.
    # NOTE: RegZeroInitHook is intentionally NOT used — it zeroes a RoI regression
    # FC, which an anchor-free FCOS head does not have.
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
