# =============================================================================
# BASE config: RVSA + DINOv3 ViT-L/16 + ViT-Adapter (DIOR-R)
# =============================================================================
# This is a SHARED base. Do not train it directly -- use the stage1/stage2
# configs that inherit from it.
#
# Combines the end-to-end RVSA detector (VSA Transformer + Hungarian matching on
# rotated boxes) with the DINOv3 ViT-Adapter backbone:
#   1. BACKBONE -> DINOv3ViTAdapter (frozen ViT-L/16 + Spatial Prior Module +
#                  4 deformable InteractionBlocks fusing ViT layers [5,11,17,23]).
#                  Outputs a 4-level pyramid @ 1024 channels, strides [4,8,16,32].
#   2. NECK     -> FPN fusing the 1024-d adapter pyramid down to 4 levels @ 256
#                  (num_outs=4 matches the RVSA head's num_feature_levels=4).
#   3. HEAD     -> RVSAHead with VSATransformer (6 enc + 6 dec layers),
#                  RotatedHungarianAssigner (Focal cls + RotatedL1 + RotatedIoU).
#   4. DATA     -> TRAINVAL: train on the FULL train+val pool, use the held-out
#                  test split for both model selection (val) and final eval (test).
#   5. PRECISION-> bf16 for the heavy frozen ViT (inside the backbone); the
#                  trainable head/adapter train in fp32 (fp16 AMP disabled).
#   6. FREEZE   -> two-stage: see stage1 (frozen ViT) / stage2 (end-to-end).
#
# NOTE: RVSA is end-to-end (no RPN/RoI), so this base does NOT use
# RegZeroInitHook (that is RoI-regression specific). Stage 1/2 differ only in
# freeze_vit + lr/schedule.
# =============================================================================

_base_ = []

custom_imports = dict(
    imports=[
        'models.backbones.dinov3_vit_adapter',   # registers DINOv3ViTAdapter
        'models.layers.vsa_attention',           # registers VariedSizeAttention
        'models.layers.vsa_transformer',         # registers VSATransformer/Encoder/Decoder
        'models.layers.rotated_match',           # registers Rotated* assigner/costs/loss
        'models.dense_heads.rvsa_head',          # registers RVSAHead
        'models.detectors.rvsa',                 # registers RVSADetector
        'models.datasets.dior',
        'models.pipelines.albu_metadata',
    ],
    allow_failed_imports=False,
)

model = dict(
    type='RVSADetector',

    # -------------------- Backbone: DINOv3 ViT-Adapter -----------------------
    # Frozen ViT-L/16 (1024-d, 24 blocks) + Spatial Prior Module + 4 deformable
    # InteractionBlocks fusing ViT layers [5,11,17,23]. Outputs a 4-level pyramid
    # @ 1024 channels (full embed_dim), strides [4,8,16,32], fed into the FPN.
    # bf16_vit runs the frozen ViT in bfloat16 (official DINOv3 eval recipe).
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
    # Adapter pyramid is 4 levels @ 1024 -> FPN fuses to 4 levels @ 256, matching
    # the RVSA head's num_feature_levels=4 (no extra P5 needed for set prediction).
    neck=dict(
        type='FPN',
        in_channels=[1024, 1024, 1024, 1024],
        out_channels=256,
        start_level=0,
        add_extra_convs=False,
        num_outs=4,
    ),

    # -------------------- RVSA Head (end-to-end set prediction) --------------
    bbox_head=dict(
        type='RVSAHead',
        num_classes=20,
        in_channels=256,
        num_query=300,
        num_reg_fcs=2,
        sync_cls_avg_factor=True,
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='RotatedIoULoss', mode='linear', loss_weight=2.0),
        train_cfg=dict(
            assigner=dict(
                type='RotatedHungarianAssigner',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='RotatedL1Cost', weight=5.0),
                iou_cost=dict(type='RotatedIoUCost', weight=2.0))),
        test_cfg=dict(max_per_img=300),
        transformer=dict(
            type='VSATransformer',
            embed_dims=256,
            num_feature_levels=4,
            two_stage_num_proposals=300,
            encoder=dict(
                type='VSAEncoder',
                num_layers=6,
                embed_dims=256,
                feedforward_channels=2048,
                num_heads=8,
                num_levels=4,
                num_points=4,
                dilation_rates=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5],
            ),
            decoder=dict(
                type='VSADecoder',
                num_layers=6,
                embed_dims=256,
                feedforward_channels=2048,
                num_heads=8,
                num_levels=4,
                num_points=4,
                dilation_rates=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5],
                return_intermediate=True,
            ),
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
    # Calibrated for rvsa_vitl_adapter_trainval.sh default (samples_per_gpu=8).
    # Re-scale lr in the stage configs if you change GPU count or this value.
    samples_per_gpu=4,
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
    dict(type='EMAHook', momentum=0.9998, priority='ABOVE_NORMAL'),
]

# bf16 replaces fp16: the frozen ViT runs under bfloat16 autocast inside the
# adapter backbone (see bf16_vit=True). Disable the global fp16 AMP hook so the
# trainable head/adapter train in stable fp32 (RVSAHead losses are force_fp32).
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
