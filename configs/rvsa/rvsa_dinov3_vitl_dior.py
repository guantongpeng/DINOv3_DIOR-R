# =============================================================================
# RVSA (Rotated Varied-Size Attention) with DINOv3 ViT-L for DIOR-R
# =============================================================================
# Backbone:  DINOv3 ViT-L/16 (1024-dim, SAT493M pretrained)
# Neck:      ViTDetFPN (1024->256, strides 4/8/16/32)
# Head:      RVSAHead with VSA Transformer (end-to-end set prediction)
#
# Architecture: Backbone -> FPN -> VSA Encoder -> VSA Decoder -> Cls+Reg
# =============================================================================

_base_ = []

custom_imports = dict(
    imports=[
        'models.backbones.dinov3_wrapper',
        'models.necks.vitdet_fpn',
        'models.layers.vsa_attention',
        'models.layers.vsa_transformer',
        'models.dense_heads.rvsa_head',
        'models.detectors.rvsa',
        'models.datasets.dior',
    ],
    allow_failed_imports=False,
)

model = dict(
    type='RVSADetector',
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
        loss_iou=dict(type='GIoULoss', loss_weight=2.0),
        train_cfg=dict(
            assigner=dict(
                type='HungarianAssigner',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))),
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

# ========================== Dataset ==========================

dataset_type = 'DIORDataset'
data_root = 'data/DIOR-R/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

image_size = (800, 800)
train_scales = [(600, 600), (800, 800), (1000, 1000)]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=train_scales, multiscale_mode='value'),
    dict(type='RRandomFlip', flip_ratio=[0.25, 0.25, 0.25],
         direction=['horizontal', 'vertical', 'diagonal'], version='le90'),
    dict(type='PhotoMetricDistortion', brightness_delta=32,
         contrast_range=(0.5, 1.5), saturation_range=(0.5, 1.5), hue_delta=18),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'],
         meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape',
                    'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                    'img_norm_cfg')),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiScaleFlipAug', img_scale=image_size, flip=False,
         transforms=[
             dict(type='RResize'),
             dict(type='Normalize', **img_norm_cfg),
             dict(type='Pad', size_divisor=32),
             dict(type='DefaultFormatBundle'),
             dict(type='Collect', keys=['img'],
                  meta_keys=('filename', 'ori_filename', 'ori_shape',
                             'img_shape', 'pad_shape', 'scale_factor',
                             'flip', 'flip_direction', 'img_norm_cfg')),
         ]),
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(type=dataset_type, ann_file=data_root + 'train/labelTxt/',
               img_prefix=data_root + 'train/images/',
               pipeline=train_pipeline, version='le90'),
    val=dict(type=dataset_type, ann_file=data_root + 'val/labelTxt/',
             img_prefix=data_root + 'val/images/',
             pipeline=test_pipeline, version='le90'),
    test=dict(type=dataset_type, ann_file=data_root + 'test/labelTxt/',
              img_prefix=data_root + 'test/images/',
              pipeline=test_pipeline, version='le90'),
)

evaluation = dict(interval=3, metric='mAP', save_best='mAP@0.50',
                  rule='greater', gpu_collect=True)

# ========================== Optimization ==========================

optimizer = dict(
    type='AdamW',
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'backbone.backbone': dict(lr_mult=0.1),
            'backbone.layer_norms': dict(lr_mult=1.0),
        },
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
    ),
)

optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
lr_config = dict(policy='CosineAnnealing', warmup='linear', warmup_iters=1000,
                 warmup_ratio=1.0 / 3, min_lr_ratio=1e-3)
runner = dict(type='EpochBasedRunner', max_epochs=50)

checkpoint_config = dict(interval=5, max_keep_ckpts=3)
log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='EMAHook', momentum=0.999, priority='ABOVE_NORMAL')]

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
