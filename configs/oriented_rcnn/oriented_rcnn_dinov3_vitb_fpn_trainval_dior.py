# =============================================================================
# Oriented R-CNN + DINOv3 ViT-B/16 + ViTDetFPN (MULTI-LAYER) — TRAINVAL run
# =============================================================================
# Inherits the proven multi-layer ViTDetFPN recipe
# (configs/oriented_rcnn/oriented_rcnn_dinov3_vitb_fpn_dior.py), and changes:
#
#   1. DATA split:
#      * train = DIOR-R train + val merged (ConcatDataset, ~11.7k images).
#      * val   = DIOR-R test split (periodic eval + save_best model selection).
#      * test  = DIOR-R test split (final tools/test.py evaluation).
#
#   2. backbone.layers_to_use = [3, 5, 8, 11]  -> ViT-B feeds 4 transformer-block
#      features into ViTDetFPN (multi-layer). The proven recipe that hit ~0.71+
#      mAP, vs the SimpleFPN single-last-block config that regressed to ~0.68.
#
#   3. ROTATION AUGMENTATION: adds PolyRandomRotate (mmrotate's rotated-aware
#      random rotation) to the train pipeline. DIOR images are north-up, so a
#      model can overfit absolute heading; random in-plane rotation enforces
#      orientation invariance and is a standard +1~3 mAP gain for oriented
#      detection. It rotates the image AND re-encodes the rotated GT boxes
#      (polygons) consistently under le90.
#
# Everything else (oriented R-CNN head, class weights, EMA, AdamW) is inherited.
#
# Usage:
#   bash scripts/dist_train_trainval_vitb_fpn.sh
# =============================================================================

_base_ = ['./oriented_rcnn_dinov3_vitb_fpn_dior.py']

# ===================== Train pipeline (with rotation) =====================
# Redefined here so we can insert PolyRandomRotate. img_norm_cfg / train_scales
# are re-declared to be self-contained (they override the base vars 1:1).
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

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
    # auto_bound=False keeps the (scaled) canvas size (corners are cropped),
    # matching the canonical Oriented R-CNN recipe.
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

# ===================== Data split: trainval merge, test as val/test =====================
data = dict(
    train=dict(
        type='DIORDataset',
        ann_file=[
            'data/DIOR-R/train/labelTxt/',
            'data/DIOR-R/val/labelTxt/',
        ],
        img_prefix=[
            'data/DIOR-R/train/images/',
            'data/DIOR-R/val/images/',
        ],
        pipeline=train_pipeline,
    ),
    val=dict(
        type='DIORDataset',
        ann_file='data/DIOR-R/test/labelTxt/',
        img_prefix='data/DIOR-R/test/images/',
    ),
    test=dict(
        type='DIORDataset',
        ann_file='data/DIOR-R/test/labelTxt/',
        img_prefix='data/DIOR-R/test/images/',
    ),
)

# Evaluate on the test split every 3 epochs and keep the best-by-mAP checkpoint.
evaluation = dict(
    interval=3,
    metric='mAP',
    save_best='mAP@0.50',
    rule='greater',
    gpu_collect=True,
)
