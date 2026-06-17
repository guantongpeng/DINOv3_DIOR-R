# =============================================================================
# RoI Transformer + DINOv3 ViT-B/16 + SimpleFPN + KFIoU loss (DIOR-R)
# =============================================================================
# Inherits the SimpleFPN config and swaps the DETECTOR from Oriented R-CNN to
# RoI Transformer (a 2-stage rotated refinement head):
#     Stage 1: RPN (horizontal proposals) + HBB head (SmoothL1)
#     Stage 2: rotated RoI feature -> RotatedKFIoUShared2FCBBoxHead + KFIoU loss
#
# Why RoI Transformer: it decouples coarse localization (stage 1) from
# angle/size refinement (stage 2), learning a rotated offset from each stage-1
# proposal. This is the strongest classic two-stage rotated head and the usual
# route to SOTA on DOTA/DIOR-R (+2~4 over a single-stage Oriented R-CNN).
#
# Backbone (DINOv3 ViT-B, single last-block feature) + neck (SimpleFPN, 4 levels
# at strides [4,8,16,32]) + data + aug + AdamW schedule are all inherited from
# the SimpleFPN config unchanged, so the comparison isolates the head.
#
# Recipe adapted from mmrotate's `roi_trans_kfiou_ln_r50_fpn_1x_dota_le90`,
# retuned for 4 FPN levels (ViT FPN) + DIOR-R (20 classes, 800px, le90).
# =============================================================================

_base_ = ['../oriented_rcnn/oriented_rcnn_dinov3_vitb_simplefpn_dior.py']

angle_version = 'le90'

model = dict(
    type='RoITransformer',

    # ---- RPN: horizontal proposals over the 4 SimpleFPN levels ----
    rpn_head=dict(
        _delete_=True,
        type='RotatedRPNHead',
        in_channels=256,
        feat_channels=256,
        version=angle_version,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32],
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0., 0., 0., 0.],
            target_stds=[1.0, 1.0, 1.0, 1.0],
        ),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
    ),

    # ---- ROI head: 2-stage refinement ----
    roi_head=dict(
        _delete_=True,
        type='RoITransRoIHead',
        version=angle_version,
        num_stages=2,
        stage_loss_weights=[1, 1],
        bbox_roi_extractor=[
            # Stage 1: horizontal RoIAlign on stage-1 HBB proposals
            dict(
                type='SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]),
            # Stage 2: rotated RoIAlign on refined rotated proposals
            dict(
                type='RotatedSingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlignRotated',
                    out_size=7,
                    sample_num=2,
                    clockwise=True),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]),
        ],
        bbox_head=[
            # Stage 1: HBB head (angle-encoded horizontal box), SmoothL1
            dict(
                type='RotatedShared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=20,
                bbox_coder=dict(
                    type='DeltaXYWHAHBBoxCoder',
                    angle_range=angle_version,
                    norm_factor=2,
                    edge_swap=True,
                    target_means=[0., 0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2, 1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(
                    type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
            # Stage 2: rotated refinement head, KFIoU loss
            dict(
                type='RotatedKFIoUShared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=20,
                bbox_coder=dict(
                    type='DeltaXYWHAOBBoxCoder',
                    angle_range=angle_version,
                    norm_factor=None,
                    edge_swap=True,
                    proj_xy=True,
                    target_means=[0., 0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1, 0.5]),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='KFLoss', fun='ln', loss_weight=0.5)),
        ],
    ),

    # ---- Training config ----
    train_cfg=dict(
        _delete_=True,
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            # Stage 1: horizontal IoU matching
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1,
                    iou_calculator=dict(type='BboxOverlaps2D')),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            # Stage 2: rotated IoU matching
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1,
                    iou_calculator=dict(type='RBboxOverlaps2D')),
                sampler=dict(
                    type='RRandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
        ],
    ),

    # ---- Testing config ----
    test_cfg=dict(
        _delete_=True,
        rpn=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type=angle_version, iou_thr=0.1),
            max_per_img=2000),
    ),
)
