# =============================================================================
# Oriented R-CNN + DINOv3 ViT-B/16 + SimpleFPN + KFIoU loss (DIOR-R)
# =============================================================================
# Inherits the SimpleFPN config and ONLY swaps the regression head/loss:
#   SmoothL1 (DeltaXYWHAOBBoxCoder)  ->  KFIoU (RotatedKFIoUShared2FCBBoxHead).
#
# Why KFIoU: the angle-aware Kalman-Filter-IoU regression loss directly optimizes
# a differentiable surrogate of rotated IoU instead of decoupled (dx,dy,dw,dh,dθ)
# SmoothL1. It is the standard drop-in upgrade for two-stage rotated detectors
# (typically +1~3 mAP@0.5 on aerial datasets vs SmoothL1), with no architecture
# change and low risk.
#
# Recipe follows mmrotate's `roi_trans_kfiou_ln` stage-2 head exactly:
#   - DeltaXYWHAOBBoxCoder(norm_factor=None, edge_swap=True, proj_xy=True,
#     target_stds=[0.05,0.05,0.1,0.1,0.5])
#   - KFLoss(fun='ln', loss_weight=0.5)
#
# Backbone / neck / aug / schedule are identical to the SimpleFPN config so the
# only variable is the regression loss -> clean A/B comparison.
# =============================================================================

_base_ = ['./oriented_rcnn_dinov3_vitb_simplefpn_dior.py']

model = dict(
    roi_head=dict(
        bbox_head=dict(
            _delete_=True,
            type='RotatedKFIoUShared2FCBBoxHead',
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
                target_means=[0., 0., 0., 0., 0.],
                target_stds=[0.05, 0.05, 0.1, 0.1, 0.5],
            ),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0,
            ),
            loss_bbox=dict(type='KFLoss', fun='ln', loss_weight=0.5),
        ),
    ),
)
