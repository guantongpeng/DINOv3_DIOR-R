"""Rotated matching components for end-to-end rotated detection (RVSA).

Provides:
    - RotatedL1Cost:  L1 matching cost on 5-dim normalized rotated boxes.
    - RotatedIoUCost: rotated IoU matching cost on absolute rotated boxes.
    - RotatedHungarianAssigner: one-to-one Hungarian matching for 5-dim le90 boxes.

All rotated boxes use the (cx, cy, w, h, theta) convention with theta in radians.
"""

import math

import torch
from scipy.optimize import linear_sum_assignment

from mmrotate.core.bbox.iou_calculators import rbbox_overlaps
from mmdet.core.bbox.match_costs.builder import MATCH_COST, build_match_cost
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.assigners.assign_result import AssignResult
from mmdet.core.bbox.assigners.base_assigner import BaseAssigner

PI = math.pi


def theta_to_norm(theta):
    """Map an angle in [-pi/2, pi/2) (le90) to [0, 1]."""
    return (theta + PI / 2) / PI


def norm_to_theta(value):
    """Map a value in [0, 1] back to an angle in [-pi/2, pi/2)."""
    return value * PI - PI / 2


def rbboxes_to_norm(rbboxes, img_h, img_w):
    """Normalize absolute (cx, cy, w, h, theta) boxes to [0, 1] range.

    Args:
        rbboxes (Tensor): shape (..., 5), absolute le90 rotated boxes.
        img_h (int): image height.
        img_w (int): image width.

    Returns:
        Tensor: same shape, normalized so cx/cy/w/h in [0, 1] and theta in [0, 1].
    """
    norm = rbboxes.new_empty(rbboxes.shape)
    norm[..., 0] = rbboxes[..., 0] / img_w
    norm[..., 1] = rbboxes[..., 1] / img_h
    norm[..., 2] = rbboxes[..., 2] / img_w
    norm[..., 3] = rbboxes[..., 3] / img_h
    norm[..., 4] = theta_to_norm(rbboxes[..., 4])
    return norm


def norm_to_rbboxes(norm_boxes, img_h, img_w):
    """Inverse of :func:`rbboxes_to_norm`: [0, 1] -> absolute rotated boxes."""
    abs_boxes = norm_boxes.new_empty(norm_boxes.shape)
    abs_boxes[..., 0] = norm_boxes[..., 0] * img_w
    abs_boxes[..., 1] = norm_boxes[..., 1] * img_h
    abs_boxes[..., 2] = norm_boxes[..., 2] * img_w
    abs_boxes[..., 3] = norm_boxes[..., 3] * img_h
    abs_boxes[..., 4] = norm_to_theta(norm_boxes[..., 4])
    return abs_boxes


@MATCH_COST.register_module()
class RotatedL1Cost:
    """L1 cost on 5-dim normalized rotated boxes.

    Both ``bbox_pred`` and ``gt_bboxes`` must already be normalized to [0, 1]
    (see :func:`rbboxes_to_norm`).
    """

    def __init__(self, weight=1.0):
        self.weight = weight

    def __call__(self, bbox_pred, gt_bboxes):
        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight


@MATCH_COST.register_module()
class RotatedIoUCost:
    """IoU cost on absolute 5-dim rotated boxes (cx, cy, w, h, theta).

    Uses rotated 2D IoU; higher IoU yields lower cost.
    """

    def __init__(self, weight=1.0):
        self.weight = weight

    def __call__(self, bboxes, gt_bboxes):
        ious = rbbox_overlaps(bboxes, gt_bboxes, mode='iou', is_aligned=False)
        cost = -ious
        return cost * self.weight


@BBOX_ASSIGNERS.register_module()
class RotatedHungarianAssigner(BaseAssigner):
    """One-to-one Hungarian matcher for 5-dim rotated (le90) boxes.

    The total matching cost is the weighted sum of:
        - classification cost (FocalLossCost by default),
        - regression L1 cost on normalized 5-dim boxes,
        - rotated IoU cost on absolute boxes.

    Args:
        cls_cost (dict): Classification match cost config.
        reg_cost (dict): Regression L1 cost config on normalized 5-dim boxes.
        iou_cost (dict): Rotated IoU cost config on absolute boxes.
    """

    def __init__(self,
                 cls_cost=dict(type='FocalLossCost', weight=2.0),
                 reg_cost=dict(type='RotatedL1Cost', weight=5.0),
                 iou_cost=dict(type='RotatedIoUCost', weight=2.0)):
        self.cls_cost = build_match_cost(cls_cost)
        self.reg_cost = build_match_cost(reg_cost)
        self.iou_cost = build_match_cost(iou_cost)

    def assign(self,
               bbox_pred,
               cls_pred,
               gt_bboxes,
               gt_labels,
               img_meta,
               gt_bboxes_ignore=None,
               eps=1e-7):
        """Compute one-to-one matching between predictions and ground truth.

        Args:
            bbox_pred (Tensor): normalized (cx, cy, w, h, theta) predictions in
                [0, 1], shape (num_query, 5).
            cls_pred (Tensor): classification logits, shape (num_query, C).
            gt_bboxes (Tensor): absolute le90 rotated boxes, shape (num_gt, 5).
            gt_labels (Tensor): shape (num_gt,).
            img_meta (dict): image meta containing ``img_shape``.
            gt_bboxes_ignore (Tensor, optional): ignored boxes. Must be None.
        """
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)

        assigned_gt_inds = bbox_pred.new_full(
            (num_bboxes,), -1, dtype=torch.long)
        assigned_labels = bbox_pred.new_full(
            (num_bboxes,), -1, dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            if num_gts == 0:
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)

        img_h, img_w = img_meta['img_shape'][:2]

        # 1. classification cost
        cls_cost = self.cls_cost(cls_pred, gt_labels)

        # 2. regression L1 cost on normalized 5-dim boxes
        normalize_gt_bboxes = rbboxes_to_norm(gt_bboxes, img_h, img_w)
        reg_cost = self.reg_cost(bbox_pred, normalize_gt_bboxes)

        # 3. rotated IoU cost on absolute boxes (detach for matching)
        pred_abs = norm_to_rbboxes(bbox_pred.detach(), img_h, img_w)
        iou_cost = self.iou_cost(pred_abs, gt_bboxes)

        cost = cls_cost + reg_cost + iou_cost
        cost = cost.detach().cpu()

        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(bbox_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(bbox_pred.device)

        assigned_gt_inds[:] = 0
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels)
