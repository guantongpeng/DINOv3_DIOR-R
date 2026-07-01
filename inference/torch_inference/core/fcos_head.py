"""Pure-PyTorch RotatedFCOSHead (ports mmrotate RotatedFCOSHead test path).

Anchor-free: per FPN level produces cls / ltrb-reg (norm_on_bbox: clamp(min=0)*stride)
/ centerness / angle (scale_angle). Decode via DistanceAnglePointCoder, then
rotated multi-class NMS with centerness as score factor. Same param names as the
trained checkpoint so weights load directly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import box_ops


class Scale(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(scale)))

    def forward(self, x):
        return x * self.scale


class ConvGNReLU(nn.Module):
    """Conv2d(3x3, no bias) + GroupNorm(32) + ReLU. Exposes .conv/.gn for keys."""

    def __init__(self, in_ch, out_ch, num_groups=32):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.gn = nn.GroupNorm(num_groups, out_ch)

    def forward(self, x):
        return F.relu(self.gn(self.conv(x)), inplace=True)


def fcos_points(feat_h, feat_w, stride, device, dtype):
    """FCOS reference points for one level: (H*W, 2) in (x, y), row-major."""
    shift_x = (torch.arange(feat_w, device=device, dtype=dtype) + 0.5) * stride
    shift_y = (torch.arange(feat_h, device=device, dtype=dtype) + 0.5) * stride
    yy, xx = torch.meshgrid(shift_y, shift_x, indexing='ij')
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)


class RotatedFCOSHead(nn.Module):
    def __init__(self, num_classes=20, in_channels=256, feat_channels=256,
                 stacked_convs=4, strides=(4, 8, 16, 32, 64), norm_on_bbox=True,
                 centerness_on_reg=True, scale_angle=True, num_groups=32):
        super().__init__()
        self.num_classes = num_classes
        self.strides = list(strides)
        self.norm_on_bbox = norm_on_bbox
        self.centerness_on_reg = centerness_on_reg
        self.is_scale_angle = scale_angle

        self.cls_convs = nn.ModuleList(
            [ConvGNReLU(in_channels, feat_channels, num_groups) for _ in range(stacked_convs)])
        self.reg_convs = nn.ModuleList(
            [ConvGNReLU(in_channels, feat_channels, num_groups) for _ in range(stacked_convs)])
        self.conv_cls = nn.Conv2d(feat_channels, num_classes, 3, padding=1)
        self.conv_reg = nn.Conv2d(feat_channels, 4, 3, padding=1)
        self.conv_centerness = nn.Conv2d(feat_channels, 1, 3, padding=1)
        self.conv_angle = nn.Conv2d(feat_channels, 1, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])
        if scale_angle:
            self.scale_angle = Scale(1.0)

    def forward(self, feats):
        cls_scores, bbox_preds, angle_preds, centernesses = [], [], [], []
        for i, x in enumerate(feats):
            cls_feat = x
            for m in self.cls_convs:
                cls_feat = m(cls_feat)
            reg_feat = x
            for m in self.reg_convs:
                reg_feat = m(reg_feat)
            cls_score = self.conv_cls(cls_feat)
            bbox_pred = self.conv_reg(reg_feat)
            centerness = self.conv_centerness(reg_feat if self.centerness_on_reg else cls_feat)
            bbox_pred = self.scales[i](bbox_pred).float()
            if self.norm_on_bbox:
                bbox_pred = bbox_pred.clamp(min=0)
                bbox_pred = bbox_pred * self.strides[i]   # inference (not training)
            angle_pred = self.conv_angle(reg_feat)
            if self.is_scale_angle:
                angle_pred = self.scale_angle(angle_pred).float()
            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
            angle_preds.append(angle_pred)
            centernesses.append(centerness)
        return cls_scores, bbox_preds, angle_preds, centernesses

    @torch.no_grad()
    def get_bboxes_single(self, cls_scores, bbox_preds, angle_preds, centernesses,
                          img_shape, scale_factor, nms_pre=2000, score_thr=0.05,
                          nms_iou=0.1, max_per_img=2000, rescale=True, device='cuda'):
        mlvl_bboxes, mlvl_scores, mlvl_centerness = [], [], []
        for lvl, (cs, bp, ap, cn) in enumerate(zip(cls_scores, bbox_preds, angle_preds, centernesses)):
            H, W = cs.shape[-2:]
            scores = cs.permute(1, 2, 0).reshape(-1, self.num_classes).sigmoid()
            cent = cn.permute(1, 2, 0).reshape(-1).sigmoid()
            bp = bp.permute(1, 2, 0).reshape(-1, 4)
            ap = ap.permute(1, 2, 0).reshape(-1, 1)
            bp = torch.cat([bp, ap], dim=1)            # (N,5) = l,t,r,b,angle
            points = fcos_points(H, W, self.strides[lvl], device, scores.dtype)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * cent[:, None]).max(dim=1)
                _, topk = max_scores.topk(nms_pre)
                points, bp, scores, cent = points[topk], bp[topk], scores[topk], cent[topk]
            bboxes = box_ops.distance_angle_point_decode(points, bp)   # [x,y,w,h,a] le90
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(cent)
        bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            sf = bboxes.new_tensor(scale_factor)
            bboxes[..., :4] = bboxes[..., :4] / sf
        scores = torch.cat(mlvl_scores)
        centerness = torch.cat(mlvl_centerness)
        scores = torch.cat([scores, scores.new_zeros(scores.size(0), 1)], dim=1)  # bg col
        dets, labels = box_ops.multiclass_nms_rotated(
            bboxes, scores, score_thr, nms_iou, max_num=max_per_img, score_factors=centerness)
        return dets, labels
