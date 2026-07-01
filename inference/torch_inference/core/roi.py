"""Pure-PyTorch OrientedStandardRoIHead (ports mmrotate RoI stage).

Contains:
  * roi_align_rotated  -- faithful port of the mmcv CUDA op (aligned=False,
                          clockwise -> -theta, sample_num=2, d2 bilinear)
  * RotatedSingleRoIExtractor (level assignment by sqrt(w*h), finest_scale=56)
  * RotatedShared2FCBBoxHead (shared_fcs -> fc_cls/fc_reg)
  * OrientedStandardRoIHead.simple_test -> per-class [N,6] arrays (eval format)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import box_ops


# ---------------------------------------------------------------------------
# RoIAlignRotated
# ---------------------------------------------------------------------------
def roi_align_rotated(feat, rois, out_size=7, spatial_scale=0.25,
                      sampling_ratio=2, clockwise=True, aligned=False):
    """feat: (C,H,W) single image; rois: (n,6) [b_ind,cx,cy,w,h,a]; -> (n,C,o,o)."""
    n = rois.shape[0]
    C, H, W = feat.shape
    if n == 0:
        return feat.new_zeros(n, C, out_size, out_size)
    dev = feat.device
    offset = 0.5 if aligned else 0.0
    cx = rois[:, 1] * spatial_scale - offset
    cy = rois[:, 2] * spatial_scale - offset
    rw = rois[:, 3] * spatial_scale
    rh = rois[:, 4] * spatial_scale
    theta = rois[:, 5]
    if clockwise:
        theta = -theta
    if not aligned:
        rw = rw.clamp(min=1.0)
        rh = rh.clamp(min=1.0)
    bin_h = rh / out_size
    bin_w = rw / out_size
    o = out_size
    sr = sampling_ratio
    ph = torch.arange(o, device=dev).float()
    sub = (torch.arange(sr, device=dev).float() + 0.5) / sr
    # yy[n,ph,iy], xx[n,pw,ix]
    yy = (-rh / 2)[:, None, None] + ph[None, :, None] * bin_h[:, None, None] + sub[None, None, :] * bin_h[:, None, None]
    xx = (-rw / 2)[:, None, None] + ph[None, :, None] * bin_w[:, None, None] + sub[None, None, :] * bin_w[:, None, None]
    # combine over (ph,pw,iy,ix)
    yy = yy[:, :, None, :, None]   # (n,ph,1,iy,1)
    xx = xx[:, None, :, None, :]   # (n,1,pw,1,ix)
    y = yy * torch.cos(theta)[:, None, None, None, None] - xx * torch.sin(theta)[:, None, None, None, None] + cy[:, None, None, None, None]
    x = yy * torch.sin(theta)[:, None, None, None, None] + xx * torch.cos(theta)[:, None, None, None, None] + cx[:, None, None, None, None]
    # -> sample points (n, o*o*sr*sr)
    y = y.reshape(n, -1)
    x = x.reshape(n, -1)
    flat = feat.reshape(C, -1)                       # (C, H*W)
    valid = (y > -1.0) & (y < H) & (x > -1.0) & (x < W)
    yc = y.clamp(0, H - 1)
    xc = x.clamp(0, W - 1)
    y_low = torch.floor(yc).long()
    x_low = torch.floor(xc).long()
    y_high = (y_low + 1).clamp(max=H - 1)
    x_high = (x_low + 1).clamp(max=W - 1)
    ly = yc - y_low
    lx = xc - x_low
    hy = 1.0 - ly
    hx = 1.0 - lx
    il = (y_low * W + x_low)
    ih = (y_low * W + x_high)
    il2 = (y_high * W + x_low)
    ih2 = (y_high * W + x_high)
    vl = hy * hx
    vr = hy * lx
    vll = ly * hx
    vrr = ly * lx
    def g(idx):
        return flat[:, idx.reshape(-1)].reshape(C, n, -1).permute(1, 2, 0)   # (n,s,C)
    val = vl[:, :, None] * g(il) + vr[:, :, None] * g(ih) + vll[:, :, None] * g(il2) + vrr[:, :, None] * g(ih2)
    val = val * valid[:, :, None]
    val = val.reshape(n, o, o, sr, sr, C).mean(dim=(3, 4))   # average pooling
    return val.permute(0, 3, 1, 2).contiguous()              # (n,C,o,o)


class RotatedSingleRoIExtractor(nn.Module):
    def __init__(self, out_channels=256, featmap_strides=(4, 8, 16, 32),
                 out_size=7, finest_scale=56):
        super().__init__()
        self.out_channels = out_channels
        self.featmap_strides = list(featmap_strides)
        self.out_size = out_size
        self.finest_scale = finest_scale

    def map_roi_levels(self, rois, num_levels):
        scale = torch.sqrt(rois[:, 3] * rois[:, 4])
        lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-6))
        return lvls.clamp(min=0, max=num_levels - 1).long()

    @torch.no_grad()
    def forward(self, feats, rois):
        num_levels = len(feats)
        n = rois.shape[0]
        roi_feats = feats[0].new_zeros(n, self.out_channels, self.out_size, self.out_size)
        if n == 0:
            return roi_feats
        if num_levels == 1:
            feat = feats[0][0]
            return roi_align_rotated(feat, rois, self.out_size, 1.0 / self.featmap_strides[0])
        target_lvls = self.map_roi_levels(rois, num_levels)
        for i in range(num_levels):
            mask = target_lvls == i
            inds = mask.nonzero(as_tuple=False).squeeze(1)
            if inds.numel() > 0:
                feat = feats[i][0]   # single image batch
                roi_feats[inds] = roi_align_rotated(feat, rois[inds], self.out_size,
                                                    1.0 / self.featmap_strides[i])
        return roi_feats


# ---------------------------------------------------------------------------
# bbox head
# ---------------------------------------------------------------------------
class RotatedShared2FCBBoxHead(nn.Module):
    def __init__(self, in_channels=256, fc_out_channels=1024, roi_feat_size=7,
                 num_classes=20, reg_class_agnostic=True):
        super().__init__()
        self.num_classes = num_classes
        self.reg_class_agnostic = reg_class_agnostic
        self.shared_fcs = nn.ModuleList([
            nn.Linear(in_channels * roi_feat_size * roi_feat_size, fc_out_channels),
            nn.Linear(fc_out_channels, fc_out_channels)])
        out_dim = 5 if reg_class_agnostic else num_classes * 5
        self.fc_cls = nn.Linear(fc_out_channels, num_classes + 1)
        self.fc_reg = nn.Linear(fc_out_channels, out_dim)

    def forward(self, roi_feats):
        x = roi_feats.flatten(1)
        for fc in self.shared_fcs:
            x = F.relu(fc(x))
        return self.fc_cls(x), self.fc_reg(x)


# ---------------------------------------------------------------------------
# roi head (test path)
# ---------------------------------------------------------------------------
class OrientedStandardRoIHead(nn.Module):
    test_cfg = dict(nms_pre=2000, min_bbox_size=0, score_thr=0.05,
                    nms_iou=0.1, max_per_img=2000)

    def __init__(self, bbox_roi_extractor=None, bbox_head=None):
        super().__init__()
        self.bbox_roi_extractor = bbox_roi_extractor
        self.bbox_head = bbox_head

    def _bbox_forward(self, feats, rois):
        # RoI extraction uses only the first len(featmap_strides) FPN levels
        # (mmrotate: x[:bbox_roi_extractor.num_inputs]).
        n_inp = len(self.bbox_roi_extractor.featmap_strides)
        roi_feats = self.bbox_roi_extractor(feats[:n_inp], rois)
        cls_score, bbox_pred = self.bbox_head(roi_feats)
        return cls_score, bbox_pred

    @torch.no_grad()
    def simple_test_bboxes(self, feats, proposals_list, img_metas, rescale=True):
        rois = box_ops.rbbox2roi(proposals_list)
        cls_score, bbox_pred = self._bbox_forward(feats, rois)
        num_per = [p.size(0) for p in proposals_list]
        cls_score = cls_score.split(num_per, 0)
        bbox_pred = bbox_pred.split(num_per, 0)
        det_bboxes, det_labels = [], []
        for i in range(len(proposals_list)):
            b, l = self._get_bboxes_single(
                rois.split(num_per, 0)[i] if rois.size(0) > 0 else rois,
                cls_score[i], bbox_pred[i], img_metas[i]['img_shape'],
                img_metas[i]['scale_factor'], rescale)
            det_bboxes.append(b)
            det_labels.append(l)
        return det_bboxes, det_labels

    def _get_bboxes_single(self, rois, cls_score, bbox_pred, img_shape, scale_factor, rescale):
        if cls_score is None or cls_score.numel() == 0:
            return rois.new_zeros((0, 6)), rois.new_zeros((0,), dtype=torch.long)
        scores = F.softmax(cls_score, dim=-1)
        bboxes = box_ops.delta_xywha_decode(
            rois[:, 1:6] if rois.size(0) > 0 else rois.new_zeros((0, 5)),
            bbox_pred, means=(0.,) * 5, stds=(0.1, 0.1, 0.2, 0.2, 0.1),
            max_shape=img_shape, edge_swap=True, proj_xy=True)
        if rescale and bboxes.size(0) > 0:
            sf = bboxes.new_tensor(scale_factor)
            bboxes = bboxes.view(bboxes.size(0), -1, 5)
            bboxes[..., :4] = bboxes[..., :4] / sf
            bboxes = bboxes.view(bboxes.size(0), -1)
        dets, labels = box_ops.multiclass_nms_rotated(
            bboxes, scores, self.test_cfg['score_thr'], self.test_cfg['nms_iou'],
            max_num=self.test_cfg['max_per_img'])
        return dets, labels

    @torch.no_grad()
    def simple_test(self, feats, proposal_list, img_metas, rescale=True):
        det_bboxes, det_labels = self.simple_test_bboxes(feats, proposal_list, img_metas, rescale)
        return [box_ops.rbbox2result(det_bboxes[i], det_labels[i], self.bbox_head.num_classes)
                for i in range(len(det_bboxes))]
