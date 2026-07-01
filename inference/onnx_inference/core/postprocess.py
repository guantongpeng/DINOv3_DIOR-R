"""Single-stage head post-processing (PyTorch) for the exported ONNX outputs.

Supports the two anchor-free heads sharing the DINOv3 ViT-Adapter backbone:

  * **FCOS**  : head outputs (cls, bbox[ltrb], angle, centerness). Bbox is the
    head's already-scaled pixel distances; decode via the rotated
    ``DistanceAnglePointCoder``. Score = centerness (applied as NMS score factor).
  * ``YOLO26``: head outputs (cls, bbox[raw ltrb], angle[raw], obj). Bbox needs
    ``clamp(max=15).exp()`` then a *horizontal* ltrb decode; angle =
    ``(sigmoid-0.5)*pi``; score = ``cls*obj`` baked in before NMS.

Both then run the same per-class rotated NMS. Rotated IoU / NMS reuse the
validated ``inference/torch_inference/box_ops.py`` (pure PyTorch -> CPU & CUDA).
"""

import math
import os.path as osp
import sys

import numpy as np
import torch

_HERE = osp.dirname(osp.abspath(__file__))            # .../inference/onnx_inference/core
_INF = osp.dirname(osp.dirname(_HERE))                # .../inference  (torch_inference lives here)
if _INF not in sys.path:
    sys.path.insert(0, _INF)

from torch_inference.core.box_ops import box_iou_rotated  # noqa: E402  (validated vs mmcv, greedy fallback)

try:  # fast reference rotated NMS (CUDA + CPU); greedy fallback if unavailable
    from mmcv.ops import nms_rotated as _mmcv_nms_rotated
    _HAS_MMCV_NMS = True
except Exception:  # pragma: no cover
    _HAS_MMCV_NMS = False

NUM_CLASSES = 20
STRIDES = (4, 8, 16, 32, 64)
POINT_OFFSET = 0.5  # MlvlPointGenerator default


def grid_priors(featmap_sizes, strides=STRIDES, offset=POINT_OFFSET, device='cpu'):
    """MlvlPointGenerator.grid_priors: list of (H*W, 2) point grids [x, y]."""
    priors = []
    for (H, W), stride in zip(featmap_sizes, strides):
        shift_x = (torch.arange(W, device=device, dtype=torch.float32) + offset) * stride
        shift_y = (torch.arange(H, device=device, dtype=torch.float32) + offset) * stride
        sy, sx = torch.meshgrid(shift_y, shift_x, indexing='ij')
        priors.append(torch.stack([sx.reshape(-1), sy.reshape(-1)], dim=-1))
    return priors


def norm_angle_le90(angle):
    return (angle + math.pi / 2) % math.pi - math.pi / 2


# ----------------------------------------------------------------------------
# distance2obb  (FCOS: DistanceAnglePointCoder.distance2obb, le90)
# ----------------------------------------------------------------------------
def distance2obb(points, distance):
    """points (N,2), distance (N,5)=[l,t,r,b,angle] -> (N,5) [cx,cy,w,h,a]."""
    dist, angle = distance[:, :4], distance[:, 4:5]
    cos, sin = torch.cos(angle), torch.sin(angle)
    rot = torch.cat([cos, -sin, sin, cos], dim=1).reshape(-1, 2, 2)
    wh = dist[:, :2] + dist[:, 2:]
    offset_t = (dist[:, 2:] - dist[:, :2]) / 2.0
    offset = torch.bmm(rot, offset_t.unsqueeze(2)).squeeze(2)
    ctr = points + offset
    return torch.cat([ctr, wh, norm_angle_le90(angle)], dim=-1)


# ----------------------------------------------------------------------------
# YOLO26 horizontal ltrb decode + sigmoid angle
# ----------------------------------------------------------------------------
def decode_yolo26(ltrb, angle_raw, points):
    """ltrb (N,4) already exp()'d; angle_raw (N,1); points (N,2) -> (N,5)."""
    px, py = points[:, 0], points[:, 1]
    l, t, r, b = ltrb[:, 0], ltrb[:, 1], ltrb[:, 2], ltrb[:, 3]
    x1, y1, x2, y2 = px - l, py - t, px + r, py + b
    w = (x2 - x1).clamp(min=1.0)
    h = (y2 - y1).clamp(min=1.0)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    angle = (torch.sigmoid(angle_raw) - 0.5) * math.pi
    return torch.stack([cx, cy, w, h, angle.squeeze(-1)], dim=-1)


# ----------------------------------------------------------------------------
# rotated NMS (shared)
# ----------------------------------------------------------------------------
def _nms_rotated(boxes, scores, iou_thr):
    order = torch.argsort(scores, descending=True)
    boxes = boxes[order]
    n = boxes.shape[0]
    if n == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    iou = box_iou_rotated(boxes, boxes)
    keep = []
    suppressed = torch.zeros(n, dtype=torch.bool, device=boxes.device)
    for i in range(n):
        if suppressed[i]:
            continue
        keep.append(i)
        if i + 1 < n:
            suppressed[i + 1:] |= (iou[i, i + 1:] >= iou_thr)
    return order[torch.as_tensor(keep, dtype=torch.long, device=boxes.device)]


def multiclass_nms_rotated(multi_bboxes, multi_scores, score_thr, iou_thr,
                           max_num, score_factors=None):
    """Per-class rotated NMS. multi_scores (N, nc+1) [bg col ignored].
    score_factors: (N,) applied to scores for NMS ranking (None for YOLO26).

    Uses mmcv's fast ``nms_rotated`` (the reference op, via the class-offset
    trick so different classes never suppress each other) when available, else
    a greedy pure-torch fallback.
    """
    num_classes = multi_scores.shape[1] - 1
    scores = multi_scores[:, :-1]
    labels = torch.arange(num_classes, device=scores.device).view(1, -1).expand_as(scores)
    bboxes = multi_bboxes[:, None].expand(multi_bboxes.shape[0], num_classes, 5).reshape(-1, 5)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)
    valid_mask = scores > score_thr
    if score_factors is not None:
        sf = score_factors.view(-1, 1).expand(multi_bboxes.shape[0], num_classes).reshape(-1)
        scores = scores * sf
    inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
    bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]
    if bboxes.numel() == 0:
        return torch.cat([bboxes, scores[:, None]], -1), labels

    if _HAS_MMCV_NMS:
        max_coord = bboxes[:, :2].max() + bboxes[:, 2:4].max()
        offsets = labels.to(bboxes) * (max_coord + 1)
        bf = bboxes.clone()
        bf[:, :2] = bf[:, :2] + offsets[:, None]
        _, keep = _mmcv_nms_rotated(bf, scores, iou_thr)
        if max_num > 0:
            keep = keep[:max_num]
        keep = keep.to(torch.long)
        bboxes, scores, labels = bboxes[keep], scores[keep], labels[keep]
        return torch.cat([bboxes, scores[:, None]], 1), labels

    # greedy fallback (per class) -- slower, only used without mmcv
    det_b, det_s, det_l = [], [], []
    for c in range(num_classes):
        m = labels == c
        if not m.any():
            continue
        cb, cs = bboxes[m], scores[m]
        keep = _nms_rotated(cb, cs, iou_thr)
        det_b.append(cb[keep]); det_s.append(cs[keep])
        det_l.append(torch.full((keep.numel(),), c, dtype=torch.long, device=labels.device))
    if not det_b:
        return bboxes.new_zeros((0, 6)), labels.new_zeros((0,))
    det_b = torch.cat(det_b); det_s = torch.cat(det_s); det_l = torch.cat(det_l)
    if max_num > 0 and det_b.shape[0] > max_num:
        _, top = det_s.sort(descending=True)
        det_b, det_s, det_l = det_b[top[:max_num]], det_s[top[:max_num]], det_l[top[:max_num]]
    return torch.cat([det_b, det_s[:, None]], 1), det_l


def _rescale(bboxes, scale_factor):
    if isinstance(scale_factor, np.ndarray):
        scale_factor = torch.as_tensor(scale_factor, dtype=bboxes.dtype, device=bboxes.device)
    sf = scale_factor if scale_factor.numel() == 4 else scale_factor[:4]
    bboxes[..., :4] = bboxes[..., :4] / sf
    return bboxes


def _to_result(dets, labels, num_classes=NUM_CLASSES):
    dets = dets.cpu().numpy()
    labels = labels.cpu().numpy()
    result = [np.zeros((0, 6), dtype=np.float32) for _ in range(num_classes)]
    if dets.shape[0]:
        for c in range(num_classes):
            m = labels == c
            if m.any():
                result[c] = dets[m]
    return result


# ----------------------------------------------------------------------------
# per-head decoders
# ----------------------------------------------------------------------------
def postprocess_fcos(outputs, meta, cfg, strides=STRIDES):
    fourth = cfg.get('fourth_key', 'ctr')
    cls_scores, bbox_preds, angle_preds, cents = [], [], [], []
    sizes = []
    for lvl in range(len(strides)):
        cls_scores.append(outputs[f'f{lvl}_cls'][0])
        bbox_preds.append(outputs[f'f{lvl}_bbox'][0])
        angle_preds.append(outputs[f'f{lvl}_angle'][0])
        cents.append(outputs[f'f{lvl}_{fourth}'][0])
        sizes.append(cls_scores[-1].shape[-2:])
    points = grid_priors(sizes, strides, POINT_OFFSET, cls_scores[0].device)
    mlvl_b, mlvl_s, mlvl_c = [], [], []
    for cs, bp, ap, cn, pt in zip(cls_scores, bbox_preds, angle_preds, cents, points):
        H, W = cs.shape[-2:]
        scores = cs.permute(1, 2, 0).reshape(H * W, -1).sigmoid()
        ctr = cn.permute(1, 2, 0).reshape(H * W).sigmoid()
        bp = bp.permute(1, 2, 0).reshape(H * W, 4)
        ap = ap.permute(1, 2, 0).reshape(H * W, 1)
        bp = torch.cat([bp, ap], dim=1)
        nms_pre = cfg.get('nms_pre', -1)
        if nms_pre > 0 and scores.shape[0] > nms_pre:
            mx, _ = (scores * ctr[:, None]).max(dim=1)
            _, tk = mx.topk(nms_pre)
            pt, bp, scores, ctr = pt[tk], bp[tk], scores[tk], ctr[tk]
        mlvl_b.append(distance2obb(pt, bp)); mlvl_s.append(scores); mlvl_c.append(ctr)
    mlvl_b = _rescale(torch.cat(mlvl_b), meta['scale_factor'])
    mlvl_s = torch.cat(mlvl_s); mlvl_c = torch.cat(mlvl_c)
    multi_scores = torch.cat([mlvl_s, mlvl_s.new_zeros(mlvl_s.shape[0], 1)], dim=1)
    dets, labels = multiclass_nms_rotated(
        mlvl_b, multi_scores, cfg['score_thr'], cfg['nms_iou_thr'],
        cfg['max_per_img'], score_factors=mlvl_c)
    return _to_result(dets, labels, cfg.get('num_classes', NUM_CLASSES))


def postprocess_yolo26(outputs, meta, cfg, strides=STRIDES):
    cls_scores, bbox_preds, angle_preds, objs = [], [], [], []
    sizes = []
    for lvl in range(len(strides)):
        cls_scores.append(outputs[f'f{lvl}_cls'][0])
        bbox_preds.append(outputs[f'f{lvl}_bbox'][0])
        angle_preds.append(outputs[f'f{lvl}_angle'][0])
        objs.append(outputs[f'f{lvl}_obj'][0])
        sizes.append(cls_scores[-1].shape[-2:])
    points = grid_priors(sizes, strides, POINT_OFFSET, cls_scores[0].device)
    mlvl_b, mlvl_s = [], []
    for cs, bp, ap, ob, pt in zip(cls_scores, bbox_preds, angle_preds, objs, points):
        H, W = cs.shape[-2:]
        cls_sig = cs.permute(1, 2, 0).reshape(H * W, -1).sigmoid()
        obj_sig = ob.permute(1, 2, 0).reshape(H * W).sigmoid()
        scores = cls_sig * obj_sig[:, None]            # (HW, nc) -- obj baked in
        ltrb = bp.permute(1, 2, 0).reshape(H * W, 4).clamp(max=15.0).exp()
        araw = ap.permute(1, 2, 0).reshape(H * W, 1)
        bboxes = decode_yolo26(ltrb, araw, pt)
        nms_pre = cfg.get('nms_pre', -1)
        if nms_pre > 0 and scores.shape[0] > nms_pre:
            mx, _ = scores.max(dim=1)
            _, tk = mx.topk(nms_pre)
            bboxes, scores = bboxes[tk], scores[tk]
        mlvl_b.append(bboxes); mlvl_s.append(scores)
    mlvl_b = _rescale(torch.cat(mlvl_b), meta['scale_factor'])
    mlvl_s = torch.cat(mlvl_s)
    multi_scores = torch.cat([mlvl_s, mlvl_s.new_zeros(mlvl_s.shape[0], 1)], dim=1)
    dets, labels = multiclass_nms_rotated(
        mlvl_b, multi_scores, cfg['score_thr'], cfg['nms_iou_thr'],
        cfg['max_per_img'])            # no score_factors: obj already in score
    return _to_result(dets, labels, cfg.get('num_classes', NUM_CLASSES))


_DISPATCH = {'fcos': postprocess_fcos, 'yolo26': postprocess_yolo26}


def postprocess_single(outputs, meta, head, cfg):
    """Dispatch on head type. cfg must carry score_thr/nms_iou_thr/max_per_img/
    nms_pre (and fourth_key for FCOS, num_classes)."""
    fn = _DISPATCH.get(head)
    if fn is None:
        raise ValueError(f'unsupported single-stage head: {head}')
    return fn(outputs, meta, cfg)
