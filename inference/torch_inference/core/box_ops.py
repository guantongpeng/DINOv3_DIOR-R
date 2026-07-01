"""Pure-PyTorch rotated bounding-box operations (no openmmlab).

Ported verbatim (math only) from mmrotate/mmdet so detection + mAP match the
reference framework:
  * le90  <-> polygon  / hbb conversions (torch + cv2.minAreaRect for GT)
  * MidpointOffsetCoder.delta2bbox  (RPN decode)
  * DeltaXYWHAOBBoxCoder.delta2bbox (RoI head decode, edge_swap + proj_xy)
  * box_iou_rotated  (vectorised convex-polygon intersection, matches mmcv)
  * horizontal NMS (torchvision) + rotated multi-class NMS (greedy, mmrotate)
  * VOC-style average_precision ('11points' & 'area')
"""

import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import batched_nms as _tv_batched_nms
from torchvision.ops import nms as _tv_nms

PI = math.pi


# ----------------------------------------------------------------------------
# angle normalisation + le90 conversions
# ----------------------------------------------------------------------------
def norm_angle(angle, angle_range='le90'):
    if angle_range == 'oc':
        return angle
    if angle_range == 'le135':
        return (angle + PI / 4) % PI - PI / 4
    if angle_range == 'le90':
        return (angle + PI / 2) % PI - PI / 2
    raise NotImplementedError(angle_range)


def obb2poly_le90(rboxes):
    """[x,y,w,h,a] (N,5) -> polygon [x0,y0,...,y3] (N,8)."""
    N = rboxes.shape[0]
    if N == 0:
        return rboxes.new_zeros((0, 8))
    x_ctr, y_ctr, width, height, angle = (rboxes[:, 0], rboxes[:, 1],
                                          rboxes[:, 2], rboxes[:, 3],
                                          rboxes[:, 4])
    tl_x, tl_y, br_x, br_y = -width * 0.5, -height * 0.5, width * 0.5, height * 0.5
    rects = torch.stack([tl_x, br_x, br_x, tl_x, tl_y, tl_y, br_y, br_y],
                        dim=0).reshape(2, 4, N).permute(2, 0, 1)
    sin, cos = torch.sin(angle), torch.cos(angle)
    M = torch.stack([cos, -sin, sin, cos], dim=0).reshape(2, 2, N).permute(2, 0, 1)
    polys = M.matmul(rects).permute(2, 1, 0).reshape(-1, N).transpose(1, 0)
    polys[:, ::2] += x_ctr.unsqueeze(1)
    polys[:, 1::2] += y_ctr.unsqueeze(1)
    return polys.contiguous()


def obb2xyxy_le90(obboxes):
    center, w, h, theta = torch.split(obboxes, [2, 1, 1, 1], dim=-1)
    Cos, Sin = torch.cos(theta), torch.sin(theta)
    x_bias = torch.abs(w / 2 * Cos) + torch.abs(h / 2 * Sin)
    y_bias = torch.abs(w / 2 * Sin) + torch.abs(h / 2 * Cos)
    bias = torch.cat([x_bias, y_bias], dim=-1)
    return torch.cat([center - bias, center + bias], dim=-1)


def poly2obb_le90(polys):
    """polygon [..,8] -> [x,y,w,h,a] le90 (torch)."""
    polys = polys.reshape(*polys.shape[:-1], 8) if polys.shape[-1] == 8 else polys.reshape(-1, 8)
    pt1, pt2, pt3, pt4 = polys[..., :8].chunk(4, -1)
    edge1 = torch.sqrt((pt1[..., 0] - pt2[..., 0]) ** 2 + (pt1[..., 1] - pt2[..., 1]) ** 2)
    edge2 = torch.sqrt((pt2[..., 0] - pt3[..., 0]) ** 2 + (pt2[..., 1] - pt3[..., 1]) ** 2)
    a1 = torch.atan2(pt2[..., 1] - pt1[..., 1], pt2[..., 0] - pt1[..., 0])
    a2 = torch.atan2(pt4[..., 1] - pt1[..., 1], pt4[..., 0] - pt1[..., 0])
    angles = polys.new_zeros(polys.shape[0])
    angles[edge1 > edge2] = a1[edge1 > edge2].squeeze(-1) if a1.dim() > angles.dim() else a1[edge1 > edge2]
    angles[edge1 <= edge2] = a2[edge1 <= edge2].squeeze(-1) if a2.dim() > angles.dim() else a2[edge1 <= edge2]
    angles = norm_angle(angles, 'le90')
    x_ctr = (pt1[..., 0] + pt3[..., 0]) / 2.0
    y_ctr = (pt1[..., 1] + pt3[..., 1]) / 2.0
    edges = torch.stack([edge1.squeeze(-1), edge2.squeeze(-1)], dim=1)
    width, _ = torch.max(edges, 1)
    height, _ = torch.min(edges, 1)
    return torch.stack([x_ctr.squeeze(-1), y_ctr.squeeze(-1), width, height, angles], 1)


def poly2obb_np_le90(poly):
    """numpy: 8 coords -> (x,y,w,h,a) le90, via cv2.minAreaRect (matches mmrotate)."""
    bboxps = np.array(poly).reshape((4, 2))
    rbbox = cv2.minAreaRect(bboxps)
    x, y, w, h, a = rbbox[0][0], rbbox[0][1], rbbox[1][0], rbbox[1][1], rbbox[2]
    if w < 2 or h < 2:
        return None
    a = a / 180.0 * np.pi
    if w < h:
        w, h = h, w
        a += np.pi / 2
    while not (np.pi / 2 > a >= -np.pi / 2):
        if a >= np.pi / 2:
            a -= np.pi
        else:
            a += np.pi
    return x, y, w, h, a


# ----------------------------------------------------------------------------
# result / roi formatting
# ----------------------------------------------------------------------------
def rbbox2result(bboxes, labels, num_classes):
    if bboxes.shape[0] == 0:
        return [np.zeros((0, 6), dtype=np.float32) for _ in range(num_classes)]
    bboxes = bboxes.cpu().numpy()
    labels = labels.cpu().numpy()
    return [bboxes[labels == i, :] for i in range(num_classes)]


def rbbox2roi(bbox_list):
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        if bboxes.size(0) > 0:
            img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
            rois = torch.cat([img_inds, bboxes[:, :5]], dim=-1)
        else:
            rois = bboxes.new_zeros((0, 6))
        rois_list.append(rois)
    return torch.cat(rois_list, 0)


# ----------------------------------------------------------------------------
# box coders (decode only -- inference path)
# ----------------------------------------------------------------------------
def distance_angle_point_decode(points, pred):
    """DistanceAnglePointCoder.distance2obb (le90). points (N,2), pred (N,5)
    = (left, top, right, bottom, angle) -> [cx, cy, w, h, a] le90."""
    dist = pred[:, :4]
    angle = pred[:, 4:5]
    cos, sin = torch.cos(angle), torch.sin(angle)
    rot = torch.cat([cos, -sin, sin, cos], dim=1).reshape(-1, 2, 2)
    wh = dist[:, :2] + dist[:, 2:]                       # (l+r, t+b)
    offset_t = (dist[:, 2:] - dist[:, :2]) / 2.0         # (r-l, b-t)/2
    offset = torch.bmm(rot, offset_t.unsqueeze(2)).squeeze(2)
    ctr = points + offset
    return torch.cat([ctr, wh, norm_angle(angle, 'le90')], dim=-1)


def midpoint_offset_decode(rois, deltas, means=(0.,) * 6, stds=(1.,) * 6,
                           wh_ratio_clip=16.0 / 1000, version='le90'):
    """MidpointOffsetCoder delta2bbox: rois (N,4) HBB, deltas (N,6) -> obb (N,5)."""
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 6)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 6)
    denorm = deltas * stds + means
    dx, dy, dw, dh, da, db = (denorm[:, 0::6], denorm[:, 1::6], denorm[:, 2::6],
                              denorm[:, 3::6], denorm[:, 4::6], denorm[:, 5::6])
    max_ratio = abs(math.log(wh_ratio_clip))
    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    dh = dh.clamp(min=-max_ratio, max=max_ratio)
    px = ((rois[:, 0] + rois[:, 2]) * 0.5).unsqueeze(1).expand_as(dx)
    py = ((rois[:, 1] + rois[:, 3]) * 0.5).unsqueeze(1).expand_as(dy)
    pw = (rois[:, 2] - rois[:, 0]).unsqueeze(1).expand_as(dw)
    ph = (rois[:, 3] - rois[:, 1]).unsqueeze(1).expand_as(dh)
    gw = pw * dw.exp()
    gh = ph * dh.exp()
    gx = px + pw * dx
    gy = py + ph * dy
    da = da.clamp(min=-0.5, max=0.5)
    db = db.clamp(min=-0.5, max=0.5)
    ga = gx + da * gw
    _ga = gx - da * gw
    gb = gy + db * gh
    _gb = gy - db * gh
    polys = torch.stack([ga, gy - gh * 0.5, gx + gw * 0.5, gb, _ga, gy + gh * 0.5,
                         gx - gw * 0.5, _gb], dim=-1)
    gx2 = gx.expand_as(ga)
    gy2 = gy.expand_as(ga)
    center = torch.stack([gx2, gy2, gx2, gy2, gx2, gy2, gx2, gy2], dim=-1)
    cp = polys - center
    diag_len = torch.sqrt(cp[..., 0::2] ** 2 + cp[..., 1::2] ** 2)
    max_diag_len, _ = torch.max(diag_len, dim=-1, keepdim=True)
    scale = max_diag_len / diag_len
    cp = cp * scale.repeat_interleave(2, dim=-1)
    rectpolys = cp + center
    # poly2obb per-row (rectpolys: (N,nclass,8))
    nc = rectpolys.shape[1]
    flat = rectpolys.reshape(-1, 8)
    obb = poly2obb_le90(flat).reshape(deltas.shape[0], nc, 5).reshape(deltas.shape[0], -1)
    return obb


def delta_xywha_decode(rois, deltas, means=(0.,) * 5, stds=(0.1, 0.1, 0.2, 0.2, 0.1),
                       max_shape=None, wh_ratio_clip=16.0 / 1000,
                       add_ctr_clamp=False, ctr_clamp=32, angle_range='le90',
                       norm_factor=None, edge_swap=True, proj_xy=True):
    """DeltaXYWHAOBBoxCoder delta2bbox (decode). rois (N,5), deltas (N,5*nc)."""
    means = deltas.new_tensor(means).view(1, -1).repeat(1, deltas.size(1) // 5)
    stds = deltas.new_tensor(stds).view(1, -1).repeat(1, deltas.size(1) // 5)
    denorm = deltas * stds + means
    dx, dy, dw, dh, da = denorm[:, 0::5], denorm[:, 1::5], denorm[:, 2::5], denorm[:, 3::5], denorm[:, 4::5]
    if norm_factor:
        da = da * norm_factor * PI
    px = rois[:, 0].unsqueeze(1).expand_as(dx)
    py = rois[:, 1].unsqueeze(1).expand_as(dy)
    pw = rois[:, 2].unsqueeze(1).expand_as(dw)
    ph = rois[:, 3].unsqueeze(1).expand_as(dh)
    pa = rois[:, 4].unsqueeze(1).expand_as(da)
    dx_width = pw * dx
    dy_height = ph * dy
    max_ratio = abs(math.log(wh_ratio_clip))
    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    dh = dh.clamp(min=-max_ratio, max=max_ratio)
    gw = pw * dw.exp()
    gh = ph * dh.exp()
    if proj_xy:
        gx = dx * pw * torch.cos(pa) - dy * ph * torch.sin(pa) + px
        gy = dx * pw * torch.sin(pa) + dy * ph * torch.cos(pa) + py
    else:
        gx = px + dx_width
        gy = py + dy_height
    ga = norm_angle(pa + da, angle_range)
    if max_shape is not None:
        gx = gx.clamp(min=0, max=max_shape[1] - 1)
        gy = gy.clamp(min=0, max=max_shape[0] - 1)
    if edge_swap:
        w_reg = torch.where(gw > gh, gw, gh)
        h_reg = torch.where(gw > gh, gh, gw)
        th_reg = torch.where(gw > gh, ga, ga + PI / 2)
        th_reg = norm_angle(th_reg, angle_range)
        return torch.stack([gx, gy, w_reg, h_reg, th_reg], dim=-1).view_as(deltas)
    return torch.stack([gx, gy, gw, gh, ga], dim=-1).view(deltas.size())


# ----------------------------------------------------------------------------
# rotated IoU (vectorised convex-polygon intersection; matches mmcv semantics)
# ----------------------------------------------------------------------------
def _box2corners(b):
    """(N,5) [cx,cy,w,h,theta] -> (N,4,2), counter-clockwise."""
    cx, cy, w, h, a = b.unbind(-1)
    cos, sin = torch.cos(a), torch.sin(a)
    dx = torch.stack([w / 2, w / 2, -w / 2, -w / 2], -1)
    dy = torch.stack([-h / 2, h / 2, h / 2, -h / 2], -1)
    xs = cx[:, None] + dx * cos[:, None] - dy * sin[:, None]
    ys = cy[:, None] + dx * sin[:, None] + dy * cos[:, None]
    return torch.stack([xs, ys], -1)


def _ensure_ccw(poly):
    """poly (P,4,2) -> CCW order (reverse clockwise polys)."""
    a0, a1, a2, a3 = poly[:, 0], poly[:, 1], poly[:, 2], poly[:, 3]
    signed = ((a1[:, 0] - a0[:, 0]) * (a2[:, 1] - a0[:, 1])
              - (a1[:, 1] - a0[:, 1]) * (a2[:, 0] - a0[:, 0]))
    cw = signed < 0
    out = poly.clone()
    out[cw] = poly[cw][:, [3, 2, 1, 0]]
    return out


def _sh_clip(poly, count, a, b):
    """Sutherland-Hodgman: clip `poly` (P,K,2, count(P,)) by the half-plane left
    of edge a->b (inside = cross(b-a, p-a) >= 0). Returns out (P,K+1,2), oc (P,)."""
    P, K, _ = poly.shape
    dev = poly.device
    OK = max(K + 2, 9)
    out = torch.zeros(P, OK, 2, device=dev)
    oc = torch.zeros(P, dtype=torch.long, device=dev)
    edge = b - a
    rows = torch.arange(P, device=dev)
    flat = out.view(P, -1)
    for j in range(K):
        active = count > j
        if not active.any():
            continue
        cur = poly[:, j, :]
        if j == 0:
            prev = poly[rows, (count - 1).clamp(min=0), :]
        else:
            prev = poly[:, j - 1, :]
        cur_in = (edge[:, 0] * (cur[:, 1] - a[:, 1])
                  - edge[:, 1] * (cur[:, 0] - a[:, 0])) >= -1e-9
        prev_in = (edge[:, 0] * (prev[:, 1] - a[:, 1])
                   - edge[:, 1] * (prev[:, 0] - a[:, 0])) >= -1e-9
        r = cur - prev
        denom = r[:, 0] * edge[:, 1] - r[:, 1] * edge[:, 0]
        t = (edge[:, 1] * (a[:, 0] - prev[:, 0])
             - edge[:, 0] * (a[:, 1] - prev[:, 1])) / (denom + 1e-12)
        ipt = prev + t[:, None] * r
        # emit logic (standard SH)
        emit1 = active & cur_in & (~prev_in)       # crossing in -> intersection + cur
        emit2 = active & (~cur_in) & prev_in        # crossing out -> intersection
        emit_cur = active & cur_in & (~prev_in)
        # first vertex: intersection when entering
        m = active & (~prev_in) & cur_in
        idx = oc.clone()
        flat[rows[m], idx[m] * 2] = ipt[m, 0]
        flat[rows[m], idx[m] * 2 + 1] = ipt[m, 1]
        oc = oc + m.long()
        # cur when inside
        m = active & cur_in
        idx = oc.clone()
        flat[rows[m], idx[m] * 2] = cur[m, 0]
        flat[rows[m], idx[m] * 2 + 1] = cur[m, 1]
        oc = oc + m.long()
        # intersection when leaving
        m = active & prev_in & (~cur_in)
        idx = oc.clone()
        flat[rows[m], idx[m] * 2] = ipt[m, 0]
        flat[rows[m], idx[m] * 2 + 1] = ipt[m, 1]
        oc = oc + m.long()
    return out, oc


def _poly_area(poly, count):
    """Shoelace area of polygons poly(P,K,2) with per-row vertex count(P,).
    Vertices are stored contiguously in slots 0..count-1 (closed polygon)."""
    P, K, _ = poly.shape
    dev = poly.device
    ar = torch.arange(K, device=dev)[None, :]
    nxt_idx = torch.where(ar < (count[:, None] - 1), ar + 1,
                          torch.zeros_like(ar))           # last valid vertex -> 0
    nxt = torch.gather(poly, 1, nxt_idx[..., None].expand(-1, -1, 2))
    valid = ar < count[:, None]
    cross = poly[..., 0] * nxt[..., 1] - poly[..., 1] * nxt[..., 0]
    signed_sum = (cross * valid.float()).sum(1)
    area = signed_sum.abs() / 2.0
    return torch.where(count >= 3, area, torch.zeros_like(area))


def box_iou_rotated(boxes1, boxes2, mode='iou'):
    """IoU between two sets of le90 rbboxes. boxes1 (N,5), boxes2 (M,5) -> (N,M)."""
    boxes1 = boxes1[:, :5]
    boxes2 = boxes2[:, :5]
    N, M = boxes1.shape[0], boxes2.shape[0]
    if N == 0 or M == 0:
        return boxes1.new_zeros((N, M))
    P = N * M
    a = _box2corners(boxes1)[:, None].expand(N, M, 4, 2).reshape(P, 4, 2)
    b = _box2corners(boxes2)[None].expand(N, M, 4, 2).reshape(P, 4, 2)
    a = _ensure_ccw(a)
    b = _ensure_ccw(b)
    count = torch.full((P,), 4, dtype=torch.long, device=boxes1.device)
    poly = a
    for i in range(4):
        poly, count = _sh_clip(poly, count, b[:, i, :], b[:, (i + 1) % 4, :])
    inter = _poly_area(poly, count).reshape(N, M)
    area1 = (boxes1[:, 2] * boxes1[:, 3])[:, None]
    area2 = (boxes2[:, 2] * boxes2[:, 3])[None, :]
    if mode == 'iou':
        return inter / (area1 + area2 - inter + 1e-9)
    return inter


# ----------------------------------------------------------------------------
# NMS
# ----------------------------------------------------------------------------
def batched_nms(boxes, scores, idxs, iou_threshold):
    """Horizontal boxes (N,4); delegates to torchvision (same as mmcv.batched_nms)."""
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    return _tv_batched_nms(boxes, scores, idxs, iou_threshold)


def multiclass_nms_rotated(multi_bboxes, multi_scores, score_thr, nms_iou_thr,
                           max_num=-1, score_factors=None):
    """Rotated multiclass NMS (mmrotate semantics). Returns (dets(N,6), labels(N,)).

    multi_bboxes (N,5) or (N,nc*5); multi_scores (N,nc+1) [bg is last col].
    score_factors (N,) optional centerness factor (FCOS): multiplies scores for
    NMS/ranking while score_thr filtering uses the raw scores.
    """
    num_classes = multi_scores.size(1) - 1
    if multi_bboxes.shape[1] > 5:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 5)
    else:
        bboxes = multi_bboxes[:, None].expand(multi_scores.size(0), num_classes, 5)
    scores = multi_scores[:, :-1]
    out_b, out_s, out_l = [], [], []
    for c in range(num_classes):
        sc = scores[:, c]
        m = sc > score_thr
        if not m.any():
            continue
        bb = bboxes[m][:, c, :]
        ss = sc[m]
        if score_factors is not None:
            ss = ss * score_factors[m]
        keep = _nms_rotated(bb, ss, nms_iou_thr)
        out_b.append(bb[keep])
        out_s.append(ss[keep])
        out_l.append(ss.new_full((keep.numel(),), c, dtype=torch.long))
    if not out_b:
        z = multi_bboxes.new_zeros((0, 5))
        return torch.cat([z, multi_bboxes.new_zeros((0, 1))], 1), \
            multi_bboxes.new_zeros((0,), dtype=torch.long)
    bboxes = torch.cat(out_b)
    scores = torch.cat(out_s)
    labels = torch.cat(out_l)
    if max_num > 0 and scores.size(0) > max_num:
        _, idx = scores.sort(descending=True)
        idx = idx[:max_num]
        bboxes, scores, labels = bboxes[idx], scores[idx], labels[idx]
    return torch.cat([bboxes, scores[:, None]], 1), labels


def _nms_rotated(boxes, scores, iou_thr):
    """Greedy rotated NMS. boxes (N,5) le90, scores (N,). Returns keep indices."""
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    order = torch.argsort(scores, descending=True)
    boxes = boxes[order]
    n = boxes.shape[0]
    iou = box_iou_rotated(boxes, boxes)        # NxN, computed once
    keep = []
    suppressed = torch.zeros(n, dtype=torch.bool, device=boxes.device)
    for i in range(n):
        if suppressed[i]:
            continue
        keep.append(i)
        if i + 1 < n:
            suppressed[i + 1:] |= (iou[i, i + 1:] >= iou_thr)
    keep = torch.as_tensor(keep, dtype=torch.int64, device=boxes.device)
    return order[keep]


# ----------------------------------------------------------------------------
# VOC average precision
# ----------------------------------------------------------------------------
def average_precision(recalls, precisions, mode='11points'):
    no_scale = False
    if recalls.ndim == 1:
        no_scale = True
        recalls = recalls[np.newaxis, :]
        precisions = precisions[np.newaxis, :]
    num_scales = recalls.shape[0]
    ap = np.zeros(num_scales, dtype=np.float32)
    if mode == 'area':
        zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
        ones = np.ones((num_scales, 1), dtype=recalls.dtype)
        mrec = np.hstack((zeros, recalls, ones))
        mpre = np.hstack((zeros, precisions, zeros))
        for i in range(mpre.shape[1] - 1, 0, -1):
            mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
        for i in range(num_scales):
            ind = np.where(mrec[i, 1:] != mrec[i, :-1])[0]
            ap[i] = np.sum((mrec[i, ind + 1] - mrec[i, ind]) * mpre[i, ind + 1])
    elif mode == '11points':
        for i in range(num_scales):
            for thr in np.arange(0, 1 + 1e-3, 0.1):
                precs = precisions[i, recalls[i, :] >= thr]
                prec = precs.max() if precs.size > 0 else 0
                ap[i] += prec
        ap /= 11
    if no_scale:
        ap = ap[0]
    return ap
