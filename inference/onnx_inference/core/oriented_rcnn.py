"""Two-stage OrientedRCNN ONNX inference (DINOv3 ViT-Adapter backbone).

The detector is split into two ONNX graphs:

  * **stage A** (``model.onnx``): backbone + FPN + RPN heads -> 4 FPN feature
    maps + 5 RPN cls maps + 5 RPN reg maps.
  * **stage B** (``model_head.onnx``): RoI bbox head (2x FC -> cls + reg),
    with a dynamic number of RoIs.

Everything that cannot go into ONNX (anchor grid, RPN decode + horizontal NMS,
rotated RoIAlign, RoI decode + rotated NMS) is reused verbatim from
``inference/torch_inference`` (rpn.py / roi.py / box_ops.py) -- the same
pure-PyTorch path the reference uses, so results match.
"""

import os.path as osp
import sys

import numpy as np
import torch

_HERE = osp.dirname(osp.abspath(__file__))            # .../inference/onnx_inference/core
_INF = osp.dirname(osp.dirname(_HERE))                # .../inference  (torch_inference lives here)
if _INF not in sys.path:
    sys.path.insert(0, _INF)

from torch_inference.core import box_ops  # noqa: E402
from . import postprocess  # noqa: E402  (sibling within the core package)
from torch_inference.core.rpn import OrientedRPNHead  # noqa: E402
from torch_inference.core.roi import (RotatedSingleRoIExtractor, RotatedShared2FCBBoxHead,  # noqa: E402
                                      OrientedStandardRoIHead)

try:  # mmcv's exact CUDA rotated RoIAlign (matches the reference); torch fallback
    from mmcv.ops import roi_align_rotated as _mmcv_roi_align
    _HAS_MMCV_ROI = True
except Exception:  # pragma: no cover
    _HAS_MMCV_ROI = False


def extract_roi_feats(feats, rois, featmap_strides=(4, 8, 16, 32),
                      out_size=7, finest_scale=56):
    """Multi-level rotated RoIAlign. feats: list of (1,C,H,W); rois: (n,6)."""
    n = rois.shape[0]
    num_levels = len(feats)
    if n == 0:
        return feats[0].new_zeros(0, feats[0].shape[1], out_size, out_size)
    if _HAS_MMCV_ROI:
        scale = torch.sqrt(rois[:, 3] * rois[:, 4])
        lvls = torch.floor(torch.log2(scale / finest_scale + 1e-6)).clamp(0, num_levels - 1).long()
        out = feats[0].new_zeros(n, feats[0].shape[1], out_size, out_size)
        for i in range(num_levels):
            inds = (lvls == i).nonzero(as_tuple=False).squeeze(1)
            if inds.numel() > 0:
                out[inds] = _mmcv_roi_align(feats[i], rois[inds], (out_size, out_size),
                                            1.0 / featmap_strides[i], 2, False, True)
        return out
    # pure-torch fallback (torch_inference)
    extractor = RotatedSingleRoIExtractor(out_channels=feats[0].shape[1],
                                          featmap_strides=featmap_strides,
                                          out_size=out_size, finest_scale=finest_scale)
    return extractor(feats, rois)


def make_glue(num_classes, device='cpu'):
    """Build the weight-free helper objects (anchors, decoders, RoIAlign)."""
    dev = torch.device(device)
    rpn = OrientedRPNHead(in_channels=256, feat_channels=256, num_anchors=3,
                          scales=(8.0,), ratios=(0.5, 1.0, 2.0),
                          strides=(4, 8, 16, 32, 64))
    rpn.base_anchors = [b.to(dev) for b in rpn.base_anchors]
    extractor = RotatedSingleRoIExtractor(
        out_channels=256, featmap_strides=(4, 8, 16, 32),
        out_size=7, finest_scale=56)
    bbox_head = RotatedShared2FCBBoxHead(
        in_channels=256, fc_out_channels=1024, roi_feat_size=7,
        num_classes=num_classes, reg_class_agnostic=True)
    roi_head = OrientedStandardRoIHead(bbox_roi_extractor=extractor,
                                       bbox_head=bbox_head)
    return rpn, roi_head


@torch.no_grad()
def run_image(sess_a, sess_b, names_a, names_b, img_np, meta, cfg, rpn, roi_head,
              dev='cpu'):
    """img_np: (1,3,800,800) float32 numpy. Returns per-class list of (n,6).
    dev: torch device for the post-processing tensors (cuda when on GPU)."""
    dev = torch.device(dev)
    a_out = sess_a.run(names_a, {'input': img_np})
    fpn = [torch.from_numpy(a_out[i]).to(dev) for i in range(4)]         # 4x (1,256,H,W)
    rpn_cls = [torch.from_numpy(a_out[4 + l])[0].to(dev) for l in range(5)]  # (A,H,W)
    rpn_reg = [torch.from_numpy(a_out[9 + l])[0].to(dev) for l in range(5)]  # (A*6,H,W)

    rpn_cfg = cfg['test_cfg']['rpn']
    proposals = rpn.get_bboxes_single(
        rpn_cls, rpn_reg, meta['img_shape'],
        nms_pre=rpn_cfg['nms_pre'], nms_iou=rpn_cfg['nms_iou'],
        max_per_img=rpn_cfg['max_per_img'], min_bbox_size=0,
        device=rpn_cls[0].device)
    proposals = proposals[:, :5]                                         # (n,5)

    rois = box_ops.rbbox2roi([proposals])                               # (n,6)
    roi_feats = extract_roi_feats(fpn, rois)                           # (n,256,7,7)

    cls_np, reg_np = sess_b.run(names_b, {'roi_feats': roi_feats.cpu().numpy()})
    cls_score = torch.from_numpy(cls_np).to(dev)                        # (n,21)
    bbox_pred = torch.from_numpy(reg_np).to(dev)                        # (n,5)

    rcnn = cfg['test_cfg']['rcnn']
    # RoI decode + NMS (replicated from OrientedStandardRoIHead._get_bboxes_single
    # but using the memory-efficient mmcv rotated NMS instead of the greedy O(N^2)
    # fallback, which spikes memory under GPU contention).
    import torch.nn.functional as F
    scores = F.softmax(cls_score, dim=-1)                       # (n, nc+1)
    bboxes = box_ops.delta_xywha_decode(
        rois[:, 1:6], bbox_pred, means=(0.,) * 5,
        stds=(0.1, 0.1, 0.2, 0.2, 0.1), max_shape=meta['img_shape'],
        edge_swap=True, proj_xy=True)
    if bboxes.size(0) > 0:
        sf = scores.new_tensor(meta['scale_factor'])
        bboxes = bboxes.view(bboxes.size(0), -1, 5)
        bboxes[..., :4] = bboxes[..., :4] / sf
        bboxes = bboxes.view(bboxes.size(0), -1)
    dets, labels = postprocess.multiclass_nms_rotated(
        bboxes, scores, rcnn['score_thr'], rcnn['nms_iou'],
        rcnn['max_per_img'])
    return box_ops.rbbox2result(dets, labels, cfg['num_classes'])
