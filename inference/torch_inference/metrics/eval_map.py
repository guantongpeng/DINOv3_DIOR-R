"""Pure-PyTorch DOTA-style rotated mAP (replaces mmrotate.eval_rbbox_map).

Mirrors ``mmrotate/core/evaluation/eval_map.eval_rbbox_map``:
  * tpfp_default uses rotated IoU (our box_iou_rotated -> matches mmcv box_iou_rotated)
  * VOC '11points' average precision (use_07_metric=True, as DIORDataset uses)
"""

import numpy as np
import torch

from ..core.box_ops import average_precision, box_iou_rotated


def _tpfp_default(det_bboxes, gt_bboxes, gt_bboxes_ignore, iou_thr, area_ranges):
    num_dets = det_bboxes.shape[0]
    num_gts = gt_bboxes.shape[0]
    num_scales = len(area_ranges)
    tp = np.zeros((num_scales, num_dets), dtype=np.float32)
    fp = np.zeros((num_scales, num_dets), dtype=np.float32)
    if num_gts == 0:
        if area_ranges == [(None, None)]:
            fp[...] = 1
        return tp, fp
    ious = box_iou_rotated(
        torch.from_numpy(det_bboxes).float(),
        torch.from_numpy(gt_bboxes).float()).numpy()
    ious_max = ious.max(axis=1)
    ious_argmax = ious.argmax(axis=1)
    sort_inds = np.argsort(-det_bboxes[:, -1])
    gt_ignore_inds = np.zeros(num_gts, dtype=bool)   # cls_gts already excludes ignored boxes
    for k, (min_area, max_area) in enumerate(area_ranges):
        gt_covered = np.zeros(num_gts, dtype=bool)
        if min_area is None:
            gt_area_ignore = np.zeros(num_gts, dtype=bool)
        else:
            raise NotImplementedError
        for i in sort_inds:
            if ious_max[i] >= iou_thr:
                m = ious_argmax[i]
                if not (gt_ignore_inds[m] or gt_area_ignore[m]):
                    if not gt_covered[m]:
                        gt_covered[m] = True
                        tp[k, i] = 1
                    else:
                        fp[k, i] = 1
            elif min_area is None:
                fp[k, i] = 1
    return tp, fp


def _get_cls_results(det_results, annotations, class_id):
    cls_dets = [img_res[class_id] for img_res in det_results]
    cls_gts, cls_gts_ignore = [], []
    for ann in annotations:
        gt_inds = ann['labels'] == class_id
        cls_gts.append(ann['bboxes'][gt_inds, :])
        if ann.get('labels_ignore', None) is not None and len(ann['labels_ignore']):
            ig = ann['labels_ignore'] == class_id
            cls_gts_ignore.append(ann['bboxes_ignore'][ig, :])
        else:
            cls_gts_ignore.append(np.zeros((0, 5), dtype=np.float32))
    return cls_dets, cls_gts, cls_gts_ignore


def eval_rbbox_map(det_results, annotations, iou_thr=0.5, use_07_metric=True,
                   classes=None, nproc=4):
    """det_results: list per image of [per-class [N,6] arrays].
    annotations: list per image of dict(bboxes,labels,bboxes_ignore,labels_ignore).
    Returns (mean_ap, [per-class dict])."""
    num_imgs = len(det_results)
    num_classes = len(det_results[0])
    area_ranges = [(None, None)]
    eval_results = []
    for c in range(num_classes):
        cls_dets, cls_gts, cls_gts_ignore = _get_cls_results(det_results, annotations, c)
        tpfp = [_tpfp_default(cls_dets[i], cls_gts[i], cls_gts_ignore[i], iou_thr, area_ranges)
                for i in range(num_imgs)]
        tp = np.hstack([t[0] for t in tpfp])
        fp = np.hstack([t[1] for t in tpfp])
        num_gts = 0
        for g in cls_gts:
            num_gts += g.shape[0]
        if len(cls_dets) == 0 or np.vstack(cls_dets).size == 0:
            cls_dets_cat = np.zeros((0, 6), dtype=np.float32)
            num_dets = 0
        else:
            cls_dets_cat = np.vstack(cls_dets)
            num_dets = cls_dets_cat.shape[0]
        sort_inds = np.argsort(-cls_dets_cat[:, -1]) if num_dets else np.array([], dtype=int)
        tp = tp[:, sort_inds] if num_dets else tp
        fp = fp[:, sort_inds] if num_dets else fp
        tp = np.cumsum(tp, axis=1)
        fp = np.cumsum(fp, axis=1)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts, eps)
        precisions = tp / np.maximum((tp + fp), eps)
        recalls = recalls[0, :]
        precisions = precisions[0, :]
        ap = average_precision(recalls, precisions, mode='11points' if use_07_metric else 'area')
        eval_results.append({'num_gts': num_gts, 'num_dets': num_dets,
                             'recall': recalls, 'precision': precisions, 'ap': float(ap)})
    aps = [r['ap'] for r in eval_results if r['num_gts'] > 0]
    mean_ap = float(np.mean(aps)) if aps else 0.0
    return mean_ap, eval_results
