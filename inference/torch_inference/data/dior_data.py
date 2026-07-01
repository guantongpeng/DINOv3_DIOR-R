"""Native DIOR-R data loading + test-time preprocessing (no mmrotate pipeline).

Everything here is plain cv2 / numpy / torch. The transforms reproduce the
mmrotate ``test_pipeline`` *bit-for-bit* so that model outputs -- and therefore
the mAP -- match ``tools/test.py``:

    LoadImageFromFile (BGR, uint8)
    RResize(img_scale=(800,800), keep_ratio=True)        # aspect-ratio preserved
    Normalize(mean, std, to_rgb=True)                    # BGR->RGB + standardize
    Pad(size_divisor=32, pad_val=0)                      # bottom/right pad
    DefaultFormatBundle                                  # HWC uint8/float -> CHW tensor

Ground-truth boxes are parsed from DOTA-style ``labelTxt`` files and converted to
``le90`` ``[cx, cy, w, h, theta]`` via the *same* ``poly2obb_np`` the DIORDataset
uses, so the evaluation annotations are identical.
"""

import glob
import os
import os.path as osp

import cv2
import numpy as np
import torch

from ..core.box_ops import poly2obb_np_le90 as _poly2obb_np_le90  # pure-torch/numpy (no mmrotate)

# DIOR-R 20 classes -- order must match the trained model head.
CLASSES = (
    'airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
    'chimney', 'dam', 'Expressway-Service-area', 'Expressway-toll-station',
    'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship', 'stadium',
    'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill',
)

CLS_MAP = {c: i for i, c in enumerate(CLASSES)}

# Defaults taken from the stage2 config (img_norm_cfg / test_pipeline).
DEFAULT_IMG_SCALE = (800, 800)
DEFAULT_MEAN = (123.675, 116.28, 103.53)
DEFAULT_STD = (58.395, 57.12, 57.375)
DEFAULT_PAD_DIVISOR = 32


def build_image_list(data_root, split='test', img_ext='.jpg',
                     filter_empty_gt=True, difficulty=100):
    """Return the ordered list of (img_path, gt_dict) for a split.

    Reproduces DIORDataset + DOTADataset filtering so the image set matches the
    mmrotate evaluation exactly:
      * annotation files of size 0 are skipped (DIORDataset.load_annotations);
      * images with no valid GT box after parsing are skipped (DOTADataset
        _filter_imgs, since filter_empty_gt=True).

    The returned order is deterministic (sorted by image id). mAP is
    order-invariant, so this is fine as long as predictions & GT stay aligned.
    """
    img_dir = osp.join(data_root, split, 'images')
    ann_dir = osp.join(data_root, split, 'labelTxt')
    ann_files = sorted(glob.glob(osp.join(ann_dir, '*.txt')))

    samples = []
    for ann_file in ann_files:
        if osp.getsize(ann_file) == 0 and filter_empty_gt:
            continue

        img_id = osp.splitext(osp.basename(ann_file))[0]
        img_path = osp.join(img_dir, img_id + img_ext)

        gt_bboxes, gt_labels = [], []
        gt_bboxes_ignore, gt_labels_ignore = [], []
        with open(ann_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 10:
                    continue
                poly = np.array(parts[:8], dtype=np.float32)
                try:
                    res = _poly2obb_np_le90(poly)
                    if res is None:
                        continue
                    x, y, w, h, a = res
                except Exception:
                    continue
                cls_name = parts[8]
                if cls_name not in CLS_MAP:
                    continue
                label = CLS_MAP[cls_name]
                try:
                    diff = int(parts[9])
                except (ValueError, IndexError):
                    diff = 0
                if diff > difficulty:
                    gt_bboxes_ignore.append([x, y, w, h, a])
                    gt_labels_ignore.append(label)
                else:
                    gt_bboxes.append([x, y, w, h, a])
                    gt_labels.append(label)

        if filter_empty_gt and len(gt_labels) == 0:
            # mirrors DOTADataset._filter_imgs (keep only if labels.size > 0)
            continue

        ann = {
            'bboxes': np.array(gt_bboxes, dtype=np.float32).reshape(-1, 5),
            'labels': np.array(gt_labels, dtype=np.int64),
            'bboxes_ignore': np.array(gt_bboxes_ignore, dtype=np.float32).reshape(-1, 5),
            'labels_ignore': np.array(gt_labels_ignore, dtype=np.int64),
        }
        samples.append((img_id, img_path, ann))

    return samples


def preprocess(img_bgr, img_scale=DEFAULT_IMG_SCALE, mean=DEFAULT_MEAN,
               std=DEFAULT_STD, pad_divisor=DEFAULT_PAD_DIVISOR):
    """Apply RResize(keep_ratio) -> Normalize(to_rgb) -> Pad(divisor).

    Returns ``(tensor[1,C,H,W], img_meta)`` where ``img_meta`` carries the exact
    fields the roi/rpn heads need to decode + rescale boxes back to the original
    image space:
        scale_factor = [w_scale, h_scale, w_scale, h_scale]
    """
    h, w = img_bgr.shape[:2]
    max_long = max(img_scale)
    max_short = min(img_scale)
    scale_factor = min(max_long / max(h, w), max_short / min(h, w))
    new_w = int(round(w * scale_factor))
    new_h = int(round(h * scale_factor))

    # RResize -- cv2 bilinear == mmcv 'bilinear' backend.
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Normalize(to_rgb=True): BGR->RGB then (x-mean)/std, float32.
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    normed = (resized[:, :, ::-1].astype(np.float32) - mean) / std

    # Pad(size_divisor): pad bottom/right to a multiple of the divisor with 0.
    pad_h = int(np.ceil(new_h / pad_divisor)) * pad_divisor
    pad_w = int(np.ceil(new_w / pad_divisor)) * pad_divisor
    padded = np.zeros((pad_h, pad_w, 3), dtype=np.float32)
    padded[:new_h, :new_w] = normed

    # DefaultFormatBundle: HWC -> CHW tensor.
    tensor = torch.from_numpy(np.ascontiguousarray(padded.transpose(2, 0, 1)))

    w_scale = new_w / w
    h_scale = new_h / h
    img_meta = {
        'filename': None,
        'ori_filename': None,
        'ori_shape': (h, w, 3),
        'img_shape': (new_h, new_w, 3),
        'pad_shape': (pad_h, pad_w, 3),
        'scale_factor': np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32),
        'flip': False,
        'flip_direction': None,
        'img_norm_cfg': dict(mean=[float(m) for m in mean],
                             std=[float(s) for s in std], to_rgb=True),
        'batch_input_shape': (pad_h, pad_w),
    }
    return tensor, img_meta


def load_image(path):
    """cv2 BGR uint8 read (same as LoadImageFromFile)."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f'Failed to read image: {path}')
    return img
