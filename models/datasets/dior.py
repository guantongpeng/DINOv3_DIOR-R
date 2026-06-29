# Copyright (c) OpenMMLab. All rights reserved.
"""DIOR-R Dataset for mmrotate.

DIOR-R is a large-scale benchmark dataset for oriented object detection
in aerial/remote sensing images. It contains 23,463 images and 192,472
oriented bounding box instances across 20 object categories.

The annotations are in DOTA format (txt files with 8 corner coordinates
per line: x1 y1 x2 y2 x3 y3 x4 y4 category difficult).
"""

import glob
import os
import os.path as osp

import numpy as np
from mmrotate.core import poly2obb_np
from mmrotate.core.evaluation import eval_rbbox_map
from mmrotate.datasets.builder import ROTATED_DATASETS
from mmrotate.datasets.dota import DOTADataset


@ROTATED_DATASETS.register_module()
class DIORDataset(DOTADataset):
    """DIOR-R dataset for oriented object detection.

    DIOR-R contains 20 remote sensing object categories with oriented
    bounding box annotations in DOTA format.

    Args:
        ann_file (str): Path to annotation folder containing .txt files.
        pipeline (list[dict]): Processing pipeline.
        version (str, optional): Angle representations. Defaults to 'le90'.
        difficulty (int, optional): Difficulty threshold for filtering
            ground truth boxes. Boxes with difficulty > this value are
            ignored. Default: 100 (keep all).
        filter_empty_gt (bool): Whether to filter images without GT boxes.
            Default: True.
        img_ext (str): Image file extension. Default: '.jpg'.
    """

    CLASSES = (
        'airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
        'chimney', 'dam', 'Expressway-Service-area',
        'Expressway-toll-station', 'golffield', 'groundtrackfield', 'harbor',
        'overpass', 'ship', 'stadium', 'storagetank', 'tenniscourt',
        'trainstation', 'vehicle', 'windmill',
    )

    PALETTE = [
        (165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
        (138, 43, 226), (255, 128, 0), (255, 0, 255), (0, 255, 255),
        (255, 193, 193), (0, 51, 153), (255, 250, 205), (0, 139, 139),
        (255, 255, 0), (147, 116, 116), (0, 0, 255), (220, 20, 60),
        (128, 128, 0), (255, 215, 0), (128, 128, 128), (64, 224, 208),
    ]

    def __init__(self,
                 ann_file,
                 pipeline,
                 version='le90',
                 difficulty=100,
                 filter_empty_gt=True,
                 img_ext='.jpg',
                 **kwargs):
        self.img_ext = img_ext
        super().__init__(
            ann_file=ann_file,
            pipeline=pipeline,
            version=version,
            difficulty=difficulty,
            filter_empty_gt=filter_empty_gt,
            **kwargs,
        )

    def load_annotations(self, ann_folder):
        """Load annotations from DOTA-format txt files.

        Overrides DOTADataset.load_annotations to:
        1. Support custom image extensions (DIOR uses .jpg).
        2. Use DIOR-R 20-class mapping.

        Args:
            ann_folder (str): Folder containing DOTA format .txt files.

        Returns:
            list[dict]: List of data info dicts.
        """
        cls_map = {c: i for i, c in enumerate(self.CLASSES)}
        ann_files = glob.glob(osp.join(ann_folder, '*.txt'))
        data_infos = []

        if not ann_files:
            # Test phase: no annotation files, img_ids may be set externally
            self.img_ids = []
            return []

        for ann_file in ann_files:
            data_info = {}
            img_id = osp.splitext(osp.basename(ann_file))[0]
            img_name = img_id + self.img_ext
            data_info['filename'] = img_name
            data_info['ann'] = {}

            gt_bboxes = []
            gt_labels = []
            gt_polygons = []
            gt_bboxes_ignore = []
            gt_labels_ignore = []
            gt_polygons_ignore = []

            if osp.getsize(ann_file) == 0 and self.filter_empty_gt:
                continue

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
                        x, y, w, h, a = poly2obb_np(poly, self.version)
                    except Exception:
                        continue

                    cls_name = parts[8]
                    difficulty = int(parts[9])

                    if cls_name not in cls_map:
                        continue

                    label = cls_map[cls_name]

                    if difficulty > self.difficulty:
                        gt_bboxes_ignore.append([x, y, w, h, a])
                        gt_labels_ignore.append(label)
                        gt_polygons_ignore.append(poly)
                    else:
                        gt_bboxes.append([x, y, w, h, a])
                        gt_labels.append(label)
                        gt_polygons.append(poly)

            if gt_bboxes:
                data_info['ann']['bboxes'] = np.array(gt_bboxes, dtype=np.float32)
                data_info['ann']['labels'] = np.array(gt_labels, dtype=np.int64)
                data_info['ann']['polygons'] = np.array(gt_polygons, dtype=np.float32)
            else:
                data_info['ann']['bboxes'] = np.zeros((0, 5), dtype=np.float32)
                data_info['ann']['labels'] = np.array([], dtype=np.int64)
                data_info['ann']['polygons'] = np.zeros((0, 8), dtype=np.float32)

            if gt_bboxes_ignore:
                data_info['ann']['bboxes_ignore'] = np.array(
                    gt_bboxes_ignore, dtype=np.float32)
                data_info['ann']['labels_ignore'] = np.array(
                    gt_labels_ignore, dtype=np.int64)
                data_info['ann']['polygons_ignore'] = np.array(
                    gt_polygons_ignore, dtype=np.float32)
            else:
                data_info['ann']['bboxes_ignore'] = np.zeros(
                    (0, 5), dtype=np.float32)
                data_info['ann']['labels_ignore'] = np.array(
                    [], dtype=np.int64)
                data_info['ann']['polygons_ignore'] = np.zeros(
                    (0, 8), dtype=np.float32)

            data_infos.append(data_info)

        self.img_ids = [osp.splitext(info['filename'])[0]
                        for info in data_infos]
        return data_infos

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None,
                 nproc=4,
                 classwise=False,
                 **kwargs):
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        if metric not in ('mAP', 'mAP_multi', 'mAP_coco'):
            raise KeyError(f'metric {metric} is not supported')

        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = {}
        nproc = min(nproc, os.cpu_count() or 4)

        if metric == 'mAP':
            mean_ap, cls_results = eval_rbbox_map(
                results, annotations,
                scale_ranges=scale_ranges, iou_thr=iou_thr,
                dataset=self.CLASSES, logger=logger, nproc=nproc)
            eval_results[f'mAP@{iou_thr:.2f}'] = mean_ap
            if classwise:
                for i, cls_result in enumerate(cls_results):
                    eval_results[f'{self.CLASSES[i]}'] = float(cls_result['ap'])

        elif metric == 'mAP_multi':
            for thr in [0.5, 0.75]:
                mean_ap, _ = eval_rbbox_map(
                    results, annotations,
                    scale_ranges=scale_ranges, iou_thr=thr,
                    dataset=self.CLASSES, logger=None, nproc=nproc)
                eval_results[f'mAP@{thr:.2f}'] = mean_ap

        elif metric == 'mAP_coco':
            thresholds = np.arange(0.5, 1.0, 0.05)
            aps = []
            for thr in thresholds:
                mean_ap, _ = eval_rbbox_map(
                    results, annotations,
                    scale_ranges=scale_ranges, iou_thr=thr,
                    dataset=self.CLASSES, logger=None, nproc=nproc)
                aps.append(mean_ap)
                eval_results[f'mAP@{thr:.2f}'] = mean_ap
            eval_results['mAP@50:95'] = float(np.mean(aps))

        return eval_results
