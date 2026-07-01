"""Gather sharded ONNX-inference results and evaluate DOTA-style rotated mAP.

Reuses the pure-PyTorch ``eval_rbbox_map`` from ``inference/torch_inference`` so
the metric matches the mmrotate / torch reference exactly. Report the overall
mAP@0.50 plus per-class AP.

Usage:
    python inference/onnx_inference/evaluate.py --results-dir inference/onnx_inference/results \\
        --num-shards 8 --iou-thr 0.5
"""

import argparse
import glob
import os
import os.path as osp
import pickle
import sys

PROJ = osp.dirname(osp.dirname(osp.abspath(__file__)))   # .../inference  (torch_inference lives here)
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from torch_inference.data import dior_data  # noqa: E402
from torch_inference.metrics.eval_map import eval_rbbox_map  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--results-dir', default='inference/onnx_inference/results')
    p.add_argument('--num-shards', type=int, required=True)
    p.add_argument('--data-root', default='data/DIOR-R')
    p.add_argument('--split', default='test', choices=['test', 'val', 'train'])
    p.add_argument('--iou-thr', type=float, default=0.5)
    p.add_argument('--out', default=None)
    return p.parse_args()


def gather(results_dir, num_shards, n_expected):
    by_index = {}
    for shard in range(num_shards):
        pattern = osp.join(results_dir, f'part_{shard:03d}_of_{num_shards:03d}.pkl')
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f'Missing shard results: {pattern}')
        with open(matches[0], 'rb') as f:
            by_index.update(pickle.load(f))
    missing = [i for i in range(n_expected) if i not in by_index]
    if missing:
        raise RuntimeError(f'{len(missing)} images missing predictions, e.g. {missing[:5]}')
    return [by_index[i] for i in range(n_expected)]


def main():
    args = parse_args()
    samples = dior_data.build_image_list(args.data_root, split=args.split)
    n = len(samples)
    print(f'Dataset ({args.split}): {n} images')
    det_results = gather(args.results_dir, args.num_shards, n)
    annotations = [s[2] for s in samples]

    mean_ap, results = eval_rbbox_map(
        det_results, annotations, iou_thr=args.iou_thr, use_07_metric=True,
        classes=list(dior_data.CLASSES))
    lines = ['=' * 60, f'mAP@{args.iou_thr:.2f} = {mean_ap:.4f}', '-' * 60]
    for name, r in zip(dior_data.CLASSES, results):
        lines.append(f'  {name:<30s} gts={r["num_gts"]:>5d} dets={r["num_dets"]:>6d} AP={r["ap"]:.4f}')
    lines += ['-' * 60, f'mAP@{args.iou_thr:.2f} (mean) = {mean_ap:.4f}', '=' * 60]
    report = '\n'.join(lines)
    print(report)
    if args.out:
        with open(args.out, 'w') as f:
            f.write(report + '\n')
        print(f'Saved metrics to {args.out}')


if __name__ == '__main__':
    main()
