"""ONNX-runtime inference for the DINOv3 ViT-Adapter detectors on DIOR-R.

Supports all three heads via a meta sidecar (``<onnx>.meta.json`` written by
``export/convert.py``):
  * single-stage (FCOS / YOLO26): one ONNX, ``core/postprocess.py`` decodes + NMS.
  * two-stage (OrientedRCNN): stage-A ONNX + stage-B ONNX, ``core/oriented_rcnn.py``
    reuses the validated torch RPN/RoI glue.

Device modes: GPU (one shard per GPU process) or CPU.

Usage:
    CUDA_VISIBLE_DEVICES=0 python inference/onnx_inference/inference.py \\
        --onnx inference/onnx_inference/models/model.onnx --shard 0 --num-shards 1
    DEVICE=cpu ... --device cpu
"""

import argparse
import json
import os
import os.path as osp
import pickle
import sys
import time

import numpy as np
import torch

_HERE = osp.dirname(osp.abspath(__file__))            # .../inference/onnx_inference
_INF = osp.dirname(_HERE)                              # .../inference  (torch_inference lives here)
if _INF not in sys.path:
    sys.path.insert(0, _INF)

import onnxruntime as ort  # noqa: E402
from tqdm import tqdm  # noqa: E402

from torch_inference.data import dior_data  # noqa: E402  (preprocessing + GT)
from core import postprocess  # noqa: E402


def build_session(onnx_path, device):
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(os.environ.get('OMP_NUM_THREADS', 0)) or so.intra_op_num_threads
    providers = (['CUDAExecutionProvider', 'CPUExecutionProvider']
                 if device == 'gpu' else ['CPUExecutionProvider'])
    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
    return sess, sess.get_inputs()[0].name, [o.name for o in sess.get_outputs()]


def load_meta(onnx_path):
    meta_path = osp.splitext(onnx_path)[0] + '.meta.json'
    with open(meta_path) as f:
        return json.load(f)


class SingleStageRunner:
    def __init__(self, onnx_path, device, meta):
        self.sess, self.in_name, self.out_names = build_session(onnx_path, device)
        self.meta = meta
        self.dev = torch.device('cuda' if device == 'gpu' and torch.cuda.is_available() else 'cpu')

    def run(self, img_bgr):
        tensor, m = dior_data.preprocess(img_bgr)
        inp = tensor.unsqueeze(0).numpy().astype(np.float32)
        outs = self.sess.run(self.out_names, {self.in_name: inp})
        out_dict = {n: torch.from_numpy(o).to(self.dev) for n, o in zip(self.out_names, outs)}
        cfg = dict(self.meta['test_cfg'], fourth_key=self.meta.get('fourth_key', 'ctr'),
                   num_classes=self.meta['num_classes'])
        return postprocess.postprocess_single(out_dict, m, self.meta['detector'], cfg)


class OrientedRCNNRunner:
    def __init__(self, onnx_path, device, meta):
        base = osp.dirname(onnx_path)
        a_path = osp.join(base, meta['files']['stage_a'])
        b_path = osp.join(base, meta['files']['stage_b'])
        self.sess_a, self.in_a, self.out_a = build_session(a_path, device)
        self.sess_b, self.in_b, self.out_b = build_session(b_path, device)
        from core import oriented_rcnn
        self.dev = torch.device('cuda' if device == 'gpu' and torch.cuda.is_available() else 'cpu')
        self.rpn, self.roi_head = oriented_rcnn.make_glue(
            meta['num_classes'], 'cuda' if self.dev.type == 'cuda' else 'cpu')
        self.cfg = meta

    def run(self, img_bgr):
        tensor, m = dior_data.preprocess(img_bgr)
        inp = tensor.unsqueeze(0).numpy().astype(np.float32)
        from core import oriented_rcnn
        return oriented_rcnn.run_image(self.sess_a, self.sess_b, self.out_a,
                                       self.out_b, inp, m, self.cfg,
                                       self.rpn, self.roi_head, self.dev)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--onnx', required=True)
    p.add_argument('--data-root', default='data/DIOR-R')
    p.add_argument('--split', default='test', choices=['test', 'val', 'train'])
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--device', default='auto', choices=['auto', 'gpu', 'cpu'])
    p.add_argument('--out-dir', default='inference/onnx_inference/results')
    p.add_argument('--limit', type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    if args.device == 'auto':
        args.device = 'gpu' if torch.cuda.is_available() else 'cpu'

    meta = load_meta(args.onnx)
    detector = meta['detector']
    print(f'[meta] detector={detector} num_classes={meta["num_classes"]} '
          f'files={meta["files"]}')

    samples = dior_data.build_image_list(args.data_root, split=args.split)
    samples = [s for i, s in enumerate(samples) if i % args.num_shards == args.shard]
    if args.limit:
        samples = samples[:args.limit]
    print(f'[shard {args.shard}/{args.num_shards}] {len(samples)} images on {args.device}')

    if detector == 'oriented_rcnn':
        runner = OrientedRCNNRunner(args.onnx, args.device, meta)
    else:
        runner = SingleStageRunner(args.onnx, args.device, meta)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = osp.join(args.out_dir, f'part_{args.shard:03d}_of_{args.num_shards:03d}.pkl')
    results = {}
    t0 = time.time()
    for j, (img_id, img_path, _ann) in enumerate(tqdm(samples, desc=f'shard {args.shard}')):
        img_bgr = dior_data.load_image(img_path)
        results[args.shard + j * args.num_shards] = runner.run(img_bgr)
    with open(out_path, 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    dt = time.time() - t0
    print(f'[shard {args.shard}] wrote {out_path} | {len(results)} imgs | {dt:.1f}s '
          f'({len(results)/max(dt,1e-6):.2f} img/s)')


if __name__ == '__main__':
    main()
