"""Run pure-PyTorch inference on DIOR-R and dump per-image detection results.

Loads images itself, applies the native test preprocessing (dior_data), and runs
the pure OrientedRCNN detector (``simple_test``) -- no mmrotate/mm*/config
machinery at all. Results are pickled as ``{global_index: det_result}`` where
``det_result`` is the per-class list of ``[N, 6]`` arrays (cx, cy, w, h, theta,
score) in original-image coordinates. Use --shard/--num-shards to split across
GPUs (see run.sh).
"""

import argparse
import os
import os.path as osp
import pickle
import sys
import time

_HERE = osp.dirname(osp.abspath(__file__))            # .../inference/torch_inference
_INF = osp.dirname(_HERE)                              # .../inference
if _INF not in sys.path:
    sys.path.insert(0, _INF)

import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from torch_inference.data import dior_data  # noqa: E402
from torch_inference import model as model_lib  # noqa: E402

_ARGS = None


def parse_args():
    p = argparse.ArgumentParser(description='Pure-PyTorch OrientedRCNN inference on DIOR-R')
    p.add_argument('--config', default='ignored', help='kept for CLI compat (architecture is hard-coded)')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data-root', default='data/DIOR-R')
    p.add_argument('--split', default='test', choices=['test', 'val', 'train'])
    p.add_argument('--img-scale', type=int, nargs=2, default=[800, 800])
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--out-dir', default='inference/torch_inference/results')
    p.add_argument('--limit', type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def infer_one(model, img_bgr, device):
    tensor, meta = dior_data.preprocess(img_bgr, img_scale=tuple(_ARGS.img_scale))
    tensor = tensor.unsqueeze(0).to(device)
    det = model.simple_test(tensor, [meta], rescale=True)
    return det[0]


def main():
    global _ARGS
    _ARGS = args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    samples = dior_data.build_image_list(args.data_root, split=args.split)
    samples = [s for i, s in enumerate(samples) if i % args.num_shards == args.shard]
    if args.limit:
        samples = samples[:args.limit]
    print(f'[shard {args.shard}/{args.num_shards}] {len(samples)} images on {device}')

    model, classes = model_lib.build_model(args.checkpoint, num_classes=20, device=device)
    print(f'[shard {args.shard}] model ready ({len(classes)} classes)')

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = osp.join(args.out_dir, f'part_{args.shard:03d}_of_{args.num_shards:03d}.pkl')
    results = {}
    t0 = time.time()
    for j, (img_id, img_path, _ann) in enumerate(tqdm(samples, desc=f'shard {args.shard}')):
        img_bgr = dior_data.load_image(img_path)
        det = infer_one(model, img_bgr, device)
        results[args.shard + j * args.num_shards] = det
    with open(out_path, 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    dt = time.time() - t0
    print(f'[shard {args.shard}] wrote {out_path} | {len(results)} imgs | {dt:.1f}s '
          f'({len(results)/max(dt,1e-6):.2f} img/s)')


if __name__ == '__main__':
    main()
