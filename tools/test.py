#!/usr/bin/env python3
"""
Evaluation script for Oriented R-CNN with DINOv3 backbone on DIOR-R.

Performs evaluation using mAP (mean Average Precision) metric with
oriented bounding box IoU calculation.

Usage:
    # Evaluate on test set
    python tools/test.py configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_dior.py \
        work_dirs/oriented_rcnn_dinov3_fpn_dior/epoch_36.pth \
        --eval mAP

    # Evaluate with multi-scale testing
    python tools/test.py configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_dior.py \
        work_dirs/oriented_rcnn_dinov3_fpn_dior/epoch_36.pth \
        --eval mAP --aug-test

    # Save detection results
    python tools/test.py configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_dior.py \
        work_dirs/oriented_rcnn_dinov3_fpn_dior/epoch_36.pth \
        --out results.pkl --eval mAP

    # Visualize predictions
    python tools/test.py configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_dior.py \
        work_dirs/oriented_rcnn_dinov3_fpn_dior/epoch_36.pth \
        --show --show-dir vis_results

    # Multi-GPU evaluation
    bash tools/dist_test.sh configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_dior.py \
        work_dirs/oriented_rcnn_dinov3_fpn_dior/epoch_36.pth 4 --eval mAP

Environment:
    source /home/guantp/pro/olmoearth_pretrain/.venv/bin/activate
"""

import argparse
import os
import os.path as osp
import sys
import time

# Add project root to Python path for custom imports
_proj_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import mmcv
import mmrotate
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (
    get_dist_info,
    init_dist,
    load_checkpoint,
    wrap_fp16_model,
)

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (
    build_dataloader,
    build_dataset,
    replace_ImageToTensor,
)
from mmdet.models import build_detector
from mmdet.utils import setup_multi_processes

from mmrotate import __version__ as mmrotate_version


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate Oriented R-CNN with DINOv3 on DIOR-R'
    )
    parser.add_argument(
        'config',
        help='Path to test config file',
    )
    parser.add_argument(
        'checkpoint',
        help='Path to checkpoint file',
    )
    parser.add_argument(
        '--out',
        help='Output result file (pickle format)',
    )
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Fuse conv and bn layers for faster inference',
    )
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        default=['mAP'],
        choices=['mAP', 'recall', 'bbox'],
        help='Evaluation metrics',
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Show detection results',
    )
    parser.add_argument(
        '--show-dir',
        help='Directory to save visualized results',
    )
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='Score threshold for visualization',
    )
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='Collect results from all GPUs',
    )
    parser.add_argument(
        '--tmpdir',
        help='Temporary directory for collecting results',
    )
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override config settings',
    )
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='Custom evaluation options',
    )
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='Job launcher for distributed testing',
    )
    parser.add_argument(
        '--local_rank',
        type=int,
        default=0,
        help='Local rank for distributed testing',
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Validate checkpoint
    assert args.checkpoint.endswith('.pth'), \
        f'Checkpoint must be a .pth file, got {args.checkpoint}'
    assert osp.exists(args.checkpoint), \
        f'Checkpoint not found: {args.checkpoint}'

    # Load config
    cfg = Config.fromfile(args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Set cudnn benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # Update config based on args
    cfg.model.train_cfg = None

    # Initialize distributed
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # Create work directory
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    # Build dataloader
    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
    if samples_per_gpu > 1:
        # Replace 'ImageToTensor' with 'DefaultFormatBundle'
        cfg.data.test.pipeline = replace_ImageToTensor(
            cfg.data.test.pipeline
        )

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
    )

    # Build model
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))

    # Fuse conv+bn for faster inference
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    # Load checkpoint
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    # Handle legacy checkpoints
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    # Old versions saved class names in checkpoints
    if 'class_names' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['class_names']

    # Wrap model
    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )

    # Run evaluation
    outputs = single_gpu_test(
        model, data_loader, args.show, args.show_dir, args.show_score_thr
    )

    # Save results
    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nSaving results to {args.out}')
            mmcv.dump(outputs, args.out)

        # Evaluate
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # Use 'bbox' for compatibility
            for key in [
                'interval', 'tmpdir', 'start', 'gpu_collect',
                'save_best', 'rule',
            ]:
                eval_kwargs.pop(key, None)

            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            metric_results = dataset.evaluate(outputs, **eval_kwargs)

            # Print results
            print('\n' + '=' * 60)
            print('Evaluation Results')
            print('=' * 60)
            for metric_name, value in metric_results.items():
                print(f'  {metric_name}: {value:.4f}')
            print('=' * 60)


if __name__ == '__main__':
    main()
