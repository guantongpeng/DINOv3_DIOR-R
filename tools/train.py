#!/usr/bin/env python3
"""
Training script for Oriented R-CNN with DINOv3 backbone on DIOR-R.

This script handles:
    - Distributed training (multi-GPU)
    - Mixed precision training (fp16)
    - Checkpoint saving and resuming
    - Logging and visualization

Usage:
    # Single GPU training
    python tools/train.py configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_dior.py

    # Multi-GPU training (4 GPUs)
    bash tools/dist_train.sh configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_dior.py 4

    # Resume from checkpoint
    python tools/train.py configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_dior.py \
        --resume-from work_dirs/oriented_rcnn_dinov3_fpn_dior/latest.pth

    # Specify work directory
    python tools/train.py configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_dior.py \
        --work-dir work_dirs/my_experiment

Environment:
    source /home/guantp/pro/olmoearth_pretrain/.venv/bin/activate
"""

import argparse
import copy
import os
os.environ.setdefault('OPENCV_LOG_LEVEL', 'ERROR')
os.environ.setdefault('OPENCV_IO_LOG_LEVEL', 'ERROR')
import os.path as osp
import sys
import time
import warnings

# Add project root to Python path for custom imports
_proj_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import mmcv
import mmrotate
import torch

import cv2
try:
    cv2.setLogLevel(0)
except AttributeError:
    pass


# Monkey-patch: fix mmcv 1.x compatibility with PyTorch 2.7+
# PyTorch 2.7 _get_stream expects torch.device, but mmcv 1.x passes raw ints.
# mmcv caches its own import reference, so we must patch both locations.
def _install_get_stream_patch():
    import torch.nn.parallel._functions as _torch_fns
    _orig = _torch_fns._get_stream
    def _patched(device):
        if isinstance(device, int):
            device = torch.device('cuda', device)
        return _orig(device)
    _torch_fns._get_stream = _patched
    # Also patch mmcv's cached reference
    try:
        import mmcv.parallel._functions as _mmcv_fns
        _mmcv_fns._get_stream = _patched
    except Exception:
        pass

_install_get_stream_patch()


# Monkey-patch: fix mmcv MMDistributedDataParallel compatibility with PyTorch 2.7+
# PyTorch 2.7's DDP.forward() calls _run_ddp_forward(), which accesses
# _use_replicated_tensor_module (new PyTorch attribute that mmcv doesn't set).
def _install_mmddp_patch():
    from mmcv.parallel import MMDistributedDataParallel
    if not hasattr(MMDistributedDataParallel, '_use_replicated_tensor_module'):
        MMDistributedDataParallel._use_replicated_tensor_module = False

_install_mmddp_patch()

from mmcv import Config, DictAction
from mmcv.runner import (
    HOOKS,
    DistSamplerSeedHook,
    EpochBasedRunner,
    Fp16OptimizerHook,
    OptimizerHook,
    build_optimizer,
    build_runner,
    get_dist_info,
    init_dist,
    set_random_seed,
)
from mmcv.utils import build_from_cfg, get_git_hash

from mmdet import __version__ as mmdet_version
from mmdet.apis import train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import (
    collect_env,
    get_root_logger,
    setup_multi_processes,
)

from mmrotate import __version__ as mmrotate_version


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train Oriented R-CNN with DINOv3 on DIOR-R'
    )
    parser.add_argument(
        'config',
        help='Path to training config file',
    )
    parser.add_argument(
        '--work-dir',
        help='Directory to save logs and checkpoints',
    )
    parser.add_argument(
        '--resume-from',
        help='Resume from checkpoint',
    )
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='Auto resume from latest checkpoint',
    )
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Disable evaluation during training',
    )
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='Number of GPUs to use',
    )
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        default=None,
        help='Specific GPU IDs to use',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed',
    )
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='Enable deterministic mode',
    )
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override config settings (key=value pairs)',
    )
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='Job launcher for distributed training',
    )
    parser.add_argument(
        '--local_rank',
        type=int,
        default=0,
        help='Local rank for distributed training (legacy)'
    )
    parser.add_argument(
        '--local-rank',
        dest='local_rank',          # 将值存储到 args.local_rank
        type=int,
        default=0,
        help='Local rank for distributed training (new torch.distributed.launch)'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    cfg = Config.fromfile(args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Set cudnn benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # Set work directory with timestamp suffix to avoid overwriting
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        run_ts = time.strftime('%Y%m%d_%H%M%S')
        cfg.work_dir = osp.join(
            './work_dirs',
            f"{osp.splitext(osp.basename(args.config))[0]}_{run_ts}",
        )

    # Auto resume
    if args.auto_resume:
        cfg.resume_from = osp.join(cfg.work_dir, 'latest.pth')
    elif args.resume_from is not None:
        cfg.resume_from = args.resume_from

    # Set GPU IDs
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(args.gpus)

    # Initialize distributed training
    if args.launcher == 'none':
        distributed = False
        if args.gpus > 1:
            warnings.warn(
                f'Using {args.gpus} GPUs without distributed mode. '
                f'Use --launcher pytorch for multi-GPU training.'
            )
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # Re-set gpu_ids after init_dist
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # Create work directory
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    # Dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))

    # Set up logging
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # Initialize wandb logger if available
    if cfg.get('wandb', False):
        try:
            import wandb
            wandb_cfg = cfg.wandb if isinstance(cfg.wandb, dict) else {}
            wandb.init(
                project=wandb_cfg.get('project', 'mmrotate-dino'),
                name=wandb_cfg.get('name', osp.basename(cfg.work_dir)),
                config=cfg._cfg_dict,
                dir=cfg.work_dir,
            )
            logger.info('Wandb logging enabled')
        except ImportError:
            logger.warning('wandb not installed, skipping wandb logging')

    # Log environment info
    meta = {}
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n'
                + dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    meta['mmdet_version'] = mmdet_version
    meta['mmrotate_version'] = mmrotate_version

    # Log config
    logger.info(f'Config:\n{cfg.pretty_text}')

    # Set random seed
    set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed

    # Set up multi-process
    setup_multi_processes(cfg)

    # Build model
    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'),
    )
    model.init_weights()

    # Log model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    logger.info(
        f'Model Statistics:\n'
        f'  Total parameters: {total_params / 1e6:.2f}M\n'
        f'  Trainable parameters: {trainable_params / 1e6:.2f}M\n'
        f'  Frozen parameters: {(total_params - trainable_params) / 1e6:.2f}M'
    )

    # Build datasets
    datasets = [build_dataset(cfg.data.train)]

    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))

    if cfg.checkpoint_config is not None:
        # Save mmdet/mmrotate version and config info in checkpoint metadata
        cfg.checkpoint_config.meta = dict(
            mmdet_version=mmdet_version,
            mmrotate_version=mmrotate_version,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
        )

    # Add output attribute to model for compatibility
    model.CLASSES = datasets[0].CLASSES

    if distributed:
        # Enable find_unused_parameters only when necessary (frozen backbone
        # or progressive O2O training create params outside the loss graph).
        # Otherwise skip it to avoid the ~10-20% DDP overhead.
        frozen_stages = cfg.model.get('backbone', {}).get('frozen_stages', -1)
        progressive_cfg = cfg.model.get('train_cfg', {}).get('progressive_loss', None)
        if frozen_stages >= 0 or progressive_cfg is not None:
            cfg.find_unused_parameters = True
            logger.info(
                'find_unused_parameters=True (frozen_stages=%s, progressive_loss=%s)',
                frozen_stages, progressive_cfg is not None,
            )
        else:
            cfg.find_unused_parameters = False
            logger.info('find_unused_parameters=False (no frozen stages or progressive loss)')

    # Train
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta,
    )

    # Finish wandb run
    if cfg.get('wandb', False):
        try:
            import wandb
            wandb.finish()
        except ImportError:
            pass


if __name__ == '__main__':
    main()
