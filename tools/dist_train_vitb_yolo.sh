#!/usr/bin/env bash
# =============================================================================
# Distributed Training Script for Oriented R-CNN with DINOv3 ViT-B/16 on DIOR-R
# =============================================================================
# Usage:
#   bash tools/dist_train_vitb.sh
#
# Examples (override args):
#   bash tools/dist_train_vitb.sh --work-dir work_dirs/my_exp
#   bash tools/dist_train_vitb.sh --resume-from work_dirs/.../latest.pth
# =============================================================================

set -e

CONFIG='/mnt/ht2-nas2/00-model/guantp/dino/mm_dino/configs/yolo26/yolo26_dinov3_fpn_dior.py'

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NO_ALBUMENTATIONS_UPDATE=1

if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: Config file not found: ${CONFIG}"
    exit 1
fi

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run \
    --nproc_per_node=8 \
    --master_port=29503 \
    $(dirname "$0")/train.py \
    ${CONFIG} \
    --launcher pytorch \
    "$@"

echo ""
echo "Training completed!"
