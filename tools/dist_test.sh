#!/usr/bin/env bash
# =============================================================================
# Distributed Testing Script for Oriented R-CNN with DINOv3 Backbone
# =============================================================================
# Usage:
#   bash tools/dist_test.sh <config> <checkpoint> <num_gpus> [optional_args]
#
# Examples:
#   bash tools/dist_test.sh configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_dior.py \
#       work_dirs/oriented_rcnn_dinov3_fpn_dior/epoch_36.pth 4 --eval mAP
# =============================================================================

set -e

CONFIG='configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_dior.py'
CHECKPOINT='/mnt/ht2-nas2/00-model/guantp/dino/mm_dino/work_dirs/oriented_rcnn_dinov3_fpn_dior/latest.pth'
# Set environment variables
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NO_ALBUMENTATIONS_UPDATE=1

MMDET_PYTHON="/home/users_model/miniconda3/envs/mmdet/bin/python"

# Check that the config file exists
if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: Config file not found: ${CONFIG}"
    exit 1
fi

# python -m torch.distributed.launch \
CUDA_VISIBLE_DEVICES=4,5,6,7 ${MMDET_PYTHON} -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_port=29503 \
    $(dirname "$0")/test.py \
    ${CONFIG} \
    ${CHECKPOINT} \
    --launcher pytorch \
    "$@"

echo ""
echo "Testing completed!"
