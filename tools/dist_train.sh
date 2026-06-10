#!/usr/bin/env bash
# =============================================================================
# Distributed Training Script for Oriented R-CNN with DINOv3 Backbone
# =============================================================================
# Usage:
#   bash tools/dist_train.sh <config> <num_gpus> [optional_args]
#
# Examples:
#   bash tools/dist_train.sh configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_dior.py 4
#   bash tools/dist_train.sh configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_dior.py 8 --work-dir work_dirs/my_exp
#   bash tools/dist_train.sh configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_dior.py 4 --resume-from work_dirs/.../latest.pth
# =============================================================================

set -e

CONFIG='configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_dior.py'

# Set environment variables for better performance
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NO_ALBUMENTATIONS_UPDATE=1
# Activate the correct conda environment
# Using the full path to the mmdet environment's Python to avoid environment activation issues
MMDET_PYTHON="/home/users_model/miniconda3/envs/mmdet/bin/python"

# Check that the config file exists
if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: Config file not found: ${CONFIG}"
    exit 1
fi

CUDA_VISIBLE_DEVICES=4,5,6,7 ${MMDET_PYTHON} -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_port=29504 \
    $(dirname "$0")/train.py \
    ${CONFIG} \
    --launcher pytorch

echo ""
echo "Training completed!"
