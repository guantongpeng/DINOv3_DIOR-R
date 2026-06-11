#!/usr/bin/env bash
# =============================================================================
# Distributed Training Script for Oriented R-CNN + DINOv3 on Star-1021+Extend3
# =============================================================================
set -e

CONFIG='configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_star.py'
MSTAR_PYTHON="/home/users_model/miniconda3/envs/mmdet/bin/python"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NO_ALBUMENTATIONS_UPDATE=1

if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: Config file not found: ${CONFIG}"
    exit 1
fi

CUDA_VISIBLE_DEVICES=2,4,5,6 ${MSTAR_PYTHON} -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_port=29506 \
    $(dirname "$0")/train.py \
    ${CONFIG} \
    --launcher pytorch \
    "$@"

echo ""
echo "Training completed!"
