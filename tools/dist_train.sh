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

CONFIG=$1
GPUS=$2
shift 2  # Remove first two args, remaining are passed to train.py

if [ -z "$CONFIG" ] || [ -z "$GPUS" ]; then
    echo "Usage: bash tools/dist_train.sh <config> <num_gpus> [optional_args]"
    echo ""
    echo "Example:"
    echo "  bash tools/dist_train.sh configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_dior.py 4"
    exit 1
fi

# Get the Python virtual environment
VENV_PATH="/home/guantp/pro/olmoearth_pretrain/.venv"
if [ -f "${VENV_PATH}/bin/activate" ]; then
    source "${VENV_PATH}/bin/activate"
    echo "Activated Python environment: ${VENV_PATH}"
fi

# Set environment variables for better performance
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

# Print training information
echo "================================================"
echo "Distributed Training"
echo "================================================"
echo "Config:        ${CONFIG}"
echo "GPUs:          ${GPUS}"
echo "Extra args:    $@"
echo "Python:        $(which python)"
echo "CUDA devices:  ${CUDA_VISIBLE_DEVICES}"
echo "================================================"
echo ""

# Check that the config file exists
if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: Config file not found: ${CONFIG}"
    exit 1
fi

# Check GPU count vs available GPUs
AVAILABLE_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
if [ $GPUS -gt $AVAILABLE_GPUS ]; then
    echo "WARNING: Requested ${GPUS} GPUs but only ${AVAILABLE_GPUS} visible."
    echo "         Setting GPUS to ${AVAILABLE_GPUS}"
    GPUS=$AVAILABLE_GPUS
fi

# Build port for distributed training
PORT=${PORT:-29500}

# Launch distributed training
echo "Launching distributed training with ${GPUS} GPUs..."

python -m torch.distributed.launch \
    --nproc_per_node=${GPUS} \
    --master_port=${PORT} \
    $(dirname "$0")/train.py \
    ${CONFIG} \
    --launcher pytorch \
    --gpus ${GPUS} \
    "$@"

echo ""
echo "Training completed!"
