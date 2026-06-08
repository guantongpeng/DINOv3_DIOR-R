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

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
shift 3  # Remove first three args, remaining are passed to test.py

if [ -z "$CONFIG" ] || [ -z "$CHECKPOINT" ] || [ -z "$GPUS" ]; then
    echo "Usage: bash tools/dist_test.sh <config> <checkpoint> <num_gpus> [optional_args]"
    echo ""
    echo "Example:"
    echo "  bash tools/dist_test.sh configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_dior.py \\"
    echo "      work_dirs/oriented_rcnn_dinov3_fpn_dior/epoch_36.pth 4 --eval mAP"
    exit 1
fi

# Get the Python virtual environment
VENV_PATH="/home/guantp/pro/olmoearth_pretrain/.venv"
if [ -f "${VENV_PATH}/bin/activate" ]; then
    source "${VENV_PATH}/bin/activate"
    echo "Activated Python environment: ${VENV_PATH}"
fi

# Set environment variables
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

# Print testing information
echo "================================================"
echo "Distributed Testing"
echo "================================================"
echo "Config:        ${CONFIG}"
echo "Checkpoint:    ${CHECKPOINT}"
echo "GPUs:          ${GPUS}"
echo "Extra args:    $@"
echo "================================================"
echo ""

# Validate files exist
if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: Config file not found: ${CONFIG}"
    exit 1
fi

if [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: Checkpoint not found: ${CHECKPOINT}"
    exit 1
fi

# Build port for distributed testing
PORT=${PORT:-29501}

# Launch distributed testing
echo "Launching distributed testing with ${GPUS} GPUs..."

python -m torch.distributed.launch \
    --nproc_per_node=${GPUS} \
    --master_port=${PORT} \
    $(dirname "$0")/test.py \
    ${CONFIG} \
    ${CHECKPOINT} \
    --launcher pytorch \
    "$@"

echo ""
echo "Testing completed!"
