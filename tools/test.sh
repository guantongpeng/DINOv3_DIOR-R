#!/usr/bin/env bash
# =============================================================================
# Test Script for Oriented R-CNN with DINOv3 Backbone on DIOR-R
# =============================================================================
# Usage:
#   bash tools/test.sh
#
# Options (set via environment variables):
#   NUM_GPUS=4           Number of GPUs for distributed testing (default: 1)
#   SAVE_VIS=1           Save detection images with boxes and class names
#   SAVE_VIS=0           Only compute metrics, no image saving (faster)
#   SHOW_SCORE_THR       Score threshold for visualized boxes (default: 0.3)
#   VIS_DIR              Directory to save visualized images
#
# Note: SAVE_VIS=1 forces single-GPU mode (visualization not supported in
#       distributed mode). For fastest evaluation, use SAVE_VIS=0 NUM_GPUS=N.
#
# Examples:
#   # Single GPU, metrics only (fastest single GPU)
#   SAVE_VIS=0 bash tools/test.sh
#
#   # 4 GPUs distributed, metrics only (fastest overall)
#   SAVE_VIS=0 NUM_GPUS=4 CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/test.sh
#
#   # 8 GPUs distributed, metrics only
#   SAVE_VIS=0 NUM_GPUS=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash tools/test.sh
#
#   # Single GPU + save visualizations
#   SAVE_VIS=1 bash tools/test.sh
# =============================================================================

set -e

CONFIG=${CONFIG:-'configs/oriented_rcnn/oriented_rcnn_dinov3_vitb_fpn_dior.py'}
WORK_DIR=${WORK_DIR:-'/mnt/ht2-nas2/00-model/guantp/dino/mm_dino/work_dirs/oriented_rcnn_dinov3_vitb_fpn_dior_20260615_104700'}

# Prefer the best-on-val checkpoint (from save_best='mAP@0.50'); fall back to
# latest.pth if no best checkpoint exists. Override with TEST_CKPT=<path>.
BEST_CKPT=$(ls -1 "${WORK_DIR}"/best_mAP*.pth 2>/dev/null | head -1)
if [ -n "${BEST_CKPT}" ]; then
    DEFAULT_CKPT="${BEST_CKPT}"
else
    DEFAULT_CKPT="${WORK_DIR}/latest.pth"
fi
CHECKPOINT=${TEST_CKPT:-"${DEFAULT_CKPT}"}

# GPU settings
NUM_GPUS=${NUM_GPUS:-1}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
MASTER_PORT=${MASTER_PORT:-29510}

# Result saving
RESULT_FILE=${RESULT_FILE:-"${WORK_DIR}/test_results.txt"}  # Metrics output txt
CLASSWISE=${CLASSWISE:-1}            # 1=output per-class AP, 0=overall only

# Visualization settings
SAVE_VIS=${SAVE_VIS:-1}              # 1=save detection images, 0=metrics only
SHOW_SCORE_THR=${SHOW_SCORE_THR:-0.3}  # Score threshold for drawn boxes
VIS_DIR=${VIS_DIR:-"${WORK_DIR}/vis_test_results"}  # Output dir for visualized images

# Python
MMDET_PYTHON="/root/miniconda3/envs/mmdet/bin/python"

# Environment
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NO_ALBUMENTATIONS_UPDATE=1

# -------------------- Validation --------------------
if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: Config file not found: ${CONFIG}"
    exit 1
fi

if [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: Checkpoint file not found: ${CHECKPOINT}"
    exit 1
fi

# -------------------- Sanity Checks --------------------
if [ "${SAVE_VIS}" = "1" ] && [ "${NUM_GPUS}" -gt 1 ]; then
    echo "WARNING: SAVE_VIS=1 not compatible with multi-GPU, falling back to NUM_GPUS=1"
    NUM_GPUS=1
fi

# -------------------- Build Command --------------------
# Shared args
EXTRA_ARGS=""
EXTRA_ARGS="${EXTRA_ARGS} --eval mAP"
EXTRA_ARGS="${EXTRA_ARGS} --cfg-options work_dir=${WORK_DIR}"

if [ "${CLASSWISE}" = "1" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --eval-options 'classwise=True'"
fi

if [ "${SAVE_VIS}" = "1" ]; then
    mkdir -p "${VIS_DIR}"
    echo "Saving detection visualizations to: ${VIS_DIR}"
    EXTRA_ARGS="${EXTRA_ARGS} --show-score-thr ${SHOW_SCORE_THR}"
    EXTRA_ARGS="${EXTRA_ARGS} --show-dir ${VIS_DIR}"
else
    echo "Visualization saving disabled (SAVE_VIS=0)"
fi

if [ "${NUM_GPUS}" -gt 1 ]; then
    # Multi-GPU distributed mode
    echo "Running distributed test on ${NUM_GPUS} GPUs (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
    mkdir -p "${WORK_DIR}"

    CMD="CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ${MMDET_PYTHON} -m torch.distributed.run"
    CMD="${CMD} --nproc_per_node=${NUM_GPUS}"
    CMD="${CMD} --master_port=${MASTER_PORT}"
    CMD="${CMD} tools/test.py"
    CMD="${CMD} ${CONFIG}"
    CMD="${CMD} ${CHECKPOINT}"
    CMD="${CMD} --launcher pytorch"
    CMD="${CMD} --gpu-collect"
    CMD="${CMD} ${EXTRA_ARGS}"
else
    # Single GPU mode
    echo "Running single-GPU test (GPU ${CUDA_VISIBLE_DEVICES})"

    CMD="CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ${MMDET_PYTHON} tools/test.py"
    CMD="${CMD} ${CONFIG}"
    CMD="${CMD} ${CHECKPOINT}"
    CMD="${CMD} ${EXTRA_ARGS}"
fi

echo ""
echo "================================================"
echo "Running: ${CMD}"
echo "Results will be saved to: ${RESULT_FILE}"
echo "================================================"
echo ""

eval ${CMD} 2>&1 | tee "${RESULT_FILE}"

echo ""
echo "Testing completed!"
echo "Results saved to: ${RESULT_FILE}"
if [ "${SAVE_VIS}" = "1" ]; then
    echo "Visualized results saved to: ${VIS_DIR}"
fi
