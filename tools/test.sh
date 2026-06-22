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

CONFIG=${CONFIG:-'/mnt/ht2-nas2/00-model/guantp/dino/mm_dino/work_dirs/yolo26_dinov3_fpn_dior_bugfix/yolo26_dinov3_fpn_dior.py'}
WORK_DIR=${WORK_DIR:-'/mnt/ht2-nas2/00-model/guantp/dino/mm_dino/work_dirs/yolo26_dinov3_fpn_dior_bugfix/'}

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
# MASTER_PORT is resolved below (auto-picked when unset, verified when set).

# Result saving
RESULT_FILE=${RESULT_FILE:-"${WORK_DIR}/test_results.txt"}  # Metrics output txt
CLASSWISE=${CLASSWISE:-1}            # 1=output per-class AP, 0=overall only

# Visualization settings
SAVE_VIS=${SAVE_VIS:-1}              # 1=save detection images, 0=metrics only
SHOW_SCORE_THR=${SHOW_SCORE_THR:-0.3}  # Score threshold for drawn boxes
VIS_DIR=${VIS_DIR:-"${WORK_DIR}/vis_test_results"}  # Output dir for visualized images

# Python
MMDET_PYTHON="/root/miniconda3/envs/mmdet/bin/python"

# -------------------- Master Port Resolution --------------------
# Auto-select a free master port when MASTER_PORT is unset; verify it is free
# when explicitly provided (avoids cryptic torch EADDRINUSE crashes).
_port_is_free() {
    "${MMDET_PYTHON}" - "$1" <<'PY'
import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.bind(("", int(sys.argv[1])))
except OSError:
    sys.exit(1)
finally:
    s.close()
sys.exit(0)
PY
}

_find_free_port() {
    "${MMDET_PYTHON}" - <<'PY'
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
PY
}

if [ -z "${MASTER_PORT+x}" ]; then
    MASTER_PORT=$(_find_free_port)
    echo "Auto-selected free MASTER_PORT=${MASTER_PORT}"
else
    if ! _port_is_free "${MASTER_PORT}"; then
        echo "ERROR: MASTER_PORT=${MASTER_PORT} is already in use (likely another distributed job)."
        FREE_PORT=$(_find_free_port)
        echo "       A free port is: ${FREE_PORT}"
        echo "       Re-run with: MASTER_PORT=${FREE_PORT} bash tools/test.sh"
        exit 1
    fi
fi

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
