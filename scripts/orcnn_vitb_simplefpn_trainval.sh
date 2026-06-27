#!/usr/bin/env bash
# =============================================================================
# Oriented R-CNN + DINOv3 ViT-B/16 + SimpleFPN (DIOR-R) — TRAINVAL final run
# =============================================================================
# Config: configs/oriented_rcnn/oriented_rcnn_dinov3_vitb_simplefpn_trainval_dior.py
#
# Trains on the FULL DIOR-R trainval pool (train + val merged, ~11.7k images) to
# maximize test mAP. The held-out TEST split is used both for periodic eval /
# save_best model selection during training and for the final tools/test.py run.
#
# NOTE on eval cost: val == test here, so every evaluation runs over the full
# (~11.7k-image) test set. Use EVAL_INTERVAL to control how often this happens.
#
# Usage:
#   bash scripts/orcnn_vitb_simplefpn_trainval.sh
#
# Common overrides (environment variables):
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7   # which GPUs to use
#   SAMPLES_PER_GPU=8                       # batch size per GPU
#   MAX_EPOCHS=120                          # schedule length
#   EVAL_INTERVAL=3                         # epochs between test-set evals
#   MASTER_PORT=29508                       # DDP port (change if 'port in use')
#   RESUME=work_dirs/.../latest.pth         # resume from a checkpoint
#   WORK_DIR=work_dirs/my_run               # custom output dir
#
# Examples:
#   bash scripts/orcnn_vitb_simplefpn_trainval.sh
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/orcnn_vitb_simplefpn_trainval.sh
#   EVAL_INTERVAL=5 bash scripts/orcnn_vitb_simplefpn_trainval.sh     # less frequent (cheaper) eval
#   RESUME=work_dirs/.../latest.pth bash scripts/orcnn_vitb_simplefpn_trainval.sh
# =============================================================================

set -e

# ----------------------------- configuration --------------------------------
CONFIG='configs/oriented_rcnn/oriented_rcnn_dinov3_vitb_simplefpn_trainval_dior.py'

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '
' | wc -l)
MASTER_PORT=${MASTER_PORT:-29508}
SAMPLES_PER_GPU=${SAMPLES_PER_GPU:-16}
MAX_EPOCHS=${MAX_EPOCHS:-200}
EVAL_INTERVAL=${EVAL_INTERVAL:-3}
WORK_DIR=${WORK_DIR:-"work_dirs/oriented_rcnn_dinov3_vitb_simplefpn_trainval_dior_$(date +%Y%m%d_%H%M%S)"}

# Tuning knobs passed to the config at runtime
EXTRA_CFG=""
EXTRA_CFG="${EXTRA_CFG} data.samples_per_gpu=${SAMPLES_PER_GPU}"
EXTRA_CFG="${EXTRA_CFG} runner.max_epochs=${MAX_EPOCHS}"
EXTRA_CFG="${EXTRA_CFG} evaluation.interval=${EVAL_INTERVAL}"

# ----------------------------- environment ----------------------------------
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NO_ALBUMENTATIONS_UPDATE=1

# ----------------------------- sanity checks --------------------------------
if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: Config not found: ${CONFIG}"
    exit 1
fi


mkdir -p "${WORK_DIR}"

# ----------------------------- build command --------------------------------
CMD="CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
CMD="${CMD} python -m torch.distributed.run"
CMD="${CMD} --nproc_per_node=${NUM_GPUS}"
CMD="${CMD} --master_port=${MASTER_PORT}"
CMD="${CMD} $(dirname "$0")/../tools/train.py"
CMD="${CMD} ${CONFIG}"
CMD="${CMD} --launcher pytorch"
CMD="${CMD} --work-dir ${WORK_DIR}"
CMD="${CMD} --cfg-options ${EXTRA_CFG}"

if [ -n "${RESUME}" ]; then
    if [ ! -f "${RESUME}" ]; then
        echo "ERROR: RESUME checkpoint not found: ${RESUME}"
        exit 1
    fi
    CMD="${CMD} --resume-from ${RESUME}"
fi

echo "================================================"
echo "Oriented R-CNN + DINOv3 ViT-B/16 + SimpleFPN (TRAINVAL)"
echo "Data       : train + val merged  |  eval on test split"
echo "GPUs       : ${CUDA_VISIBLE_DEVICES} (${NUM_GPUS})"
echo "Batch/GPU  : ${SAMPLES_PER_GPU}   (effective batch = $((SAMPLES_PER_GPU * NUM_GPUS)))"
echo "Epochs     : ${MAX_EPOCHS}   (eval every ${EVAL_INTERVAL})"
echo "Work dir   : ${WORK_DIR}"
[ -n "${RESUME}" ] && echo "Resuming from: ${RESUME}"
echo "------------------------------------------------"
echo "${CMD}"
echo "================================================"

eval "${CMD} 2>&1 | tee ${WORK_DIR}/train.log"

echo ""
echo "Training finished. Results in: ${WORK_DIR}"
echo "Best checkpoint (by test mAP): ${WORK_DIR}/best_mAP*.pth"
echo ""
echo "Final eval on the official DIOR-R test set (no aug, classwise AP):"
echo "  CONFIG='configs/oriented_rcnn/oriented_rcnn_dinov3_vitb_simplefpn_trainval_dior.py' \\"
echo "  TEST_CKPT=${WORK_DIR}/best_mAP_epoch_*.pth \\"
echo "  WORK_DIR=${WORK_DIR} SAVE_VIS=0 NUM_GPUS=${NUM_GPUS} bash scripts/test.sh"
