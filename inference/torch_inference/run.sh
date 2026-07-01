#!/usr/bin/env bash
# =============================================================================
# Pure-PyTorch inference + DOTA mAP evaluation for
#   Oriented R-CNN + DINOv3 ViT-L/16 Adapter  on  DIOR-R
#
# No mmrotate/mmcv/mmdet/dinov3 -- everything is torch + cv2 + numpy +
# torchvision, implemented under inference/torch_inference/. One process per GPU;
# each shard dumps a pkl; evaluate.py gathers and computes mAP@0.50.
#
# Usage:
#   bash inference/torch_inference/run.sh                                  # all visible GPUs
#   CUDA_VISIBLE_DEVICES=0 bash inference/torch_inference/run.sh           # single GPU
#   CHECKPOINT=/path/to/x.pth DATA_ROOT=/path/to/DIOR-R bash inference/torch_inference/run.sh
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

CHECKPOINT="${CHECKPOINT:-work_dirs/oriented_rcnn_dinov3_vitl_adapter_dior_20260626_092528/stage2/best_mAP@0.50_epoch_28.pth}"
DATA_ROOT="${DATA_ROOT:-data/DIOR-R}"
SPLIT="${SPLIT:-test}"
IOU_THR="${IOU_THR:-0.5}"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    NGPU_TOTAL="$(nvidia-smi -L 2>/dev/null | wc -l)"
    [ -z "${NGPU_TOTAL}" ] || [ "${NGPU_TOTAL}" -lt 1 ] && NGPU_TOTAL=8
    GPU_LIST="$(seq -s, 0 $((NGPU_TOTAL-1)) | sed 's/,$//')"
else
    GPU_LIST="${CUDA_VISIBLE_DEVICES}"
fi
NUM_GPUS="${NUM_GPUS:-$(echo "${GPU_LIST}" | tr ',' '\n' | grep -c .)}"
GPU_ARR=(${GPU_LIST//,/ })

OUT_DIR="${SCRIPT_DIR}/results"
rm -f "${OUT_DIR}"/part_*.pkl "${OUT_DIR}"/shard_*.log 2>/dev/null || true
mkdir -p "${OUT_DIR}"

PY="${PY:-/root/miniconda3/envs/olmoearth/bin/python}"
export NO_ALBUMENTATIONS_UPDATE=1
export OMP_NUM_THREADS=2

echo "================================================"
echo "Checkpoint: ${CHECKPOINT}"
echo "Data:       ${DATA_ROOT} [${SPLIT}]"
echo "GPUs:       ${GPU_LIST}  (${NUM_GPUS} shards)   [pure torch]"
echo "================================================"

# 1) inference: one process per GPU shard
PIDS=()
for ((i=0; i<NUM_GPUS; i++)); do
    GPU="${GPU_ARR[$i]}"
    CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" "${SCRIPT_DIR}/inference.py" \
        --checkpoint "${CHECKPOINT}" --data-root "${DATA_ROOT}" --split "${SPLIT}" \
        --shard "${i}" --num-shards "${NUM_GPUS}" --out-dir "${OUT_DIR}" \
        > "${OUT_DIR}/shard_${i}.log" 2>&1 &
    PIDS+=($!)
    echo "Launched shard ${i} on GPU ${GPU} (pid ${PIDS[-1]})"
done

FAIL=0
for pid in "${PIDS[@]}"; do
    wait "${pid}" || { echo "ERROR: process ${pid} failed (see ${OUT_DIR}/shard_*.log)"; FAIL=1; }
done
[ "${FAIL}" -ne 0 ] && exit 1
echo "All shards done."

# 2) evaluation
"${PY}" "${SCRIPT_DIR}/evaluate.py" \
    --results-dir "${OUT_DIR}" --num-shards "${NUM_GPUS}" \
    --data-root "${DATA_ROOT}" --split "${SPLIT}" --iou-thr "${IOU_THR}" \
    --out "${OUT_DIR}/metrics_${SPLIT}.txt"

echo "Done. See ${OUT_DIR}/metrics_${SPLIT}.txt"
