#!/usr/bin/env bash
# =============================================================================
# ONNX inference + DOTA-mAP evaluation for
#   Rotated FCOS + DINOv3 ViT-L/16 Adapter  on  DIOR-R
#
# Runs the exported ONNX model (backbone+neck+head) with onnxruntime. The FCOS
# decode + rotated NMS is done in PyTorch (core/postprocess.py). Supports:
#   * multi-GPU : one process per GPU, each on a shard (fastest)
#   * single-GPU
#   * CPU       (set DEVICE=cpu)
#
# Usage:
#   bash inference/onnx_inference/run.sh                                   # all visible GPUs
#   CUDA_VISIBLE_DEVICES=0 bash inference/onnx_inference/run.sh            # single GPU
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash inference/onnx_inference/run.sh
#   DEVICE=cpu bash inference/onnx_inference/run.sh                        # CPU inference
#
# Convert first (only needs to run once):
#   CONVERT=1 bash inference/onnx_inference/run.sh
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# -------- configuration (override via env) --------
ONNX="${ONNX:-${SCRIPT_DIR}/models/model.onnx}"
CONFIG="${CONFIG:-work_dirs/rotated_fcos_dinov3_vitl_adapter_trainval_dior_20260629_092912/stage2/rotated_fcos_dinov3_vitl_adapter_stage2_trainval_dior.py}"
CHECKPOINT="${CHECKPOINT:-work_dirs/rotated_fcos_dinov3_vitl_adapter_trainval_dior_20260629_092912/stage2/best_mAP@0.50_epoch_39.pth}"
DATA_ROOT="${DATA_ROOT:-data/DIOR-R}"
SPLIT="${SPLIT:-test}"
IOU_THR="${IOU_THR:-0.5}"
DEVICE="${DEVICE:-gpu}"          # gpu | cpu  (gpu => one shard per visible GPU)
LIMIT="${LIMIT:-0}"

PY="${PY:-/root/miniconda3/envs/olmoearth/bin/python}"
export NO_ALBUMENTATIONS_UPDATE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/results/$(basename "${ONNX}" .onnx)}"
mkdir -p "${OUT_DIR}"

# -------- optional: (re)convert the checkpoint to ONNX --------
if [ "${CONVERT:-0}" = "1" ] || [ ! -f "${ONNX}" ]; then
    echo "================ CONVERT ================"
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} "${PY}" "${SCRIPT_DIR}/export/convert.py" \
        --config "${CONFIG}" --checkpoint "${CHECKPOINT}" --out "${ONNX}" --verify 1
fi

# -------- GPU list / shard count --------
if [ "${DEVICE}" = "cpu" ]; then
    NUM_GPUS=0
else
    if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
        NGPU_TOTAL="$(nvidia-smi -L 2>/dev/null | wc -l)"
        [ -z "${NGPU_TOTAL}" ] || [ "${NGPU_TOTAL}" -lt 1 ] && NGPU_TOTAL=8
        GPU_LIST="$(seq -s, 0 $((NGPU_TOTAL-1)) | sed 's/,$//')"
    else
        GPU_LIST="${CUDA_VISIBLE_DEVICES}"
    fi
    NUM_GPUS="$(echo "${GPU_LIST}" | tr ',' '\n' | grep -c .)"
    [ "${NUM_GPUS}" -lt 1 ] && { echo "no GPUs visible"; exit 1; }
fi

echo "================================================"
echo "ONNX:       ${ONNX}"
echo "Data:       ${DATA_ROOT} [${SPLIT}]"
echo "Device:     ${DEVICE}  ($([ "${DEVICE}" = "cpu" ] && echo CPU || echo "${NUM_GPUS} shard(s), GPUs ${GPU_LIST}"))"
echo "IoU thr:    ${IOU_THR}"
echo "================================================"

LIMIT_ARG=""
[ "${LIMIT}" -gt 0 ] && LIMIT_ARG="--limit ${LIMIT}"

# -------- 1) inference --------
if [ "${DEVICE}" = "cpu" ]; then
    echo "[*] CPU inference (single process)"
    "${PY}" "${SCRIPT_DIR}/inference.py" --onnx "${ONNX}" \
        --data-root "${DATA_ROOT}" --split "${SPLIT}" \
        --device cpu --shard 0 --num-shards 1 --out-dir "${OUT_DIR}" ${LIMIT_ARG} \
        2>&1 | tee "${OUT_DIR}/cpu.log"
    NUM_SHARDS=1
else
    PIDS=()
    GPU_ARR=(${GPU_LIST//,/ })
    for ((i=0; i<NUM_GPUS; i++)); do
        GPU="${GPU_ARR[$i]}"
        CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" "${SCRIPT_DIR}/inference.py" \
            --onnx "${ONNX}" --data-root "${DATA_ROOT}" --split "${SPLIT}" \
            --device gpu --shard "${i}" --num-shards "${NUM_GPUS}" \
            --out-dir "${OUT_DIR}" ${LIMIT_ARG} \
            > "${OUT_DIR}/shard_${i}.log" 2>&1 &
        PIDS+=($!)
        echo "Launched shard ${i} on GPU ${GPU} (pid ${PIDS[-1]})"
    done
    FAIL=0
    for pid in "${PIDS[@]}"; do
        if ! wait "${pid}"; then
            echo "ERROR: inference process ${pid} failed (see ${OUT_DIR}/shard_*.log)"; FAIL=1
        fi
    done
    [ "${FAIL}" -ne 0 ] && exit 1
    echo "All shards done."
    NUM_SHARDS="${NUM_GPUS}"
fi

# -------- 2) evaluation --------
"${PY}" "${SCRIPT_DIR}/evaluate.py" \
    --results-dir "${OUT_DIR}" --num-shards "${NUM_SHARDS}" \
    --data-root "${DATA_ROOT}" --split "${SPLIT}" \
    --iou-thr "${IOU_THR}" \
    --out "${OUT_DIR}/metrics_${SPLIT}.txt"

echo "Done. See ${OUT_DIR}/metrics_${SPLIT}.txt"
