#!/usr/bin/env bash
# =============================================================================
# Two-stage training: Rotated FCOS + DINOv3 ViT-L/16 + ViT-Adapter (DIOR-R)
# =============================================================================
# ViT-L/16 (1024-d, 24 blocks) with the DINOv3 ViT-Adapter, anchor-free Rotated
# FCOS head. Pretrained weights:
#   data/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
#
# Stage 1: frozen ViT  -> train adapter (SPM + deformable interactions) + FPN + FCOS head
# Stage 2: load stage-1 best, unfreeze ViT, end-to-end fine-tune (backbone @ 0.1x lr)
#
# lr is calibrated for effective batch 128 (8 GPUs x samples_per_gpu=16, the
# script default). If you change SAMPLES_PER_GPU / NUM_GPUS, re-scale S1_LR /
# S2_LR proportionally (e.g. batch 64 -> halve them).
#
# Usage:
#   bash scripts/fcos_vitl_adapter_trainval.sh                      # stage1 -> stage2
#   STAGE=1 bash scripts/fcos_vitl_adapter_trainval.sh              # stage 1 only
#   STAGE=2 STAGE1_CKPT=work_dirs/.../best_mAP_epoch_30.pth \
#       bash scripts/fcos_vitl_adapter_trainval.sh                  # stage 2 only
#   STAGE=1 RESUME=1 WORK_DIR=work_dirs/.../rotated_fcos_dinov3_vitl_adapter_trainval_dior_<ts> \
#       bash scripts/fcos_vitl_adapter_trainval.sh                  # resume interrupted stage 1
#
# Common overrides (environment variables):
#   CUDA_VISIBLE_DEVICES=0,1,2,3   # GPUs to use
#   SAMPLES_PER_GPU=4              # batch/GPU (ViT-L is memory-heavy; lower if OOM)
#   S1_EPOCHS=36                   # stage-1 schedule length
#   S2_EPOCHS=24                   # stage-2 schedule length
#   EVAL_INTERVAL=3                # epochs between test-set evals
#   MASTER_PORT=29512              # DDP port
#   WORK_DIR=work_dirs/fcos_run    # shared output root (stage1/ and stage2/ inside)
#   STAGE1_CKPT=<path>             # explicit stage-1 ckpt for stage 2 (else: best)
#   S1_LR=4e-4 / S2_LR=2e-4        # override base lr (defaults assume batch 128)
#   RESUME=1                       # resume interrupted stage from <WORK_DIR>/<stage>/latest.pth
#                            (requires WORK_DIR = the EXISTING run dir, not a fresh timestamp)
# =============================================================================

set -e

# ----------------------------- configuration --------------------------------
CONFIG_S1='configs/fcos/rotated_fcos_dinov3_vitl_adapter_stage1_trainval_dior.py'
CONFIG_S2='configs/fcos/rotated_fcos_dinov3_vitl_adapter_stage2_trainval_dior.py'

STAGE=${STAGE:-all}                 # all | 1 | 2
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '
' | wc -l)
MASTER_PORT=${MASTER_PORT:-29512}
SAMPLES_PER_GPU=${SAMPLES_PER_GPU:-8}   # ViT-L/16 + Adapter is memory-heavy; lower if OOM
S1_EPOCHS=${S1_EPOCHS:-36}
S2_EPOCHS=${S2_EPOCHS:-48}
EVAL_INTERVAL=${EVAL_INTERVAL:-3}
STAGE1_CKPT=${STAGE1_CKPT:-}
RESUME=${RESUME:-0}                 # 1 = resume interrupted stage from <WORK_DIR>/<stage>/latest.pth
S1_LR=${S1_LR:-}                    # override stage-1 base lr (default 4e-4 in config for batch 128)
S2_LR=${S2_LR:-}                    # override stage-2 base lr (default 2e-4 in config for batch 128)
WORK_DIR=${WORK_DIR:-"work_dirs/rotated_fcos_dinov3_vitl_adapter_trainval_dior_$(date +%Y%m%d_%H%M%S)"}

S1_DIR="${WORK_DIR}/stage1"
S2_DIR="${WORK_DIR}/stage2"

# ----------------------------- environment ----------------------------------
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NO_ALBUMENTATIONS_UPDATE=1
export NCCL_DEBUG=WARN
# Reduce CUDA fragmentation -> fewer OOMs (esp. stage 2 with unfrozen ViT).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Resolve a python that has torch. The plain `python` may land in the base conda
# env (no torch) -> "No module named 'torch'". Fall back to the mmdet env.
PYTHON=${PYTHON:-python}
if ! ${PYTHON} -c "import torch" >/dev/null 2>&1; then
    if [ -x /root/miniconda3/envs/mmdet/bin/python ]; then
        PYTHON=/root/miniconda3/envs/mmdet/bin/python
        echo "NOTE: '${PYTHON:-python}' check: default python lacks torch; using ${PYTHON}"
    else
        echo "ERROR: no python with torch found. 'conda activate mmdet' or set PYTHON=<path>."
        exit 1
    fi
fi

# ----------------------------- sanity checks --------------------------------
for c in "${CONFIG_S1}" "${CONFIG_S2}"; do
    if [ ! -f "${c}" ]; then echo "ERROR: config not found: ${c}"; exit 1; fi
done


mkdir -p "${S1_DIR}" "${S2_DIR}"

run_stage () {
    local cfg="$1"; local wd="$2"; shift 2
    local extra="$*"
    local cmd="CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    cmd="${cmd} ${PYTHON} -m torch.distributed.run"
    cmd="${cmd} --nproc_per_node=${NUM_GPUS}"
    cmd="${cmd} --master_port=${MASTER_PORT}"
    cmd="${cmd} $(dirname "$0")/../tools/train.py"
    cmd="${cmd} ${cfg}"
    cmd="${cmd} --launcher pytorch"
    cmd="${cmd} --work-dir ${wd}"
    cmd="${cmd} --cfg-options ${extra}"
    echo "================================================"
    echo "${cmd}"
    echo "================================================"
    eval "${cmd} 2>&1 | tee ${wd}/train.log"
}

# ----------------------------- stage 1 --------------------------------------
if [ "${STAGE}" = "all" ] || [ "${STAGE}" = "1" ]; then
    EXTRA_S1=""
    EXTRA_S1="${EXTRA_S1} data.samples_per_gpu=${SAMPLES_PER_GPU}"
    EXTRA_S1="${EXTRA_S1} runner.max_epochs=${S1_EPOCHS}"
    EXTRA_S1="${EXTRA_S1} evaluation.interval=${EVAL_INTERVAL}"
    S1_MODE="(fresh)"
    if [ "${RESUME}" = "1" ]; then
        if [ -f "${S1_DIR}/latest.pth" ]; then
            EXTRA_S1="${EXTRA_S1} resume_from=${S1_DIR}/latest.pth"
            S1_MODE="(resume from ${S1_DIR}/latest.pth)"
        else
            echo "WARNING: RESUME=1 but no ${S1_DIR}/latest.pth found -> starting stage 1 fresh."
        fi
    fi
    if [ -n "${S1_LR}" ]; then
        EXTRA_S1="${EXTRA_S1} optimizer.lr=${S1_LR}"
    fi

    echo "########## STAGE 1: frozen DINOv3 ViT ##########"
    echo "Mode      : ${S1_MODE}"
    echo "GPUs      : ${CUDA_VISIBLE_DEVICES} (${NUM_GPUS})"
    echo "Batch/GPU : ${SAMPLES_PER_GPU} (effective = $((SAMPLES_PER_GPU * NUM_GPUS)))"
    echo "Epochs    : ${S1_EPOCHS} (eval every ${EVAL_INTERVAL})"
    echo "Work dir  : ${S1_DIR}"
    run_stage "${CONFIG_S1}" "${S1_DIR}" "${EXTRA_S1}"
fi

# ----------------------------- link stage 1 -> stage 2 ----------------------
if [ "${STAGE}" = "all" ] || [ "${STAGE}" = "2" ]; then
    EXTRA_S2=""
    EXTRA_S2="${EXTRA_S2} data.samples_per_gpu=${SAMPLES_PER_GPU}"
    EXTRA_S2="${EXTRA_S2} runner.max_epochs=${S2_EPOCHS}"
    EXTRA_S2="${EXTRA_S2} evaluation.interval=${EVAL_INTERVAL}"

    S2_MODE="(fresh)"
    if [ "${RESUME}" = "1" ] && [ -f "${S2_DIR}/latest.pth" ]; then
        # Resume an interrupted stage-2 run: keep optimizer + epoch state, do NOT
        # reload stage-1 weights (load_from would reset the model).
        EXTRA_S2="${EXTRA_S2} resume_from=${S2_DIR}/latest.pth"
        S2_MODE="(resume from ${S2_DIR}/latest.pth)"
        STAGE1_CKPT="(resuming, not reloaded)"
    else
        # Resolve the stage-1 checkpoint to load into a fresh stage-2 run.
        if [ -z "${STAGE1_CKPT}" ]; then
            # Prefer the best-by-mAP checkpoint produced by save_best.
            STAGE1_CKPT=$(ls -t "${S1_DIR}"/best_mAP@*_epoch_*.pth 2>/dev/null | head -1 || true)
            if [ -z "${STAGE1_CKPT}" ]; then
                STAGE1_CKPT=$(ls -t "${S1_DIR}"/epoch_*.pth 2>/dev/null | head -1 || true)
            fi
        fi
        if [ -z "${STAGE1_CKPT}" ]; then
            echo "ERROR: no stage-1 checkpoint found in ${S1_DIR}."
            echo "       Run stage 1 first, or pass STAGE1_CKPT=<path>."
            exit 1
        fi
        if [ ! -f "${STAGE1_CKPT}" ]; then
            echo "ERROR: stage-1 checkpoint not found: ${STAGE1_CKPT}"; exit 1
        fi
        EXTRA_S2="${EXTRA_S2} load_from=${STAGE1_CKPT}"
    fi
    if [ -n "${S2_LR}" ]; then
        EXTRA_S2="${EXTRA_S2} optimizer.lr=${S2_LR}"
    fi

    echo "########## STAGE 2: end-to-end fine-tune ##########"
    echo "GPUs       : ${CUDA_VISIBLE_DEVICES} (${NUM_GPUS})"
    echo "Batch/GPU  : ${SAMPLES_PER_GPU} (effective = $((SAMPLES_PER_GPU * NUM_GPUS)))"
    echo "Epochs     : ${S2_EPOCHS} (eval every ${EVAL_INTERVAL})"
    echo "Mode       : ${S2_MODE}"
    echo "Load from  : ${STAGE1_CKPT}"
    echo "Work dir   : ${S2_DIR}"
    run_stage "${CONFIG_S2}" "${S2_DIR}" "${EXTRA_S2}"
fi

echo ""
echo "Two-stage training finished."
echo "Stage 1 dir : ${S1_DIR}"
echo "Stage 2 dir : ${S2_DIR}"
echo "Best ckpt   : ${S2_DIR}/best_mAP*.pth"
echo ""
echo "Final eval on the official DIOR-R test set (no aug, classwise AP):"
echo "  CONFIG='${CONFIG_S2}' \\"
echo "  TEST_CKPT=${S2_DIR}/best_mAP_epoch_*.pth \\"
echo "  WORK_DIR=${S2_DIR} SAVE_VIS=0 NUM_GPUS=${NUM_GPUS} bash scripts/test.sh"
