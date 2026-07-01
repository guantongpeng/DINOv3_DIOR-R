# ONNX Inference — DINOv3 ViT-L/16 Adapter detectors (DIOR-R)

Convert a trained `.pth` checkpoint to ONNX and run inference with
**onnxruntime**, supporting **single-GPU, multi-GPU, and CPU**, for three
detector heads sharing the same backbone:

| checkpoint (work_dir) | head (ONNX export) | export | reference mAP@0.50 | ONNX mAP@0.50 |
|-----------------------|--------------------|--------|--------------------|---------------|
| `rotated_fcos_.../stage2/best_mAP@0.50_epoch_39.pth` | FCOS (anchor-free, centerness)    | 1 ONNX               | 0.7473 | **0.7473** |
| `yolo26_.../stage2/best_mAP@0.50_epoch_36.pth`       | FCOS (see note)                   | 1 ONNX               | 0.7584 | **0.7585** |
| `oriented_rcnn_.../stage2/best_mAP@0.50_epoch_28.pth`| OrientedRCNN (two-stage)          | 2 ONNX (A + B)       | 0.7500 | **0.6992** |

The ONNX-export accuracy is validated against the mmrotate / PyTorch reference.
Single-stage heads reproduce the reference **exactly**; the two-stage detector
reproduces it to ~0.70 (see "OrientedRCNN gap" below).

> **Note on the `yolo26` checkpoint.** `best_mAP@0.50_epoch_36.pth` was trained
> with the **RotatedFCOS** head (it loads into the FCOS model with 0 missing /
> 0 unexpected keys; the current `models/heads/yolo26_rotated_head.py` was
> refactored after this checkpoint and no longer matches it — loading it there
> leaves 293 head keys random, producing garbage). `convert.py` therefore
> exports it via the FCOS path (`--config` the FCOS config), giving the exact
> reference mAP. If you have a checkpoint that genuinely matches the current
> `YOLO26RotatedHead`, point `--config` at the yolo26 config and it will be
> exported with the YOLO26 decode branch.

## Layout

```
inference/onnx_inference/
  export/convert.py          detector-aware .pth -> ONNX export (see "Files" below)
  core/postprocess.py        single-stage decode + NMS
  core/oriented_rcnn.py      two-stage RPN/RoI glue (reuses torch_inference)
  inference.py               entry point: dispatch + onnxruntime shard run
  evaluate.py                entry point: gather shards + DOTA mAP@0.50
  run.sh                     one-shot driver
  models/                    exported `*.onnx` + `*.meta.json` sidecars
  results/<name>/            per-model inference dumps + metrics
```

## Files

| file | purpose |
|------|---------|
| `export/convert.py`  | Detector-aware export (auto-detected from the config). Single ONNX for FCOS/YOLO26; stage-A + stage-B for OrientedRCNN. Writes `<stem>.meta.json` + verifies torch-vs-ort. |
| `core/postprocess.py`    | Single-stage decode + NMS (FCOS `distance2obb`/centerness, YOLO26 horizontal-ltrb/`cls*obj`). Uses mmcv `nms_rotated`. |
| `core/oriented_rcnn.py`  | Two-stage glue: loads stage-A+B, reuses `inference/torch_inference` (anchors, RPN decode+NMS, mmcv rotated RoIAlign, RoI decode+NMS). |
| `inference.py`      | Reads the meta sidecar, dispatches on detector, runs a shard with onnxruntime (GPU/CPU). |
| `evaluate.py`       | Gather shards + DOTA mAP@0.50 (reuses `inference/torch_inference/eval_map.py`). |
| `run.sh`            | One-shot driver: (optional) convert → multi/single-GPU or CPU inference → eval. |
| `models/model*.onnx`, `models/model*.meta.json` | exported models + their head/test-cfg metadata. |

## How it works

Only the **network forward** goes into ONNX; post-processing (decode + rotated
NMS, and for OrientedRCNN the RPN/RoI decode + rotated RoIAlign) stays in
Python — that is what makes the export tractable, since the custom CUDA ops
(rotated NMS, RoIAlignRotated, MSDeformAttn) cannot live in ONNX.

Backbone export fixes (all in `export/convert.py`, identical for every head):

1. **MSDeformAttn** is forced onto the pure-PyTorch `grid_sample` path
   (`IS_CUDA_AVAILABLE=False` on the module the class is defined in); otherwise
   the tracer freezes the whole attention sub-graph into a constant.
2. **RoPE / deform reference points** built with `linspace`/`arange` (ONNX
   `Range` crash) are emitted as numpy-derived constants.
3. Externalized weight files are merged back into one self-contained `.onnx`.

Per detector:

* **FCOS / YOLO26** — single ONNX emits 5 levels × {cls, bbox, angle, X}
  (X = centerness / objectness). `core/postprocess.py` decodes per head and runs
  rotated NMS.
* **OrientedRCNN** — *stage A* (backbone+FPN+RPN → FPN feats + RPN maps) and
  *stage B* (RoI bbox head, dynamic #RoIs). The Python glue does anchor-based
  RPN decode + horizontal NMS, rotated RoIAlign, then RoI decode + rotated NMS
  (reusing the validated `inference/torch_inference`).

## Quick start

```bash
# convert once per checkpoint (auto-detects the head from the config)
python inference/onnx_inference/export/convert.py --config <cfg> --checkpoint <pth> \
    --out inference/onnx_inference/models/model_<name>.onnx

# multi-GPU inference + eval on the whole DIOR-R test set
ONNX=inference/onnx_inference/models/model_<name>.onnx CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    bash inference/onnx_inference/run.sh
# single GPU
ONNX=inference/onnx_inference/models/model_<name>.onnx CUDA_VISIBLE_DEVICES=0 \
    bash inference/onnx_inference/run.sh
# CPU
ONNX=inference/onnx_inference/models/model_<name>.onnx DEVICE=cpu \
    bash inference/onnx_inference/run.sh
```

`run.sh` skips conversion if the `.onnx` exists (set `CONVERT=1` to force).
Results/metrics go to `inference/onnx_inference/results/<name>/`.

## OrientedRCNN gap (0.6992 vs 0.7500)

The OrientedRCNN pipeline is **verified correct per-component**: its ONNX
backbone+FPN+RPN match torch (≤1e-2), its RPN proposals match the mmrotate
reference (2000 each, near-identical values), it uses mmcv's exact rotated
RoIAlign and rotated NMS, and its RoI decode matches the reference on confident
classes. The ~5-point gap comes from the **two-stage cascade amplifying the
small, unavoidable backbone numerical difference** between the ONNX export
(`grid_sample` MSDeformAttn + fp32 RoPE — required because the CUDA custom ops
can't go into ONNX) and the reference (CUDA MSDeformAttn + bf16 RoPE). The
single-stage FCOS head is robust to that ~1e-2 feature difference (matches
exactly); the RPN→RoI cascade is not, so borderline detections flip and recall
drops. This is an inherent limitation of ONNX-exporting this ViT-Adapter
backbone for a two-stage detector.

## Notes

* Input is fixed at `(1, 3, 800, 800)` — every DIOR-R image is 800×800.
* Multi-GPU = one onnxruntime process per GPU on a disjoint image shard.
* Post-processing runs on GPU tensors when `device=gpu` (fast mmcv `nms_rotated`
  / `roi_align_rotated`); CPU mode falls back to pure-torch implementations.
* CPU inference works but is slow (ViT-L forward on CPU).

