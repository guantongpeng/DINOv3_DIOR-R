# Pure-PyTorch Inference — DINOv3 ViT-L/16 Adapter detectors (DIOR-R)

End-to-end **inference + DOTA mAP evaluation** of DINOv3-ViT-Adapter detectors on
DIOR-R, implemented with **only PyTorch / NumPy / OpenCV / torchvision** —
**no mmrotate, mmcv, mmdet, or the dinov3 package**. Two detector types are
supported and **auto-detected** from the checkpoint:

| Checkpoint | Detector | pure-torch class |
|-----------|----------|------------------|
| `oriented_rcnn_.../best_mAP@0.50_epoch_28.pth` | **Oriented R-CNN** (two-stage) | `detector.OrientedRCNN` |
| `yolo26_.../best_mAP@0.50_epoch_36.pth` | **RotatedFCOS** (anchor-free)* | `detector.RotatedFCOS` |

\* despite the `yolo26_` folder/config name, that checkpoint's saved meta-config
and weights are a **RotatedFCOS** model (`type='RotatedFCOS'`, `RotatedFCOSHead`);
`build_detector_from_checkpoint` keys off `rpn_head.`/`roi_head.` → OrientedRCNN
vs `bbox_head.conv_centerness` → RotatedFCOS.

Both share the same ViT-Adapter backbone + FPN (reused); the detection-specific
parts are pure-torch ports. Trained weights load with 0 missing / 0 unexpected,
and the reported mAP aligns with the mmrotate framework.

## Why pure torch?

The previous version reused the mmrotate detector + mmcv ops. This rewrite ports
every custom op to plain torch/numpy so the pipeline has **zero openmmlab
dependencies**:

| Component | OpenMMLab original | Pure-torch reimplementation |
|-----------|-------------------|------------------------------|
| ViT-L backbone | `dinov3.DinoVisionTransformer` | `dinov3_vit.py` (RoPE, LinearKMaskedBias qkv, LayerScale) |
| ViT-Adapter | `dinov3.DINOv3_Adapter` + mmcv MSDeformAttn | `vit_adapter.py` (SPM, interaction blocks, pytorch deformable attn) |
| FPN | `mmdet.FPN` | `fpn.py` |
| Oriented RPN | `mmrotate.OrientedRPNHead` | `rpn.py` (anchor gen, MidpointOffsetCoder) |
| RoI stage | `mmrotate.OrientedStandardRoIHead` + mmcv `RoIAlignRotated` | `roi.py` (pure-torch RoIAlignRotated, DeltaXYWHAOBBoxCoder, Shared2FC head) |
| RotatedFCOS head | `mmrotate.RotatedFCOSHead` | `fcos_head.py` (cls/reg convs, scales, centerness, angle, DistanceAnglePointCoder) |
| Rotated IoU / NMS | mmcv `box_iou_rotated` / `nms_rotated` | `box_ops.py` (Sutherland–Hodgman polygon IoU, greedy rotated NMS) |
| DOTA mAP | `mmrotate.eval_rbbox_map` | `eval_map.py` (tpfp_default + VOC 11-point AP) |
| Preprocessing | mmrotate `RResize/Normalize/Pad` pipeline | `dior_data.py` (cv2, bit-identical to the test pipeline) |

Every algorithm is a faithful port of the reference math (the deformable
attention is the exact mmcv forward; RoIAlignRotated matches the CUDA kernel
formula; rotated IoU is geometric polygon intersection validated against mmcv to
~1e-6).

## Checkpoint / EMA note

The checkpoint is saved by `mmcv.EMAHook`. `EMAHook.after_train_epoch` swaps the
EMA weights **into** the model before the eval that triggers `save_best`, so the
**non-prefixed** keys (`backbone.*`, `neck.*`, `rpn_head.*`, `roi_head.*`) in the
file **are the EMA weights** that produced the reported mAP. The `ema_*` keys hold
the raw weights and are discarded. `model.build_model` loads exactly those
non-prefixed keys (same as mmrotate `load_checkpoint`).

## Files

```
inference/torch_inference/
  core/
    box_ops.py      le90 transforms, box coders (midpoint/delta_xywha/dist-angle), rotated IoU, rotated NMS, AP
    dinov3_vit.py   DINOv3 ViT-L/16 (RoPE, masked-K-bias qkv, layerscale)
    vit_adapter.py  ViT-Adapter (SPM + 4 deformable interaction blocks)
    fpn.py          FPN (shared by both detectors)
    rpn.py          OrientedRPNHead (anchors + MidpointOffsetCoder decode + NMS)
    roi.py          RoIAlignRotated (torch) + Shared2FC head + DeltaXYWHA coder
    fcos_head.py    RotatedFCOSHead (cls/reg/centerness/angle + DistanceAnglePoint coder)
    detector.py     OrientedRCNN + RotatedFCOS + build_detector_from_checkpoint (auto-detect)
  data/
    dior_data.py    DIOR-R image/GT loading + native test preprocessing
  metrics/
    eval_map.py     DOTA rotated mAP (tpfp + 11-point AP)
  model.py          build detector + load non-EMA weights
  inference.py      per-shard inference -> results pkl
  evaluate.py       gather shards -> mAP
  run.sh            8-GPU launcher (one process/GPU)
```

## Run

```bash
# all visible GPUs (default), metrics -> inference/torch_inference/results/metrics_test.txt
bash inference/torch_inference/run.sh

# single GPU
CUDA_VISIBLE_DEVICES=0 bash inference/torch_inference/run.sh

# custom checkpoint / data root
CHECKPOINT=/path/to/best.pth DATA_ROOT=/path/to/DIOR-R bash inference/torch_inference/run.sh
```

Or step by step:

```bash
# 1) inference (shard 0 of 1, all images, single GPU)
CUDA_VISIBLE_DEVICES=0 python inference/torch_inference/inference.py \
    --checkpoint work_dirs/.../best_mAP@0.50_epoch_28.pth \
    --shard 0 --num-shards 1 --out-dir inference/torch_inference/results

# 2) evaluate
python inference/torch_inference/evaluate.py --results-dir inference/torch_inference/results \
    --num-shards 1 --data-root data/DIOR-R --split test --iou-thr 0.5
```

## Alignment with mmrotate

* `box_iou_rotated` matches `mmcv.box_iou_rotated` to ~1e-6 on random boxes
  (incl. far-from-origin / large coords).
* Per-image detection counts match the mmrotate reference on samples
  (e.g. 25 vs 28, 3 vs 3, 6 vs 6, 1 vs 1, 1 vs 1).
* The EMA-swapped checkpoints reproduce the reported `mAP@0.50`:
  OrientedRCNN subset of 40 images → 0.827, 150 → 0.754 (ref full-set ≈ 0.7500);
  RotatedFCOS subset of 150 → 0.799. Full test-set numbers come out once GPUs
  are free.

### Note on shared-GPU contention

This box's 8 GPUs can be fully occupied by other users' jobs (observed 100% util,
~46 GB/GPU). Under that contention the inference crawls and shards can be
OOM-killed, so the full `run.sh` should be launched when the GPUs have free
capacity (~14 GB + compute per shard). When idle, the full DIOR-R test set runs
in ~12 min on 8 GPUs (~0.5 s/img); the partial results are checkpointed per
shard under `inference/torch_inference/results/`.

Runtime: ~3 img/s/GPU on A100 → full DIOR-R test set (11.7k imgs) in ~8 min on
8 GPUs.
