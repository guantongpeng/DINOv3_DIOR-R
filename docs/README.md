# 文档索引 | Documentation Index

本目录收录 `mm_dino` 项目的全部说明文档。按主题分组，可直接点击跳转。

## 架构与组件

| 文档 | 内容 |
|------|------|
| [model_architecture.md](model_architecture.md) | 整体架构详解（Backbone / Neck / Head 数据流、各检测器概览、训练配置速查） |
| [vit_adapter_explained.md](vit_adapter_explained.md) | ViT-Adapter 原理、实现细节、三种 Neck 对比、早期训练下降根因诊断与修正建议 |

## 检测器详解

| 文档 | 检测器 | 配置目录 |
|------|--------|----------|
| [oriented_rcnn_dinov3_dior.md](oriented_rcnn_dinov3_dior.md) | Oriented R-CNN（ViT-B/L FPN / SimpleFPN / KFIoU / ViT-Adapter / Swin-L） | `configs/oriented_rcnn/` |
| [rotated_fcos_dinov3_dior.md](rotated_fcos_dinov3_dior.md) | Rotated FCOS（ViT-L + ViT-Adapter 两阶段） | `configs/fcos/` |
| [yolo26_detection_head.md](yolo26_detection_head.md) | YOLO26（O2M+O2O 双头，NMS-free） | `configs/yolo26/` |
| [yolo26_bugfix_inference_o2o.md](yolo26_bugfix_inference_o2o.md) | YOLO26 推理 O2O 阶段 bug 修复记录 | — |
| [orcnn_nan_debug.md](orcnn_nan_debug.md) | Oriented R-CNN 训练 NaN 调试记录 | — |

> RVSA、RoI Transformer 目前无独立文档，其配置见 `configs/rvsa/`、`configs/roi_trans/`，骨干/Neck 细节见
> [model_architecture.md](model_architecture.md) 与 [vit_adapter_explained.md](vit_adapter_explained.md)。

## DINOv3 骨干与权重

| 文档 | 内容 |
|------|------|
| [dinov3_local_checkpoint.md](dinov3_local_checkpoint.md) | 本地 checkpoint 加载机制（官方格式 → 内部封装） |
| [dinov3_weight_verification.md](dinov3_weight_verification.md) | 权重加载校验流程与脚本用法 |

## 数据与工具

| 文档 | 内容 |
|------|------|
| [custom_25class_dataset.md](custom_25class_dataset.md) | 25 类自定义数据集接入说明 |
| [yolo2dota_tool.md](yolo2dota_tool.md) | YOLO OBB → DOTA 标注格式转换工具 |

## 工程兼容性与变更记录

| 文档 | 内容 |
|------|------|
| [pytorch27_compatibility_fixes.md](pytorch27_compatibility_fixes.md) | PyTorch 2.7 / mmcv monkey-patch 兼容修复 |
| [config_fixes_2026_06_25.md](config_fixes_2026_06_25.md) | 配置修复记录（2026-06-25） |
| [code-review-2026-06-16.md](code-review-2026-06-16.md) | 代码审查记录（2026-06-16） |

---

> 项目总览与快速上手见根目录 [README.md](../README.md)。
