# YOLO26 旋转目标检测头适配 DIOR-R 数据集

## 概述

本项目将 YOLO26 的目标检测头适配到 DIOR-R 旋转框目标检测任务中，结合 DINOv3 骨干网络和 ViTDetFPN 特征金字塔，构建一个端到端的旋转目标检测器。

> 排查与修复记录见 [YOLO26 训练效果差的排查与修复](./yolo26_bugfix_inference_o2o.md)（NMS 推理 / O2O 初始化爆炸 / O2O 分支未启用 三个 Bug）。

## 模型架构

```
输入图像 (800×800)
    │
    ▼
DINOv3 ViT-Base 骨干网络 (官方 dinov3_vitb16 封装)
  - patch_size=16, embed_dim=768, depth=12
  - frozen_stages=0：全参数微调（非冻结前 8 层）
  - 输出特征索引: [3, 5, 8, 11]
  - 所有输出分辨率: stride=16 (50×50)
    │
    ▼
ViTDetFPN 特征金字塔
  - 将同分辨率ViT特征融合为多尺度金字塔
  - P0: stride=4  (200×200) - 上采样
  - P1: stride=8  (100×100) - 上采样
  - P2: stride=16 (50×50)   - 直通
  - P3: stride=32 (25×25)   - 下采样
  - 含 SE 通道注意力 + top-down 融合
    │
    ▼
YOLO26RotatedHead 检测头
  ├── O2M分支 (One-to-Many): 用于训练
  │   ├── cls_branch: 2×Conv(3x3,BN,SiLU) → Conv2d(1x1, nc)
  │   ├── reg_branch: 2×Conv(3x3,BN,SiLU) → Conv2d(1x1, 4)
  │   ├── angle_branch: 2×Conv(3x3,BN,SiLU) → Conv2d(1x1, 1)
  │   └── obj_branch: 2×Conv(3x3,BN,SiLU) → Conv2d(1x1, 1)
  │
  └── O2O分支 (One-to-One): 用于推理
      └── 相同的架构，独立参数
    │
    ▼
输出: (N, max_det, 6) = [x, y, w, h, angle, score]
```

## YOLO26 核心特性

### 1. 双头架构 (Dual-Head Architecture)

| 分支 | 用途 | 标签分配 | 后处理 |
|------|------|----------|--------|
| O2M (One-to-Many) | 训练 | Task-Aligned Assigner (TAL) | 需要 NMS |
| O2O (One-to-One) | 推理 | Hungarian Matching | 无需 NMS |

训练时两个分支同时运行，O2M 提供丰富的学习信号，O2O 学习产生干净、无重叠的预测。推理时仅使用 O2O 分支。

### 2. 无锚框设计 (Anchor-Free)

- 每个像素点直接预测目标的中心偏移、宽高和角度
- 无需手动设计锚框尺寸和比例
- 使用 FCOS 风格的 left/top/right/bottom 距离回归

### 3. 无 NMS 端到端推理 (NMS-Free)

- O2O 分支使用匈牙利匹配进行一对一标签分配
- 推理时只需置信度阈值 + Top-K 选择
- 消除了 NMS 的超参数调优和后处理延迟

### 4. 无 DFL (Distribution Focal Loss)

- 移除了 DFL 模块，回归头更轻量
- 直接预测 bbox 距离（l,t,r,b），不需要分布建模
- 减少了参数量和计算量

### 5. 角度编码

```
angle = (sigmoid(pred) - 0.25) × π
范围: [-π/4, 3π/4]
```

### 6. 渐进损失 (Progressive Loss)

```
Epoch  0-12:  O2O weight = 0    (仅 O2M)
Epoch 12-30:  O2O weight = 递增 (O2M + O2O)
Epoch 30-36:  O2O weight = 1.0  (O2M + O2O 等权重)
```

通过 `ProgressiveLossHook` 在每个 epoch 开始前更新权重。

### 7. 任务对齐标签分配 (Task-Aligned Label Assignment, TAL)

```
alignment = cls_score^α × IoU^β

其中 α=1.0, β=6.0 (IoU 权重更高)
```

选择 alignment 最高的 top-K 个锚点分配给每个 GT，解决冲突时取 alignment 最大的匹配。

## 文件结构

```
models/
├── __init__.py                          # 注册所有模型模块
├── backbones/
│   ├── dinov3_wrapper.py                # 官方 DINOv3 封装（YOLO26 配置使用）
│   └── vit_dinov3.py                    # timm 版 DINOv3 骨干
├── datasets/
│   └── dior.py                          # DIOR-R 数据集加载器
├── detectors/
│   ├── __init__.py
│   └── dinov3_yolo26.py                # DINOv3+YOLO26 完整检测器
├── heads/
│   ├── __init__.py
│   └── yolo26_rotated_head.py          # YOLO26 旋转检测头
├── hooks.py                             # 自定义训练钩子（ProgressiveLossHook）
├── necks/
│   └── vitdet_fpn.py                    # ViTDetFPN 特征金字塔（YOLO26 配置使用）
configs/
└── yolo26/
    └── yolo26_dinov3_fpn_dior.py        # DIOR-R 训练配置
```

## 训练

### 环境准备

```bash
source /home/guantp/pro/olmoearth_pretrain/.venv/bin/activate
```

### 单 GPU 训练

```bash
python tools/train.py configs/yolo26/yolo26_dinov3_fpn_dior.py \
    --work-dir work_dirs/yolo26_dinov3_fpn_dior
```

### 多 GPU 训练

```bash
bash scripts/dist_train.sh configs/yolo26/yolo26_dinov3_fpn_dior.py 4
```

### 从检查点恢复训练

```bash
python tools/train.py configs/yolo26/yolo26_dinov3_fpn_dior.py \
    --resume-from work_dirs/yolo26_dinov3_fpn_dior/latest.pth
```

## 推理

```bash
python tools/test.py configs/yolo26/yolo26_dinov3_fpn_dior.py \
    work_dirs/yolo26_dinov3_fpn_dior/best_mAP.pth \
    --eval mAP
```

## 配置说明

### 关键超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `num_classes` | 20 | DIOR-R 类别数 |
| `in_channels` | 256 | 输入特征通道数 |
| `feat_channels` | 128 | 检测头隐藏层通道数 |
| `stacked_convs` | 2 | 每个分支的卷积层数 |
| `strides` | [8, 16, 32, 64] | FPN 特征步长 |
| `tal_topk` | 13 | TAL 分配中每个 GT 的 top-K 锚点数 |
| `tal_alpha` | 1.0 | TAL 分类权重 |
| `tal_beta` | 2.0 | TAL IoU 权重（降低以提升训练初期稳定性） |
| `o2o_start_epoch` | 12 | 开始激活 O2O 分支的 epoch |
| `o2o_end_epoch` | 30 | O2O 分支达到全权重的 epoch |

### Loss 函数

| Loss | 类型 | 权重 | 说明 |
|------|------|------|------|
| `loss_cls` | FocalLoss (sigmoid) | 1.0 | 分类损失，平衡正负样本 |
| `loss_bbox` | RotatedIoULoss | 2.5 | 旋转框 IoU 损失（在 exp 解码后的框上计算） |
| `loss_angle` | SmoothL1Loss (β=0.05) | 1.0 | 解码后角度 vs GT 角度的回归损失 |
| `loss_obj` | BCEWithLogitsLoss | 1.0 | 二值目标性损失 (FG=1, BG=0) |

### BBox 解码

训练和推理使用一致的解码流程：

```python
# Step 1: exp() 确保距离为正
l, t, r, b = exp(bbox_pred_raw)

# Step 2: 从网格点 + 距离解码 (x, y, w, h)
x1, y1 = px - l, py - t     # 左上角
x2, y2 = px + r, py + b     # 右下角
cx, cy = (x1+x2)/2, (y1+y2)/2
w, h = (x2-x1), (y2-y1)

# Step 3: 角度解码
angle = (sigmoid(angle_raw) - 0.25) × π  # 范围 [-π/4, 3π/4]
```

## DIOR-R 数据集

- **图像**: 23,463 张遥感图像
- **实例**: 192,472 个旋转边界框
- **类别**: 20 个遥感目标类别
- **格式**: DOTA 格式 (8 个角点坐标 + 类别名)

### 20个类别

airplane, airport, baseballfield, basketballcourt, bridge, chimney, dam,
Expressway-Service-area, Expressway-toll-station, golffield, groundtrackfield,
harbor, overpass, ship, stadium, storagetank, tenniscourt, trainstation,
vehicle, windmill

### 数据预处理

1. `RResize`: 调整图像到 800×800
2. `RRandomFlip`: 随机翻转 (水平/垂直/对角各 25% 概率)
3. `Normalize`: 使用 ImageNet 均值和标准差
4. `Pad`: 填充到 32 的倍数

## 技术参考

- [YOLO26: Unified Real-Time End-to-End Vision Models](https://arxiv.org/abs/2606.03748)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [DINOv3](https://github.com/facebookresearch/dinov3)
- [mmrotate](https://github.com/open-mmlab/mmrotate)
- [DIOR-R Dataset](https://arxiv.org/abs/1909.00133)

## 与原始 Oriented R-CNN 配置的区别

| 特性 | Oriented R-CNN | YOLO26 |
|------|---------------|--------|
| 检测框架 | 两阶段 (RPN + ROI) | 单阶段 (Anchor-Free) |
| 锚框 | 需要手动设计 | 无需锚框 |
| 标签分配 | MaxIoU Assigner | Task-Aligned Assigner |
| NMS | 需要 (RPN + RCNN) | 无需 (O2O 分支) |
| DFL | 不使用 | 不使用 |
| 角度预测 | 在 ROI Head 中 | 直接在密集预测中 |
| 推理速度 | 较慢 (两阶段) | 较快 (单阶段, 无NMS) |
| 训练速度 | 较慢 | 较快 (无需 RPN 训练) |
