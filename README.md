# Oriented R-CNN + DINOv3 Backbone for DIOR-R Fine-tuning

基于 [MMRotate](https://github.com/open-mmlab/mmrotate) 框架，使用 **DINOv3** (Meta AI) 作为骨干网络，**Oriented R-CNN** 作为检测头，对 **DIOR-R** 遥感图像数据集进行旋转目标检测微调。

## 模型架构

```
输入 (800×800) → ViT-Base DINOv3 (1024px) → SimpleFPN → Oriented RPN → Oriented RoI Head → 旋转检测框
```

| 组件 | 配置 | 说明 |
|------|------|------|
| Backbone | ViT-Base DINOv3 | patch=16, embed=768, depth=12, frozen_stages=8, img_size=1024 |
| Neck | SimpleFPN | 从同分辨率 ViT 特征构建 stride [8,16,32,64] 金字塔 |
| RPN | OrientedRPNHead | 旋转锚点生成 + 中点偏移编码 |
| RoI Head | OrientedStandardRoIHead | RotatedRoIAlign + 旋转框回归 (20类) |

## 环境要求

- Python 3.12+
- PyTorch 2.7.1 (CUDA 12.8)
- MMCV 1.7.2 / MMRotate 0.3.4 / MMDetection 2.28.2
- timm >= 1.0

```bash
source /home/guantp/pro/olmoearth_pretrain/.venv/bin/activate
```

> **PyTorch 2.7 兼容性说明**：`tools/train.py` 内置了两个 monkey-patch 以适配 PyTorch 2.7+：
> 1. `_get_stream` 补丁 — 解决 mmcv 传递 int 给 `torch.device` 的问题
> 2. `_use_replicated_tensor_module` 补丁 — 解决 MMDistributedDataParallel 缺少新属性的问题
>
> 同时请确保 config 中 `mp_start_method = 'spawn'`（非 `fork`），因为 CUDA 不支持 fork。

## 项目结构

```
mm_dino/
├── configs/oriented_rcnn/
│   └── oriented_rcnn_dinov3_fpn_dior.py   # 训练配置
├── models/
│   ├── backbones/vit_dinov3.py            # DINOv3 ViT 骨干网络
│   └── necks/simple_fpn.py                # 多尺度特征金字塔
├── tools/
│   ├── train.py                           # 训练脚本 (含 PyTorch 2.7 兼容补丁)
│   ├── test.py                            # 评估脚本
│   ├── dist_train.sh                      # 分布式训练
│   └── dist_test.sh                       # 分布式测试
├── data/prepare_dior.py                   # 数据集准备
└── docs/oriented_rcnn_dinov3_dior.md      # 详细文档
```

## 快速开始

### 1. 准备数据集

从 [DIOR 官网](https://gcheng-nwpu.github.io/) 下载 DIOR-R 数据集，解压到 `data/DIOR-R/`：

```bash
python data/prepare_dior.py --data_root ./data/DIOR-R
```

期望目录结构：
```
data/DIOR-R/
├── train/
│   ├── images/          # 训练图像
│   └── labelTxt/        # DOTA 格式标注
├── val/
│   ├── images/          # 验证图像
│   └── labelTxt/        # 验证标注
├── test/
│   ├── images/          # 测试图像
│   └── labelTxt/        # 测试标注
└── ImageSets/           # train/val/test 划分
```

### 2. 训练

```bash
# 方式一：直接运行 dist_train.sh（使用 4,5,6,7 号 GPU，4 卡）
bash tools/dist_train.sh

# 方式二：单 GPU 训练
python tools/train.py configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_dior.py

# 从检查点恢复
python tools/train.py configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_dior.py \
    --resume-from work_dirs/oriented_rcnn_dinov3_fpn_dior/epoch_6.pth
```

### 3. 评估

```bash
python tools/test.py configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_dior.py \
    work_dirs/oriented_rcnn_dinov3_fpn_dior/epoch_6.pth --eval mAP
```

### 4. 训练参数调优

```bash
# 调整学习率
python tools/train.py ... --cfg-options optimizer.lr=5e-5

# 调整冻结层数
python tools/train.py ... --cfg-options "model.backbone.frozen_stages=4"

# 跳过验证加速训练
python tools/train.py ... --no-validate

# 显存不足时减小批次
python tools/train.py ... --cfg-options data.samples_per_gpu=4 data.workers_per_gpu=2
```

## DIOR-R 数据集 (20 类)

| airplane | airport | baseballfield | basketballcourt | bridge |
| chimney | dam | Expressway-Service-area | Expressway-toll-station | golffield |
| groundtrackfield | harbor | overpass | ship | stadium |
| storagetank | tenniscourt | trainstation | vehicle | windmill |

## DINOv3 模型变体

| 变体 | embed_dim | depth | 参数量 | 推荐 |
|------|-----------|-------|--------|------|
| `vit_small_patch16_dinov3` | 384 | 12 | 22M | 快速实验 |
| `vit_base_patch16_dinov3` | 768 | 12 | 86M | ⭐ 推荐 |
| `vit_large_patch16_dinov3` | 1024 | 24 | 304M | 高精度 |
| `vit_huge_plus_patch16_dinov3` | 1280 | 32 | 632M | 最佳精度 |

切换方式：修改 config 中 `model.backbone.model_name`。

## 训练配置

| 配置 | 值 | 说明 |
|------|-----|------|
| 优化器 | AdamW (lr=1e-4, weight_decay=0.05) | backbone lr_mult=0.1 |
| 学习率调度 | CosineAnnealing + 500 iter warmup | min_lr_ratio=1e-3 |
| 层级衰减 | backbone lr_mult=0.1 | 冻结前 8 层 |
| 批次大小 | 16/GPU × 4 GPU = 64 | workers_per_gpu=4 |
| 训练轮数 | 100 | evaluation interval=6 |
| 混合精度 | fp16 (loss_scale=512) | |
| 梯度裁剪 | max_norm=35 | |
| 多进程方式 | spawn | 必须用 spawn（CUDA 不支持 fork） |

## 参考

- [DINOv3](https://github.com/facebookresearch/dinov3) — Meta AI 自监督 ViT
- [Oriented R-CNN](https://arxiv.org/abs/2108.05699) — ICCV 2021
- [MMRotate](https://github.com/open-mmlab/mmrotate) — OpenMMLab 旋转目标检测
- [DIOR-R](https://gcheng-nwpu.github.io/) — 遥感旋转目标检测基准
- [ViTDet](https://arxiv.org/abs/2203.16527) — ViT 用于目标检测
