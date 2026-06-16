# Oriented R-CNN + DINOv3 Backbone for Remote Sensing OBB Detection

基于 [MMRotate](https://github.com/open-mmlab/mmrotate) 框架，使用 **DINOv3** (Meta AI) 作为骨干网络，**Oriented R-CNN** 作为检测头，支持多个遥感图像旋转目标检测数据集：

- **DIOR-R** (20 类)
- **Star-1021+Extend3** (25 类)

## 模型架构

```
输入 (1024×1024) → ViT-Base DINOv3 → ViTDetFPN → Oriented RPN → Oriented RoI Head → 旋转检测框
```

| 组件 | 配置 | 说明 |
|------|------|------|
| Backbone | ViT-Base DINOv3 | patch=16, embed=768, depth=12, img_size=1024, drop_path=0.1 |
| Neck | **ViTDetFPN** | 渐进式上采样 + top-down 融合 + SE 注意力, 输出 stride [4,8,16,32] |
| RPN | OrientedRPNHead | 旋转锚点生成 + 中点偏移编码 |
| RoI Head | OrientedStandardRoIHead | RotatedRoIAlign + 旋转框回归 (20/25类) |

> **2025-06-12 架构升级**: SimpleFPN → ViTDetFPN, 修复 DINOv3 checkpoint 权重加载(162/162 key 匹配),
> 输入分辨率 800→1024, 新增多尺度训练和 photometric 数据增强。

## 环境要求

- Python 3.12+
- PyTorch 2.7.1 (CUDA 12.8)
- MMCV 1.7.2 / MMRotate 0.3.4 / MMDetection 2.28.2
- timm >= 1.0

```bash
conda activate mmdet
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
│   ├── oriented_rcnn_dinov3_fpn_dior.py   # DIOR-R 训练配置
│   └── oriented_rcnn_dinov3_fpn_star.py   # Star-1021+Extend3 训练配置
├── models/
│   ├── backbones/vit_dinov3.py            # DINOv3 ViT 骨干网络
│   ├── datasets/
│   │   ├── dior.py                        # DIOR-R 数据集类
│   │   └── star.py                        # Star-1021+Extend3 数据集类
│   └── necks/
│       ├── simple_fpn.py                # SimpleFPN (旧版 neck)
│       └── vitdet_fpn.py                # ViTDetFPN (新版 neck, 推荐)
├── tools/
│   ├── train.py                           # 训练脚本 (含 PyTorch 2.7 兼容补丁)
│   ├── test.py                            # 评估脚本
│   ├── dist_train.sh                      # 分布式训练 (DIOR-R)
│   ├── dist_train_star.sh                 # 分布式训练 (Star-1021+Extend3)
│   └── dist_test.sh                       # 分布式测试
├── data/
│   ├── prepare_dior.py                    # DIOR-R 数据集准备
│   └── prepare_star.py                    # Star-1021+Extend3 数据集准备 (YOLO→DOTA)
└── docs/
    ├── oriented_rcnn_dinov3_dior.md       # DIOR-R 详细文档
    ├── oriented_rcnn_dinov3_star.md       # Star-1021+Extend3 详细文档
    └── dinov3_local_checkpoint.md         # 本地 checkpoint 加载说明
```

## 快速开始

### DIOR-R 数据集

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

### Star-1021+Extend3 数据集

Star-1021+Extend3 标注为 YOLO OBB 格式，需先转为 DOTA 格式。本地数据目录通过 symlink 映射原始数据集 (`/mnt/ht2-nas2/00-model/Datasets/star-1021_1016+extend3/`)：

```bash
python data/prepare_star.py \
    --data_root data/star-1021_1016+extend3
```



### 2. 训练

```bash
# DIOR-R — 多 GPU 分布式训练（使用 4,5,6,7 号 GPU）
bash tools/dist_train.sh

# Star-1021+Extend3 — 多 GPU 分布式训练（使用 4,5,6,7 号 GPU）
bash tools/dist_train_star.sh

# 单 GPU 训练
python tools/train.py configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_dior.py
python tools/train.py configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_star.py

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

## Star-1021+Extend3 数据集 (25 类)

| 两栖攻击舰 | 侦察机 | 加油机 | 反潜巡逻机 | 商业客机 |
| 坦克 | 导弹快艇 | 巡洋舰 | 扫雷艇 | 护卫舰 |
| 机场 | 武装直升机 | 民用客轮 | 登陆舰 | 空天战斗机 |
| 航空母舰 | 补给舰 | 装甲运输车 | 轰炸机 | 运输机 |
| 通用直升机 | 重型运输车 | 隐身战斗机 | 预警机 | 驱逐舰 |

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
| 优化器 | AdamW (lr=1e-4, weight_decay=0.05) | 全参数统一学习率 |
| 学习率调度 | CosineAnnealing + 500 iter warmup | min_lr_ratio=1e-3 |
| 批次大小 | 4/GPU × 4 GPU = 16 | workers_per_gpu=4 |
| 训练轮数 | 36 | evaluation interval=3 |
| 输入分辨率 | 1024×1024 (多尺度训练 800-1200) | ViT 特征 64×64 |
| 数据增强 | RandomFlip + PhotoMetricDistortion | 多尺度 + 色彩抖动 |
| 测试增强 | 多尺度 [800, 1024] | |
| 混合精度 | fp16 (loss_scale=512) | |
| 梯度裁剪 | max_norm=35 | |
| DropPath | 0.1 | backbone 正则化 |
| 多进程方式 | spawn | CUDA 不支持 fork |

## 参考

- [DINOv3](https://github.com/facebookresearch/dinov3) — Meta AI 自监督 ViT
- [Oriented R-CNN](https://arxiv.org/abs/2108.05699) — ICCV 2021
- [MMRotate](https://github.com/open-mmlab/mmrotate) — OpenMMLab 旋转目标检测
- [DIOR-R](https://gcheng-nwpu.github.io/) — 遥感旋转目标检测基准
- [ViTDet](https://arxiv.org/abs/2203.16527) — ViT 用于目标检测
