# Oriented R-CNN + DINOv3 Backbone for Star-1021+Extend3 Fine-tuning

## 概览 | Overview

本项目基于 [MMRotate](https://github.com/open-mmlab/mmrotate) 框架，使用 **DINOv3** (Meta AI) 作为骨干网络，**Oriented R-CNN** 作为检测头，对 **Star-1021+Extend3** 遥感图像数据集进行旋转目标检测微调。

### 模型架构

```
输入图像 (800×800)
    │
    ▼
┌─────────────────────────────┐
│  ViT-Base DINOv3 Backbone   │  ← 预训练权重 (frozen_stages=8)
│  - patch_size=16            │
│  - embed_dim=768            │
│  - depth=12                 │
│  - 输出4层特征 (stride=16)   │
└──────────┬──────────────────┘
           │ 4× [B, 256, 50, 50]
           ▼
┌─────────────────────────────┐
│  SimpleFPN Neck             │  ← 多尺度特征金字塔
│  - stride 8, 16, 32, 64    │
│  - 反卷积上采样 + 卷积下采样  │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  Oriented RPN Head          │  ← 旋转区域提议网络
│  - 生成旋转候选框             │
│  - NMS过滤                  │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  Oriented Standard RoI Head │  ← 旋转RoI检测头
│  - RotatedRoIAlign          │
│  - 旋转边界框回归 (5参数)     │
│  - 25类分类                 │
└──────────┬──────────────────┘
           │
           ▼
     检测结果 (cx, cy, w, h, a)
```

## 项目结构 | Project Structure

```
mm_dino/
├── configs/
│   └── oriented_rcnn/
│       └── oriented_rcnn_dinov3_fpn_star.py   # 训练配置
├── models/
│   ├── __init__.py
│   ├── backbones/
│   │   ├── __init__.py
│   │   └── vit_dinov3.py                      # DINOv3 ViT 骨干网络
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── dior.py                            # DIOR-R 数据集类
│   │   └── star.py                            # Star-1021+Extend3 数据集类
│   └── necks/
│       ├── __init__.py
│       └── simple_fpn.py                      # SimpleFPN 特征金字塔
├── tools/
│   ├── train.py                               # 训练脚本
│   ├── test.py                                # 评估脚本
│   ├── dist_train.sh                          # 分布式训练脚本 (DIOR-R)
│   ├── dist_train_star.sh                     # 分布式训练脚本 (Star-1021+Extend3)
│   ├── dist_test.sh                           # 分布式测试脚本
│   └── yolo2dota.py                           # YOLO→DOTA 格式转换
├── data/
│   ├── prepare_dior.py                        # DIOR-R 数据集准备
│   └── prepare_star.py                        # Star-1021+Extend3 数据集准备
└── docs/
    ├── oriented_rcnn_dinov3_dior.md           # DIOR-R 文档
    └── oriented_rcnn_dinov3_star.md           # 本文档
```

## 环境要求 | Requirements

### 已验证环境
- **Python**: 3.12+
- **PyTorch**: 2.7.1 (CUDA 12.8)
- **MMCV**: 1.7.2 / 2.1.0
- **MMDetection**: 2.28.2
- **MMRotate**: 0.3.4
- **timm**: 1.0.20

### 激活虚拟环境
```bash
conda activate mmdet
```

### 安装依赖
```bash
pip install timm>=0.9.0
# 其他依赖已包含在上述虚拟环境中
```

## 数据集准备 | Dataset Preparation

### Star-1021+Extend3 数据集简介

Star-1021+Extend3 是一个扩展的遥感图像旋转目标检测数据集：
- **类别数量**: 25个
- **图像格式**: TIFF 文件 (.tif)
- **标注格式**: YOLO OBB 格式 (四点归一化坐标 + class_id)
- **数据划分**: train / val / test

### 25个目标类别

| 序号 | 类别名 | 中文 |
|------|--------|------|
| 0 | 两栖攻击舰 | Amphibious Assault Ship |
| 1 | 侦察机 | Reconnaissance Aircraft |
| 2 | 加油机 | Tanker Aircraft |
| 3 | 反潜巡逻机 | Anti-submarine Patrol Aircraft |
| 4 | 商业客机 | Commercial Airliner |
| 5 | 坦克 | Tank |
| 6 | 导弹快艇 | Missile Boat |
| 7 | 巡洋舰 | Cruiser |
| 8 | 扫雷艇 | Minesweeper |
| 9 | 护卫舰 | Frigate |
| 10 | 机场 | Airport |
| 11 | 武装直升机 | Attack Helicopter |
| 12 | 民用客轮 | Civilian Passenger Ship |
| 13 | 登陆舰 | Landing Ship |
| 14 | 空天战斗机 | Aerospace Fighter |
| 15 | 航空母舰 | Aircraft Carrier |
| 16 | 补给舰 | Supply Ship |
| 17 | 装甲运输车 | Armored Transport Vehicle |
| 18 | 轰炸机 | Bomber |
| 19 | 运输机 | Transport Aircraft |
| 20 | 通用直升机 | Utility Helicopter |
| 21 | 重型运输车 | Heavy Transport Vehicle |
| 22 | 隐身战斗机 | Stealth Fighter |
| 23 | 预警机 | Early Warning Aircraft |
| 24 | 驱逐舰 | Destroyer |

### 数据转换

由于 Star-1021+Extend3 的标注格式为 YOLO OBB（归一化四点坐标），需要先转换为 DOTA 格式才能用于 MMRotate 训练。

```bash
# 转换 train/val/test 三个子集
python data/prepare_star.py \
    --data_root data/star-1021_1016+extend3 \
    --splits train val test
```

### 期望的目录结构

原始数据集 (只读):
```
/mnt/ht2-nas2/00-model/Datasets/star-1021_1016+extend3/
├── train/
│   ├── images/              # 训练图像 (.tif)
│   ├── labels/              # YOLO OBB 标注 (原始)
├── val/
│   ├── images/
│   ├── labels/
├── test/
│   ├── images/
│   ├── labels/
└── DIOR-obb_star_1021-extend3.yaml  # 原始 YOLO 配置文件
```

项目本地数据目录 (images/labels 由 symlink 映射):
```
data/star-1021_1016+extend3/
├── train/
│   ├── images/ → (symlink)   # 原始图像
│   ├── labels/ → (symlink)   # 原始 YOLO 标注
│   └── labelTxt/             # DOTA 标注 (由 prepare_star.py 生成)
├── val/
│   ├── images/ → (symlink)
│   ├── labels/ → (symlink)
│   └── labelTxt/
├── test/
│   ├── images/ → (symlink)
│   ├── labels/ → (symlink)
│   └── labelTxt/
└── classes.txt               # 类别列表 (由 prepare_star.py 生成)
```

### DOTA标注格式

每行标注格式：
```
x1 y1 x2 y2 x3 y3 x4 y4 category difficult
```
- `(x1,y1) ... (x4,y4)`: 四个角点坐标 (绝对像素坐标)
- `category`: 目标类别名（中文）
- `difficult`: 困难标记 (0 或 1)

### 备注：无需 train/val 划分

Star-1021+Extend3 数据集已经提供了 train/val/test 三个子集的划分，直接使用即可。

## 模型详情 | Model Details

### DINOv3 骨干网络 (ViTDinoV3)

DINOv3 是 Meta AI 提出的自监督视觉Transformer预训练模型。

**关键参数 (ViT-Base)**:
| 参数 | 值 | 说明 |
|------|-----|------|
| patch_size | 16 | 每个patch的大小 |
| embed_dim | 768 | Token嵌入维度 |
| depth | 12 | Transformer块数量 |
| num_heads | 12 | 注意力头数量 |
| 预训练数据 | LVD-142M | 大规模网页数据 |

**输出配置**:
- `out_indices = (3, 5, 7, 11)`: 从第3、5、7、11个Transformer块提取特征
- `out_channels = 256`: 输出通道统一到256维
- `frozen_stages = 8`: 冻结前8个Transformer块 (保留预训练知识)
- `img_size = 1024`: 位置编码插值目标尺寸

### SimpleFPN 特征金字塔

由于ViT输出所有特征图具有相同的空间分辨率 (stride=16)，使用SimpleFPN创建多尺度金字塔。

**处理流程**:
```
输入: 4个特征图 @ stride 16 (50×50)
    │
    ├─→ Level 0: 反卷积上采样 → stride 8  (100×100)
    ├─→ Level 1: 保留原分辨率 → stride 16 (50×50)
    ├─→ Level 2: stride-2 卷积 → stride 32 (25×25)
    └─→ Level 3: stride-2 卷积 → stride 64 (13×13)
```

### Oriented R-CNN 检测头

Oriented R-CNN 是专为旋转目标检测设计的二阶段检测器。

**第一阶段 - Oriented RPN**:
- 旋转锚点生成器 (3种宽高比)
- 中点偏移编码器 (MidpointOffsetCoder)
- 生成旋转区域提议

**第二阶段 - Oriented RoI Head**:
- RotatedRoIAlign: 旋转RoI特征提取
- 全连接分类头 (25类)
- DeltaXYWHAOBBoxCoder: 旋转边界框回归 (cx, cy, w, h, a)

### 训练策略

| 配置项 | 值 | 说明 |
|--------|-----|------|
| 优化器 | AdamW | 带权重衰减 |
| 学习率 | 1e-4 | 骨干网络使用0.1×倍率 |
| 学习率策略 | CosineAnnealing | 余弦退火 |
| Warmup | 150 iter | 线性预热 |
| 批次大小 | 16/GPU | 800×800图像 |
| 训练轮数 | 200 | - |
| 混合精度 | fp16 | 加速训练 |
| 梯度裁剪 | max_norm=35 | 稳定训练 |

## 使用方法 | Usage

### 1. 数据预处理

```bash
# 首先转换标注格式
python data/prepare_star.py \
    --data_root data/star-1021_1016+extend3
```

### 2. 单GPU训练

```bash
conda activate mmdet

# 基础训练
python tools/train.py configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_star.py

# 指定工作目录
python tools/train.py configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_star.py \
    --work-dir work_dirs/oriented_rcnn_dinov3_fpn_star

# 从检查点恢复
python tools/train.py configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_star.py \
    --resume-from work_dirs/oriented_rcnn_dinov3_fpn_star/latest.pth
```

### 3. 多GPU分布式训练

```bash
# 使用快捷脚本（4卡，GPU 4,5,6,7）
bash tools/dist_train_star.sh

# 通用方式：4 GPU训练
bash tools/dist_train.sh configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_star.py 4

# 8 GPU训练
bash tools/dist_train.sh configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_star.py 8 \
    --work-dir work_dirs/oriented_rcnn_dinov3_fpn_star

# 指定GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_train.sh \
    configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_star.py 4
```

### 4. 调优训练参数

```bash
# 覆盖学习率
python tools/train.py configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_star.py \
    --cfg-options optimizer.lr=5e-5

# 覆盖批次大小
python tools/train.py configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_star.py \
    --cfg-options data.samples_per_gpu=4

# 调整冻结层数
python tools/train.py configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_star.py \
    --cfg-options "model.backbone.frozen_stages=4"
```

### 5. 模型评估

```bash
# 单GPU评估 (默认 mAP_coco: mAP@50:95 + 10 个 IoU 阈值)
python tools/test.py configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_star.py \
    work_dirs/oriented_rcnn_dinov3_fpn_star/epoch_200.pth \
    --eval mAP_coco

# 多GPU评估
bash tools/dist_test.sh configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_star.py \
    work_dirs/oriented_rcnn_dinov3_fpn_star/epoch_200.pth 4 \
    --eval mAP_coco

# 使用其他评估模式
python tools/test.py configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_star.py \
    work_dirs/oriented_rcnn_dinov3_fpn_star/epoch_200.pth \
    --eval mAP_multi   # mAP@0.50 + mAP@0.75

# 保存检测结果
python tools/test.py configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_star.py \
    work_dirs/oriented_rcnn_dinov3_fpn_star/epoch_200.pth \
    --out results.pkl --eval mAP_coco
```

**支持的评估模式**:
| 模式 | 输出指标 | 说明 |
|------|---------|------|
| `mAP` | mAP@<iou_thr> | 单个 IoU 阈值 (默认 0.5) |
| `mAP_multi` | mAP@0.50, mAP@0.75 | 常用双阈值 |
| `mAP_coco` | mAP@50:95 + 10 个阈值 | COCO 风格多阈值平均 |

### 6. 模型推理

```python
import torch
from mmdet.apis import init_detector, inference_detector

# 加载模型
config_file = 'configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_star.py'
checkpoint_file = 'work_dirs/oriented_rcnn_dinov3_fpn_star/epoch_200.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# 推理单张图片
img = 'data/star-1021_1016+extend3/test/images/000000293.tif'
result = inference_detector(model, img)
```

## 预训练模型选择 | DINOv3 Model Variants

| 模型 | embed_dim | depth | 参数量 | 推荐场景 |
|------|-----------|-------|--------|---------|
| vit_small_patch16_dinov3 | 384 | 12 | 22M | 快速实验/资源受限 |
| vit_base_patch16_dinov3 | 768 | 12 | 86M | **推荐** (平衡性能与效率) |
| vit_large_patch16_dinov3 | 1024 | 24 | 304M | 追求更高精度 |
| vit_huge_plus_patch16_dinov3 | 1280 | 32 | 632M | 最佳精度 (需要高端GPU) |

### 切换模型变体

修改config中的backbone配置：
```python
# 使用ViT-Small (更快，精度略低)
backbone=dict(
    type='ViTDinoV3',
    model_name='vit_small_patch16_dinov3',  # 切换模型
    out_indices=(3, 5, 7, 11),
    out_channels=256,
    frozen_stages=6,  # 相应调整
    ...
)
```

## 性能优化 | Optimization Tips

### 显存优化
1. **减小批次大小**: `data.samples_per_gpu=1`
2. **使用梯度检查点**: `model.backbone.with_cp=True`
3. **减少FPN层数**: `model.neck.num_outs=3`
4. **使用ViT-Small**: 显存需求约减半

### 训练加速
1. **启用混合精度**: `fp16 = dict(loss_scale=512.0)` (默认已启用)
2. **增加workers**: `data.workers_per_gpu=8`
3. **使用更多GPU**: 线性扩展

### 精度优化
1. **减小冻结层数**: `frozen_stages=4` (更多层参与训练)
2. **使用更大backbone**: ViT-Large 或 ViT-Huge
3. **多尺度训练**: 添加 `img_scale=[(800,800), (1024,1024)]`
4. **延长训练**: `runner.max_epochs=300`
5. **EMA**: 添加指数移动平均hook

### 常见配置调整

```bash
# 显存不足? 使用以下配置
python tools/train.py configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_star.py \
    --cfg-options \
        data.samples_per_gpu=1 \
        model.backbone.with_cp=True \
        model.neck.num_outs=3

# 精度不够? 使用以下配置
python tools/train.py configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_star.py \
    --cfg-options \
        runner.max_epochs=300 \
        model.backbone.frozen_stages=4 \
        optimizer.lr=5e-5
```

## 故障排除 | Troubleshooting

### 1. CUDA内存不足 (OOM)
```
RuntimeError: CUDA out of memory
```
**解决方案**:
- 减小 `data.samples_per_gpu` 到 1
- 启用梯度检查点: `model.backbone.with_cp=True`
- 减小图像尺寸: `image_size = (600, 600)`
- 使用ViT-Small替代ViT-Base

### 2. timm导入错误
```
ImportError: timm is required for ViTDinoV3 backbone
```
**解决方案**:
```bash
pip install timm>=0.9.0
```

### 3. 数据路径错误
```
FileNotFoundError: .../labelTxt/...
```
**解决方案**:
- 确保已运行 `python data/prepare_star.py` 进行数据转换
- 检查 `data_root` 路径是否正确

### 4. TIFF 图片读取问题
```
[WARNING] Cannot read image
```
**解决方案**:
- 确保系统已安装 libtiff
- `apt-get install libtiff5-dev` (Ubuntu) 或 `yum install libtiff-devel` (CentOS)
- 如果 PIL 无法读取 .tif，检查 `img_ext` 配置是否正确

### 5. 分布式训练端口冲突
```
Address already in use
```
**解决方案**:
```bash
# 使用不同端口
PORT=29501 bash tools/dist_train.sh configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_star.py 4
```

## 参考 | References

1. **DINOv3**: [Meta AI DINOv3](https://github.com/facebookresearch/dinov3)
   - Oquab, M., et al. "DINOv3: All are worth 1 word." arXiv 2025.

2. **Oriented R-CNN**: [Oriented R-CNN for Object Detection](https://openaccess.thecvf.com/content/ICCV2021/papers/Xie_Oriented_R-CNN_for_Object_Detection_ICCV_2021_paper.pdf)
   - Xie, X., et al. "Oriented R-CNN for Object Detection." ICCV 2021.

3. **MMRotate**: [OpenMMLab MMRotate](https://github.com/open-mmlab/mmrotate)
   - Zhou, Y., et al. "MMRotate: A Rotated Object Detection Benchmark using PyTorch." ACM MM 2022.

4. **ViTDet**: [Exploring Plain Vision Transformer Backbones for Object Detection](https://arxiv.org/abs/2203.16527)
   - Li, Y., et al. "Exploring Plain Vision Transformer Backbones for Object Detection." ECCV 2022.

## 许可 | License

本项目代码遵循 Apache 2.0 许可。使用的预训练模型和数据集遵循各自的许可条款。