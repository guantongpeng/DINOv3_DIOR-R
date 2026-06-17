# Oriented R-CNN + DINOv3 Backbone for DIOR-R Fine-tuning

> **更新说明 (2026-06)**：本文档最初描述基于 `timm` 的 `ViTDinoV3` 骨干（`out_indices=(3,5,7,11)`、`img_size=1024`）。
> 当前的 DIOR-R 配置已迁移到 **Meta 官方 DINOv3 封装**（`DinoVisionTransformerBackbone`）：
> - ViT-B 配置：`dinov3_vitb16`，`layers_to_use=[3,5,8,11]`，`frozen_stages=0`，输入 **800×800**（多尺度 600/800/1000）。
> - ViT-L 配置：`dinov3_vitl16`，`layers_to_use=[5,11,17,23]`，`frozen_stages=0`，输入 **800×800**。
>
> 下文 `img_size=1024`、`out_indices=(3,5,7,11)` 等描述对应历史 timm 配置；以下方当前配置文件为准：
> `configs/oriented_rcnn/oriented_rcnn_dinov3_vitb_fpn_dior.py`、`configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_dior.py`。

## 概览 | Overview

本项目基于 [MMRotate](https://github.com/open-mmlab/mmrotate) 框架，使用 **DINOv3** (Meta AI) 作为骨干网络，**Oriented R-CNN** 作为检测头，对 **DIOR-R** 遥感图像数据集进行旋转目标检测微调。

### 模型架构

```
输入图像 (1024×1024)
    │
    ▼
┌─────────────────────────────┐
│  ViT-Base DINOv3 Backbone   │  ← 预训练权重 (全参数微调)
│  - patch_size=16            │
│  - embed_dim=768            │
│  - depth=12                 │
│  - drop_path=0.1            │
│  - 输出4层特征 (stride=16)   │
└──────────┬──────────────────┘
           │ 4× [B, 256, 64, 64]
           ▼
┌─────────────────────────────┐
│  ViTDetFPN Neck             │  ← 强特征金字塔
│  - stride 4, 8, 16, 32     │
│  - 渐进式上采样 + top-down  │
│  - SE 通道注意力            │
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
│  - 20类分类                 │
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
│       └── oriented_rcnn_dinov3_fpn_dior.py   # DIOR-R 训练配置
├── models/
│   ├── __init__.py
│   ├── backbones/
│   │   ├── __init__.py
│   │   └── vit_dinov3.py                      # DINOv3 ViT 骨干网络
│   ├── datasets/
│   │   ├── __init__.py
│   │   └── dior.py                            # DIOR-R 数据集类
│   └── necks/
│       ├── __init__.py
│       ├── simple_fpn.py                      # SimpleFPN (旧版 neck)
│       └── vitdet_fpn.py                      # ViTDetFPN (新版 neck, 推荐)
├── tools/
│   ├── train.py                               # 训练脚本
│   ├── test.py                                # 评估脚本
│   ├── dist_train.sh                          # 分布式训练脚本 (DIOR-R)
│   ├── dist_test.sh                           # 分布式测试脚本
│   └── yolo2dota.py                           # YOLO→DOTA 格式转换
├── data/
│   └── prepare_dior.py                        # DIOR-R 数据集准备
└── docs/
    └── oriented_rcnn_dinov3_dior.md           # 本文档
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

### DIOR-R 数据集简介

DIOR-R 是一个大规模遥感图像旋转目标检测基准数据集：
- **图像数量**: 23,463张
- **实例数量**: 192,472个旋转边界框
- **类别数量**: 20个
- **图像尺寸**: 800×800像素
- **标注格式**: DOTA格式 (四点坐标 + 类别)

### 20个目标类别

| 序号 | 类别名 (英文) | 中文 |
|------|-------------|------|
| 1 | airplane | 飞机 |
| 2 | airport | 机场 |
| 3 | baseballfield | 棒球场 |
| 4 | basketballcourt | 篮球场 |
| 5 | bridge | 桥梁 |
| 6 | chimney | 烟囱 |
| 7 | dam | 水坝 |
| 8 | Expressway-Service-area | 高速公路服务区 |
| 9 | Expressway-toll-station | 高速公路收费站 |
| 10 | golffield | 高尔夫球场 |
| 11 | groundtrackfield | 田径场 |
| 12 | harbor | 港口 |
| 13 | overpass | 立交桥 |
| 14 | ship | 船舶 |
| 15 | stadium | 体育场 |
| 16 | storagetank | 储罐 |
| 17 | tenniscourt | 网球场 |
| 18 | trainstation | 火车站 |
| 19 | vehicle | 车辆 |
| 20 | windmill | 风车 |

### 下载与解压

从以下渠道下载DIOR-R数据集：

1. **官方渠道** (推荐): https://gcheng-nwpu.github.io/
2. **OpenDataLab**: https://opendatalab.com/DIOR
3. **PapersWithCode**: https://paperswithcode.com/dataset/dior

下载后解压到 `data/DIOR-R/` 目录：
```bash
# 创建目录
mkdir -p data/DIOR-R

# 解压数据集
unzip DIOR-R.zip -d data/DIOR-R/
# 或
unzip DIOR.zip -d data/DIOR-R/
```

### 运行数据准备脚本

```bash
# 创建目录结构并验证数据
python data/prepare_dior.py --data_root ./data/DIOR-R

# 创建 train/val 分割
python data/prepare_dior.py --data_root ./data/DIOR-R --val_ratio 0.1 --seed 42
```

### 期望的目录结构

```
data/DIOR-R/
├── trainval/
│   ├── images/              # 训练+验证图像
│   │   ├── 00001.jpg
│   │   └── ...
│   └── labelTxt/            # DOTA格式标注
│       ├── 00001.txt
│       └── ...
├── test/
│   ├── images/              # 测试图像
│   │   ├── 00001.jpg
│   │   └── ...
│   └── labelTxt/            # 测试标注
│       ├── 00001.txt
│       └── ...
└── ImageSets/               # 训练/验证/测试划分
    ├── train.txt
    ├── val.txt
    └── test.txt
```

### DOTA标注格式

每行标注格式：
```
x1 y1 x2 y2 x3 y3 x4 y4 category difficult
```
- `(x1,y1) ... (x4,y4)`: 四个角点坐标 (顺时针，从左上角开始)
- `category`: 目标类别名
- `difficult`: 困难标记 (0或1)

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
| 预训练数据 | LVD-1689M | 大规模网页数据 |
| drop_path | 0.1 | 随机深度正则化 |

**输出配置**:
- `out_indices = (3, 5, 7, 11)`: 从第3、5、7、11个Transformer块提取特征
- `out_channels = 256`: 输出通道统一到256维
- `img_size = 1024`: 位置编码插值目标尺寸
- **全参数微调**: 所有 ViT 参数参与训练（无冻结层）

**Checkpoint 加载修复**:

官方 DINOv3 checkpoint 使用与 timm 不同的命名约定。`ViTDinoV3._remap_dinov3_official_to_timm()`
自动完成 key 映射：

| 官方 key | timm key |
|----------|----------|
| `blocks.X.ls1.gamma` | `blocks.X.gamma_1` |
| `blocks.X.ls2.gamma` | `blocks.X.gamma_2` |
| `storage_tokens` | `reg_token` |

加载日志应显示 `162/162 keys matched, 0 missing`。若显示缺失则表示映射未生效。

### ViTDetFPN 特征金字塔

ViTDetFPN 是专门为 ViT backbone 设计的强 FPN，相比 SimpleFPN 有以下改进：

| 特性 | SimpleFPN (旧) | ViTDetFPN (新) |
|------|---------------|---------------|
| 输出 stride | [8, 16, 32, 64] | **[4, 8, 16, 32]** |
| P0 分辨率 (1024²) | 128×128 | **256×256** (4×) |
| 上采样 | 单层 ConvTranspose2d | bilinear + 3×3 conv |
| 跨尺度融合 | 可选 (导致降点) | 强制 top-down |
| 通道注意力 | 无 | SE Block |
| ViT 适配 | 无 | Pre-norm 层 |

**处理流程**:
```
输入: 4个特征图 @ stride 16 (64×64)
    │              │              │              │
  f0(block3)   f1(block5)   f2(block7)   f3(block11)
    │              │              │              │
 lateral0     lateral1      lateral2      lateral3
    │              │              │              │
    │              │              │          downsample
    │              │              │              │
    │              │          upsample 2× <── P2 (s16)
    │              │              │
    │          upsample 2× <── SE + fusion
    │              │
upsample 2× <── SE + fusion
    │
 SE + fusion
    │
  P0 (s4)      P1 (s8)       P2 (s16)      P3 (s32)
```

### Oriented R-CNN 检测头

Oriented R-CNN 是专为旋转目标检测设计的二阶段检测器。

**第一阶段 - Oriented RPN**:
- 旋转锚点生成器 (3种宽高比)
- 中点偏移编码器 (MidpointOffsetCoder)
- 生成旋转区域提议

**第二阶段 - Oriented RoI Head**:
- RotatedRoIAlign: 旋转RoI特征提取
- 全连接分类头 (20类)
- DeltaXYWHAOBBoxCoder: 旋转边界框回归 (cx, cy, w, h, a)

### 训练策略

| 配置项 | 值 | 说明 |
|--------|-----|------|
| 优化器 | AdamW | lr=1e-4, weight_decay=0.05 |
| 学习率 | 1e-4 | 全参数统一（无层级衰减） |
| 学习率策略 | CosineAnnealing | 余弦退火, min_lr_ratio=1e-3 |
| Warmup | 500 iter | 线性预热 |
| 批次大小 | 4/GPU × 4 = 16 | 800×800 图像（vitb 配置 samples_per_gpu=16） |
| 训练轮数 | 300 (vitb/vitl) | 每 3 epoch 评估 |
| 输入分辨率 | 800×800 | 多尺度训练 [600, 800, 1000] |
| 数据增强 | RandomFlip + PhotoMetricDistortion | 多尺度 + 色彩抖动 |
| 测试增强 | 多尺度 [800, 1024] | |
| 混合精度 | fp16 (loss_scale=512) | 加速训练 |
| 梯度裁剪 | max_norm=35 | 稳定训练 |
| DropPath | 0.1 | backbone 正则化 |

## 使用方法 | Usage

### 1. 单GPU训练

```bash
conda activate mmdet

# 基础训练
python tools/train.py configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_dior.py

# 指定工作目录
python tools/train.py configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_dior.py \
    --work-dir work_dirs/my_experiment

# 从检查点恢复
python tools/train.py configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_dior.py \
    --resume-from work_dirs/oriented_rcnn_dinov3_fpn_dior/latest.pth
```

### 2. 多GPU分布式训练

```bash
# 4 GPU训练
bash tools/dist_train.sh configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_dior.py 4

# 8 GPU训练
bash tools/dist_train.sh configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_dior.py 8 \
    --work-dir work_dirs/my_experiment

# 指定GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_train.sh \
    configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_dior.py 4
```

### 3. 调优训练参数

```bash
# 覆盖学习率
python tools/train.py configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_dior.py \
    --cfg-options optimizer.lr=5e-5

# 覆盖批次大小
python tools/train.py configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_dior.py \
    --cfg-options data.samples_per_gpu=4

# 调整冻结层数
python tools/train.py configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_dior.py \
    --cfg-options "model.backbone.frozen_stages=4"
```

### 4. 模型评估

```bash
# 单GPU评估
python tools/test.py configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_dior.py \
    work_dirs/oriented_rcnn_dinov3_fpn_dior/epoch_36.pth \
    --eval mAP

# 多GPU评估
bash tools/dist_test.sh configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_dior.py \
    work_dirs/oriented_rcnn_dinov3_fpn_dior/epoch_36.pth 4 \
    --eval mAP

# 保存检测结果
python tools/test.py configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_dior.py \
    work_dirs/oriented_rcnn_dinov3_fpn_dior/epoch_36.pth \
    --out results.pkl --eval mAP

# 可视化检测结果
python tools/test.py configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_dior.py \
    work_dirs/oriented_rcnn_dinov3_fpn_dior/epoch_36.pth \
    --show --show-dir vis_results --show-score-thr 0.3
```

### 5. 模型推理

```python
import torch
from mmdet.apis import init_detector, inference_detector
from mmrotate.core import visualize

# 加载模型
config_file = 'configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_dior.py'
checkpoint_file = 'work_dirs/oriented_rcnn_dinov3_fpn_dior/epoch_36.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# 推理单张图片
img = 'data/DIOR-R/test/images/00001.jpg'
result = inference_detector(model, img)

# 可视化结果
visualize(img, result, out_file='result.jpg')
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

# 使用ViT-Large (更高精度，需要更多显存)
backbone=dict(
    type='ViTDinoV3',
    model_name='vit_large_patch16_dinov3',
    out_indices=(5, 11, 17, 23),  # 24层中提取
    out_channels=256,
    frozen_stages=16,
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
4. **延长训练**: `runner.max_epochs=72`
5. **EMA**: 添加指数移动平均hook

### 常见配置调整

```bash
# 显存不足? 使用以下配置
python tools/train.py configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_dior.py \
    --cfg-options \
        data.samples_per_gpu=1 \
        model.backbone.with_cp=True \
        model.neck.num_outs=3

# 精度不够? 使用以下配置
python tools/train.py configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_dior.py \
    --cfg-options \
        runner.max_epochs=72 \
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
FileNotFoundError: data/DIOR-R/trainval/images/
```
**解决方案**:
- 按照 "数据集准备" 章节重新组织数据
- 运行 `python data/prepare_dior.py` 验证

### 4. 预训练权重下载失败
```
Error loading pretrained weights
```
**解决方案**:
- 检查网络连接 (timm需要联网下载权重)
- 手动下载权重到 `~/.cache/torch/hub/checkpoints/`
- 或设置 `pretrained=False` 从零开始训练

### 5. 分布式训练端口冲突
```
Address already in use
```
**解决方案**:
```bash
# 使用不同端口
PORT=29501 bash tools/dist_train.sh ... 4
```

## 参考 | References

1. **DINOv3**: [Meta AI DINOv3](https://github.com/facebookresearch/dinov3)
   - Oquab, M., et al. "DINOv3: All are worth 1 word." arXiv 2025.

2. **Oriented R-CNN**: [Oriented R-CNN for Object Detection](https://openaccess.thecvf.com/content/ICCV2021/papers/Xie_Oriented_R-CNN_for_Object_Detection_ICCV_2021_paper.pdf)
   - Xie, X., et al. "Oriented R-CNN for Object Detection." ICCV 2021.

3. **MMRotate**: [OpenMMLab MMRotate](https://github.com/open-mmlab/mmrotate)
   - Zhou, Y., et al. "MMRotate: A Rotated Object Detection Benchmark using PyTorch." ACM MM 2022.

4. **DIOR-R**: [DIOR Dataset](https://gcheng-nwpu.github.io/)
   - Li, K., et al. "Object Detection in Optical Remote Sensing Images: A Survey and A New Benchmark." ISPRS 2020.

5. **ViTDet**: [Exploring Plain Vision Transformer Backbones for Object Detection](https://arxiv.org/abs/2203.16527)
   - Li, Y., et al. "Exploring Plain Vision Transformer Backbones for Object Detection." ECCV 2022.

## 许可 | License

本项目代码遵循 Apache 2.0 许可。使用的预训练模型和数据集遵循各自的许可条款。
