# DINOv3 骨干的遥感旋转目标检测 (mm_dino)

基于 [MMRotate](https://github.com/open-mmlab/mmrotate) 框架，以 **DINOv3** (Meta AI) ViT 为骨干网络，在统一的
`Backbone → Neck → Head` 范式下集成多种旋转目标检测头，支持多个遥感图像 OBB（有向边界框）检测数据集：

- **DIOR-R** (20 类)

## 支持的检测器与配置

同一套 DINOv3 特征可对接不同检测头，便于横向对比。骨干以 **DINOv3 ViT** 为主，另提供 **Swin-Large** 作为非自监督 ViT 的对照基线。

| 检测器 | 类型 | 骨干 / Neck | 配置 |
|--------|------|-------------|------|
| **Oriented R-CNN** | 两阶段（旋转 RPN + RoI） | ViT-L / ViTDetFPN | `configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_train_dior.py` |
| **Oriented R-CNN** | 两阶段 | ViT-B / ViTDetFPN | `configs/oriented_rcnn/oriented_rcnn_dinov3_vitb_fpn_train_dior.py` |
| **Oriented R-CNN** | 两阶段 | ViT-B / SimpleFeaturePyramid | `configs/oriented_rcnn/oriented_rcnn_dinov3_vitb_simplefpn_train_dior.py` |
| **Oriented R-CNN** | 两阶段 | ViT-B / **ViT-Adapter**（两阶段训练） | `configs/oriented_rcnn/oriented_rcnn_dinov3_vitb_adapter_stage{1,2}_trainval_dior.py` |
| **Oriented R-CNN** | 两阶段 | ViT-L / **ViT-Adapter**（两阶段训练） | `configs/oriented_rcnn/oriented_rcnn_dinov3_vitl_adapter_stage{1,2}_trainval_dior.py` |
| **Oriented R-CNN** | 两阶段 | **Swin-L** / FPN（全参微调，对照基线） | `configs/oriented_rcnn/oriented_rcnn_swin_large_trainval_dior.py` |
| **Oriented R-CNN** | 两阶段（KFIoU） | ViT-B / SimpleFeaturePyramid | `configs/oriented_rcnn/oriented_rcnn_dinov3_vitb_simplefpn_kfiou_train_dior.py` |
| **Rotated FCOS** | 单阶段无锚框（per-pixel l/t/r/b + 角度 + centerness） | ViT-L / **ViT-Adapter**（两阶段训练） | `configs/fcos/rotated_fcos_dinov3_vitl_adapter_stage{1,2}_trainval_dior.py` |
| **RoI Transformer** | 两阶段（水平 RPN + 旋转细化，KFIoU） | ViT-B / SimpleFeaturePyramid | `configs/roi_trans/roi_trans_dinov3_vitb_simplefpn_kfiou_train_dior.py` |
| **RVSA** | Transformer（VSA 注意力，集合预测） | ViT-L / ViTDetFPN | `configs/rvsa/rvsa_dinov3_vitl_trainval_dior.py` |
| **RVSA** | Transformer（VSA 注意力，集合预测） | ViT-L / **ViT-Adapter**（两阶段训练） | `configs/rvsa/rvsa_dinov3_vitl_adapter_stage{1,2}_trainval_dior.py` |
| **YOLO26** | 单阶段无锚框（O2M+O2O 双头，NMS-free） | ViT-B / ViTDetFPN | `configs/yolo26/yolo26_dinov3_fpn_train_dior.py` |

> **trainval 配置**：名称含 `_trainval` 的配置采用「train+val 合并训练 / test 划分用于验证与最终评估」配方，与同名 `_train` 配置（仅用 train 训练、val 验证）相对。
> 多数检测器同时提供 `_train` 与 `_trainval` 两个版本（如 Oriented R-CNN 的 ViT-B/L FPN、SimpleFPN），按需选用。

以最常用的 **Oriented R-CNN + ViTDetFPN** 为例：

```
输入 (800×800) → DINOv3 ViT → ViTDetFPN → Oriented RPN → Oriented RoI Head → 旋转检测框
```

| 组件 | 配置 | 说明 |
|------|------|------|
| Backbone | DINOv3 ViT-B/L | 官方 Meta 封装 `DinoVisionTransformerBackbone`；ViT-B 取 blocks [3,5,8,11]，ViT-L 取 [5,11,17,23] |
| 对照 Backbone | **Swin Transformer v1-L** | ImageNet-22k 预训练、window12/384，全参微调（`oriented_rcnn_swin_large_*`） |
| Neck | **ViTDetFPN**（默认） | 渐进式上采样 + top-down 融合 + SE 注意力, 输出 stride [4,8,16,32] |
| 备选 Neck | SimpleFeaturePyramid / SimpleFPN | 单层 ViT 特征 → 金字塔 / 4 层 ViT 特征 → [8,16,32,64] |
| 进阶 Neck | **ViT-Adapter** | 冻结 ViT + 空间先验 + 多尺度可变形注意力，多层 ViT 特征交互；显存/算力开销大 |
| 检测头 | OrientedRPN + OrientedStandardRoIHead / RotatedFCOS / YOLO26 / RVSA / RoITrans | 见上表 |

> **ViT-Adapter 说明**：用可变形注意力把冻结 ViT 的多层特征重建为多尺度金字塔，理论上对密集预测（检测/分割）最优，但训练成本高、对超参敏感。
> 详见 [docs/vit_adapter_explained.md](docs/vit_adapter_explained.md)（含原理、实现、**早期训练效果下降的根因诊断与修正建议**）。

## 环境要求

- Python 3.12+
- PyTorch 2.7.1 (CUDA 12.8)
- MMCV 1.7.2 / MMRotate 0.3.4 / MMDetection 2.28.2
- timm >= 1.0

```bash
cd third_party/openmmlab/mmrotate
pip install -v -e . --no-build-isolation
```

> **PyTorch 2.7 兼容性说明**：`tools/train.py` 与 `tools/test.py` 内置了 monkey-patch 以适配 PyTorch 2.7+：
> 1. `_get_stream` / `Scatter.forward` 补丁 — 解决 mmcv 传递 int 给 `torch.device` 的问题
> 2. `_use_replicated_tensor_module` 补丁 — 解决 MMDistributedDataParallel 缺少新属性的问题
>
> 同时请确保 config 中 `mp_start_method = 'spawn'`（非 `fork`），因为 CUDA 不支持 fork。详见
> [docs/pytorch27_compatibility_fixes.md](docs/pytorch27_compatibility_fixes.md)。

## 项目结构

```
mm_dino/
├── configs/
│   ├── oriented_rcnn/                     # Oriented R-CNN 配置（ViT-B/L，多 Neck）
│   │   ├── oriented_rcnn_dinov3_fpn_train_dior.py          # ViT-L + ViTDetFPN
│   │   ├── oriented_rcnn_dinov3_vitb_fpn_train_dior.py     # ViT-B + ViTDetFPN
│   │   ├── oriented_rcnn_dinov3_vitb_simplefpn_train_dior.py
│   │   ├── _oriented_rcnn_dinov3_vitb_adapter_base_trainval_dior.py   # ViT-Adapter 共享基座
│   │   ├── oriented_rcnn_dinov3_vitb_adapter_stage1_trainval_dior.py  # 两阶段 stage1（冻结 ViT）
│   │   ├── oriented_rcnn_dinov3_vitb_adapter_stage2_trainval_dior.py  # 两阶段 stage2（端到端微调）
│   │   ├── _oriented_rcnn_dinov3_vitl_adapter_base_trainval_dior.py   # ViT-L ViT-Adapter 共享基座
│   │   ├── oriented_rcnn_dinov3_vitl_adapter_stage{1,2}_trainval_dior.py  # ViT-L 两阶段训练
│   │   ├── oriented_rcnn_swin_large_trainval_dior.py   # Swin-Large 对照基线（全参微调）
│   │   └── oriented_rcnn_dinov3_vitb_simplefpn_kfiou_train_dior.py
│   ├── fcos/                              # Rotated FCOS 配置（ViT-L + ViT-Adapter）
│   │   ├── _rotated_fcos_dinov3_vitl_adapter_base_trainval_dior.py   # ViT-Adapter 共享基座
│   │   ├── rotated_fcos_dinov3_vitl_adapter_stage1_trainval_dior.py  # 两阶段 stage1（冻结 ViT）
│   │   └── rotated_fcos_dinov3_vitl_adapter_stage2_trainval_dior.py  # 两阶段 stage2（端到端微调）
│   ├── roi_trans/                         # RoI Transformer 配置
│   ├── rvsa/                              # RVSA 配置
│   │   ├── rvsa_dinov3_vitl_trainval_dior.py          # ViT-L + ViTDetFPN
│   │   ├── _rvsa_dinov3_vitl_adapter_base_trainval_dior.py   # ViT-Adapter 共享基座
│   │   ├── rvsa_dinov3_vitl_adapter_stage1_trainval_dior.py  # 两阶段 stage1（冻结 ViT）
│   │   └── rvsa_dinov3_vitl_adapter_stage2_trainval_dior.py  # 两阶段 stage2（端到端微调）
│   └── yolo26/                            # YOLO26 配置
├── models/
│   ├── backbones/
│   │   ├── dinov3_wrapper.py             # 官方 Meta DINOv3 封装（DIOR 配置使用）
│   │   ├── dinov3_vit_adapter.py         # ViT-Adapter backbone（可变形注意力多层融合）
│   │   └── vit_dinov3.py                 # timm 版 DINOv3 封装
│   ├── necks/
│   │   ├── vitdet_fpn.py                # ViTDetFPN（推荐）
│   │   ├── simple_feature_pyramid.py    # SimpleFeaturePyramid（ViTDet 标准配方）
│   │   ├── passthrough_neck.py          # PassthroughNeck（ViT-Adapter 配套透传 neck）
│   │   └── simple_fpn.py                # SimpleFPN
│   ├── heads/yolo26_rotated_head.py     # YOLO26 旋转检测头
│   ├── dense_heads/rvsa_head.py         # RVSA head
│   ├── detectors/
│   │   ├── dinov3_yolo26.py             # DINOv3 + YOLO26 检测器
│   │   └── rvsa.py                      # RVSA 检测器
│   ├── layers/                          # VSA 注意力 / Transformer
│   ├── datasets/
│   │   └── dior.py                      # DIOR-R 数据集
│   ├── pipelines/albu_metadata.py       # Albu 增强 pipeline
│   └── hooks.py                         # ProgressiveLossHook / RegZeroInitHook（回归头零初始化）
├── tools/                              # Python 工具脚本（训练/评估入口）
│   ├── train.py                          # 训练入口（含 PyTorch 2.7 兼容补丁）
│   ├── test.py                           # 评估入口
│   ├── plot_loss.py                      # 训练曲线绘制
│   ├── verify_dinov3_weights.py          # DINOv3 权重加载校验
│   ├── verify_adapter_alignment.py       # ViT-Adapter 空间对齐校验
│   └── yolo2dota.py                      # YOLO OBB → DOTA 标注转换
├── scripts/                            # 分布式训练/评估 shell 脚本（统一入口）
│   ├── orcnn_vitl_fpn_train.sh                     # Oriented R-CNN (ViT-L) 分布式训练
│   ├── orcnn_vitb_fpn_train.sh                # Oriented R-CNN (ViT-B) 分布式训练
│   ├── orcnn_vitb_adapter_trainval.sh    # ViT-Adapter 两阶段训练（ViT-B，stage1→stage2）
│   ├── orcnn_vitl_adapter_trainval.sh # ViT-Adapter 两阶段训练（ViT-L）
│   ├── fcos_vitl_adapter_trainval.sh  # Rotated FCOS + ViT-Adapter 两阶段训练（ViT-L）
│   ├── yolo26_vitb_train.sh           # YOLO26 (ViT-B) 分布式训练
│   ├── roitrans_vitb_train.sh            # RoI Transformer 分布式训练
│   ├── orcnn_vitb_kfiou_train.sh               # KFIoU 分布式训练
│   ├── orcnn_vitb_simplefpn_train.sh           # SimpleFPN 分布式训练
│   ├── orcnn_vitb_simplefpn_trainval.sh            # trainval 训练
│   ├── orcnn_vitb_fpn_trainval.sh   # ViT-B + ViTDetFPN trainval 训练
│   ├── orcnn_swin_large_trainval.sh # Swin-Large trainval 训练
│   ├── rvsa_vitl_adapter_trainval.sh # RVSA (ViT-L) + ViT-Adapter 两阶段训练
│   ├── test.sh                           # 分布式评估脚本
│   └── train_pipeline.sh                 # 一键串联多个 trainval 训练流水线
├── data/
│   ├── prepare_dior.py                   # DIOR-R 数据集准备
│   ├── convert_dior_xml_to_dota.py       # DIOR XML → DOTA 标注转换
│   └── weights/                          # 预训练权重（见下表）
├── docs/
│   ├── README.md                         # 文档索引
│   ├── model_architecture.md             # 整体架构详解
│   ├── vit_adapter_explained.md          # ViT-Adapter 原理/实现/早期训练下降诊断
│   ├── oriented_rcnn_dinov3_dior.md      # Oriented R-CNN (DIOR-R) 详细文档
│   ├── rotated_fcos_dinov3_dior.md       # Rotated FCOS (DIOR-R) 详细文档
│   ├── yolo26_detection_head.md          # YOLO26 检测头文档
│   ├── yolo26_bugfix_inference_o2o.md    # YOLO26 推理 O2O 修复记录
│   ├── dinov3_local_checkpoint.md        # 本地 checkpoint 加载说明
│   ├── dinov3_weight_verification.md     # 权重校验说明
│   ├── custom_25class_dataset.md         # 25 类自定义数据集说明
│   ├── yolo2dota_tool.md                 # YOLO→DOTA 工具说明
│   ├── pytorch27_compatibility_fixes.md  # PyTorch 2.7 兼容修复
│   ├── config_fixes_2026_06_25.md        # 配置修复记录
│   ├── orcnn_nan_debug.md                # Oriented R-CNN NaN 调试记录
│   └── code-review-2026-06-16.md         # 代码审查记录
└── third_party/                         # 第三方库（DINOv3 / OpenMMLab）
    ├── dinov3/                          # 官方 DINOv3 源码（backbone 经 DINOV3_SRC 引用）
    └── openmmlab/
        └── mmrotate/                    # MMRotate（editable 安装指向此目录）
```

## 快速开始

### 1. 准备预训练权重

将 DINOv3 官方权重放入 `data/weights/`：

| 权重文件 | 模型 | 预训练数据 | 用途 |
|----------|------|-----------|------|
| `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` | ViT-B/16 (768d, 12 blocks) | LVD-1689M | ViT-B 全部配置 |
| `dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth` | ViT-L/16 (1024d, 24 blocks) | SAT-493M | Oriented R-CNN / RVSA 非 Adapter 的 ViT-L 配置 |
| `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` | ViT-L/16 (1024d, 24 blocks) | LVD-1689M | ViT-Adapter 系列两阶段配置（orcnn/rvsa/fcos ViT-L Adapter） |
| `dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth` | ViT-H+/16 (1280d, 32 blocks) | LVD-1689M | 高精度实验（需手动配置） |

> Swin-Large 对照基线使用外部 ImageNet-22k 权重
> `swin_large_patch4_window12_384_22k.pth`（见 `oriented_rcnn_swin_large_trainval_dior.py` 顶部 `SWIN_PRETRAIN`）。

### 2. 准备数据集

#### DIOR-R

从 [DIOR 官网](https://gcheng-nwpu.github.io/) 下载，解压到 `data/DIOR-R/` 后转换：

```bash
python data/prepare_dior.py --data_root ./data/DIOR-R
```

期望目录结构：
```
data/DIOR-R/
├── train/{images,labelTxt}/
├── val/{images,labelTxt}/
├── test/{images,labelTxt}/
└── ImageSets/           # train/val/test 划分
```

### 3. 训练

```bash
# Oriented R-CNN (ViT-L, DIOR-R) — 6 GPU
bash scripts/orcnn_vitl_fpn_train.sh

# Oriented R-CNN (ViT-B, DIOR-R) — 6 GPU
bash scripts/orcnn_vitb_fpn_train.sh

# YOLO26 (ViT-B, DIOR-R) — 8 GPU
bash scripts/yolo26_vitb_train.sh

# RoI Transformer (ViT-B, DIOR-R)
bash scripts/roitrans_vitb_train.sh

# ViT-Adapter 两阶段（ViT-B, DIOR-R）：stage1 冻结 ViT 训 adapter → stage2 端到端微调
bash scripts/orcnn_vitb_adapter_trainval.sh
# 只跑某一阶段：STAGE=1 bash scripts/orcnn_vitb_adapter_trainval.sh

# ViT-Adapter 两阶段（ViT-L, DIOR-R）
bash scripts/orcnn_vitl_adapter_trainval.sh

# Rotated FCOS + ViT-Adapter 两阶段（ViT-L, DIOR-R）
bash scripts/fcos_vitl_adapter_trainval.sh

# RVSA + ViT-Adapter 两阶段（ViT-L, DIOR-R）
bash scripts/rvsa_vitl_adapter_trainval.sh

# Swin-Large 对照基线（trainval，全参微调）
bash scripts/orcnn_swin_large_trainval.sh

# 一键串联当前主推训练流水线（RVSA + FCOS 的 ViT-L Adapter 两阶段）
bash scripts/train_pipeline.sh

# Rotated FCOS + ViT-Adapter 两阶段（ViT-L, DIOR-R）：stage1 冻结 ViT → stage2 端到端微调
bash scripts/fcos_vitl_adapter_trainval.sh
# 只跑某一阶段：STAGE=1 bash scripts/fcos_vitl_adapter_trainval.sh

# 单 GPU
python tools/train.py configs/oriented_rcnn/oriented_rcnn_dinov3_vitb_fpn_train_dior.py

# 从检查点恢复
python tools/train.py configs/oriented_rcnn/oriented_rcnn_dinov3_vitb_fpn_train_dior.py \
    --resume-from work_dirs/.../latest.pth
```

### 4. 评估

```bash
python tools/test.py configs/oriented_rcnn/oriented_rcnn_dinov3_vitb_fpn_train_dior.py \
    work_dirs/.../best_mAP_epoch_*.pth --eval mAP
```

### 5. 常用调参

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

## 数据集类别

### DIOR-R 数据集 (20 类)

| airplane | airport | baseballfield | basketballcourt | bridge |
| chimney | dam | Expressway-Service-area | Expressway-toll-station | golffield |
| groundtrackfield | harbor | overpass | ship | stadium |
| storagetank | tenniscourt | trainstation | vehicle | windmill |

## DINOv3 骨干网络

| 变体 | 模型名 | embed_dim | depth | 参数量 | 抽取层 |
|------|--------|-----------|-------|--------|--------|
| ViT-S/16 | `dinov3_vits16` | 384 | 12 | 22M | — |
| ViT-B/16 ⭐ | `dinov3_vitb16` | 768 | 12 | 86M | [3,5,8,11] |
| ViT-L/16 | `dinov3_vitl16` | 1024 | 24 | 304M | [5,11,17,23] |
| ViT-H+/16 | `dinov3_vith16plus` | 1280 | 32 | 632M | [7,15,23,31]（需手动配置） |

切换方式：修改 config 中 `model.backbone.model_name` 与 `layers_to_use`（并匹配 Neck 的 `in_channels`）。

> **两套封装**：DIOR 配置使用 `models/backbones/dinov3_wrapper.py`（导入官方 `dinov3` 仓库）；
> `models/backbones/vit_dinov3.py` 为 timm 封装（支持官方 checkpoint key 自动重映射）。

## 训练配置（Oriented R-CNN / DIOR-R 参考）

| 配置 | 值 | 说明 |
|------|-----|------|
| 优化器 | AdamW (lr=1e-4, weight_decay=0.05) | 分组学习率（backbone lr_mult=0.25） |
| 学习率调度 | CosineAnnealing + 500 iter warmup | min_lr_ratio=1e-3 |
| 批次大小 | 16/GPU（ViT-L 配置 4/GPU） | workers_per_gpu=4 |
| 训练轮数 | 300 | evaluation interval=3 |
| 输入分辨率 | 800×800 (多尺度训练 600-1000) | ViT 特征 50×50 |
| 数据增强 | RandomFlip + PhotoMetricDistortion + Albu | 多尺度 + 色彩抖动 |
| 混合精度 | fp16 (loss_scale=512) | |
| 梯度裁剪 | max_norm=35 | |
| 多进程方式 | spawn | CUDA 不支持 fork |
| EMA | momentum=0.999 | |

## 参考

- [DINOv3](https://github.com/facebookresearch/dinov3) — Meta AI 自监督 ViT
- [Oriented R-CNN](https://arxiv.org/abs/2108.05699) — ICCV 2021
- [Rotated FCOS](https://arxiv.org/abs/2111.02104) — 无锚框旋转检测（分离角度 + centerness）
- [RoI Transformer](https://arxiv.org/abs/1812.00155) — CVPR 2019
- [RVSA](https://arxiv.org/abs/2211.06550) — 旋转多尺度注意力
- [YOLO26](https://arxiv.org/abs/2606.03748) — 无锚框旋转检测
- [Swin Transformer](https://arxiv.org/abs/2103.14030) — 对照基线骨干
- [ViT-Adapter](https://arxiv.org/abs/2205.08534) — ViT 密集预测适配器
- [MMRotate](https://github.com/open-mmlab/mmrotate) — OpenMMLab 旋转目标检测
- [DIOR-R](https://gcheng-nwpu.github.io/) — 遥感旋转目标检测基准
- [ViTDet](https://arxiv.org/abs/2203.16527) — ViT 用于目标检测
