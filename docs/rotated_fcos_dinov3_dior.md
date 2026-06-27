# Rotated FCOS + DINOv3 ViT-Adapter for DIOR-R

> 相关代码：`configs/fcos/`（base / stage1 / stage2）
> 训练脚本：`scripts/fcos_vitl_adapter_trainval.sh`
> 依赖骨干：`models/backbones/dinov3_vit_adapter.py`（与 Oriented R-CNN / RVSA 共用）

## 概览

本配置把 **Rotated FCOS**（无锚框、逐像素回归）检测头接到与 Oriented R-CNN 完全相同的
**DINOv3 ViT-L/16 + ViT-Adapter + FPN** 骨干上，用于在固定骨干条件下**横向对比两阶段锚框检测器
（Oriented R-CNN）与单阶段无锚框检测器（FCOS）**。

```
输入图像 (800×800，所有尺度为 32 的倍数)
    │
    ▼
┌─────────────────────────────────────────┐
│  DINOv3 ViT-L/16 (冻结 ViT, bf16)       │  ← 自监督预训练
│  + ViT-Adapter                          │
│     · SPM（轻量 CNN 空间先验）            │
│     · 4× InteractionBlock (MSDeformAttn) │
│       交互 ViT 层 [5,11,17,23]           │
│  → 4 级金字塔 @ 1024 通道, stride [4,8,16,32]
└──────────┬──────────────────────────────┘
           │ 4× [B, 1024, Hi, Wi]
           ▼
┌─────────────────────────────────────────┐
│  FPN (lateral + top-down, +P5 stride-64)│  → 5 级输出 @ 256 通道
│  strides [4,8,16,32,64]                 │
└──────────┬──────────────────────────────┘
           │ 5× [B, 256, Hi, Wi]
           ▼
┌─────────────────────────────────────────┐
│  RotatedFCOSHead (无锚框, 逐像素)         │
│  · 每像素回归 l,t,r,b + 角度 + centerness │
│  · separate_angle=True (GIoU + L1 angle) │
│  · 20 类分类                              │
└──────────┬──────────────────────────────┘
           │
           ▼
     检测结果 (cx, cy, w, h, θ) le90
```

## 设计要点

| 组件 | 选择 | 说明 |
|------|------|------|
| Backbone | DINOv3 ViT-Adapter（ViT-L/16） | 与 Oriented R-CNN/RVSA 同一 backbone，保证骨干变量唯一 |
| 冻结 ViT 精度 | bfloat16（`bf16_vit=True`） | 官方 DINOv3 eval 配方，省激活显存；可训练部分保持 fp32 |
| Neck | `FPN`（mmdet 标准 5 级） | 在 1024 通道 adapter 金字塔上做 lateral+top-down，并 `add_extra_convs='on_output'` 补 P5（stride 64） |
| Head | `RotatedFCOSHead` | 无锚框；`separate_angle=True`：水平框用 GIoU + 角度用 L1（mmrotate 旋转 FCOS 标准配方） |
| 角度编码 | `DistanceAnglePointCoder`（le90） | 由 l/t/r/b + angle 解码为旋转框 |
| 训练策略 | 两阶段 | stage1 冻结 ViT 训 adapter+FPN+FCOS；stage2 解冻 ViT 端到端微调 |

### 为什么用 5 级 FPN（含 stride 64）

FCOS 按回归范围（`regress_ranges`）把不同尺寸目标分配到不同 FPN 层。本项目在 adapter 4 级金字塔
(stride 4/8/16/32) 之上补一层 P5 (stride 64)，得到 `[4,8,16,32,64]` 五级，与 head 的
`strides` / `regress_ranges` 一一对应，覆盖从超小目标到超大目标。

### `separate_angle=True`（分离角度）

mmrotate 旋转 FCOS 的标准做法：不直接回归 5 维旋转框，而是
- 水平分量 l/t/r/b 用 `DistancePointBBoxCoder` + **GIoU loss**；
- 角度分量单独用 **L1 loss**（`loss_angle`，权重 0.2）。
推理时再把两者解码为旋转框。这比直接回归 5 参数更稳，尤其对角度边界情况。

## 两阶段训练

| 阶段 | ViT 状态 | 学习率 | 典型轮数 | 说明 |
|------|----------|--------|---------|------|
| Stage 1 | 冻结（`freeze_vit=True`） | base 4e-4（batch 128 校准） | 36 | 只训练 SPM + 4 交互 block + FPN + FCOS 头 |
| Stage 2 | 解冻（`freeze_vit=False`） | base 2e-4，ViT `lr_mult=0.1`（→2e-5） | 48 | 端到端微调，加载 stage1 最优 checkpoint |

> 学习率按 **effective batch 128**（8 GPU × samples_per_gpu=16）校准。改 batch 需等比缩放 lr。

### 配置文件

| 文件 | 角色 |
|------|------|
| `_rotated_fcos_dinov3_vitl_adapter_base_trainval_dior.py` | 共享 base（不要直接训练） |
| `rotated_fcos_dinov3_vitl_adapter_stage1_trainval_dior.py` | Stage 1：冻结 ViT |
| `rotated_fcos_dinov3_vitl_adapter_stage2_trainval_dior.py` | Stage 2：端到端微调 |

## 数据划分（trainval 配方）

| 子集 | 来源 | 用途 |
|------|------|------|
| train | DIOR-R **train + val 合并**（~11.7k 张） | 训练 |
| val | DIOR-R **test** 划分 | 周期性评估 + `save_best` 模型选择 |
| test | DIOR-R **test** 划分 | `tools/test.py` 最终评估 |

> 此配方下 val == test，每次评估都跑完整测试集；用 `evaluation.interval` 控制评估频率。

## 训练 / 评估

### 一键两阶段训练

```bash
# stage1 -> stage2（自动加载 stage1 最优 checkpoint）
bash scripts/fcos_vitl_adapter_trainval.sh

# 只跑某一阶段
STAGE=1 bash scripts/fcos_vitl_adapter_trainval.sh
STAGE=2 STAGE1_CKPT=work_dirs/.../best_mAP_epoch_30.pth bash scripts/fcos_vitl_adapter_trainval.sh

# 从中断处恢复（需指明已存在的 run 目录）
STAGE=1 RESUME=1 WORK_DIR=work_dirs/.../rotated_fcos_dinov3_vitl_adapter_trainval_dior_<ts> \
    bash scripts/fcos_vitl_adapter_trainval.sh
```

常用环境变量覆盖：

| 变量 | 默认 | 说明 |
|------|------|------|
| `CUDA_VISIBLE_DEVICES` | 0,..,7 | GPU 列表（NUM_GPUS 自动推导） |
| `SAMPLES_PER_GPU` | 8 | ViT-L+Adapter 显存重，OOM 时调小 |
| `S1_EPOCHS` / `S2_EPOCHS` | 36 / 48 | 各阶段轮数 |
| `EVAL_INTERVAL` | 3 | 测试集评估间隔 |
| `S1_LR` / `S2_LR` | （config 内） | 覆盖 base lr（默认按 batch 128） |
| `WORK_DIR` | 时间戳目录 | 输出根目录（内含 `stage1/`、`stage2/`） |

### 手动单卡训练

```bash
# Stage 1
python tools/train.py configs/fcos/rotated_fcos_dinov3_vitl_adapter_stage1_trainval_dior.py

# Stage 2（手动指定 stage1 checkpoint）
python tools/train.py configs/fcos/rotated_fcos_dinov3_vitl_adapter_stage2_trainval_dior.py \
    --cfg-options load_from=work_dirs/.../stage1/best_mAP_epoch_30.pth
```

### 评估

```bash
# stage2 最优 checkpoint 在测试集上的最终评估（含 classwise AP）
CONFIG=configs/fcos/rotated_fcos_dinov3_vitl_adapter_stage2_trainval_dior.py \
TEST_CKPT=work_dirs/.../stage2/best_mAP_epoch_*.pth \
WORK_DIR=work_dirs/.../stage2 SAVE_VIS=0 bash scripts/test.sh
```

## 关键超参速查

| 参数 | 值 | 说明 |
|------|-----|------|
| `interaction_indexes` | `[5,11,17,23]` | ViT-L 每 quarter 末层 |
| `deform_num_heads` / `n_points` | 16 / 4 | 可变形注意力配置 |
| `strides`（head） | `[4,8,16,32,64]` | 必须与 FPN 输出一致 |
| `regress_ranges` | `((-1,64),(64,128),(128,256),(256,512),(512,1e8))` | 各 FPN 层目标尺寸区间 |
| `norm_on_bbox` / `centerness_on_reg` | True / True | FCOS 标准配方 |
| `loss_bbox` | GIoU (1.0) | 水平框 |
| `loss_angle` | L1 (0.2) | 角度 |
| `loss_centerness` | CE-sigmoid (1.0) | centerness |
| `EMAHook` momentum | 0.9998 | 短 schedule 下比 0.9999 更有效 |
| 梯度裁剪 | max_norm=10 | |
| 多进程方式 | spawn | CUDA 不支持 fork |

> **注意**：FCOS 是无锚框头，**不使用** `RegZeroInitHook`（该 hook 仅用于 RoI 回归 FC）；
> `train_cfg` 必须为 `None`（无 assigner/sampler/anchor）。

## 故障排除

| 现象 | 原因 | 解决 |
|------|------|------|
| `H,W must be divisible by 32` | 输入非 32 整除 | 训练尺度改 32 倍数；`Pad size_divisor=32` |
| CUDA OOM | ViT-L + adapter + 5 级 FPN 显存重 | `SAMPLES_PER_GPU=2`；`with_cp=True`；保持 `bf16_vit=True` |
| head `strides` 与 FPN `num_outs` 不匹配 | 改 FPN 级数时未同步 head | 两者必须一致（5 级） |
| `expected Float but found BFloat16` | ViT bf16 输出未上转 | adapter 内部已 `.float()`；自定义分支需手动上转 |

## 参考

- **Rotated FCOS**（mmrotate 实现）：分离角度 + centerness 的旋转无锚框检测器，`mmrotate.models.dense_heads.RotatedFCOSHead`
- **FCOS**：Tian et al., "FCOS: Fully Convolutional One-Stage Object Detection", ICCV 2019, arXiv:1904.01355
- **ViT-Adapter**：详见 [vit_adapter_explained.md](vit_adapter_explained.md)
- **DINOv3**：[Meta AI DINOv3](https://github.com/facebookresearch/dinov3)
- **横向对比**：固定本骨干，可改用 `configs/oriented_rcnn/oriented_rcnn_dinov3_vitl_adapter_stage{1,2}_trainval_dior.py` 跑两阶段锚框检测器对照
