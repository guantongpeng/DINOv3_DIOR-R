# 配置与代码修复报告（2026-06-25）

> **范围**：全项目 configs/ 配置修复 + backbone 代码修复 + 训练脚本修复
> **背景**：DINOv3 ViT-Adapter 方案训练效果明显弱于 vit-adapter 参考实现，本文档记录根因分析与全部修复。

---

## 目录

1. [根因分析：为什么效果比 vit-adapter 差](#1-根因分析为什么效果比-vit-adapter-差)
2. [修复一：全参微调失效（no_grad）](#2-修复一全参微调失效no_grad)
3. [修复二：FPN 替换 PassthroughNeck](#3-修复二fpn-替换-passthroughneck)
4. [修复三：旋转检测配置（NMS / 角度 / 编码器 / RoI）](#4-修复三旋转检测配置)
5. [修复四：训练超参（grad_clip / find_unused_parameters）](#5-修复四训练超参)
6. [修复五：数据增强（PolyRandomRotate）](#6-修复五数据增强)
7. [修复六：全量配置清理（移除 class_weight 等）](#7-修复六全量配置清理)
8. [修复七：DDP + gradient checkpointing 冲突](#8-修复七ddp--gradient-checkpointing-冲突)
9. [修复八：训练脚本（NUM_GPUS 自动推导 / NCCL）](#9-修复八训练脚本)
10. [修改文件清单](#10-修改文件清单)

---

## 1. 根因分析：为什么效果比 vit-adapter 差

将当前 `DINOv3ViTAdapter` + mmrotate 0.x 配置与参考实现 `vit-adapter/dinov3_dior_orcnn_cosine.py`（mmrotate 1.x）逐项对比后，定位到以下问题（按影响排序）：

| # | 问题 | 影响 |
|---|------|------|
| 1 | `DINOv3_Adapter.forward` 无条件用 `torch.no_grad()` 包裹 ViT，`finetune_vit` 属性从未被读取 | **"全参微调"完全失效**，stage2 与 stage1 实际等价 |
| 2 | Neck 用 `PassthroughNeck`（恒等）+ backbone 内嵌 1×1 Conv 1024→256 | 丢失多尺度融合，没有 P5，特征在融合前就降维 |
| 3 | 测试 rcnn NMS `iou_thr=0.5` | 旋转框 NMS 阈值过松，重复框未抑制 |
| 4 | RPN `target_stds=[1,1,1,1,1,1]`（角度 1.0） | 角度回归归一化过松，收敛慢 |
| 5 | bbox_coder 缺 `edge_swap=True, proj_xy=True` | 与标准 Oriented R-CNN 不一致 |
| 6 | `find_unused_parameters` 未启用（freeze_vit 场景） | DDP 可能静默跳过梯度同步 |
| 7 | `grad_clip max_norm=35` | 参考实现注释明确指出 35 会让 mAP→0.005 |
| 8 | 无旋转增强 | 旋转检测缺少 orientation invariance |
| 9 | RoI `out_size=14` / `roi_feat_size=14` | 与标准 Oriented R-CNN（7）不一致 |
| 10 | loss_cls 带 `class_weight` 手动类别权重 | 压低多数类 AP，拖累 macro-mAP |

---

## 2. 修复一：全参微调失效（no_grad）

### 问题

`DINOv3_Adapter.forward`（`third_party/dinov3/.../dinov3_adapter.py`）中：

```python
with torch.autocast("cuda", torch.bfloat16):
    with torch.no_grad():                                    # ← 无条件 no_grad
        all_layers = self.backbone.get_intermediate_layers(
            x, n=self.interaction_indexes, return_class_token=True
        )
```

Wrapper (`dinov3_vit_adapter.py:274`) 设置了 `self.adapter.finetune_vit = not freeze_vit`，但这个属性**从未被 forward 读取**。即使 `freeze_vit=False` + `requires_grad_(True)`，ViT 在 `no_grad` 上下文下不构建计算图，**收不到任何梯度**。

### 修复

`DINOv3_Adapter.__init__` 新增默认属性：
```python
self.finetune_vit = False
```

`forward` 改为条件执行：
```python
if self.finetune_vit:
    # 全参微调：fp32 + 构建计算图
    all_layers = self.backbone.get_intermediate_layers(
        x, n=self.interaction_indexes, return_class_token=True)
else:
    # 冻结特征提取：bf16 autocast + no_grad（官方 eval 配方）
    with torch.autocast("cuda", torch.bfloat16):
        with torch.no_grad():
            all_layers = self.backbone.get_intermediate_layers(
                x, n=self.interaction_indexes, return_class_token=True)
```

### 验证

ViT-B adapter `freeze_vit=False` 模式下，162/175 个 ViT 参数收到非零梯度（之前为 0）。

**文件**: `third_party/dinov3/dinov3/eval/segmentation/models/backbone/dinov3_adapter.py`

---

## 3. 修复二：FPN 替换 PassthroughNeck

### 问题

原配置：
```python
# backbone 内嵌 1×1 Conv embed_dim→256 + GroupNorm
# neck = PassthroughNeck（恒等）
```

adapter 输出的完整 embed_dim（ViT-L=1024, ViT-B=768）特征**在送入检测头之前就被降维到 256**，丢失了大量信息。同时没有 FPN 的 lateral + top-down 融合，也没有 stride-64 的 P5 层。

### 修复

backbone 不再做投影（`out_channels=None`），直接输出完整 embed_dim 特征：

```python
# models/backbones/dinov3_vit_adapter.py
def __init__(self, ..., out_channels: Optional[int] = None, ...):
    if out_channels is not None:
        self.out_proj = ...  # 仅 PassthroughNeck 旧配置需要
    else:
        self.out_proj = None  # 直接返回 embed_dim，给 FPN 融合
```

配置改用标准 FPN：

```python
neck=dict(
    type='FPN',
    in_channels=[1024, 1024, 1024, 1024],   # ViT-L; ViT-B 用 768
    out_channels=256,
    start_level=0,
    add_extra_convs='on_output',
    num_outs=5,                               # 多出 P5 (stride 64)
    relu_before_extra_convs=True,
),
```

RPN strides 同步更新为 `[4, 8, 16, 32, 64]`（5 级）。

### 验证

FPN 输出 5 个 level，256-d，stride [4,8,16,32,64]，RPN 正常生成 2000 proposals。

**文件**:
- `models/backbones/dinov3_vit_adapter.py`（`out_channels` 可选化）
- `configs/oriented_rcnn/_oriented_rcnn_dinov3_vitl_adapter_base_dior.py`
- `configs/oriented_rcnn/_oriented_rcnn_dinov3_vitb_adapter_base_dior.py`

> **注意**：ViTDetFPN / SimpleFPN 配置（非 adapter）的 neck **不修改**——它们本身就是合理的 ViTDet 多尺度方案，只有 adapter 配置需要替换 PassthroughNeck。

---

## 4. 修复三：旋转检测配置

### 4.1 测试 NMS

```python
# 修改前
nms=dict(type='nms', iou_thr=0.5)
# 修改后
nms=dict(iou_thr=0.1)
```

`RotatedShared2FCBBoxHead.get_bboxes` 内部已调用 `multiclass_nms_rotated`，该函数始终使用旋转 NMS（忽略 `type` 字段，仅读 `iou_thr`）。0.5 过松导致重复框保留，0.1 是标准 Oriented R-CNN 配方。

### 4.2 RPN 角度归一化

```python
# 修改前
target_stds=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
# 修改后
target_stds=[1.0, 1.0, 1.0, 1.0, 0.5, 0.5]
```

角度偏移量（后两项）的归一化系数从 1.0 降到 0.5，收紧角度回归的目标分布，加速角度收敛。与标准 mmrotate Oriented R-CNN 一致。

### 4.3 bbox 编码器

```python
# 修改前
bbox_coder=dict(type='DeltaXYWHAOBBoxCoder', angle_range='le90',
    target_means=[0,0,0,0,0], target_stds=[0.1,0.1,0.2,0.2,0.1])
# 修改后
bbox_coder=dict(type='DeltaXYWHAOBBoxCoder', angle_range='le90',
    norm_factor=None, edge_swap=True, proj_xy=True,
    target_means=[0,0,0,0,0], target_stds=[0.1,0.1,0.2,0.2,0.1])
```

- `edge_swap=True`：编码前自动将宽高对齐到角度范围
- `proj_xy=True`：回归目标做 xy 投影补偿
- 与标准 mmrotate 0.x `oriented_rcnn_r50_fpn_1x_dota_le90.py` 一致

### 4.4 RoI 特征尺寸

```python
# 修改前：out_size=14, roi_feat_size=14
# 修改后：out_size=7, roi_feat_size=7
```

标准 Oriented R-CNN 用 7×7 RoI 特征。14×14 不但偏离标准配方，在继承配置中（如 KFIoU）还导致 extractor 与 head 尺寸不匹配的潜在 bug。

### 4.5 移除 gpu_assign_thr

```python
# 修改前
assigner=dict(..., gpu_assign_thr=200)
# 修改后
assigner=dict(...)  # 移除该参数
```

`gpu_assign_thr=200` 强制当 GT 数 > 200 时切回 CPU assigner，在 DIOR-R 上不必要且拖慢训练。标准 Oriented R-CNN 不使用此参数。

**文件**: 全部 `configs/oriented_rcnn/*.py`（adapter base + ViTDetFPN + SimpleFPN）、`configs/oriented_rcnn/oriented_rcnn_swin_large_trainval_dior.py`

---

## 5. 修复四：训练超参

### 5.1 grad_clip max_norm 35 → 10

```python
# 修改前
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# 修改后
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
```

参考实现（`vit-adapter/dinov3_dior_orcnn_cosine.py`）注释明确记载：

> max_norm=35 曾把 epoch32 的爆炸梯度 (pre-clip 256) 仅削到 35，仍毁掉该轮模型使 mAP 跌到 0.005；降到 10 可在不动正常更新的前提下兜住尖峰。

正常训练 grad_norm 中位 ~1.0（p99 ~4），max_norm=10 不影响正常梯度更新，仅截断异常尖峰。

### 5.2 find_unused_parameters

`tools/train.py` 中 DDP `find_unused_parameters` 的自动判断逻辑原仅检查 `frozen_stages`：

```python
# 修改前
frozen_stages = cfg.model.get('backbone', {}).get('frozen_stages', -1)
if frozen_stages >= 0 or progressive_cfg is not None:
    cfg.find_unused_parameters = True
```

`DINOv3ViTAdapter` 用 `freeze_vit` 而非 `frozen_stages`，导致 stage1（冻结 ViT）不会启用 `find_unused_parameters`。

```python
# 修改后
backbone_cfg = cfg.model.get('backbone', {})
frozen_stages = backbone_cfg.get('frozen_stages', -1)
freeze_vit = backbone_cfg.get('freeze_vit', False)
if frozen_stages >= 0 or progressive_cfg is not None or freeze_vit:
    cfg.find_unused_parameters = True
```

- stage1 (`freeze_vit=True`) → `find_unused_parameters=True`
- stage2 (`freeze_vit=False`) → `find_unused_parameters=False`（全部参数参与梯度）

**文件**:
- `tools/train.py`
- 全部 `configs/oriented_rcnn/*stage*.py`（grad_clip）

---

## 6. 修复五：数据增强（PolyRandomRotate）

### 问题

原 adapter/FPS/SimpleFPN 训练管线缺少旋转增强。DIOR-R 图像是 north-up 的，模型容易过拟合绝对朝向。旋转增强（PolyRandomRotate）是 Oriented R-CNN 的标准配方，通常 +1~3 mAP。

### 修复

在 train_pipeline 的 RRandomFlip 后插入：

```python
dict(
    type='PolyRandomRotate',
    rotate_ratio=0.5,
    angles_range=180,
    auto_bound=False,
    version='le90',
),
```

同时旋转图像和 GT 框（polygon → 重新编码为 le90），保证标注一致性。

**文件**: 全部 adapter base config、ViTDetFPN config、SimpleFPN config、`yolo26_dinov3_fpn_dior.py`、`rvsa_dinov3_vitl_dior.py`（Swin-L 和 trainval_fpn 原本已有）

---

## 7. 修复六：全量配置清理

### 7.1 移除 class_weight（手动类别权重）

**问题**：3 个配置在 `loss_cls` 中设置了 `class_weight=[1.0, 0.25, 0.88, ...]`（21 个值，按逆频率平方根手动调参），将多数类（ship=0.12, vehicle=0.14, storagetank=0.15）的损失权重大幅压低。

**危害**：DIOR-R 用 macro-mAP（每类等权），压低多数类损失直接拉低它们的 AP，而它们在 macro-mAP 中与稀有类同等重要。

**修复**：移除 `class_weight` 字段，使用均匀交叉熵（uniform CE）。`SimpleFPN` 配置原本已移除。

**影响配置**:
- `oriented_rcnn_dinov3_vitb_fpn_dior.py`
- `oriented_rcnn_dinov3_fpn_dior.py`
- `oriented_rcnn_swin_large_trainval_dior.py`

### 7.2 统一修复汇总

以下修复应用到全部适用的配置文件（14 个 config，含继承）：

| 修复项 | 说明 |
|--------|------|
| 移除 `class_weight` | 均匀 CE，不压低多数类 AP |
| `test nms iou_thr=0.1` | 旋转 NMS 标准阈值 |
| `target_stds=[1,1,1,1,0.5,0.5]` | RPN 角度归一化 |
| `bbox_coder` 加 `edge_swap/proj_xy` | 编码对齐标准 |
| `grad_clip=10` | 截断梯度尖峰 |
| 移除 `gpu_assign_thr` | 恢复 GPU assigner |
| `RoI out_size=7` | 对齐标准 7×7 |
| 加 `PolyRandomRotate` | 旋转增强 |

### 7.3 架构特殊处理

| 配置 | 特殊处理 |
|------|----------|
| RVSA（DETR 式） | 保留 `grad_clip=0.1`（集合预测需要极小裁剪），仅加旋转增强 |
| YOLO26 | 已有 `nms_rotated iou_thr=0.1`，仅修 grad_clip + 加旋转增强 |
| Swin-L | 原本已有正确 angle_stds / PolyRandomRotate / nms / out_size=7 |
| KFIoU / RoI Trans | 继承 SimpleFPN base，自动获得修复；额外修正 `roi_feat_size=14→7` 维度对齐 |

---

## 8. 修复七：DDP + gradient checkpointing 冲突

### 问题

运行 `dist_train_adapter_twostage_vitl.sh` 时报错：

```
RuntimeError: Expected to mark a variable ready only once.
Parameter at index 144 with name backbone.adapter.interactions.3.extra_extractors.1.ffn.fc2.bias
has been marked as ready twice.
```

**根因**：`DINOv3_Adapter` 的 `Extractor` 使用 `torch.utils.checkpoint.checkpoint()`（默认 `use_reentrant=True`，可重入检查点）。反向传播时检查点重新执行前向计算，导致 DDP 的梯度 hook 对同一参数触发两次。

参考 vit-adapter 实现中直接用 `with_cp=False` 规避（注释："Disable checkpointing for DDP compatibility"）。

### 修复

PyTorch 2.7.1 支持 `use_reentrant=False`（非可重入检查点），不会重复触发 DDP hook，**同时保留显存节省**：

```python
# Extractor.forward (dinov3_adapter.py:152)
query = cp.checkpoint(_inner_forward, query, feat, use_reentrant=False)

# SpatialPriorModule.forward (dinov3_adapter.py:299)
outs = cp.checkpoint(_inner_forward, x, use_reentrant=False)
```

### 验证

ViT-B adapter `with_cp=True` + `freeze_vit=True`：前向 + 反向正常，137/157 adapter 参数收到梯度，无 "marked ready twice" 错误。

**文件**: `third_party/dinov3/dinov3/eval/segmentation/models/backbone/dinov3_adapter.py`

---

## 9. 修复八：训练脚本

### 9.1 NUM_GPUS 自动推导

**问题**：所有脚本要求单独设置 `NUM_GPUS=N`，且需要手动保证与 `CUDA_VISIBLE_DEVICES` 列表一致。设错会导致 DDP 进程数不匹配。

**修复**：`NUM_GPUS` 从 `CUDA_VISIBLE_DEVICES` 自动推导：

```bash
# 修改前
NUM_GPUS=${NUM_GPUS:-8}
# ...后续还有 NGPU_LIST 对比 + warning 逻辑...

# 修改后
NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
```

移除了 NGPU_LIST sanity check 整段逻辑和所有 `# NUM_GPUS=N` 注释。现在只需设 `CUDA_VISIBLE_DEVICES=0,1,2,3` 即可自动用 4 卡。

**文件**: `scripts/` 下全部 11 个 `.sh` 脚本

### 9.2 NCCL 日志降噪

**问题**：NCCL 输出大量 `INFO` 级通信通道日志刷屏。

**修复**：训练脚本添加 `export NCCL_DEBUG=WARN`。

**文件**: `scripts/dist_train_adapter_twostage_vitl.sh`、`scripts/dist_train_adapter_twostage.sh`

---

## 10. 修改文件清单

### 代码文件（3 个）

| 文件 | 修改内容 |
|------|----------|
| `third_party/dinov3/.../dinov3_adapter.py` | `finetune_vit` 条件 no_grad；`use_reentrant=False` checkpoint |
| `models/backbones/dinov3_vit_adapter.py` | `out_channels` 可选化（None = 返回 embed_dim） |
| `tools/train.py` | `find_unused_parameters` 增加 `freeze_vit` 检测 |

### 配置文件（14 个）

| 文件 | 关键修改 |
|------|----------|
| `_oriented_rcnn_dinov3_vitl_adapter_base_dior.py` | FPN 替换 PassthroughNeck, NMS, angle_stds, bbox_coder, RoI=7, PolyRandomRotate |
| `_oriented_rcnn_dinov3_vitb_adapter_base_dior.py` | 同上（768-d） |
| `oriented_rcnn_dinov3_vitl_adapter_stage{1,2}_dior.py` | grad_clip 10 |
| `oriented_rcnn_dinov3_vitb_adapter_stage{1,2}_dior.py` | grad_clip 10 |
| `oriented_rcnn_dinov3_vitb_fpn_dior.py` | 移除 class_weight, NMS, angle_stds, bbox_coder, grad_clip, gpu_thr, RoI=7, PolyRandomRotate |
| `oriented_rcnn_dinov3_fpn_dior.py` | 同上（ViT-L） |
| `oriented_rcnn_dinov3_vitb_fpn_trainval_dior.py` | 继承 vitb_fpn，自动获得修复 |
| `oriented_rcnn_dinov3_vitb_simplefpn_dior.py` | NMS, angle_stds, bbox_coder, grad_clip, gpu_thr, RoI=7, PolyRandomRotate |
| `oriented_rcnn_dinov3_vitb_simplefpn_trainval_dior.py` | 同上 |
| `oriented_rcnn_dinov3_vitb_simplefpn_kfiou_dior.py` | roi_feat_size 14→7 |
| `oriented_rcnn_swin_large_trainval_dior.py` | 移除 class_weight, grad_clip, bbox_coder, gpu_thr |
| `roi_trans/roi_trans_dinov3_vitb_simplefpn_kfiou_dior.py` | 继承 simplefpn，自动获得修复 |
| `rvsa/rvsa_dinov3_vitl_dior.py` | 加 PolyRandomRotate（grad_clip=0.1 保留） |
| `yolo26/yolo26_dinov3_fpn_dior.py` | grad_clip 10, 加 PolyRandomRotate |

### 脚本文件（12 个）

| 文件 | 修改内容 |
|------|----------|
| `scripts/dist_train_adapter_twostage_vitl.sh` | NUM_GPUS 自动推导, NCCL_DEBUG=WARN |
| `scripts/dist_train_adapter_twostage.sh` | NUM_GPUS 自动推导, NCCL_DEBUG=WARN |
| `scripts/dist_train_*.sh`（其余 8 个） | NUM_GPUS 自动推导 |
| `scripts/test.sh` | NUM_GPUS 自动推导, 注释更新 |
