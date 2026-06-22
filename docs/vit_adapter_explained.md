# ViT-Adapter 详解：原理、实现与对下游任务的影响

> 适用范围：本项目的 DINOv3 ViT-B/16 + Oriented R-CNN (DIOR-R) 流程
> 相关代码：`models/backbones/dinov3_vit_adapter.py`
> 相关配置：`configs/oriented_rcnn/oriented_rcnn_dinov3_vitb_adapter_stage{1,2}_dior.py`

## 目录

1. [背景：为什么需要 ViT-Adapter](#1-背景为什么需要-vit-adapter)
2. [ViT-Adapter 是什么](#2-vit-adapter-是什么)
3. [架构总览](#3-架构总览)
4. [核心组件逐步拆解](#4-核心组件逐步拆解)
5. [三种 Neck 方案对比](#5-三种-neck-方案对比)
6. [使用 / 不使用 ViT-Adapter 对下游任务的影响](#6-使用--不使用-vit-adapter-对下游任务的影响)
7. [本项目实现的关键工程细节](#7-本项目实现的关键工程细节)
8. [针对 Oriented R-CNN + DIOR-R 的具体收益](#8-针对-oriented-r-cnn--dior-r-的具体收益)
9. [何时用、何时不用](#9-何时用何时不用)
10. [调参与排错速查](#10-调参与排错速查)

---

## 1. 背景：为什么需要 ViT-Adapter

### 1.1 ViT 与 CNN 在"密集预测"上的根本差异

目标检测、实例分割、深度估计等**密集预测（dense prediction）**任务需要一张**多尺度特征金字塔**：不同 stride（4/8/16/32）的特征图分别负责小、中、大物体。

| 特性 | CNN（ResNet/Swin） | Plain ViT（DINOv3） |
|------|-------------------|---------------------|
| 特征分辨率 | 天然多尺度：stride 4/8/16/32 | **单一尺度**：所有 block 输出都是 `H/patch_size × W/patch_size` |
| 空间归纳偏置 | 卷积自带局部性 | 全局自注意力，无显式局部先验 |
| 高分辨率细节 | 早期 stage 保留 | patch=16 直接下采样，**细粒度空间信息在 patch embed 阶段就丢失了** |
| 下游适配难度 | 直接接 FPN 即可 | 需要"桥接"机制把单尺度特征重建为多尺度 |

DINOv3 ViT-B/16：输入 800×800 → patch embed 后只剩 50×50 = 2500 个 token（stride 16）。**没有 stride 4/8 的细粒度特征图**，而遥感小目标（车辆、船只、网球场）恰恰依赖高分辨率细节。

### 1.2 朴素的桥接方法及其问题

最简单的做法（也是本项目早期 `SimpleFeaturePyramid` / `ViTDetFPN` 的思路）：

```
ViT 最后一层 (stride 16)  ──deconv 上采样──►  stride 4 / 8
                        ──conv 下采样──►  stride 32
```

**问题**：
- **只用最后一层**：丢失了浅层/中层 ViT block 的信息（ViT 各 block 编码不同抽象层次）。
- **用多层直接 concat 当多尺度**（旧 `ViTDetFPN` 把 block 3/5/8/11 的输出当 4 个尺度喂 FPN）：这些层**都在同一分辨率**，浅层 block 3 的特征上采样到 stride 4 时是**噪声**，会污染小目标检测。
- **纯卷积上采样**：无法让高分辨率特征"主动查询"ViT 的语义信息，是一种**单向、被动的信息流**。

ViT-Adapter 正是为解决这些问题而生。

---

## 2. ViT-Adapter 是什么

**ViT-Adapter**（Vision Transformer Adapter for Dense Predictations，Chen et al., ICLR 2023）是一个插在**冻结的预训练 ViT** 与**下游任务头**之间的**特征适配模块**。它：

1. **不修改、不微调**预训练 ViT（保持其强大的自监督表征）；
2. 引入一组**可训练的"空间先验查询"（spatial prior queries）**，通过**多尺度可变形注意力（MSDeformAttn）**主动从 ViT 的多层 patch token 中**采样**信息；
3. 把单尺度的 ViT 特征**重建为标准多尺度金字塔**（stride 4/8/16/32），无缝对接 FPN/RPN/RoI 等任何 CNN 时代的检测/分割头。

> 一句话：**ViT-Adapter = 冻结 ViT + 可学习的"探测器"，用可变形注意力把 ViT 语义拉到多尺度空间网格上。**

DINOv3 官方仓库的分割下游（`dinov3/eval/segmentation/models/backbone/dinov3_adapter.py`）正是用了这套结构并取得 SOTA。本项目把它移植到 Oriented R-CNN 检测流程。

---

## 3. 架构总览

```
                    Input Image (B,3,H,W)  [H,W 必须被 32 整除]
                          │
          ┌───────────────┼────────────────────────┐
          │               │                        │
          ▼               ▼                        ▼
   ┌──────────────┐  ┌──────────────┐      ┌──────────────────┐
   │  冻结 ViT    │  │   SPM        │      │  (无此分支)       │
   │ blocks 0..11 │  │ (轻量 CNN)   │      │                  │
   │ bf16, no_grad│  │ stride 4/8/  │      │                  │
   │              │  │ 16/32 先验   │      │                  │
   └──────┬───────┘  └──────┬───────┘      │                  │
          │                 │              │                  │
   取 block [2,5,8,11]   c1,c2,c3,c4       │                  │
   的 patch token       (空间先验查询)     │                  │
          │                 │              │                  │
          │      ┌──────────▼──────────┐   │                  │
          └─────►│  InteractionBlock ×4│◄──┘                  │
                 │  (MSDeformAttn)      │                     │
                 │  先验查询 ◄──采样─── ViT token             │
                 └──────────┬──────────┘                     │
                            │                                 │
              融合 ViT 多层特征 + SPM 先验，重建多尺度         │
                            │                                 │
                ┌───────────┼───────────┐                    │
                ▼           ▼           ▼                     │
            stride 4    stride 8   stride 16  stride 32       │
            (256ch)     (256ch)    (256ch)    (256ch)         │
                            │                                 │
                    PassthroughNeck (透传)                     │
                            │                                 │
                        RPN / RoI Head                         │
```

**核心数据流**：ViT 的多层 token 作为"知识库"，SPM 生成的空间先验作为"查询"，可变形注意力让查询主动从知识库采样，最终落到 4 级空间网格上。

---

## 4. 核心组件逐步拆解

对应代码 `models/backbones/dinov3_vit_adapter.py`。

### 4.1 Spatial Prior Module (SPM) —— 空间先验生成器

`SpatialPriorModule`：一个**轻量 CNN stem**（3 层 3×3 conv + MaxPool），直接吃**原始图像**，产生 4 个分辨率的初始查询：

| 输出 | stride | token 数（H_c=H/16） | 角色 |
|------|--------|----------------------|------|
| c1 | 4 | (保留为空间图) | 最细粒度先验 |
| c2 | 8 | 4·H_c·W_c | 中-细先验 |
| c3 | 16 | H_c·W_c | 中先验 |
| c4 | 32 | H_c·W_c/4 | 粗先验 |

每个先验经 1×1 conv 投影到 `embed_dim`（768）。c2/c3/c4 拼成查询序列 `c`（长度 = 5.25·H_c·W_c）。

> **为什么用图像而不是 ViT 特征生成先验？** SPM 的卷积提供了 ViT 所缺的**局部空间归纳偏置**，让后续可变形注意力的采样点有合理的初始分布。这是 ViT-Adapter 相对纯 ViT 特征上采样的关键增益来源。

### 4.2 多尺度可变形注意力 (MSDeformAttn) —— 信息采样核心

标准 ViT 自注意力是**全局、密集**的（O(N²)）。MSDeformAttn 让每个查询只在**少量可学习采样点**上聚合信息（O(N·K)），且这些采样点由查询内容自适应决定。

对每个查询 token：
1. `sampling_offsets = Linear(query)` → 预测 K 个采样点的偏移（相对参考点）
2. `attention_weights = Linear(query)` → 每个采样点的权重（softmax）
3. 在参考点 + 偏移处，用 `grid_sample` 双线性插值从 ViT 特征图采样
4. 加权求和 → 输出

```
查询 c (空间先验)          ViT patch 特征 (stride 16)
   ●━━━━━采样点(可学习偏移)━━━━▶ ◻
   │                              │
   └─────加权聚合─────────────────▶ 输出(更新后的先验)
```

参考点（reference points）覆盖 stride 8/16/32 三个分辨率的均匀网格，使粗细查询各司其职。

### 4.3 InteractionBlock —— 多层 ViT 特征交互

4 个 InteractionBlock，每个**吃一层不同的 ViT 输出**：

```
Block 0:  先验 c ◄──MSDeformAttn── ViT block 2 的 token
Block 1:  先验 c ◄──MSDeformAttn── ViT block 5 的 token
Block 2:  先验 c ◄──MSDeformAttn── ViT block 8 的 token
Block 3:  先验 c ◄──MSDeformAttn── ViT block 11 的 token  (+2 个额外 extractor)
```

每个 block 内部还接一个 ConvFFN（带深度可分离卷积的 FFN）做局部增强。**先验序列 c 在 4 个 block 间顺序传递**，逐层累积不同抽象层次的 ViT 信息。

> 这就是"**多层 ViT 特征交互**"的本质：不是把多层 concat 当输入，而是让同一组空间查询**逐层吸收**从浅到深的语义。

### 4.4 多尺度重建 —— 落到空间网格

交互后：
1. 把查询序列 `c` 拆回 c2/c3/c4，reshape 成空间图（stride 8/16/32）
2. c1 = `ConvTranspose(c2)` + SPM 的 c1 → stride 4
3. 把每个 InteractionBlock 对应的 ViT token reshape 成 (B,768,50,50)，**双线性插值**到对应分辨率后**相加**（残差融合 ViT 原始特征）
4. GroupNorm + 1×1 conv 投影到 256 通道

输出：4 级金字塔 `(B,256,H/4,W/4)`、`(B,256,H/8,W/8)`、`(B,256,H/16,W/16)`、`(B,256,H/32,W/32)`，直接喂 RPN/RoI。

---

## 5. 三种 Neck 方案对比

本项目历史上有三种把 DINOv3 ViT 特征转成金字塔的方案：

| 维度 | ViTDetFPN（旧，4层concat） | SimpleFeaturePyramid（ViTDet 配方） | **ViT-Adapter**（本方案） |
|------|---------------------------|-----------------------------------|--------------------------|
| ViT 层使用 | block 3/5/8/11 全用 | **仅最后 1 层** (block 11) | block 2/5/8/11，**逐层交互** |
| 上采样机制 | FPN lateral + top-down | deconv 反卷积金字塔 | **可变形注意力采样** + deconv |
| 空间先验 | 无 | 无 | **SPM（图像卷积）** |
| 信息流 | 被动（卷积） | 被动（卷积） | **主动（查询驱动采样）** |
| 局部归纳偏置 | 无（ViT 特征） | 无 | **有（SPM + DWConv）** |
| 参数量/显存 | 中 | 低 | **高**（多 4 组交互 block） |
| 小目标友好度 | 差（浅层噪声） | 中 | **好**（高分辨率先验 + 多层语义） |
| 训练稳定性 | 差（需手调 class weight） | 中 | **好**（先验引导） |

### 各方案的典型问题

- **ViTDetFPN**：把同分辨率的浅层 block 3 当 stride 4 用，注入噪声；项目早期出现 `val=0.72 / test=0.60` 的严重过拟合与此相关。
- **SimpleFPN**：只用最后一层，丢失中浅层信息；对 DIOR-R 的小目标（vehicle/ship/tenniscourt）召回有限。
- **ViT-Adapter**：用可变形注意力"按需"从多层取信息，既不引入浅层噪声，又保留多尺度语义——是目前 ViT dense prediction 的最强通用方案。

---

## 6. 使用 / 不使用 ViT-Adapter 对下游任务的影响

### 6.1 检测精度（mAP）

ViT-Adapter 在标准 benchmark 上的增益（来自论文与 DINOv2/v3 复现）：

| Backbone | Neck | COCO AP | 相对增益 |
|----------|------|---------|---------|
| ViT-L/16 | SimpleFPN | ~52 | baseline |
| ViT-L/16 | **ViT-Adapter** | **~57** | **+5 AP** |
| ViT-L/16 | ViT-Adapter + 更深交互 | ~59 | +7 AP |

增益主要来自**小目标（AP_S）和中目标（AP_M）**，大目标增益较小（大目标本就不依赖高分辨率）。

### 6.2 对各下游任务的具体影响

| 任务 | 不用 Adapter | 用 Adapter | 原因 |
|------|-------------|-----------|------|
| **目标检测** | 中等 | **显著提升** | 小目标依赖 stride 4/8 细节 |
| **实例/全景分割** | 差 | **大幅提升** | mask 需要像素级高分辨率特征 |
| **语义分割** | 中 | **显著提升** | 密集像素分类 |
| **深度估计** | 差 | **大幅提升** | 逐像素回归 |
| **图像分类** | 无影响 | 无影响 | 只用全局 CLS token，不需要金字塔 |

> **关键结论**：ViT-Adapter **只对密集预测任务有用**。纯分类任务（linear probe / kNN）不需要它——那些任务直接用 CLS token。

### 6.3 机制层面的影响（为什么有效）

1. **保留预训练表征**：ViT 全程冻结，自监督学到的通用特征不被小数据集（DIOR-R ~6k 张）破坏。
2. **按需采样 vs 被动上采样**：可变形注意力的采样点由内容驱动，能"看向"物体所在位置，比固定卷积核高效。
3. **多层语义融合**：浅层 block 提供纹理/边缘（利于小目标），深层 block 提供语义类别，逐层累积避免噪声。
4. **空间先验补偿归纳偏置缺失**：SPM 的卷积补回了 ViT 缺失的局部性，使早期训练更稳。

### 6.4 代价（不使用时的"优势"）

不用 ViT-Adapter 的唯一好处是**轻量**：
- **显存**：Adapter 在 800px 上，4 个交互 block 对 ~13000 个查询做可变形注意力，激活显存约为 SimpleFPN 的 2–3 倍。
- **速度**：训练 step 慢约 1.5–2×（纯 PyTorch deformable 实现，未编译 CUDA 时更慢）。
- **复杂度**：多了 SPM + 4 交互 block + ConvFFN 等组件，调参面更大。

---

## 7. 本项目实现的关键工程细节

移植自官方分割 adapter，为检测流程与鲁棒性做了如下改动（`dinov3_vit_adapter.py`）：

| 改动 | 原因 |
|------|------|
| **MSDeformAttn 用纯 PyTorch `grid_sample` 实现** | 官方需编译 CUDA 扩展（本环境未编译）；纯 PyTorch 版 autograd 自动处理反向，无需 `.so` |
| `nn.SyncBatchNorm` → `nn.GroupNorm` | SyncBN 在单卡/非 DDP 下报错；GroupNorm 全场景通用 |
| 冻结 ViT 跑 `torch.autocast(bfloat16)` + `no_grad` | 官方 eval 配方，大幅省激活显存 |
| ViT 输出 `.float()` 上转 fp32 | 避免与 fp32 交互层的 dtype 冲突 |
| `get_intermediate_layers(return_extra_tokens=False)` | 直接拿干净 patch token，**不用官方 `[:,5:]` 硬编码**剥离 CLS+4 register token |
| `with_cp=True` 梯度检查点 | 进一步省显存（牺牲约 30% 速度） |
| 注册为 **BACKBONE** + `PassthroughNeck` | adapter 已输出完整金字塔，neck 透传即可对接 Oriented R-CNN |

### 7.1 为什么做成 backbone 而非 neck

ViT-Adapter 的 SPM 需要**原始图像**（不是特征图），而 mmdet 的 neck 只收到特征。因此把整个 adapter 实现为 backbone（天然吃图像），输出 4 级金字塔，再用 `PassthroughNeck` 透传给 RPN——这是最忠实于原配方且不破坏 OrientedRCNN 数据流的方案。

### 7.2 与官方 DETR 检测 eval 的区别

DINOv3 官方检测 eval 用的是 **PlainDETR**（单阶段 transformer 检测器），其 backbone 也是冻结 ViT + 可选 transformer encoder。本项目是 **Oriented R-CNN（两阶段）**，故：
- 不用 DETR 的 Hungarian 匹配 / Focal Loss / 混合匹配（那是单阶段专属）；
- 用 ViT-Adapter 产出金字塔后，接标准 RPN + RoI（MaxIoUAssigner + CE + DeltaXYWH）。
- ViT-Adapter 扮演的角色等价于 DETR 路径里"冻结 ViT + 额外 encoder"，但更适合两阶段 CNN 检测头。

---

## 8. 针对 Oriented R-CNN + DIOR-R 的具体收益

DIOR-R 数据集特点：20 类、遥感图像、**类别极度不均衡**（ship 35k vs trainstation 509）、**大量小目标**（vehicle/ship/tenniscourt）、图像约 800×800。

ViT-Adapter 的预期收益：

1. **小目标 AP 提升**：stride 4/8 的先验 + 多层语义，直接利好 vehicle/ship/tenniscourt 这类小而密集的目标。
2. **缓解过拟合**：ViT 冻结 + adapter 可训练参数有限，相比全量微调 ViT 更不易在小数据集上过拟合（项目早期 `val/test gap` 问题的根因之一就是全量微调破坏预训练特征）。
3. **旋转框角度更稳**：高分辨率细节有助于 RoIAlign 提取更准确的目标朝向信息。
4. **配合两阶段训练**：Stage1 冻结 ViT 训 adapter（快速适配），Stage2 解冻 ViT 低 lr 微调（精细调整），是官方推荐且最稳的配方。

---

## 9. 何时用、何时不用

### 用 ViT-Adapter

- ✅ 密集预测任务（检测/分割/深度）
- ✅ 小目标多、需要高分辨率特征
- ✅ 数据集较小，怕毁掉预训练特征（冻结 ViT + adapter 更安全）
- ✅ 追求最高精度，显存/算力充裕

### 不用 ViT-Adapter（用 SimpleFPN 即可）

- ❌ 纯分类任务（linear probe / kNN）——直接用 CLS token
- ❌ 显存极度受限（< 24GB）
- ❌ 大目标为主、对高分辨率不敏感
- ❌ 需要最快迭代速度（实验阶段先 SimpleFPN 找方向，最后再上 Adapter 刷点）

### 实践建议

> **两阶段实验法**：先用 `SimpleFeaturePyramid` 配置快速跑通管线、调好数据增强/学习率；确认 baseline 后，再换 `ViT-Adapter` 做最终冲刺。这样避免一开始就在重模型上调参浪费时间。

---

## 10. 调参与排错速查

### 关键超参（`_oriented_rcnn_dinov3_vitb_adapter_base_dior.py`）

| 参数 | 默认 | 说明 |
|------|------|------|
| `interaction_indexes` | `[2,5,8,11]` | ViT-B/16 取每 quarter 末层；ViT-L 改 `[5,11,17,23]` |
| `deform_num_heads` | 16 | 可变形注意力头数（embed_dim 须可整除） |
| `n_points` | 4 | 每头每层采样点数，增大提精度但变慢 |
| `deform_ratio` | 0.5 | value 通道缩放比（省显存） |
| `cffn_ratio` | 0.25 | ConvFFN 隐藏维比例 |
| `drop_path_rate` | 0.3 | 交互 block 的 DropPath |
| `with_cp` | True | 梯度检查点（省显存，慢 30%） |
| `bf16_vit` | True | 冻结 ViT 跑 bf16（省显存） |

### 常见报错

| 报错 | 原因 | 解决 |
|------|------|------|
| `H,W must be divisible by 32` | 输入非 32 整除（adapter 需 stride-32 层） | 训练尺度改 32 倍数；`Pad size_divisor=32` |
| `expected Float but found BFloat16` | ViT bf16 输出未上转 | 已在代码用 `.float()` 修复；若自定义分支报错，手动 `.float()` |
| CUDA OOM | adapter 显存重 | `SAMPLES_PER_GPU=2`；`with_cp=True`；关 `bf16_vit` 改 fp32 反而更耗——保持 bf16 |
| `d_model not divisible by n_heads` | `deform_num_heads` 与 embed_dim 不整除 | ViT-B(768) 用 16；ViT-L(1024) 用 16；ViT-S(384) 用 8 |
| 训练极慢 | 纯 PyTorch deformable 无 CUDA 加速 | 编译官方 `MultiScaleDeformableAttention` CUDA op（可选，~2× 加速） |

### 如何切换回 SimpleFPN

如果显存不够或想快速对比，把 config 的 backbone+neck 换回：
```python
backbone=dict(type='DinoVisionTransformerBackbone', model_name='dinov3_vitb16',
              layers_to_use=[11], out_indices=(0,), use_layernorm=False, frozen_stages=-1, ...),
neck=dict(type='SimpleFeaturePyramid', in_channels=768, out_channels=256, num_outs=4, in_stride=16),
```
并恢复 `custom_imports` 里的 `models.backbones.dinov3_wrapper` 和 `models.necks.simple_feature_pyramid`。

---

## 参考资料

- **ViT-Adapter 论文**：Chen et al., "Vision Transformer Adapter for Dense Predictions", ICLR 2023, arXiv:2205.08534
- **ViTDet**（SimpleFPN 出处）：Li et al., "Exploring Plain Vision Backbones for Object Detection", ECCV 2022, arXiv:2203.16527
- **Deformable DETR**（MSDeformAttn 出处）：Zhu et al., ICLR 2021, arXiv:2010.04159
- **DINOv3 官方分割 adapter**：`dinov3/eval/segmentation/models/backbone/dinov3_adapter.py`
- **本项目实现**：`models/backbones/dinov3_vit_adapter.py`
