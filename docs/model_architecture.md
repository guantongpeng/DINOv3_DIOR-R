# 模型架构详解

## 目录

1. [整体架构概览](#1-整体架构概览)
2. [Backbone: ViT-DINOv3](#2-backbone-vit-dinov3)
3. [Neck: 特征金字塔网络](#3-neck-特征金字塔网络)
4. [RPN Head: 旋转区域提议网络](#4-rpn-head-旋转区域提议网络)
5. [RoI Head: 旋转RoI检测头](#5-roi-head-旋转roi检测头)
6. [训练策略与损失函数](#6-训练策略与损失函数)
7. [数据集](#7-数据集)
8. [训练配置速查](#8-训练配置速查)

---

## 1. 整体架构概览

本项目是一个基于 **DINOv3 ViT** 骨干网络的 **两阶段旋转目标检测** 框架，用于遥感图像中的有向边界框检测。

```
┌──────────────────────────────────────────────────────────────────┐
│                        Input Image                                │
│                  (3 × H × W, e.g. 1024×1024)                      │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                  ┌────────▼────────┐
                  │  ViT-DINOv3     │  Backbone (patch=16, depth=12)
                  │  ViT-Base/16    │  4 层特征 @ stride=16
                  │  Pretrained     │  embed_dim=768 → 投影至 256
                  └────────┬────────┘
                           │ 4 × [B, 256, H/16, W/16]
                  ┌────────▼────────┐
                  │  ViTDetFPN /    │  Neck (4-level pyramid)
                  │  SimpleFPN      │  ViTDetFPN: stride [4,8,16,32]
                  └────────┬────────┘  SimpleFPN: stride [8,16,32,64]
                           │ 4 × [B, 256, Hi, Wi]
                  ┌────────▼────────┐
                  │  Oriented RPN   │  Stage 1: 旋转区域提议
                  │  + AnchorGen    │  3 ratios × 1 scale × 4 levels
                  └────────┬────────┘
                           │ proposals (cx,cy,w,h,θ)
                  ┌────────▼────────┐
                  │  RoIAlignRotated│  旋转RoI对齐 (out_size=7)
                  │  + Shared2FC    │  fc→1024, cls+reg branches
                  └────────┬────────┘
                           │
                  ┌────────▼────────┐
                  │   Detection     │  (cx, cy, w, h, θ) in le90 format
                  │   Output        │  DIOR: 20 classes / Star: 25 classes
                  └─────────────────┘
```

### 数据流维度变化

| 阶段 | 输入形状 | 输出形状 | 说明 |
|------|---------|---------|------|
| Input | `[B, 3, 1024, 1024]` | — | DIOR-R 标准输入 |
| Patch Embed | `[B, 3, 1024, 1024]` | `[B, 64, 64, 768]` | patch_size=16, 64×64 patches |
| + Prefix Tokens | `[B, 64, 64, 768]` | `[B, 4101, 768]` | 1 cls + 4 reg + 4096 patches |
| Transformer Blocks | `[B, 4101, 768]` | `[B, 4101, 768]` | 12 blocks, 捕捉4个中间层 |
| Strip Tokens + Reshape | `[B, 4101, 768]` | `[B, 768, 64, 64]` | 去除前缀token |
| Output Projection | `[B, 768, 64, 64]` | `[B, 256, 64, 64]` | 1×1 conv + GN32 + GELU |
| ViTDetFPN | 4×`[B,256,64,64]` | `[B,256,256,256]`, `[B,256,128,128]`, `[B,256,64,64]`, `[B,256,32,32]` | stride=4,8,16,32 |
| RPN | 4-level features | `[N_proposals, 6]` | (cx,cy,w,h,θ) proposals |
| RoI Head | 7×7 RoI features | `[N_det, 6+cls]` | 最终检测结果 |

---

## 2. Backbone: ViT-DINOv3

### 2.1 模型概述

DINOv3 是 Meta AI 提出的自监督视觉Transformer，通过大规模预训练获得高质量的视觉表示，适用于目标检测等密集预测任务。

| 属性 | 值 |
|------|-----|
| 模型名称 | `vit_base_patch16_dinov3` |
| Patch大小 | 16×16 |
| 嵌入维度 | 768 |
| Transformer层数 | 12 |
| 注意力头数 | 12 |
| 参数量 | ~86M |
| 预训练数据 | LVD-1689M (1.689B images) |
| 位置编码 | RoPE (旋转位置编码) |
| 前缀Token | 1 cls_token + 4 reg_tokens = 5 |
| 输出特征 | 4层中间特征 @ stride=16 |

### 2.2 核心设计特点

#### RoPE 旋转位置编码
DINOv3 不使用传统绝对位置编码（`pos_embed=None`），而是采用 **RoPE (Rotary Position Embedding)**，在每个注意力块中通过旋转变换注入位置信息。RoPE 天然支持任意输入尺寸，因为它的编码由特征维度而非序列长度决定。

#### 前缀Token结构
```
Token序列: [cls_token(1), reg_tokens(4), patch_tokens(H×W)]
              ↑               ↑                ↑
          全局分类表征      寄存器Token        空间Patches
```

- **cls_token**: 保留用于图像级理解
- **reg_tokens**: DINOv3 特有的寄存器Token，用于存储全局图像统计信息
- **patch_tokens**: 空间特征，检测时提取这部分

#### `dynamic_img_size=True`
默认开启动态图像尺寸，无需固定输入分辨率。

### 2.3 代码实现 (`models/backbones/vit_dinov3.py`)

```
class ViTDinoV3(BaseModule):
    __init__:
        1. timm.create_model(model_name)  # 创建timm ViT模型
        2. _load_local_checkpoint()      # 加载本地预训练权重
        3. output_projections            # 4个 1×1 conv + GN32 + GELU
           (768→256 通道投影，适配检测neck)

    forward:
        1. patch_embed:   (B,C,H,W) → (B,H/16,W/16,768)
        2. _pos_embed:    展平 + 添加前缀Token + 生成RoPE频率
        3. norm_pre:       层归一化预处理
        4. transformer:    顺序通过12个block，在 [3,5,7,11] 处捕获特征
        5. strip tokens:   去除前缀Token (前5个)
        6. reshape:        (B,4096,768) → (B,768,64,64)
        7. projection:     1×1 conv + GN32 + GELU → (B,256,64,64)
```

#### 为什么选择 blocks [3, 5, 7, 11]?

| Block | 相对位置 | 特征特性 |
|-------|---------|---------|
| 3 (第4层) | 浅层 | 局部纹理、边缘细节 |
| 5 (第6层) | 中浅层 | 中级纹理模式 |
| 7 (第8层) | 中深层 | 部件级语义 |
| 11 (第12层) | 深层 | 全局语义、物体类别信息 |

这种"跳跃式"选择策略在降低计算量的同时保留了从细节到语义的多尺度信息。

### 2.4 权重加载机制

支持三种加载方式：

```
1. timm Hub下载 (pretrained=True):
   → timm自动从HuggingFace下载预训练权重

2. 本地pth文件 (checkpoint_path=):
   → 自动检测checkpoint格式:
     - state_dict:   训练checkpoint
     - model:        模型checkpoint
     - teacher:      Meta DINOv3 student-teacher格式
     - 原始dict:     直接作为state_dict
   → 去除 DDP "module." 前缀
   → 键名重映射 (官方→timm)

3. 从头初始化 (pretrained=False, checkpoint_path=None):
   → 仅随机初始化output projection层
```

#### 键名重映射表 (官方Meta → timm)

| 官方键名 | timm键名 | 说明 |
|---------|---------|------|
| `storage_tokens` | `reg_token` | 寄存器Token重命名 |
| `blocks.X.ls1.gamma` | `blocks.X.gamma_1` | 层缩放参数-1 |
| `blocks.X.ls2.gamma` | `blocks.X.gamma_2` | 层缩放参数-2 |
| `blocks.X.attn.qkv.bias` | **跳过** | timm使用无bias融合QKV |
| `blocks.X.attn.qkv.bias_mask` | **跳过** | timm不使用 |
| `mask_token` | **跳过** | 检测任务不需要 |
| `rope_embed.periods` | **跳过** | timm内部生成 |

验证结果：**162/162 个键完全匹配，0缺失**。

### 2.5 冻结策略 (`frozen_stages`)

```
frozen_stages=-1: 不冻结任何层（全部可训练）
frozen_stages=0:  仅冻结 patch_embed
frozen_stages=8:  冻结 patch_embed + blocks 0~7（前8层）
frozen_stages=12: 冻结整个ViT（只训练neck和检测头）
```

本项目配置 `frozen_stages=-1`，即所有层均可训练。

### 2.6 可选模型变体

| 模型 | embed_dim | depth | 参数量 | 适用场景 |
|------|-----------|-------|--------|---------|
| `vit_small_patch16_dinov3` | 384 | 12 | ~22M | 快速实验、轻量部署 |
| `vit_base_patch16_dinov3` | 768 | 12 | ~86M | 标准检测（本项目使用） |
| `vit_large_patch16_dinov3` | 1024 | 24 | ~304M | 高精度需求 |
| `vit_huge_plus_patch16_dinov3` | 1280 | 32 | ~632M | 极致性能 |

---

## 3. Neck: 特征金字塔网络

ViT输出的4个特征图具有**相同分辨率**（stride=16），需要Neck构建真正的多尺度金字塔。

### 3.1 ViTDetFPN（推荐，DIOR配置使用）

**来源**: ViTDet (Li et al., ECCV 2022) — "Exploring Plain ViT Backbones for Detection"

#### 架构图

```
ViT features (all stride-16)
f0(block 3)    f1(block 5)    f2(block 7)    f3(block 11)
    │               │               │               │
pre-norm GN+GELU pre-norm GN+GELU pre-norm GN+GELU pre-norm GN+GELU
    │               │               │               │
lateral 1×1     lateral 1×1    lateral 1×1    lateral 1×1
    │               │               │               │
upsample×2     upsample×2          │          downsample /2
    │               │               │               │
    ├─── SE+add ───┤               │               │
    │               │               │               │
    ├──────── SE+add ──────────────┤               │
    │               │               │               │
    ├─────────── SE+add ───────────────────────────┤
    │               │               │               │
 3×3 conv       3×3 conv        3×3 conv        3×3 conv
    │               │               │               │
  P0 (s4)       P1 (s8)        P2 (s16)        P3 (s32)
256×256         128×128         64×64            32×32
```

#### 核心组件

| 组件 | 功能 | 实现 |
|------|------|------|
| **Pre-Norm** | 适配ViT的LayerNorm特征到GroupNorm空间 | GN32 + GELU |
| **Lateral Conv** | 统一通道数 | 1×1 conv + GN + GELU |
| **UpsampleBlock** | 2×上采样 | Bilinear插值 + 3×3 conv + GN + GELU |
| **Downsample** | 2×下采样 | stride=2 3×3 conv + GN + GELU |
| **SEBlock** | 通道注意力融合 | GAP + FC-reduce + ReLU + FC-expand + Sigmoid |
| **FPN Conv** | 消除上采样混叠 | 3×3 conv + GN + GELU |

#### 自顶向下融合路径

```python
# 从深层到浅层逐级融合
P3 (s32) → upsample → + P2_raw → SE → P2_fused
P2_fused → upsample → + P1_raw → SE → P1_fused
P1_fused → upsample → + P0_raw → SE → P0_fused
```

每次融合后通过SE注意力块自动学习通道权重，抑制无关特征。

#### 输出特征图尺寸 (1024×1024输入)

| 层级 | Stride | 分辨率 | 感受野 | 适合目标 |
|------|--------|--------|--------|---------|
| P0 | 4 | 256×256 | 小 | 小型目标（车辆、烟囱） |
| P1 | 8 | 128×128 | 中小 | 中型目标（船只、网球场） |
| P2 | 16 | 64×64 | 中大 | 大型目标（棒球场、桥梁） |
| P3 | 32 | 32×32 | 大 | 超大型目标（机场、港口） |

### 3.2 SimpleFPN（Star配置使用）

#### 架构图

```
f0(block 3)    f1(block 5)    f2(block 7)    f3(block 11)
    │               │               │               │
lateral 1×1     lateral 1×1    lateral 1×1    lateral 1×1
    │               │               │               │
ConvTranspose 2×2/2         downsample /2   downsample /2 ×2
    │               │               │               │
  P0 (s8)       P1 (s16)       P2 (s32)        P3 (s64)
100×100         50×50           25×25            13×13
```

#### 特点

- **更简单**: 无跨尺度融合、无注意力机制
- **更高效**: 计算量更少，速度更快
- **独立生成**: 每个层级的尺度独立生成，无信息交互
- **可选融合**: `fuse_mode='top_down'` 可开启最近邻上采样融合

### 3.3 ViTDetFPN vs SimpleFPN 对比

| 特性 | ViTDetFPN (DIOR) | SimpleFPN (Star) |
|------|------------------|------------------|
| 输出Stride | `[4, 8, 16, 32]` | `[8, 16, 32, 64]` |
| 最高分辨率 | 256×256 (P0 @ s4) | 100×100 (P0 @ s8) |
| 上采样方式 | Bilinear + 3×3 conv | ConvTranspose2d |
| 跨尺度融合 | 是（top-down + add） | 可选（nearest upsample） |
| 通道注意力 | SE Block | 无 |
| Pre-Norm | 有（LN→GN适配） | 无 |
| 计算量 | 较高 | 较低 |
| 小目标性能 | 更好 | 一般 |

---

## 4. RPN Head: 旋转区域提议网络

### 4.1 架构

```
FPN Feature [B, 256, Hi, Wi]
      │
      ├─── 3×3 conv (256→256)
      │
      ├─── cls branch: 1×1 conv (256→A×1)  → sigmoid → objectness score
      │     (A = anchors per location = 3 ratios × 1 scale = 3)
      │
      └─── reg branch: 1×1 conv (256→A×6)  → delta encoding
            (cx, cy, w, h, θ offsets)
```

### 4.2 Anchor配置

| 参数 | DIOR (ViTDetFPN) | Star (SimpleFPN) |
|------|-----------------|-------------------|
| strides | `[4, 8, 16, 32]` | `[8, 16, 32, 64]` |
| scales | `[8]` | `[8]` |
| ratios | `[0.5, 1.0, 2.0]` | `[0.5, 1.0, 2.0]` |
| 每位置anchor数 | 3 | 3 |

### 4.3 Bounding Box编码器

使用 `MidpointOffsetCoder` 编码旋转框：

```
编码公式 (le90格式):
    dx = (cx - anchor_cx) / anchor_w
    dy = (cy - anchor_cy) / anchor_h
    dw = log(w / anchor_w)
    dh = log(h / anchor_h)
    dθ = θ - anchor_θ

输出格式: [dx, dy, dw, dh, dθ, dθ] (6维)
        最后两个dθ相同（历史兼容原因）
```

### 4.4 RPN训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| pos_iou_thr | 0.7 | anchor与GT的IoU > 0.7为正样本 |
| neg_iou_thr | 0.3 | anchor与GT的IoU < 0.3为负样本 |
| sampler数量 | 256 | 每图采样256个anchor |
| pos_fraction | 0.5 | 正样本比例50% |
| nms_pre | 2000 | NMS前候选数 |
| nms_iou | 0.8 | NMS IoU阈值 |
| max_per_img | 2000 | 每图最大proposal数 |

---

## 5. RoI Head: 旋转RoI检测头

### 5.1 架构详细

```
Proposals (N × 6: cx,cy,w,h,θ)
        │
        ▼
┌──────────────────────────────┐
│  RotatedSingleRoIExtractor   │  按proposal大小分配到对应FPN层
│  ├─ RoIAlignRotated (7×7)   │  旋转RoI对齐
│  └─ out_channels=256         │  输出 7×7×256 特征
└──────────────┬───────────────┘
               │ [N, 256, 7, 7]
               ▼
┌──────────────────────────────┐
│  RotatedShared2FCBBoxHead    │
│  ├─ FC1: 12544 → 1024       │  flatten 7×7×256
│  │   + ReLU                  │
│  ├─ FC2: 1024 → 1024        │
│  │   + ReLU                  │
│  ├─ cls branch: 1024 → 20/25│  Softmax分类
│  └─ reg branch: 1024 → 20*5 │  bbox回归(类别特定)
└──────────────┬───────────────┘
               │
               ▼
        最终检测结果
    (cx, cy, w, h, θ) le90
```

### 5.2 关键参数

| 参数 | DIOR | Star |
|------|------|------|
| RoI Align输出尺寸 | 7×7 | 7×7 |
| sample_num | 2 | 2 |
| fc_out_channels | 1024 | 1024 |
| 类别数 | 20 | 25 |
| BBox编码器 | DeltaXYWHAOBBoxCoder | DeltaXYWHAOBBoxCoder |
| 角度格式 | le90 | le90 |
| reg_class_agnostic | True | True |

### 5.3 Bounding Box回归编码

使用 `DeltaXYWHAOBBoxCoder`:

```
编码公式 (le90格式):
    dx = (cx - proposal_cx) / proposal_w
    dy = (cy - proposal_cy) / proposal_h
    dw = log(w / proposal_w)
    dh = log(h / proposal_h)
    dθ = θ - proposal_θ

标准化参数:
    target_means = [0, 0, 0, 0, 0]
    target_stds  = [0.1, 0.1, 0.2, 0.2, 0.1]
```

### 5.4 RoI训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| pos_iou_thr | 0.5 | proposal与GT的IoU > 0.5为正样本 |
| neg_iou_thr | 0.5 | proposal与GT的IoU < 0.5为负样本 |
| sampler数量 | 512 | 每图采样512个proposal |
| pos_fraction | 0.25 | 正样本比例25% |
| IoU计算器 | RBboxOverlaps2D | 旋转框IoU |

### 5.5 推理配置

| 参数 | 值 | 说明 |
|------|-----|------|
| score_thr | 0.05 | 分数过滤阈值 |
| nms | `iou_thr=0.1` | NMS IoU阈值（很低，保留更多框） |
| nms_pre | 2000 | NMS前候选数 |
| max_per_img | 2000 | 每图最大输出框数 |

---

## 6. 训练策略与损失函数

### 6.1 损失函数

| 阶段 | 损失 | 类型 | 权重 | 参数 |
|------|------|------|------|------|
| RPN cls | 目标性分类 | CrossEntropyLoss (sigmoid) | 1.0 | — |
| RPN reg | 提议框回归 | SmoothL1Loss | 1.0 | β=1/9 |
| RCNN cls | 类别分类 | CrossEntropyLoss (softmax) | 1.0 | — |
| RCNN reg | 检测框回归 | SmoothL1Loss | 1.0 | β=1.0 |

总损失:
```
L_total = L_rpn_cls + L_rpn_reg + L_rcnn_cls + L_rcnn_reg
```

### 6.2 优化器

| 参数 | DIOR | Star |
|------|------|------|
| 优化器 | AdamW | AdamW |
| 学习率 | 1e-4 | 1e-4 |
| β | (0.9, 0.999) | (0.9, 0.999) |
| weight_decay | 0.05 | 0.05 |
| 梯度裁剪 | max_norm=35 | max_norm=35 |

### 6.3 学习率调度

| 参数 | DIOR | Star |
|------|------|------|
| 策略 | CosineAnnealing | CosineAnnealing |
| warmup | linear, 500 iters | linear, 150 iters |
| warmup_ratio | 1/3 | 1/3 |
| min_lr_ratio | 1e-3 | 1e-3 |

### 6.4 训练参数

| 参数 | DIOR | Star |
|------|------|------|
| epochs | 100 | 200 |
| batch_size (per GPU) | 16 | 16 |
| 混合精度 | fp16 (loss_scale=512) | fp16 (loss_scale=512) |
| 输入尺寸 | 多尺度 [800,1024,1200] | 固定 800×800 |
| DropPath | 0.1 | 0.0 |

### 6.5 数据增强

**DIOR（复杂增强）**:
- 多尺度训练: [800, 1024, 1200]
- 随机翻转: 水平/垂直/对角，各25%概率
- 光度畸变: 亮度±32, 对比度0.5-1.5, 饱和度0.5-1.5, 色相±18

**Star（简单增强）**:
- 固定尺寸: 800×800
- 随机翻转: 水平/垂直/对角，各25%概率

---

## 7. 数据集

### 7.1 DIOR-R (DIOR配置)

| 属性 | 值 |
|------|-----|
| 类别数 | 20 |
| 标注格式 | DOTA txt (8坐标 + 类别 + 难度) |
| 数据路径 | `data/DIOR-R/` |
| 划分 | train / val / test |

**20类列表**:
```
airplane, airport, baseballfield, basketballcourt, bridge, chimney,
dam, Expressway-Service-area, Expressway-toll-station, golffield,
groundtrackfield, harbor, overpass, ship, stadium, storagetank,
tenniscourt, trainstation, vehicle, windmill
```

### 7.2 Star-1021+Extend3 (Star配置)

| 属性 | 值 |
|------|-----|
| 类别数 | 25 |
| 标注格式 | DOTA txt |
| 数据路径 | `data/star-1021_1016+extend3/` |
| 图像格式 | .tif / .jpg / .jpeg / .png / .bmp |
| 划分 | train / val / test |

**25类列表** (中文):
```
两栖攻击舰, 侦察机, 加油机, 反潜巡逻机, 商业客机, 坦克, 导弹快艇,
巡洋舰, 扫雷艇, 护卫舰, 机场, 武装直升机, 民用客轮, 登陆舰,
空天战斗机, 航空母舰, 补给舰, 装甲运输车, 轰炸机, 运输机,
通用直升机, 重型运输车, 隐身战斗机, 预警机, 驱逐舰
```

### 7.3 数据加载流程

```
DOTA txt → load_annotations() → poly2obb_np() → OBB格式
                                      │
                             8坐标 → (cx, cy, w, h, θ)
```

### 7.4 评测指标

| 指标 | 说明 | DIOR | Star |
|------|------|------|------|
| `mAP` | mAP@IoU=0.50 | ✓ (默认) | 可选 |
| `mAP_multi` | mAP@IoU=0.50 + 0.75 | 可选 | 可选 |
| `mAP_coco` | mAP@IoU=0.50:0.95 (10步) | 可选 | ✓ (默认) |
| `gpu_collect` | GPU all_gather (避免NFS竞争) | True | True |
| 评测间隔 | — | 每5 epoch | 每5 epoch |
| 保存最佳 | — | mAP@0.50 | mAP@50:95 |

---

## 8. 训练配置速查

### 8.1 DIOR-R 训练

```bash
# 分布式训练 (6 GPU)
bash tools/dist_train.sh

# 等效命令
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m torch.distributed.run \
    --nproc_per_node=6 --master_port=29502 \
    tools/train.py \
    configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_dior.py
```

| 配置项 | 值 |
|--------|-----|
| GPU数 | 6 |
| 总batch size | 96 (6×16) |
| 输入尺寸 | 1024×1024 (多尺度训练) |
| Neck | ViTDetFPN |
| 最大epoch | 100 |
| warmup | 500 iters |
| checkpoint间隔 | 每3 epoch |
| 评测间隔 | 每5 epoch |

### 8.2 Star-1021+Extend3 训练

```bash
# 分布式训练 (4 GPU)
bash tools/dist_train_star.sh

# 等效命令
CUDA_VISIBLE_DEVICES=2,4,5,6 python -m torch.distributed.run \
    --nproc_per_node=4 --master_port=29506 \
    tools/train.py \
    configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_star.py
```

| 配置项 | 值 |
|--------|-----|
| GPU数 | 4 |
| 总batch size | 64 (4×16) |
| 输入尺寸 | 800×800 |
| Neck | SimpleFPN |
| 最大epoch | 200 |
| warmup | 150 iters |
| checkpoint间隔 | 每5 epoch |
| 评测间隔 | 每5 epoch |

### 8.3 环境要求

```
Python >= 3.12
PyTorch >= 2.7.1  (CUDA 12.8)
MMCV == 1.7.2
MMRotate == 0.3.4
MMDetection == 2.28.2
timm >= 1.0
```

### 8.4 PyTorch 2.7 兼容性修复

训练脚本中内置了4个兼容性补丁（`tools/train.py:30-55`）：

1. **`mp_start_method='spawn'`**: 解决CUDA fork崩溃
2. **`_get_stream` 类型修复**: `int → torch.device` 适配
3. **`_use_replicated_tensor_module` 属性补充**: MMDDP缺少该属性
4. **`bbox_nms_rotated` 设备修复**: labels tensor设备不匹配

### 8.5 分布式测试

```bash
bash tools/dist_test.sh
```
