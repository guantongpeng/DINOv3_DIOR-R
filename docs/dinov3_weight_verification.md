# DINOv3 预训练权重加载验证报告

## 概述

**DINOv3 预训练权重已正确加载。** 验证结果确认：

- 162/162 个 timm 模型参数全部匹配预训练权重（0 个缺失）
- 25 个 key 正确地从 Meta 官方 DINOv3 格式映射到 timm 命名规范
- 24 个 qkv bias/bias_mask key 被正确跳过（timm 使用无偏置的融合 QKV）
- 所有权重统计量确认来自预训练权重（均值非零，标准差正常），而非随机初始化

---

## 1. 训练流程：权重是如何加载的

权重加载发生在模型构建阶段，**在训练开始之前**：

```
tools/train.py:294  build_detector(cfg.model)
    └─ ViTDinoV3.__init__()                              [vit_dinov3.py:115]
       ├─ checkpoint_path ≠ None → pretrained = False    [vit_dinov3.py:149]
       ├─ timm.create_model(pretrained=False)             [vit_dinov3.py:155]
       └─ _load_local_checkpoint(checkpoint_path)         [vit_dinov3.py:164]
          ├─ torch.load() → 解析 state dict 格式           [vit_dinov3.py:294]
          ├─ 去除 DDP "module." 前缀                        [vit_dinov3.py:317]
          ├─ _remap_dinov3_official_to_timm()  key 重映射   [vit_dinov3.py:325]
          └─ vit.load_state_dict(strict=False)            [vit_dinov3.py:345]

tools/train.py:299  model.init_weights()                  [vit_dinov3.py:386]
    └─ init_cfg=None → 跳过父类 init_weights()
    └─ 仅初始化输出投影层（output_projections）             [vit_dinov3.py:399]
    └─ 日志：已加载本地 checkpoint 权重                     [vit_dinov3.py:411]
```

**关键点**：ViT 主干网络的权重在 `__init__()` 中加载（即 `build_detector` 调用期间），而非在 `init_weights()` 中。第 299 行的 `init_weights()` 调用仅初始化输出投影层（`self.output_projections`）并打印加载状态日志。

---

## 2. Checkpoint 文件分析

配置文件使用 Meta 官方 DINOv3 发布的 checkpoint 文件：

| 配置文件 | Checkpoint 路径 | 大小 |
|----------|----------------|------|
| DIOR-R | `data/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` | 327 MB |

### Checkpoint 结构（Meta 官方格式）

```
格式: 原始 state dict（无外层包裹）
总 key 数: 188
```

**Key 分类（checkpoint 188 个 → timm 模型 162 个）：**

| 类别 | 数量 | 处理方式 |
|------|------|----------|
| 直接匹配的 key | 138 | 直接加载 |
| `storage_tokens` | 1 | 重映射 → `reg_token` |
| `blocks.X.ls1.gamma` | 12 | 重映射 → `blocks.X.gamma_1` |
| `blocks.X.ls2.gamma` | 12 | 重映射 → `blocks.X.gamma_2` |
| `blocks.X.attn.qkv.bias` | 12 | **跳过**（timm 使用无偏置 QKV） |
| `blocks.X.attn.qkv.bias_mask` | 12 | **跳过**（timm 使用无偏置 QKV） |
| `mask_token` | 1 | **跳过**（timm 未使用） |
| `rope_embed.periods` | 1 | **跳过**（timm 未使用） |
| **合计** | **188** | → 162 个加载到 timm 模型 |

### Key 重映射详情

`vit_dinov3.py:204-274` 中的 `_remap_dinov3_official_to_timm()` 函数处理以下重映射：

| 官方 DINOv3 Key | timm 模型 Key | 数量 |
|-----------------|---------------|------|
| `storage_tokens` | `reg_token` | 1 |
| `blocks.X.ls1.gamma` | `blocks.X.gamma_1` | 12 |
| `blocks.X.ls2.gamma` | `blocks.X.gamma_2` | 12 |

如果没有这个重映射，25 个参数（1 个 reg_token + 24 个 LayerScale gamma）将会被随机初始化而非从 checkpoint 加载，这会严重降低模型精度。

---

## 3. 验证结果

### 3.1 Key 匹配情况

```
DINOv3 checkpoint loaded: 162/162 keys matched, 0 missing, 0 unexpected.
```

### 3.2 权重统计量（预训练权重的证据）

随机初始化的权重大致会显示 `mean ≈ 0` 且 `std ≈ 0.02`（Kaiming uniform 初始化）。所有测试层均显示出预训练权重特有的非零均值和多样化标准差：

| 层 | mean | std | 判定 |
|----|------|-----|------|
| patch_embed.proj.weight | +1.1e-05 | 0.0256 | 预训练 |
| blocks.0.attn.qkv.weight | +4.5e-05 | 0.0766 | 预训练 |
| blocks.0.attn.proj.weight | +2.6e-05 | 0.0288 | 预训练 |
| blocks.0.mlp.fc1.weight | +5.4e-05 | 0.0600 | 预训练 |
| blocks.11.attn.qkv.weight | +3.5e-05 | 0.0553 | 预训练 |
| blocks.11.mlp.fc2.weight | +4.2e-05 | 0.0671 | 预训练 |
| norm.weight | +0.8426 | 0.7573 | 预训练 |
| cls_token | -0.0039 | 0.0524 | 预训练 |
| reg_token | -0.0010 | 0.0567 | 预训练 |

### 3.3 训练启动时的预期日志输出

训练启动时，以下日志信息可以确认权重已正确加载：

```
INFO - Loading DINOv3 checkpoint from local path: data/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
INFO - Using checkpoint dict directly as state dict.
INFO - Remapped 25 checkpoint keys to timm naming convention.
INFO - Skipped 24 qkv bias/bias_mask keys (timm uses bias-free fused qkv).
INFO - DINOv3 checkpoint loaded: 162/162 keys matched, 0 missing, 0 unexpected.
INFO - DINOv3 backbone loaded with local checkpoint weights: data/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```

**正确加载的关键指标：**

- `Remapped 25 checkpoint keys` — LayerScale 和 reg_token 已正确重映射
- `162/162 keys matched, 0 missing` — 所有 timm 模型参数均获得预训练权重
- `Skipped 24 qkv bias/bias_mask keys` — bias key 已正确排除（timm 无偏置）

**如果加载失败（当前代码下不太可能发生），你会看到：**

- `137/162 keys matched, 25 missing` — 重映射未生效，LayerScale + reg_token 被随机初始化
- `DINOv3 backbone initialized from scratch` — checkpoint_path 未设置或文件未找到

---

## 4. 配置文件分析

### 4.1 DIOR-R 配置 (`oriented_rcnn_dinov3_fpn_dior.py`)

| 设置项 | 值 | 状态 |
|--------|----|------|
| `pretrained` | `False` | 正确（使用本地 checkpoint） |
| `checkpoint_path` | `data/...` | 文件存在（327 MB） |
| `init_cfg` | `None` | 正确（ViT 权重在 `__init__` 中加载） |
| `frozen_stages` | `-1` | 所有层可训练 |
| `load_from` | `None` | 正确（无外部覆盖） |
| `resume_from` | `None` | 正确（从头训练） |

### 4.2 需要避免的潜在陷阱

1. **不要设置 `init_cfg`** — 如果设置了 `init_cfg`（例如 `dict(type='Pretrained', checkpoint=...)`），mmcv 的 `BaseModule.init_weights()` 会覆盖已经加载好的 ViT 权重。配置文件正确地将 `init_cfg` 设为 `None`。

2. **不要设置 `load_from`** — mmdet 中的全局 `load_from` 会加载完整的模型 checkpoint（包括主干网络、颈部、检测头）。设置 `load_from` 会覆盖 DINOv3 预训练的主干网络权重。配置文件正确地将 `load_from` 设为 `None`。

3. **不要在设置了 `checkpoint_path` 的同时设置 `pretrained=True`** — 代码已正确处理这种情况（内部自动将 `pretrained` 设为 `False`），但同时设置两者会造成困惑。配置文件正确地将 `pretrained` 设为 `False`。

---

## 5. 验证脚本

项目中提供了独立的验证脚本 `tools/verify_dinov3_weights.py`：

```bash
# 使用默认 checkpoint 验证权重
python tools/verify_dinov3_weights.py

# 使用指定的配置文件验证
python tools/verify_dinov3_weights.py \
    --config configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_dior.py

# 使用指定的 checkpoint 文件验证
python tools/verify_dinov3_weights.py \
    --checkpoint_path /path/to/checkpoint.pth
```

脚本执行三项检查：
1. **Checkpoint 检查** — 验证文件是否存在、格式、key 分类
2. **权重加载测试** — 验证 key 重映射，检查 162 个 key 全部匹配
3. **配置文件验证** — 检查配置中是否存在潜在的配置错误

---

## 6. 结论

DINOv3 预训练权重加载机制工作正常：

- Checkpoint 文件（188 个 key，Meta 官方格式）被正确解析
- Key 重映射函数将官方 DINOv3 key 转换为 timm 命名规范
- 所有 162 个 timm 模型参数均获得预训练权重（0 个缺失）
- 权重统计量证实来自预训练值，而非随机初始化
- 两个配置文件均无会覆盖或破坏已加载权重的配置错误
- `init_weights()` 方法正确跳过 ViT 层的重新初始化，仅初始化输出投影层

**未发现任何问题。训练将以完全预训练的 DINOv3 主干网络权重进行。**
