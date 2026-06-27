# DINOv3 使用本地 .pth 模型（不通过 HuggingFace）

## 背景

`ViTDinoV3` backbone 原先使用 `timm.create_model(model_name, pretrained=True)` 创建模型，会自动从 timm hub (HuggingFace) 下载预训练权重。

修改后支持直接加载本地 .pth 文件，无需网络连接。

## 修改内容

### 1. `models/backbones/vit_dinov3.py`

| 变更 | 说明 |
|------|------|
| 新增参数 `checkpoint_path: Optional[str] = None` | 本地 .pth 文件路径 |
| 新增方法 `_load_local_checkpoint()` | 加载并解析本地 checkpoint |
| 新增方法 `_remap_dinov3_official_to_timm()` | 官方 DINOv3 → timm key 自动映射 |
| `__init__` 逻辑调整 | 若 `checkpoint_path` 不为 None，自动将 `pretrained` 设为 `False` |

#### Key 自动映射

官方 Meta DINOv3 checkpoint 使用与 timm 不同的命名约定，`_remap_dinov3_official_to_timm()`
自动完成以下映射：

| 官方 checkpoint key | timm model key |
|---------------------|----------------|
| `storage_tokens` | `reg_token` |
| `blocks.X.ls1.gamma` | `blocks.X.gamma_1` |
| `blocks.X.ls2.gamma` | `blocks.X.gamma_2` |
| `blocks.X.attn.qkv.bias` | *(跳过 — timm 使用 bias-free qkv)* |
| `blocks.X.attn.qkv.bias_mask` | *(跳过)* |
| `mask_token` | *(跳过)* |
| `rope_embed.periods` | *(跳过)* |

> **重要**: 若日志显示 `137/162 keys matched, 25 missing` 而非 `162/162 keys matched, 0 missing`，
> 说明映射未生效。缺失的 25 个 key 为 LayerScale 参数 (`gamma_1`, `gamma_2`) 和
> `reg_token`，这些参数被随机初始化将严重影响模型精度。

#### `_load_local_checkpoint` 支持的 checkpoint 格式

| 格式 | 说明 |
|------|------|
| `{'state_dict': {...}}` | 完整训练 checkpoint（最常见） |
| `{'model': {...}}` | 部分框架使用 `model` key |
| `{'teacher': {...}}` | Meta DINOv3 官方 student-teacher 格式 |
| 直接的 state dict | 无外包裹的直接权重字典 |
| DDP `module.` 前缀 | 自动去除 |

### 2. Config 文件使用方式

```python
backbone=dict(
    type='ViTDinoV3',
    model_name='vit_base_patch16_dinov3',
    pretrained=False,                                        # 设为 False
    checkpoint_path='checkpoints/dinov3_vit_base_patch16.pth',  # 本地路径
    out_indices=(3, 5, 7, 11),
    out_channels=256,
    frozen_stages=8,
    ...
),
```

## 使用步骤

### Step 1: 下载 DINOv3 权重到本地

从 Meta 官方仓库下载：

```bash
# 示例：下载 vit_base_patch16 权重
mkdir -p checkpoints
wget https://dl.fbaipublicfiles.com/dinov3/dinov3_vit_base_patch16/dinov3_vit_base_patch16.pth \
     -O checkpoints/dinov3_vit_base_patch16.pth
```

或从其他来源下载后放到项目目录下。

### Step 2: 修改 Config

将 config 中 backbone 的 `pretrained` 设为 `False`，并添加 `checkpoint_path` 指向你的本地文件：

```python
pretrained=False,
checkpoint_path='checkpoints/dinov3_vit_base_patch16.pth',
```

### Step 3: 正常启动训练

```bash
# DIOR-R
python tools/train.py configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_train_dior.py
```

启动时会输出日志：

```
INFO - Loading DINOv3 checkpoint from local path: checkpoints/dinov3_vit_base_patch16.pth
INFO - Using checkpoint dict directly as state dict.
INFO - Remapped 25 checkpoint keys to timm naming convention.
INFO - Skipped 24 qkv bias/bias_mask keys (timm uses bias-free fused qkv).
INFO - DINOv3 checkpoint loaded: 162/162 keys matched, 0 missing, 0 unexpected.
INFO - DINOv3 backbone loaded with local checkpoint weights: checkpoints/dinov3_vit_base_patch16.pth
```

确保看到 `162/162 keys matched, 0 missing`。

## 兼容性

- **不传 `checkpoint_path`**：行为与修改前完全一致，使用 `timm` hub 下载
- **传入 `checkpoint_path`**：自动跳过 hub 下载，从本地加载
- 原 `pretrained=True` + 不传 `checkpoint_path` 的配置无需修改，仍然正常工作

## 支持的模型

| 模型 | model_name |
|------|------------|
| ViT-Small | `vit_small_patch16_dinov3` |
| ViT-Base | `vit_base_patch16_dinov3` |
| ViT-Large | `vit_large_patch16_dinov3` |
| ViT-Huge+ | `vit_huge_plus_patch16_dinov3` |
