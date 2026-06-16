# YOLO OBB → DOTA 格式转换工具

## 脚本位置

`tools/yolo2dota.py`

## 概述

将 YOLO 旋转框 (Oriented Bounding Box, OBB) 标注格式转换为 DOTA 格式，用于 mmrotate / Oriented R-CNN 训练。

## 格式对比

| | YOLO OBB (输入) | DOTA (输出) |
|---|---|---|
| **坐标** | 归一化 (0~1) | 绝对像素值 |
| **标注行** | `class_id x1 y1 x2 y2 x3 y3 x4 y4` | `x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty` |
| **类别** | 整数 ID (从 0 开始) | 字符串名称 |
| **难度** | 无 | 整数 (0=正常) |
| **图片对应** | 同名 txt 文件 | 同名 txt 文件 |

### 支持的 YOLO 输入格式

- **4 角点格式 (推荐)**: `class_id x1 y1 x2 y2 x3 y3 x4 y4`
- **cxcywha 格式**: `class_id cx cy w h angle`（自动转为 4 角点，需要 cv2 模块）

## 用法

### 单集转换

```bash
# 基本用法
python tools/yolo2dota.py \
    --yolo_dir data/Custom25/train/labels/ \
    --image_dir data/Custom25/train/images/ \
    --output_dir data/Custom25/train/labelTxt/ \
    --class_names "airplane,ship,vehicle,storage_tank,bridge,..."

# 使用类别文件
python tools/yolo2dota.py \
    --yolo_dir data/Custom25/train/labels/ \
    --image_dir data/Custom25/train/images/ \
    --output_dir data/Custom25/train/labelTxt/ \
    --classes data/Custom25/classes.txt \
    --img_ext .png \
    --difficulty 0
```

### 批量转换 (推荐)

若数据集目录结构为:
```
data/Custom25/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

```bash
python tools/yolo2dota.py \
    --batch data/Custom25/ \
    --splits train val test \
    --classes data/Custom25/classes.txt
```

输出目录结构:
```
data/Custom25/
├── train/
│   ├── images/
│   ├── labels/
│   └── labelTxt/      # ← 新生成的 DOTA 标注
├── val/
│   ├── images/
│   ├── labels/
│   └── labelTxt/      # ← 新生成的 DOTA 标注
└── test/
    ...
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--yolo_dir` | YOLO 标注文件夹路径 (单集模式) | - |
| `--image_dir` | 图片文件夹路径，用于获取图像尺寸 | - |
| `--output_dir` | DOTA 标注输出文件夹 | - |
| `--classes` | 类别列表文件路径，每行一个类别名 | - |
| `--class_names` | 逗号分隔的类别名，如 `"cls1,cls2,..."` | - |
| `--img_ext` | 图片扩展名，支持多个用逗号分隔 | `.jpg` |
| `--difficulty` | DOTA difficulty 默认值 | `0` |
| `--overwrite` | 覆盖已存在的输出文件 | `False` |
| `--batch` | 数据集根目录 (批量模式) | - |
| `--splits` | 批量模式下的子集名列表 | `train val test` |
| `--yolo_subdir` | YOLO 标注在子集下的目录名 | `labels` |
| `--image_subdir` | 图片在子集下的目录名 | `images` |
| `--output_subdir` | DOTA 标注在子集下的目录名 | `labelTxt` |

## 类别文件格式

`classes.txt` 示例 (每行一个类别名，顺序对应 YOLO 的 class_id):

```
airplane
airport
baseballfield
basketballcourt
bridge
chimney
...
```

第 1 行对应 YOLO 的 `class_id=0`，第 2 行对应 `class_id=1`，依此类推。

## 输出示例

输入 (YOLO OBB):
```
0 0.10 0.20 0.30 0.20 0.30 0.50 0.10 0.50
1 0.50 0.40 0.70 0.40 0.70 0.60 0.50 0.60
```

输出 (DOTA) — 假设图片尺寸 1000×800:
```
100.0000 160.0000 300.0000 160.0000 300.0000 400.0000 100.0000 400.0000 airplane 0
500.0000 320.0000 700.0000 320.0000 700.0000 480.0000 500.0000 480.0000 ship 0
```

## 注意事项

1. **图片必须可访问**：脚本需要读取图片获取尺寸（宽/高），确保 `image_dir` 路径正确
2. **坐标顺序保持不变**：脚本不改变角点顺序，只做归一化→绝对坐标的线性转换
3. **类别顺序一致**：`classes.txt` 或 `--class_names` 中的顺序必须与 YOLO 训练时的类别顺序完全一致
4. **空标注处理**：如果图片无目标，输出空 `.txt` 文件（保留文件以匹配 mmrotate 的数据加载逻辑）
5. **cxcywha 格式**：需要安装 `opencv-python` (`cv2`) 才能转换
