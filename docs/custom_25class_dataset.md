# 自定义 25 类数据集适配指南

本文档说明如何将现有 DIOR-R 20 类配置修改为支持自定义 25 类数据集。

---

## 修改概览

| 文件 | 操作 | 说明 |
|------|------|------|
| `models/datasets/custom25.py` | **新建** | 自定义 25 类数据集类 |
| `models/datasets/__init__.py` | **修改** | 注册新数据集类 |
| `configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_custom25.py` | **新建** | 25 类配置文件 |

共需修改/新建 **3 个文件**，涉及 **4 处关键参数**变更。

---

## 步骤 1：创建自定义数据集类

新建文件 `models/datasets/custom25.py`，继承自 `DOTADataset`（与 DIOR 相同基类），将 CLASSES 替换为你自己的 25 类名称和调色板。

### 文件：`models/datasets/custom25.py`

```python
# Copyright (c) OpenMMLab. All rights reserved.
"""Custom 25-class Dataset for mmrotate.

This dataset uses DOTA-format annotations (txt files with 8 corner
coordinates per line: x1 y1 x2 y2 x3 y3 x4 y4 category difficult).

Usage:
    Copy this file to models/datasets/custom25.py
    Update __init__.py to import Custom25Dataset
    Use dataset_type='Custom25Dataset' in your config
"""

import glob
import os
import os.path as osp

import numpy as np
from mmrotate.core import poly2obb_np
from mmrotate.datasets.builder import ROTATED_DATASETS
from mmrotate.datasets.dota import DOTADataset


@ROTATED_DATASETS.register_module()
class Custom25Dataset(DOTADataset):
    """Custom 25-class dataset for oriented object detection.

    请根据你的实际数据集修改以下 CLASSES 列表（共 25 个类别名称）。

    Args:
        ann_file (str): Path to annotation folder containing .txt files.
        pipeline (list[dict]): Processing pipeline.
        version (str, optional): Angle representations. Defaults to 'le90'.
        difficulty (int, optional): Difficulty threshold for filtering
            ground truth boxes. Boxes with difficulty > this value are
            ignored. Default: 100 (keep all).
        filter_empty_gt (bool): Whether to filter images without GT boxes.
            Default: True.
        img_ext (str): Image file extension. Default: '.jpg'.
    """

    # ============================================================
    # TODO: 替换为你的 25 个类别名称（请按实际数据集修改）
    # ============================================================
    CLASSES = (
        'class_01', 'class_02', 'class_03', 'class_04', 'class_05',
        'class_06', 'class_07', 'class_08', 'class_09', 'class_10',
        'class_11', 'class_12', 'class_13', 'class_14', 'class_15',
        'class_16', 'class_17', 'class_18', 'class_19', 'class_20',
        'class_21', 'class_22', 'class_23', 'class_24', 'class_25',
    )

    # ============================================================
    # TODO: 替换为 25 种颜色（RGB 格式，每类一个颜色）
    # 颜色用于可视化，可用工具生成 25 种不同颜色
    # ============================================================
    PALETTE = [
        (165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
        (138, 43, 226), (255, 128, 0), (255, 0, 255), (0, 255, 255),
        (255, 193, 193), (0, 51, 153), (255, 250, 205), (0, 139, 139),
        (255, 255, 0), (147, 116, 116), (0, 0, 255), (220, 20, 60),
        (128, 128, 0), (255, 215, 0), (128, 128, 128), (64, 224, 208),
        (255, 99, 71), (50, 205, 50), (238, 130, 238), (70, 130, 180),
        (210, 105, 30),
    ]

    def __init__(self,
                 ann_file,
                 pipeline,
                 version='le90',
                 difficulty=100,
                 filter_empty_gt=True,
                 img_ext='.jpg',
                 **kwargs):
        self.img_ext = img_ext
        super().__init__(
            ann_file=ann_file,
            pipeline=pipeline,
            version=version,
            difficulty=difficulty,
            filter_empty_gt=filter_empty_gt,
            **kwargs,
        )

    def load_annotations(self, ann_folder):
        """Load annotations from DOTA-format txt files.

        Overrides DOTADataset.load_annotations to:
        1. Support custom image extensions.
        2. Use custom 25-class mapping.

        Args:
            ann_folder (str): Folder containing DOTA format .txt files.

        Returns:
            list[dict]: List of data info dicts.
        """
        cls_map = {c: i for i, c in enumerate(self.CLASSES)}
        ann_files = glob.glob(osp.join(ann_folder, '*.txt'))
        data_infos = []

        if not ann_files:
            # Test phase: find all images in img_prefix
            return []

        for ann_file in ann_files:
            data_info = {}
            img_id = osp.splitext(osp.basename(ann_file))[0]
            img_name = img_id + self.img_ext
            data_info['filename'] = img_name
            data_info['ann'] = {}

            gt_bboxes = []
            gt_labels = []
            gt_polygons = []
            gt_bboxes_ignore = []
            gt_labels_ignore = []
            gt_polygons_ignore = []

            if osp.getsize(ann_file) == 0 and self.filter_empty_gt:
                continue

            with open(ann_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 10:
                        continue

                    poly = np.array(parts[:8], dtype=np.float32)
                    try:
                        x, y, w, h, a = poly2obb_np(poly, self.version)
                    except Exception:
                        continue

                    cls_name = parts[8]
                    difficulty = int(parts[9]) if len(parts) >= 10 else 0

                    if cls_name not in cls_map:
                        continue

                    label = cls_map[cls_name]

                    if difficulty > self.difficulty:
                        gt_bboxes_ignore.append([x, y, w, h, a])
                        gt_labels_ignore.append(label)
                        gt_polygons_ignore.append(poly)
                    else:
                        gt_bboxes.append([x, y, w, h, a])
                        gt_labels.append(label)
                        gt_polygons.append(poly)

            if gt_bboxes:
                data_info['ann']['bboxes'] = np.array(gt_bboxes, dtype=np.float32)
                data_info['ann']['labels'] = np.array(gt_labels, dtype=np.int64)
                data_info['ann']['polygons'] = np.array(gt_polygons, dtype=np.float32)
            else:
                data_info['ann']['bboxes'] = np.zeros((0, 5), dtype=np.float32)
                data_info['ann']['labels'] = np.array([], dtype=np.int64)
                data_info['ann']['polygons'] = np.zeros((0, 8), dtype=np.float32)

            if gt_bboxes_ignore:
                data_info['ann']['bboxes_ignore'] = np.array(
                    gt_bboxes_ignore, dtype=np.float32)
                data_info['ann']['labels_ignore'] = np.array(
                    gt_labels_ignore, dtype=np.int64)
                data_info['ann']['polygons_ignore'] = np.array(
                    gt_polygons_ignore, dtype=np.float32)
            else:
                data_info['ann']['bboxes_ignore'] = np.zeros(
                    (0, 5), dtype=np.float32)
                data_info['ann']['labels_ignore'] = np.array(
                    [], dtype=np.int64)
                data_info['ann']['polygons_ignore'] = np.zeros(
                    (0, 8), dtype=np.float32)

            data_infos.append(data_info)

        self.img_ids = [osp.splitext(info['filename'])[0]
                        for info in data_infos]
        return data_infos
```

### 核心要点

| 要素 | 说明 |
|------|------|
| `CLASSES` | 元组，25 个类别名称，**顺序决定 label id（0~24）** |
| `PALETTE` | 列表，25 种 RGB 颜色，用于可视化 |
| 注解格式 | DOTA 格式：每行 `x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty` |
| `cls_map` | 自动从 `CLASSES` 构建，注解中的类别名必须完全匹配 |

---

## 步骤 2：注册数据集类

### 文件：`models/datasets/__init__.py`

将原内容：

```python
from .dior import DIORDataset

__all__ = ['DIORDataset']
```

修改为：

```python
from .dior import DIORDataset
from .custom25 import Custom25Dataset

__all__ = ['DIORDataset', 'Custom25Dataset']
```

---

## 步骤 3：创建配置文件

基于 `configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_dior.py` 创建新配置。

### 文件：`configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_custom25.py`

需要修改的位置（与原 DIOR 配置相比）：

#### 3.1 自定义导入（第 21-28 行）

```python
custom_imports = dict(
    imports=[
        'models.backbones.vit_dinov3',
        'models.necks.simple_fpn',
        'models.datasets.custom25',       # ← 改为导入新数据集模块
    ],
    allow_failed_imports=False,
)
```

#### 3.2 类别数（第 114 行）

```python
num_classes=25,  # ← 改为 25
```

#### 3.3 数据集类型（第 208 行）

```python
dataset_type = 'Custom25Dataset'  # ← 改为新数据集类名
```

#### 3.4 数据路径（第 209 行）

```python
data_root = 'data/Custom25/'  # ← 改为你的数据集路径
```

### 完整配置文件关键差异对比

```
DIOR 配置 (20类)                          自定义配置 (25类)
─────────────────────────────────────    ─────────────────────────────────────
imports: models.datasets.dior            imports: models.datasets.custom25
num_classes: 20                          num_classes: 25
dataset_type: 'DIORDataset'             dataset_type: 'Custom25Dataset'
data_root: 'data/DIOR-R/'               data_root: 'data/Custom25/'
CLASSES: 20 个类别                       CLASSES: 25 个自定义类别
PALETTE: 20 种颜色                       PALETTE: 25 种颜色
```

---

## 步骤 4：准备数据集目录结构

按照以下结构组织数据：

```
data/Custom25/
├── train/
│   ├── images/          # 训练图片（如 .jpg, .png）
│   │   ├── 000001.jpg
│   │   ├── 000002.jpg
│   │   └── ...
│   └── labelTxt/        # DOTA 格式注解文件（.txt）
│       ├── 000001.txt
│       ├── 000002.txt
│       └── ...
├── val/                 # 验证集（结构同上）
│   ├── images/
│   └── labelTxt/
└── test/                # 测试集（结构同上）
    ├── images/
    └── labelTxt/
```

### 注解文件格式（DOTA 格式）

每行一个目标，空格分隔：

```
x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty
```

示例：
```
500.0 300.0 600.0 300.0 600.0 400.0 500.0 400.0 class_01 0
100.0 200.0 250.0 200.0 250.0 350.0 100.0 350.0 class_05 0
```

- **8 个坐标**：顺时针或逆时针排列的四边形四个顶点
- **class_name**：必须与 `CLASSES` 元组中的名称完全一致
- **difficulty**：0 表示正常目标，≥1 表示难例（difficulty > `self.difficulty` 会被忽略）

---

## 验证清单

修改完成后，按以下步骤验证：

1. [ ] `models/datasets/custom25.py` 已创建，CLASSES 包含 25 个类别
2. [ ] `models/datasets/__init__.py` 已导入 `Custom25Dataset`
3. [ ] `num_classes` 已改为 25（配置文件中 bbox_head 的 `num_classes`）
4. [ ] `dataset_type` 已改为 `'Custom25Dataset'`
5. [ ] `data_root` 指向正确的数据目录
6. [ ] `custom_imports` 中已导入 `models.datasets.custom25`
7. [ ] 数据目录结构符合要求（`train/labelTxt/`, `train/images/` 等）
8. [ ] 注解文件中类别名与 `CLASSES` 中的名称完全一致

### 快速验证脚本

```python
# 测试数据集是否能正常加载
from models.datasets.custom25 import Custom25Dataset

# 检查类别数
print(f'类别数: {len(Custom25Dataset.CLASSES)}')  # 应输出 25
print(f'类别: {Custom25Dataset.CLASSES}')

# 检查注册
from mmrotate.datasets.builder import ROTATED_DATASETS
print(f'已注册: Custom25Dataset' if 'Custom25Dataset' in ROTATED_DATASETS else '未注册!')
```

---

## 注意事项

1. **类别名称必须一致**：注解文件中的 `class_name` 字段必须与 `CLASSES` 元组中的名称**完全匹配**（区分大小写），否则该目标会被跳过
2. **label id 由 CLASSES 顺序决定**：第一个类别 id=0，第二个 id=1，依此类推，不需要改注解文件
3. **PALETTE 必须与 CLASSES 长度一致**：25 个类别需要 25 种颜色
4. **`num_classes` 只含前景类**：Oriented R-CNN 配置中 `num_classes=25` 表示 25 个前景类（背景类自动处理），不需写 26
5. **图片扩展名**：默认为 `.jpg`，如果数据集使用其他格式（如 `.png`），需要在数据类初始化时指定 `img_ext='.png'`，或在配置的 `train`/`val`/`test` 字典中添加该参数
