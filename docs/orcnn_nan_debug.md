# Oriented R-CNN 训练 NaN 问题排查与修复

> 适用配置：`dinov3_dior_oriented_rcnn_mmr0.py`、`dinov3_dior_orcnn_cosine.py`
> 排查日期：2026-06-23
> 状态：**已修复并验证**

## 1. 现象

运行 `bash run_train_oriented_rcnn.sh 0` 后，训练在第 1 个 epoch 内全部损失变为 `nan`：

```
Epoch(train) [1][ 20/733]  lr: 3.9698e-05  ...  grad_norm: nan  loss: nan  loss_rpn_cls: nan  loss_rpn_bbox: nan  loss_cls: nan  acc: 0.0000  loss_bbox: nan
Epoch(train) [1][ 40/733]  lr: 4.6399e-05  ...  grad_norm: nan  loss: nan  ...
```

且一旦出现 NaN 就再也无法恢复（参数被永久污染），后续每个 iter 都是 NaN。

### 关键观察

对比同 backbone / 同数据集的三个实验：

| 实验 | backbone 冻结 | 优化器/调度 | 结果 |
| --- | --- | --- | --- |
| `dinov3_dior_fcos_mmr0`（单阶段 FCOS） | 否 | AdamW + LinearLR/MultiStep | **正常训练** |
| `dinov3_dior_fcos_cosine`（单阶段 FCOS） | 否 | AdamW + 分组 lr + cosine | **正常训练** |
| `dinov3_dior_orcnn_cosine`（两阶段 ORCNN） | 否 | AdamW + 分组 lr + cosine | iter 20 正常，**iter 40 起 NaN** |
| `dinov3_dior_oriented_rcnn_mmr0`（两阶段 ORCNN） | 是 | AdamW + LinearLR/MultiStep | **首个日志点（iter 20）即 NaN** |

由此得到三条核心线索：

1. **仅两阶段 Oriented R-CNN 出问题，单阶段 FCOS 不受影响** —— 问题出在两阶段检测头，而不是 DINOv3 backbone / FPN / 数据加载。
2. **ORCNN cosine 在 iter 20 仍正常，到 iter 40 才 NaN** —— 不是“第 0 步就崩”，而是训练一段时间后才崩，说明不是配置结构错误，而是某个**随数据出现的触发条件**。
3. **两阶段实验里，学习率越高 / warmup 越陡，崩溃越早**（mmr0 用更高 lr，iter 20 就崩；cosine warmup 很缓，撑到 iter 40）—— 暗示崩溃需要一个“坏样本”进入 batch，而采样顺序由随机种子决定。

## 2. 排查过程

### 2.1 排除“配置/结构错误”

将本配置的 `rpn_head` / `roi_head` / `bbox_coder` / `train_cfg` 与官方参考
`mmrotate/configs/oriented_rcnn/oriented-rcnn-le90_r50_fpn_1x_dota.py`
逐项对比，**完全一致**。模型结构没有问题。

### 2.2 复现与定位：anomaly detection 探针

写一个最小探针脚本，复用 `run_train_oriented_rcnn.sh` 的环境（PYTHONPATH、`DINOV3_SRC`），
用真实模型 + 真实数据跑前向/反向：

- 用**恒定 lr=1e-4、不裁剪梯度**手写训练循环 → 跑 40 步**完全正常**；
- 改用与真实训练一致的 `model.train_step(data, optim_wrapper)`（含 `clip_grad=35`）→ 跑 60 步**完全正常**。

也就是说：**在普通 batch 上，两阶段 ORCNN 完全可以稳定训练。**
那真实训练为何崩？只剩一个可能：**某个特定样本进入 batch 时触发了 inf/nan**。

### 2.3 锁定“坏数据”

两阶段旋转框编码器 `MidpointOffsetCoder.encode`（RPN）与 `DeltaXYWHTRBBoxCoder`（RCNN）
都按 box 的宽高做归一化 / 取对数：

```python
# mmrotate .../delta_midpointoffset_rbbox_coder.py (bbox2delta)
gw = hbb[..., 2] - hbb[..., 0]   # gt 水平外接矩形宽
gh = hbb[..., 3] - hbb[..., 1]   # gt 水平外接矩形高
dw = torch.log(gw / pw)
dh = torch.log(gh / ph)
da = (ga - gx) / gw
db = (gb - gy) / gh
```

当 gt 的 `gw` 或 `gh` 为 0 时：

- `torch.log(0) = -inf`
- `(gb - gy) / 0 = inf`（或 0/0 = nan）

这些值会成为 RPN/RCNN 的**回归目标**，再经 `SmoothL1Loss` → loss 为 inf → 反向 → 梯度 inf/nan
→ `clip_grad_norm_` 算出的总范数为 nan → 所有梯度乘 nan → **全部参数永久变 nan**。

于是直接扫描 DIOR-R 标签，统计旋转框宽/高 ≤ ε 的退化标注（用 `cv2.minAreaRect` 把 8 点 quad
转成 `cx,cy,w,h,a`，同 `DOTADataset`）：

```
[train] files=5862 boxes=32634 degenerate(w/h<=0.001)=1
     ('04137.txt', 1.0, 0.0, 'ship')     # 训练集 (train) 内
[val]   files=5863 boxes=35439 degenerate(w/h<=0.001)=1
     ('07007.txt', 0.0, 1.0, 'ship')     # val 也被并入 train (ConcatDataset)
[test]  files=11738 boxes=124445 degenerate(w/h<=0.001)=2
     ('15504.txt', 0.0, 0.0, 'ship')
     ('16734.txt', 0.0, 0.0, 'ship')
```

`train_dataloader` 是 `ConcatDataset([train, val])`，所以训练流里实际有 **2 张带退化标注的图**
（`04137.jpg`、`07007.jpg`）。

### 2.4 决定性验证

写探针强制把这 2 张图塞进同一个 batch 跑一步：

```
matched bad indices: [4915, 8204] -> ['04137.jpg', '07007.jpg']
loss on degenerate batch:
   loss = nan
   loss_rpn_cls = 0.7158
   loss_rpn_bbox = 0.0
   loss_cls = 3.4371
   acc = 0.0976
   loss_bbox = nan        # ← RCNN 第二阶段回归 loss
TOTAL = nan   -> NaN? True
```

**实锤：** 退化样本使 `loss_bbox`（RCNN 回归）变 nan，进而污染全部损失。
（注：RPN 阶段未报 nan，是因为退化框 IoU≈0，没被 anchor 命中；
而 RCNN 的 sampler 设置了 `add_gt_as_proposals=True`，会把退化 gt 当作 proposal 直接回归，于是触发除零。）

### 2.5 为什么 FCOS 不受影响

单阶段 FCOS 用的是 `DistanceAnglePointCoder`，是**点到边距离**的编码方式，
回归目标只依赖特征点位置，**不除以 gt 的宽高**，所以同样的退化标注对 FCOS 无害。

## 3. 根因结论

| 项 | 说明 |
| --- | --- |
| 直接原因 | DIOR-R 标注中存在零宽/零高的退化 `ship` 框（`04137.jpg`、`07007.jpg`，以及 test 的两张） |
| 触发机制 | 退化 gt 进入 batch → `DeltaXYWHTRBBoxCoder`/`MidpointOffsetCoder` 编码时 `log(0)` 与 `/0` → 回归目标 inf/nan → loss inf → 梯度 nan → 全参数 nan |
| 为何“间歇性” | 仅当退化样本被采到时才崩；采样顺序由随机种子决定，故每次崩溃的 iter 不同（mmr0 约 20、cosine 约 40） |
| 为何 FCOS 无恙 | FCOS 的点距离编码器不除以 box 宽高 |

## 4. 修复方案

在 `ConvertBoxType`（把 gt 转成 `rbox`）之后、`Resize`/`PackDetInputs` 之前，
加一个 `FilterAnnotations`，丢弃宽或高 ≤ 1px 的退化框：

```python
train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    # 滤掉 DIOR-R 中零宽/零高的退化标注，避免旋转框编码器除零 -> NaN。
    dict(type='mmdet.FilterAnnotations', min_gt_bbox_wh=(1, 1)),
    dict(type='mmdet.Resize', scale=(800, 800), keep_ratio=True),
    ...
]
```

`val_pipeline` 同样加上该步骤（test 集也有 2 张退化图，会影响评估时的 IoU 计算）。

### 为什么用 `FilterAnnotations`

- `mmdet.FilterAnnotations` 通过 `@autocast_box_type()` 装饰器，对任意 box 类型（含 mmrotate 的
  `RotatedBoxes`）都生效；
- 它用 `gt_bboxes.widths` / `gt_bboxes.heights` 判断，对 `RotatedBoxes` 即旋转框的真实 w/h，
  正好对应编码器里会除零的那个维度；
- `min_gt_bbox_wh=(1, 1)` 只会滤掉 <1px 的退化/噪声框，DIOR-R（800×800）最小的合法目标
  （如 vehicle）也有十几像素，不会误删；
- 相比手动改 4 个标注文件，这种方式**通用、可复用**，对未来可能出现的同类脏数据同样有效。

### 修改的文件

- `dinov3_dior_oriented_rcnn_mmr0.py`（train_pipeline + val_pipeline）
- `dinov3_dior_orcnn_cosine.py`（train_pipeline + val_pipeline）

> 注：两个 FCOS 配置**无需修改**（其编码器不受影响，历史上一直正常训练）。

## 5. 验证

### 5.1 探针验证

加 `FilterAnnotations` 后，再次把 `04137.jpg`、`07007.jpg` 塞进同一 batch：

```
loss on degenerate batch:
   loss_rpn_cls = 0.6701
   loss_rpn_bbox = 0.0
   loss_cls = 3.4740
   loss_bbox = 0.2199      # ← 不再 nan
TOTAL = 4.3641   -> NaN? False
```

### 5.2 端到端训练

重新执行 `bash run_train_oriented_rcnn.sh 0`，第 1 个 epoch 前两个日志点（
此前正是这两个点开始报 nan）已恢复正常：

```
[1][ 20/733]  grad_norm: 11.1931  loss: 1.4709  loss_rpn_cls: 0.5720  loss_rpn_bbox: 0.1340  loss_cls: 0.7402  acc: 98.3276  loss_bbox: 0.0246
[1][ 40/733]  grad_norm:  6.9601  loss: 1.1157  loss_rpn_cls: 0.4115  loss_rpn_bbox: 0.1201  loss_cls: 0.5191  acc: 96.2769  loss_bbox: 0.0650
```

## 6. 经验与排查清单

遇到“训练若干 iter 后突发 NaN、且单阶段正常 / 两阶段崩溃”的情况，按以下顺序排查：

1. **区分“立即 NaN”还是“训练若干步后 NaN”**
   - 立即（iter 0）→ 多为 dtype/AMP/初始化/配置结构问题；
   - 若干步后突发且不可恢复 → 强烈怀疑**个别坏样本**或**梯度爆炸**。
2. **同 backbone 的单阶段能否训练？** 能 → 问题在两阶段特有的 RPN/ROI/编码器路径。
3. **看崩溃是否与 batch 内容相关**：用 `torch.autograd.set_detect_anomaly(True)` 复现，
   或用“恒定 lr + 不裁剪”的极简训练循环排除调度器/裁剪干扰，定位是不是“正常 batch 都没事”。
4. **怀疑坏数据时，直接扫标注**：对旋转框数据集，重点检查宽/高为 0 的退化框，
   以及长宽比极端（接近 1:∞）的框——这类框会让依赖 `log(w)`、`/w`、`/h` 的编码器产生 inf。
5. **修复优先级**：优先用通用 pipeline 过滤（`FilterAnnotations`）而非手改数据，
   保证脚本对新数据同样健壮。
6. **回归测试**：修复后务必用探针单独喂“坏样本”确认 loss 有限，再跑端到端训练跨过原先崩溃的 iter。

## 附：相关代码位置

- 编码器（除零源头）：
  - `mmrotate/models/task_modules/coders/delta_midpointoffset_rbbox_coder.py`（`MidpointOffsetCoder`，RPN 用）
  - `DeltaXYWHTRBBoxCoder`（mmdet，RCNN bbox head 用）
- 过滤算子：`mmdet/datasets/transforms/loading.py` → `FilterAnnotations`
- 退化标注：DIOR-R `train/labelTxt/04137.txt`、`val/labelTxt/07007.txt`、
  `test/labelTxt/15504.txt`、`test/labelTxt/16734.txt`（均为 `ship`）
