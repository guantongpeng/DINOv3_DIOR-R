# YOLO26 检测头训练效果差的排查与修复

> 适用配置：`configs/yolo26/yolo26_dinov3_fpn_train_dior.py`
> 关联文档：[YOLO26 旋转目标检测头](./yolo26_detection_head.md)

## 1. 问题现象

使用 YOLO26 检测头（DINOv3 ViT-B + ViTDetFPN）在 DIOR-R 上训练，效果远低于同等骨干的 Oriented R-CNN：

| 模型 | epoch 3 | epoch 60（val, mAP@0.50） |
|------|---------|---------------------------|
| Oriented R-CNN | 0.350 | **0.693**（最终 best 0.710） |
| YOLO26 | 0.014 | **0.245** |

Oriented R-CNN 在 **epoch 3 就已 0.35**，比 YOLO26 训练到最后的 0.245 还高。这是典型的「推理管线损坏」特征，而非检测头能力不足。

排查结论：**不是 YOLO26 检测头本身不行，是代码有 3 个 Bug。** 其中 O2M 分支的训练 loss 是健康收敛的（4.5 → 1.56），问题完全出在推理路径与 O2O 分支初始化上。

---

## 2. Bug 1（致命｜推理）：对 O2M head 使用了 NMS-free 路径

### 根因

`simple_test` 调用的是 O2M（one-to-many）分支（`self.bbox_head(x)`），但 `test_cfg.end2end=True` 让 `get_bboxes` 走 `_get_bboxes_end2end`（仅 score_thr + top-K，**没有 NMS**）。

O2M 训练用 `tal_topk=13`（每个 GT 分配约 13 个正样本），所以 O2M 头天然输出大量重叠框。NMS-free 把它们**全部保留**，导致假正例爆炸 + `max_per_img=300` 被一堆重复框占满，召回率与精度同时崩盘。

### 证据（epoch 60，val）

| 类别 | GT | 检测框（修复前） | 比值 |
|------|----|------------------|------|
| ship | 14322 | 81206 | 5.7× |
| vehicle | 7015 | 91430 | 13× |
| airplane | 650 | 10206 | 15.7× |

`harbor` recall 0.40、`bridge` 0.44、`golffield` 0.32 —— `max_per_img=300` 被重复框占满，密集目标排不进去。

### 修复

`configs/yolo26/yolo26_dinov3_fpn_train_dior.py:153`：`end2end=True` → `end2end=False`，让 O2M 输出走带 NMS 的 `_get_bboxes_nms`（NMS 配置 `nms_rotated, iou_thr=0.1` 原本就写在配置里，只是被 `end2end=True` 短路）。

同时把 `models/heads/yolo26_rotated_head.py:1178` 的默认值从 `True` 改为 `False`，避免缺省时静默触发 NMS-free。

### 验证（同一 checkpoint epoch_60，同一 val split，仅推理路径不同）

| 推理路径 | val mAP@0.50 |
|----------|--------------|
| NMS-free（Bug1） | **0.245** |
| 加 NMS（修复后） | **0.583** |

典型类别 AP：ship 0.292→**0.801**，vehicle 0.211→**0.689**，airplane 0.249→**0.810**。检测框数量回归正常：ship 17412/14322 = 1.2×。

> 注意：训练日志里的 `Epoch(val)` 是 val split（5863 张，batch 8 → 733 iter）；`tools/test.py` 默认评估 test split（11738 张）。比较时务必同 split。同一 epoch_60 在 test split 上 NMS 后为 0.497。

---

## 3. Bug 2（致命｜训练）：O2O 分支分类 bias 未初始化 → loss 爆炸

### 根因

`YOLO26RotatedHead.init_weights()`（`models/heads/yolo26_rotated_head.py:351`）在 `init_cfg is not None` 时执行 `super().init_weights()` 后**直接 `return`**，导致其后对 `o2o_conv_cls.bias` 的初始化**永远执行不到**。而 `init_cfg` 的 `override` 只覆盖 O2M 的 `conv_cls`，没有覆盖 `o2o_conv_cls`。

结果：`o2o_conv_cls.bias = 0` → `sigmoid(0) = 0.5`，O2O 分支所有类别、所有锚点的初始概率都是 0.5。FocalLoss 在 ~53000 锚点 × 20 类 × 0.5 概率上求和再除以 `num_pos`（O2O 每张图仅 1 个正样本/GT），loss 直接飙到约 **45000**。

### 证据（旧训练日志，progressive loss 启动瞬间）

| epoch | O2O 权重 | `o2o_loss_cls`（含权重） | 折算原始 | grad_norm |
|-------|----------|---------------------------|----------|-----------|
| 60 | 0.000 | —— | —— | 正常 |
| 61 | 0.011 | **497.56** | ≈ 45000 | **471** |
| 62 | 0.022 | **433.99** | ≈ 19700 | 412 |

O2O progressive loss 一启动，训练立刻崩溃。

### 修复

`models/heads/yolo26_rotated_head.py:351` —— 去掉 `init_weights` 的提前 `return`，让下面的分类/objectness bias 初始化**始终执行**：

```python
def init_weights(self):
    if self.init_cfg is not None:
        super().init_weights()
    else:
        # from-scratch 手动初始化
        ...
    # 始终执行：分类 / objectness 分支的 focal-loss 先验 bias
    bias_init = bias_init_with_prob(0.01)
    for conv_cls in self.conv_cls:
        nn.init.constant_(conv_cls.bias, bias_init)
    if hasattr(self, 'o2o_conv_cls'):
        nn.init.constant_(self.o2o_conv_cls.bias, bias_init)
    if hasattr(self, 'o2o_conv_obj'):
        nn.init.constant_(self.o2o_conv_obj.bias, bias_init)
```

### 验证

- 独立测试（新建 head，`o2o_weight=1.0`，随机特征）：`o2o_loss_cls` 从 ≈45000 → **1.48**。
- 真实训练 epoch 62（旧爆炸点）：`o2o_loss_cls` 497.56 → **0.008～0.012**，grad_norm 471 → 12～17。

### resume 时的特别注意

`resume-from` 会**跳过 `init_weights()`**，直接加载 checkpoint 权重。因此旧的 `epoch_60.pth` 里 O2O bias 仍是 0（sigmoid 0.5），resume 后会在 epoch 61 再次爆炸。

解决办法：对 resume 用的 checkpoint 打补丁，仅重置这两个 prediction bias（其余 O2O 参数为 BN/conv 默认值，无需动）：

```python
# /tmp/kilo/patch_ckpt_o2o.py 核心逻辑
import torch
from mmcv.cnn import bias_init_with_prob
ckpt = torch.load('.../epoch_60.pth', map_location='cpu')
sd = ckpt['state_dict']
bias_init = bias_init_with_prob(0.01)
for k in ['bbox_head.o2o_conv_cls.bias', 'bbox_head.o2o_conv_obj.bias']:
    torch.nn.init.constant_(sd[k], bias_init)   # sigmoid 0.5 -> 0.01
ckpt['state_dict'] = sd
torch.save(ckpt, '.../epoch_60_o2ofixed.pth')
```

---

## 4. Bug 3（设计缺陷）：O2O 分支训练了却从不用于推理

### 根因

`simple_test` 永远调用 `self.bbox_head(x)`（O2M 分支），从不调用 `forward_o2o`。即使 O2O 训练好了，端到端 NMS-free 推理也用不上它 —— 整个 dual-head 设计形同虚设。

### 修复

`models/detectors/dinov3_yolo26.py:161` —— 让 `simple_test` 根据 `test_cfg.end2end` 选择分支：

```python
test_cfg = self.bbox_head.test_cfg or {}
use_end2end = test_cfg.get('end2end', False)
if use_end2end and hasattr(self.bbox_head, 'forward_o2o'):
    outs = self.bbox_head.forward_o2o(x)   # O2O: NMS-free
else:
    outs = self.bbox_head(x)               # O2M: 走 NMS
```

配合 Bug1 的 `end2end=False`，训练期 eval 用 O2M+NMS（可靠指标、用于 `save_best`）；最终评估时用 `--cfg-options model.test_cfg.end2end=True` 即可切换到 O2O NMS-free 路径。

### 关于 O2O 分支的 detached 特征

`forward_train` 中 O2O 分支使用 `f.detach()` 的特征以「避免梯度冲突」。这意味着 O2O loss **只更新 O2O 分支自身参数**，不影响骨干/neck/O2M。因此：
- O2M 训练不受 O2O progressive loss 开关影响（开了只是多训一个并行头，纯计算开销，不损害 O2M）。
- O2O 头是在 O2M 训出来的骨干特征上独立学习的 NMS-free 头，可作为 O2M+NMS 之外的第二推理路径。

---

## 5. 重训方案

从打了 O2O 补丁的 `epoch_60_o2ofixed.pth` **resume**（保留已收敛的 O2M/骨干，省去前 60 epoch），继续到 epoch 200：

```bash
RESUME=work_dirs/yolo26_dinov3_fpn_train_dior_20260616_191827/epoch_60_o2ofixed.pth \
WORK_DIR=work_dirs/yolo26_dinov3_fpn_train_dior_bugfix \
MASTER_PORT=29600 bash scripts/yolo26_vitb_train.sh
```

- 8 GPU，`samples_per_gpu=16`（有效 batch 128），余弦退火继续到 epoch 200。
- progressive loss：epoch 60→150 O2O 权重 0→1 线性 ramp，150→200 保持 1.0。
- 训练期 eval 每 3 epoch，`end2end=False`（O2M+NMS），`save_best='mAP@0.50'`。
- 监控：`tail -f work_dirs/yolo26_dinov3_fpn_train_dior_bugfix/train.log`。

---

## 6. 最终评估（两条推理路径对比）

```bash
CKPT=work_dirs/yolo26_dinov3_fpn_train_dior_bugfix/best_mAP*.pth

# 路径 A：O2M + NMS（可靠，预期 > 0.60）
TEST_CKPT=$CKPT WORK_DIR=work_dirs/yolo26_dinov3_fpn_train_dior_bugfix \
  SAVE_VIS=0 NUM_GPUS=8 bash scripts/test.sh

# 路径 B：O2O NMS-free（Bug3 启用）
python -m torch.distributed.run --nproc_per_node=8 --master_port=29610 \
  tools/test.py configs/yolo26/yolo26_dinov3_fpn_train_dior.py $CKPT \
  --launcher pytorch --eval mAP \
  --cfg-options model.test_cfg.end2end=True
```

> DIOR-R 密集小目标多，两阶段（Oriented R-CNN）天然更强；单阶段 anchor-free（YOLO26/FCOS 类）在 DINOv3 上做到 0.5～0.6 是合理预期。两条路径中 O2M+NMS 通常更稳，O2O NMS-free 视训练情况而定。

---

## 7. 关键经验

1. **「单阶段 head 远低于两阶段」先怀疑推理/后处理，别急着换 head。** Oriented R-CNN epoch 3 就 0.35（高于 YOLO 最终值）是推理管线损坏的强信号。
2. **NMS-free（end2end）只对 O2O 分支有效。** 对 O2M（one-to-many）输出做 NMS-free 等于保留全部重复框。分支与后处理必须配对。
3. **`init_weights` 里 `return` 之后的代码是死代码。** 改动初始化逻辑后，务必确认目标分支的 bias 真的被设置过。
4. **`resume-from` 不跑 `init_weights`。** 修复初始化类 Bug 后，旧 checkpoint 里对应参数仍是旧值，resume 前要打补丁或从头训。
5. **比较 mAP 必须同 split。** 训练日志 `Epoch(val)` 与 `tools/test.py`（默认 test）是不同划分。
6. **focal/分类分支 bias 应初始化到负的先验值**（`bias_init_with_prob(0.01)` ≈ −4.6），否则 sigmoid≈0.5 会让初始 loss 异常大。
