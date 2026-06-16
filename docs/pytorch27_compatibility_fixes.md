# PyTorch 2.7 兼容性修复记录

本文档记录了将 MMRotate 项目适配到 **PyTorch 2.7.1** 环境时遇到的兼容性问题及其修复方案。

## 环境信息

| 组件 | 版本 |
|------|------|
| Python | 3.12 |
| PyTorch | 2.7.1 (CUDA 12.8) |
| MMCV | 1.7.2 |
| MMRotate | 0.3.4 |
| MMDetection | 2.28.2 |
| timm | >= 1.0 |

---

## 修复 1: CUDA fork 冲突

### 问题
```
c10::Error: CUDA error: initialization error
DataLoader worker (pid 1274471) is killed by signal: Aborted
```

### 根因
配置文件中 `mp_start_method = 'fork'`。`fork()` 后子进程继承父进程的 CUDA 上下文，但 CUDA **不支持 fork**，子进程中任何 CUDA 操作（包括 Tensor 析构）都会导致崩溃。

### 修复
**文件**: `configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_dior.py`

```diff
- mp_start_method = 'fork'
+ mp_start_method = 'spawn'
```

`spawn` 方式下每个 DataLoader worker 子进程独立启动，不继承 CUDA 上下文，避免冲突。

---

## 修复 2: `_get_stream` 参数类型不兼容

### 问题
```
TypeError: _get_stream(): argument 'device' must be torch.device, not int
```

### 根因
PyTorch 2.7 的 `torch.nn.parallel._functions._get_stream()` 期望 `torch.device` 类型参数，但 mmcv 1.x 传递的是原始 `int`。

### 修复
**文件**: `tools/train.py`

在文件开头添加 monkey-patch，拦截 `int` 类型并转换为 `torch.device`：

```python
def _install_get_stream_patch():
    import torch.nn.parallel._functions as _torch_fns
    _orig = _torch_fns._get_stream
    def _patched(device):
        if isinstance(device, int):
            device = torch.device('cuda', device)
        return _orig(device)
    _torch_fns._get_stream = _patched
    try:
        import mmcv.parallel._functions as _mmcv_fns
        _mmcv_fns._get_stream = _patched
    except Exception:
        pass

_install_get_stream_patch()
```

---

## 修复 3: MMDistributedDataParallel 缺少 `_use_replicated_tensor_module` 属性

### 问题
```
AttributeError: 'MMDistributedDataParallel' object has no attribute '_use_replicated_tensor_module'
```

### 根因
PyTorch 2.7 的 `DistributedDataParallel.forward()` 调用 `_run_ddp_forward()`。mmcv 的 `MMDistributedDataParallel` 重写了该方法，在第 160 行访问 `self._use_replicated_tensor_module`，但该属性是新版 PyTorch DDP 才有的，mmcv 1.x 并未设置它。

### 修复
**文件**: `tools/train.py`

在文件中添加 monkey-patch，为 `MMDistributedDataParallel` 类添加该属性：

```python
def _install_mmddp_patch():
    from mmcv.parallel import MMDistributedDataParallel
    if not hasattr(MMDistributedDataParallel, '_use_replicated_tensor_module'):
        MMDistributedDataParallel._use_replicated_tensor_module = False

_install_mmddp_patch()
```

这使 `_run_ddp_forward` 走 `else` 分支使用 `self.module`，与训练行为一致。

---

## 修复 4: NMS 后处理中 `labels` 与 `bboxes` 设备不一致

### 问题
```
RuntimeError: indices should be either on cpu or on the same device as the indexed tensor (cpu)
```
发生在 `mmrotate/mmrotate/core/post_processing/bbox_nms_rotated.py` 第 58 行：
```python
bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]
```

### 根因
`labels` 由 `torch.arange(num_classes, dtype=torch.long)` 创建，默认在 CPU 上。而 `scores`/`bboxes` 在 CUDA 上，`valid_mask.nonzero()` 生成的索引 `inds` 也在 CUDA 上。PyTorch 2.7 不允许用 CUDA 索引去索引 CPU tensor。

### 修复
**文件**: `mmrotate/mmrotate/core/post_processing/bbox_nms_rotated.py`

```diff
- labels = torch.arange(num_classes, dtype=torch.long)
+ labels = torch.arange(num_classes, dtype=torch.long, device=scores.device)
```

这确保 `labels` 与 `scores`/`bboxes`/`inds` 在同一设备上。

---

## 修改文件汇总

| 文件 | 修改内容 |
|------|----------|
| `configs/oriented_rcnn/oriented_rcnn_dinov3_fpn_dior.py` | `mp_start_method` 改为 `spawn` |
| `tools/train.py` | 添加两个 monkey-patch（`_get_stream` + `_use_replicated_tensor_module`） |
| `mmrotate/mmrotate/core/post_processing/bbox_nms_rotated.py` | `torch.arange` 添加 `device` 参数 |

## 注意事项

1. 所有 monkey-patch 仅在 mmcv 1.x + PyTorch 2.7+ 组合下需要，升级到 mmcv 2.x 后应移除
2. `mp_start_method = 'spawn'` 是 CUDA 训练的标准实践，不受 PyTorch 版本限制
3. 如果在 `spawn` 模式下遇到 `__main__` 相关的 pickling 错误，请确保 `train.py` 中有 `if __name__ == '__main__':` 保护