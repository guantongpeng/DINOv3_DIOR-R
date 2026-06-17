#!/usr/bin/env python3
"""
Plot training loss / accuracy / lr curves and validation mAP from the
MMDET / MMRotate JSON-lines logs (the ``*.log.json`` files written next to
``*.log`` under ``work_dirs/``).

Robust to the different metric-naming conventions used by the detectors in
this repo:

  * Oriented R-CNN : ``loss_rpn_cls`` / ``loss_cls`` / ``loss_bbox`` / ``acc`` / ``loss``
  * YOLO26         : ``o2m_loss_cls`` / ``o2m_loss_bbox`` / ``o2m_loss_angle`` /
                     ``o2m_loss_obj`` / ``loss``
  * RoI-Transformer: ``loss_rpn_cls`` / ``s0.loss_cls`` / ``s0.acc`` /
                     ``s1.loss_cls`` / ``s1.acc`` / ``loss``

Any numeric field that is not a bookkeeping key is treated as a plottable
metric, so new/renamed losses are picked up automatically.

Usage
-----
  # a single log file
  python tools/plot_loss.py work_dirs/xxx/yyy.log.json

  # a whole work_dir (every *.log.json inside is processed)
  python tools/plot_loss.py work_dirs/xxx

  # default: process every *.log.json under work_dirs/
  python tools/plot_loss.py

  # EMA smoothing factor in [0, 1): 0 = raw, higher = smoother (default 0.6)
  python tools/plot_loss.py work_dirs/xxx --smooth 0.6

  # only redraw the validation mAP figure
  python tools/plot_loss.py work_dirs/xxx --only val
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')  # headless-safe: never try to open a GUI window
import matplotlib.pyplot as plt  # noqa: E402

# Bookkeeping fields that must never be treated as plottable metrics. Everything
# else that is numeric (int/float, not bool) on a train/val record is a metric.
META_KEYS = {
    'mode', 'epoch', 'iter', 'memory', 'data_time', 'time',
    # 'lr' is bookkeeping but we extract it on purpose for its own curve,
    # so it is handled separately rather than left in the generic metric set.
}

DEFAULT_MAP_KEYS = ['mAP', 'mAP@0.50', 'mAP@0.75', 'mAP@50:95']
TAB10 = plt.cm.tab10.colors


def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def ema_smoothing(values, alpha):
    """Exponential moving average. ``alpha`` in [0, 1); 0 disables smoothing."""
    if alpha <= 0 or len(values) == 0:
        return list(values)
    out = []
    s = values[0]
    for v in values:
        s = alpha * s + (1.0 - alpha) * v
        out.append(s)
    return out


def load_log_data(filepath):
    """Read a JSON-lines log and return ``(train_records, val_records)``.

    Each train record is ``(global_step, {metric: value})`` where ``global_step``
    is a monotonically increasing counter (robust to resume / restarts because
    it follows file order). Each val record is ``(epoch, {mAP_metric: value})``.
    """
    train_records = []   # (step, {metric: value})
    val_records = []     # (epoch, {mAP: value})

    step = 0
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(record, dict) or 'mode' not in record:
                continue

            mode = record['mode']
            if mode == 'train':
                metrics = {k: v for k, v in record.items()
                           if k not in META_KEYS and k != 'lr' and _is_number(v)}
                if metrics:
                    train_records.append((step, metrics))
                    step += 1
            elif mode == 'val':
                map_metrics = {k: v for k, v in record.items()
                               if 'map' in k.lower() and _is_number(v)}
                if map_metrics:
                    epoch = record.get('epoch')
                    if epoch is None:  # fall back to iter, then to a counter
                        epoch = record.get('iter', len(val_records) + 1)
                    val_records.append((epoch, map_metrics))

    return train_records, val_records


def load_lr_curve(filepath):
    """Return ``[(step, lr), ...]`` extracted from train records (lr may be 0.0)."""
    lr_data = []
    step = 0
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(record, dict):
                continue
            if record.get('mode') == 'train' and 'lr' in record \
                    and _is_number(record['lr']):
                lr_data.append((step, record['lr']))
                step += 1
    return lr_data


def _group_metric_names(metric_names):
    """Split metric names into loss / acc / grad groups (by substring match)."""
    loss_keys, acc_keys, grad_keys, misc_keys = [], [], [], []
    for name in sorted(metric_names):
        low = name.lower()
        if 'loss' in low:
            loss_keys.append(name)
        elif 'acc' in low:
            acc_keys.append(name)
        elif 'grad' in low:
            grad_keys.append(name)
        else:
            misc_keys.append(name)
    return loss_keys, acc_keys, grad_keys, misc_keys


def _collect_series(train_records, names):
    """For each metric name, gather ``([steps], [values])`` skipping missing."""
    series = {}
    for name in names:
        xs, ys = [], []
        for step, metrics in train_records:
            v = metrics.get(name)
            if v is not None:
                xs.append(step)
                ys.append(v)
        series[name] = (xs, ys)
    return series


# Per-subplot target size (inches). The figure size is derived from the grid so
# that every panel keeps a readable aspect ratio no matter how many metrics.
SUBPLOT_W = 4.6
SUBPLOT_H = 3.2
MAX_COLS = 3


def _grid_shape(n, max_cols=MAX_COLS):
    """Choose a (rows, cols) grid for *n* panels, capping columns at max_cols."""
    if n <= 0:
        return 1, 1
    n_cols = min(max_cols, n)
    n_rows = (n + n_cols - 1) // n_cols
    return n_rows, n_cols


def _plot_subplots(series, names, ylabel, title, output_file,
                   smooth=0.6, sharey=False):
    n = len(names)
    n_rows, n_cols = _grid_shape(n)
    # Single panel reads better a bit wider; otherwise tile at SUBPLOT_* sizes.
    fig_w = max(SUBPLOT_W * n_cols, 8.0 if n == 1 else SUBPLOT_W * n_cols)
    fig_h = SUBPLOT_H * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h),
                             sharey=sharey, squeeze=False)
    axes = axes.flatten()
    for idx, name in enumerate(names):
        ax = axes[idx]
        xs, ys = series.get(name, ([], []))
        if xs:
            color = TAB10[idx % len(TAB10)]
            ax.plot(xs, ys, linewidth=1.0, color=color, alpha=0.35,
                    label='raw' if smooth > 0 else None)
            if smooth > 0:
                ax.plot(xs, ema_smoothing(ys, smooth),
                        linewidth=1.8, color=color, label='ema')
            if smooth > 0:
                ax.legend(loc='best', fontsize=8)
        ax.set_title(name)
        ax.set_xlabel('Step')
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.5)
    for idx in range(len(names), len(axes)):
        axes[idx].set_visible(False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_file, dpi=150)
    plt.close(fig)
    print(f'  -> {os.path.basename(output_file)}')


def plot_loss_curves(train_records, output_file, smooth=0.6):
    """Plot every loss component (and grad_norm) in its own subplot."""
    if not train_records:
        print('  No training records found, skipping loss/acc plots.')
        return False

    names = set()
    for _, metrics in train_records:
        names.update(metrics.keys())
    loss_keys, acc_keys, grad_keys, misc_keys = _group_metric_names(names)

    series = _collect_series(train_records, names)
    made = False

    if loss_keys:
        _plot_subplots(series, loss_keys, 'Loss',
                       'Training Loss', output_file, smooth=smooth)
        made = True

    if acc_keys:
        base, ext = os.path.splitext(output_file)
        _plot_subplots(series, acc_keys, 'Accuracy',
                       'Training Accuracy',
                       f'{base}_acc{ext}', smooth=smooth)
        made = True

    if grad_keys:
        base, ext = os.path.splitext(output_file)
        _plot_subplots(series, grad_keys, 'Gradient norm',
                       'Gradient Norm',
                       f'{base}_grad{ext}', smooth=smooth)
        made = True

    if misc_keys:
        base, ext = os.path.splitext(output_file)
        _plot_subplots(series, misc_keys, 'Value',
                       'Other Metrics',
                       f'{base}_misc{ext}', smooth=smooth)
        made = True

    return made


def plot_lr_curve(lr_data, output_file):
    if not lr_data:
        return False
    xs, ys = zip(*lr_data)
    if len(set(ys)) <= 1:  # constant lr is uninteresting
        return False
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(xs, ys, linewidth=1.5, color=TAB10[1])
    ax.set_xlabel('Step')
    ax.set_ylabel('Learning rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)
    print(f'  -> {os.path.basename(output_file)}')
    return True


def plot_val_map(val_records, output_file, map_keys=None):
    """Plot every available mAP metric vs epoch on one figure."""
    if not val_records:
        print('  No validation records found, skipping mAP plot.')
        return False

    sorted_records = sorted(val_records, key=lambda x: x[0])
    all_map_names = set()
    for _, m in sorted_records:
        all_map_names.update(m.keys())

    if map_keys:
        ordered = [k for k in map_keys if k in all_map_names]
        ordered += sorted(all_map_names - set(ordered))
        map_names = ordered
    else:
        map_names = sorted(all_map_names)

    epochs = [e for e, _ in sorted_records]

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, name in enumerate(map_names):
        values = [m.get(name) for _, m in sorted_records]
        color = TAB10[idx % len(TAB10)]
        ax.plot(epochs, values, marker='o', linewidth=1.6,
                color=color, label=name)
        best_i = max(range(len(values)),
                     key=lambda i: values[i] if values[i] is not None else -1)
        if values[best_i] is not None:
            ax.annotate(f'{values[best_i]:.4f}',
                        (epochs[best_i], values[best_i]),
                        textcoords='offset points', xytext=(0, 6),
                        ha='center', fontsize=8, color=color)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAP')
    ax.set_title('Validation mAP over Epochs')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)
    print(f'  -> {os.path.basename(output_file)}')
    return True


def process_log(filepath, smooth=0.6, map_keys=None, only=None):
    """Process a single ``*.log.json`` file; write figures next to it."""
    log_dir = os.path.dirname(os.path.abspath(filepath))
    print(f'Processing: {filepath}')

    train_records, val_records = load_log_data(filepath)

    wrote_something = False
    if only in (None, 'train', 'loss'):
        out = os.path.join(log_dir, 'loss_curves.png')
        if plot_loss_curves(train_records, out, smooth=smooth):
            wrote_something = True

    if only in (None, 'lr'):
        if plot_lr_curve(load_lr_curve(filepath),
                         os.path.join(log_dir, 'lr_curve.png')):
            wrote_something = True

    if only in (None, 'val', 'map'):
        out = os.path.join(log_dir, 'val_map.png')
        if plot_val_map(val_records, out, map_keys=map_keys):
            wrote_something = True

    if not wrote_something:
        print('  (no plottable data found in this log)')
    return wrote_something


def _resolve_log_files(target):
    """Expand a file/dir/None into a list of ``*.log.json`` paths."""
    if target is None:
        root = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'work_dirs')
        return sorted(glob.glob(os.path.join(root, '*', '*.log.json')))

    target = os.path.abspath(target)
    if os.path.isdir(target):
        files = sorted(glob.glob(os.path.join(target, '*.log.json')))
        if not files:  # also search one level down (work_dirs/<run>/x.log.json)
            files = sorted(glob.glob(os.path.join(target, '*', '*.log.json')))
        return files
    if os.path.isfile(target):
        return [target]
    raise FileNotFoundError(f'No such file or directory: {target}')


def main():
    parser = argparse.ArgumentParser(
        description='Plot loss / acc / lr / mAP curves from JSON-lines logs.')
    parser.add_argument(
        'target', nargs='?', default=None,
        help='A *.log.json file, a work_dir, or omit to process all logs '
             'under work_dirs/.')
    parser.add_argument('--smooth', type=float, default=0.6,
                        help='EMA smoothing factor in [0,1); 0 = raw '
                             '(default: %(default)s).')
    parser.add_argument('--map-keys', nargs='*', default=None,
                        help='Preferred mAP keys to plot (e.g. mAP mAP@0.50). '
                             'Defaults to all mAP* found in the log.')
    parser.add_argument('--only', choices=['train', 'loss', 'lr', 'val', 'map'],
                        default=None,
                        help='Only draw a subset of figures.')
    args = parser.parse_args()

    files = _resolve_log_files(args.target)
    if not files:
        print('No *.log.json files found.')
        return

    n_ok = 0
    for f in files:
        if process_log(f, smooth=args.smooth,
                       map_keys=args.map_keys, only=args.only):
            n_ok += 1
    print(f'\nDone: {n_ok}/{len(files)} log(s) produced figures.')


if __name__ == '__main__':
    main()
