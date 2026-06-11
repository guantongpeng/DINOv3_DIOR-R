#!/usr/bin/env python3
"""
Plot training loss & accuracy curves and validation mAP from a JSON lines log file.

Usage:
    python plot_log.py <log_file.json>
"""

import json
import os
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict


def load_log_data(filepath):
    """Read JSON lines file, extract train and validation records."""
    train_records = []      # list of (step, dict_of_metrics)
    val_records = []        # list of (epoch, mAP)

    step = 0
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Skip records without 'mode' field (e.g., first config block)
            if 'mode' not in record:
                continue

            mode = record['mode']
            if mode == 'train':
                # Collect all loss values (keys starting with 'loss') and 'acc'
                metrics = {k: v for k, v in record.items() if k.startswith('loss') or k == 'acc'}
                if metrics:
                    train_records.append((step, metrics))
                    step += 1
            elif mode == 'val':
                if 'mAP' in record and 'epoch' in record:
                    val_records.append((record['epoch'], record['mAP']))

    return train_records, val_records


def plot_loss_curves(train_records, output_file="loss_curves.png"):
    """Plot each loss and 'acc' in a separate subplot, each curve with distinct color."""
    if not train_records:
        print("No training records found.")
        return

    # Collect all metric names (loss* + acc) that appear in any record
    metric_names = set()
    for _, metrics in train_records:
        metric_names.update(metrics.keys())
    # Ensure 'acc' is present even if not in first records (will be handled later)
    metric_names.add('acc')
    metric_names = sorted(metric_names)   # deterministic order, 'acc' will appear alphabetically

    # Prepare data: for each metric, list of (step, value)
    metric_data = {name: ([], []) for name in metric_names}   # step list, value list
    for step, metrics in train_records:
        for name in metric_names:
            val = metrics.get(name)
            if val is not None:
                metric_data[name][0].append(step)
                metric_data[name][1].append(val)
            # If metric missing at this step, skip (no interpolation)

    # Determine subplot grid layout
    n_metrics = len(metric_names)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Use a colormap to assign distinct colors
    colors = plt.cm.tab10.colors  # 10 distinct colors, cycle if more metrics

    for idx, name in enumerate(metric_names):
        ax = axes[idx]
        steps, values = metric_data[name]
        if steps:  # only plot if data exists
            color = colors[idx % len(colors)]
            ax.plot(steps, values, linewidth=1.5, color=color)
        ax.set_title(name)
        ax.set_xlabel("Step")
        ax.set_ylabel(name if name == 'acc' else "Loss")
        ax.grid(True, linestyle='--', alpha=0.6)

    # Hide any unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Loss & accuracy curves saved to {output_file}")


def plot_val_map(val_records, output_file="val_map.png"):
    """Plot mAP vs epoch."""
    if not val_records:
        print("No validation records found.")
        return

    epochs, maps = zip(*sorted(val_records))   # sort by epoch

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, maps, marker='o', linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.title("Validation mAP over Epochs")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Validation mAP curve saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Plot loss, acc, and mAP curves from JSON lines log.")
    parser.add_argument("log_file", help="Path to the JSON lines log file")
    args = parser.parse_args()

    log_dir = os.path.dirname(os.path.abspath(args.log_file))
    train_records, val_records = load_log_data(args.log_file)

    if train_records:
        output = os.path.join(log_dir, "loss_curves.png")
        plot_loss_curves(train_records, output_file=output)
    else:
        print("No training data found, skipping loss/acc plot.")

    if val_records:
        output = os.path.join(log_dir, "val_map.png")
        plot_val_map(val_records, output_file=output)
    else:
        print("No validation data found, skipping mAP plot.")


if __name__ == "__main__":
    main()