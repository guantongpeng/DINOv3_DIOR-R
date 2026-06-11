#!/usr/bin/env python3
"""Prepare Star-1021+Extend3 Dataset for MMRotate Training.

Converts YOLO OBB labels to DOTA format and creates the expected
directory structure for mmrotate.

Usage:
    python data/prepare_star.py --data_root /mnt/ht2-nas2/00-model/Datasets/star-1021_1016+extend3
"""

import argparse
import glob
import os
import os.path as osp
import sys

import numpy as np
from PIL import Image

# 25-class Chinese names for star-1021+extend3
CLASS_NAMES = [
    "两栖攻击舰", "侦察机", "加油机", "反潜巡逻机", "商业客机",
    "坦克", "导弹快艇", "巡洋舰", "扫雷艇", "护卫舰",
    "机场", "武装直升机", "民用客轮", "登陆舰", "空天战斗机",
    "航空母舰", "补给舰", "装甲运输车", "轰炸机", "运输机",
    "通用直升机", "重型运输车", "隐身战斗机", "预警机", "驱逐舰",
]


def yolo_obb_to_dota(line, img_w, img_h, class_names):
    """Convert one YOLO OBB line to DOTA format string.

    YOLO OBB: class_id x1 y1 x2 y2 x3 y3 x4 y4 (normalized)
    DOTA:     x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty
    """
    parts = line.strip().split()
    if not parts:
        return None
    try:
        values = [float(p) for p in parts]
    except ValueError:
        return None

    cls_id = int(values[0])
    coords = values[1:]

    if len(coords) != 8:
        return None

    abs_coords = []
    for i, val in enumerate(coords):
        if i % 2 == 0:
            abs_coords.append(val * img_w)
        else:
            abs_coords.append(val * img_h)

    if cls_id < 0 or cls_id >= len(class_names):
        return None

    class_name = class_names[cls_id]
    coord_str = " ".join(f"{c:.4f}" for c in abs_coords)
    return f"{coord_str} {class_name} 0"


def convert_split(data_root, split, class_names, img_ext=".tif"):
    """Convert a single split (train/val/test) from YOLO to DOTA."""
    labels_dir = osp.join(data_root, split, "labels")
    images_dir = osp.join(data_root, split, "images")
    output_dir = osp.join(data_root, split, "labelTxt")

    if not osp.isdir(labels_dir):
        print(f"[SKIP] Labels dir not found: {labels_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    label_files = glob.glob(osp.join(labels_dir, "*.txt"))
    print(f"[{split}] Found {len(label_files)} label files")

    converted = 0
    skipped = 0

    for label_path in sorted(label_files):
        img_id = osp.splitext(osp.basename(label_path))[0]
        img_path = osp.join(images_dir, img_id + img_ext)

        if not osp.isfile(img_path):
            # Try other extensions
            for ext in [".jpg", ".png", ".bmp"]:
                alt = osp.join(images_dir, img_id + ext)
                if osp.isfile(alt):
                    img_path = alt
                    break
            else:
                print(f"[WARNING] Image not found: {img_id}")
                skipped += 1
                continue

        try:
            with Image.open(img_path) as img:
                w, h = img.size
        except Exception as e:
            print(f"[WARNING] Cannot read {img_path}: {e}")
            skipped += 1
            continue

        dota_lines = []
        with open(label_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                result = yolo_obb_to_dota(line, w, h, class_names)
                if result:
                    dota_lines.append(result)

        out_path = osp.join(output_dir, img_id + ".txt")
        if dota_lines:
            with open(out_path, "w") as f:
                f.write("\n".join(dota_lines) + "\n")
        else:
            # Empty annotation: write empty file
            open(out_path, "w").close()

        converted += 1

    print(f"[{split}] Converted: {converted}, Skipped: {skipped}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Star-1021+Extend3 dataset for MMRotate"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/mnt/ht2-nas2/00-model/Datasets/star-1021_1016+extend3",
        help="Root directory of the star dataset",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "val", "test"],
        help="Splits to convert",
    )
    args = parser.parse_args()

    print(f"Star-1021+Extend3 Dataset Preparation")
    print(f"Data root: {args.data_root}")
    print(f"Classes ({len(CLASS_NAMES)}):")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {i}: {name}")

    for split in args.splits:
        convert_split(args.data_root, split, CLASS_NAMES)

    # Write classes.txt for reference
    classes_path = osp.join(args.data_root, "classes.txt")
    with open(classes_path, "w") as f:
        for name in CLASS_NAMES:
            f.write(name + "\n")
    print(f"\nClasses list written to: {classes_path}")
    print("Done! Dataset is ready for MMRotate training.")


if __name__ == "__main__":
    main()