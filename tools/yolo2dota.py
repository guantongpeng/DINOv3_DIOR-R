#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO OBB → DOTA 标注格式转换工具
================================

将 YOLO 旋转框 (OBB) 格式的标注文件转换为 DOTA 格式，
适配 mmrotate / Oriented R-CNN 训练流程。

格式说明
--------

YOLO OBB 格式 (输入):
    每行一个目标，空格分隔，坐标为归一化值 (0~1):
        class_id x1 y1 x2 y2 x3 y3 x4 y4

DOTA 格式 (输出):
    每行一个目标，空格分隔，坐标为绝对像素值:
        x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty

用法示例
--------

    # 基本用法：指定类别列表文件
    python tools/yolo2dota.py \
        --yolo_dir data/Custom25/train/labels/ \
        --image_dir data/Custom25/train/images/ \
        --output_dir data/Custom25/train/labelTxt/ \
        --classes data/Custom25/classes.txt

    # 直接传入类别名
    python tools/yolo2dota.py \
        --yolo_dir data/Custom25/val/labels/ \
        --image_dir data/Custom25/val/images/ \
        --output_dir data/Custom25/val/labelTxt/ \
        --class_names "airplane,ship,vehicle,storage_tank,..." \
        --img_ext .png

    # 批量转换 train/val/test
    python tools/yolo2dota.py --batch data/Custom25/ --splits train val test
"""

import argparse
import glob
import os
import os.path as osp
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert YOLO OBB labels to DOTA format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ------------------------------------------------------------
    # 单个数据集转换参数
    # ------------------------------------------------------------
    parser.add_argument(
        '--yolo_dir',
        type=str,
        default=None,
        help='YOLO 标注文件夹路径（包含 .txt 文件）',
    )
    parser.add_argument(
        '--image_dir',
        type=str,
        default=None,
        help='图片文件夹路径（用于获取图像尺寸）',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='DOTA 标注输出文件夹路径',
    )
    parser.add_argument(
        '--classes',
        type=str,
        default=None,
        help='类别列表文件路径（每行一个类别名，顺序与 YOLO class_id 一致）',
    )
    parser.add_argument(
        '--class_names',
        type=str,
        default=None,
        help='逗号分隔的类别名列表，如 "airplane,ship,vehicle"（与 --classes 二选一）',
    )
    parser.add_argument(
        '--img_ext',
        type=str,
        default='.jpg',
        help='图片文件扩展名（默认: .jpg）。支持多个，用逗号分隔，如 ".jpg,.png"',
    )
    parser.add_argument(
        '--difficulty',
        type=int,
        default=0,
        help='DOTA 格式中的 difficulty 默认值（默认: 0）',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        default=False,
        help='是否覆盖已存在的输出文件',
    )

    # ------------------------------------------------------------
    # 批量转换参数
    # ------------------------------------------------------------
    parser.add_argument(
        '--batch',
        type=str,
        default=None,
        help='数据集根目录（配合 --splits 批量转换 train/val/test 三个子集）',
    )
    parser.add_argument(
        '--splits',
        type=str,
        nargs='+',
        default=['train', 'val', 'test'],
        help='批量转换的子集名称列表（默认: train val test）',
    )
    parser.add_argument(
        '--yolo_subdir',
        type=str,
        default='labels',
        help='YOLO 标注在子集下的子目录名（默认: labels）',
    )
    parser.add_argument(
        '--image_subdir',
        type=str,
        default='images',
        help='图片在子集下的子目录名（默认: images）',
    )
    parser.add_argument(
        '--output_subdir',
        type=str,
        default='labelTxt',
        help='DOTA 标注在子集下的子目录名（默认: labelTxt）',
    )

    return parser.parse_args()


def load_class_names(class_file: Optional[str] = None,
                     class_names_str: Optional[str] = None) -> List[str]:
    """加载类别名称列表。

    Args:
        class_file: 类别列表文件路径（每行一个类别名）。
        class_names_str: 逗号分隔的类别名字符串。

    Returns:
        类别名称列表，索引对应 YOLO 格式中的 class_id。
    """
    if class_file:
        with open(class_file, 'r') as f:
            classes = [line.strip() for line in f if line.strip()]
        print(f'[INFO] 从文件加载 {len(classes)} 个类别: {class_file}')
        return classes

    if class_names_str:
        classes = [name.strip() for name in class_names_str.split(',') if name.strip()]
        print(f'[INFO] 从参数加载 {len(classes)} 个类别: {classes}')
        return classes

    print('[ERROR] 请通过 --classes 或 --class_names 指定类别名称')
    sys.exit(1)


def get_image_size(image_path: str) -> Tuple[int, int]:
    """获取图片尺寸 (width, height)。"""
    try: 
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except Exception as e:
        print(f'[WARNING] 无法读取图片 {image_path}: {e}')
        return 0, 0


def find_image_file(img_id: str,
                    image_dir: str,
                    exts: List[str]) -> Optional[str]:
    """根据 img_id 在图片目录中查找对应图片文件。

    Args:
        img_id: 图片名（不含扩展名）。
        image_dir: 图片文件夹路径。
        exts: 支持的扩展名列表（含点号），如 ['.jpg', '.png']。

    Returns:
        完整图片路径，或 None。
    """
    for ext in exts:
        path = osp.join(image_dir, img_id + ext)
        if osp.isfile(path):
            return path
    return None


def yolo_obb_to_poly_abs(line: str,
                         img_w: int,
                         img_h: int) -> Optional[Tuple[int, List[float]]]:
    """解析 YOLO OBB 行并转为绝对坐标多边形。

    支持的格式：
        - 4 角点格式: class_id x1 y1 x2 y2 x3 y3 x4 y4 (归一化)
        - cxcywha 格式: class_id cx cy w h angle (归一化，需 OpenCV 转换)

    Args:
        line: YOLO 标注文件中的一行。
        img_w: 图片宽度（像素）。
        img_h: 图片高度（像素）。

    Returns:
        (class_id, [x1, y1, x2, y2, x3, y3, x4, y4]) 绝对坐标，或 None。
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

    if len(coords) == 8:
        # YOLO OBB 格式：归一化的 4 个角点 (x1,y1, x2,y2, x3,y3, x4,y4)
        abs_coords = []
        for i, val in enumerate(coords):
            if i % 2 == 0:  # x 坐标
                abs_coords.append(val * img_w)
            else:            # y 坐标
                abs_coords.append(val * img_h)
        return cls_id, abs_coords

    elif len(coords) == 5:
        # cxcywha 格式：cx, cy, w, h, angle (归一化)
        cx, cy, w, h, angle = coords
        cx_abs, cy_abs = cx * img_w, cy * img_h
        w_abs, h_abs = w * img_w, h * img_h
        # angle 通常在 YOLO OBB 中为弧度制
        import cv2
        rect = ((cx_abs, cy_abs), (w_abs, h_abs), np.degrees(angle))
        box = cv2.boxPoints(rect)  # 返回 4 个角点 (float32, shape 4x2)
        abs_coords = box.reshape(-1).tolist()
        return cls_id, abs_coords

    else:
        print(f'[WARNING] 无法解析行，坐标数为 {len(coords)}: {line.strip()[:60]}...')
        return None


def convert_single(yolo_dir: str,
                   image_dir: str,
                   output_dir: str,
                   class_names: List[str],
                   img_exts: List[str],
                   difficulty: int = 0,
                   overwrite: bool = False) -> Dict[str, int]:
    """转换一个数据集子集。

    Args:
        yolo_dir: YOLO 标注文件夹路径。
        image_dir: 图片文件夹路径。
        output_dir: 输出文件夹路径。
        class_names: 类别名称列表。
        img_exts: 支持的图片扩展名列表。
        difficulty: DOTA difficulty 字段默认值。
        overwrite: 是否覆盖已存在的输出文件。

    Returns:
        统计信息字典 {'total': int, 'skipped': int, 'converted': int, 'empty': int}。
    """
    os.makedirs(output_dir, exist_ok=True)

    yolo_files = glob.glob(osp.join(yolo_dir, '*.txt'))
    if not yolo_files:
        print(f'[ERROR] 未找到 YOLO 标注文件: {yolo_dir}')
        return {'total': 0, 'skipped': 0, 'converted': 0, 'empty': 0}

    stats = {'total': len(yolo_files), 'skipped': 0, 'converted': 0, 'empty': 0}

    for yolo_path in sorted(yolo_files):
        img_id = osp.splitext(osp.basename(yolo_path))[0]
        out_path = osp.join(output_dir, img_id + '.txt')

        # 检查是否覆盖
        if osp.exists(out_path) and not overwrite:
            stats['skipped'] += 1
            continue

        # 查找对应图片
        img_path = find_image_file(img_id, image_dir, img_exts)
        if img_path is None:
            print(f'[WARNING] 未找到图片 {img_id}，跳过')
            stats['skipped'] += 1
            continue

        img_w, img_h = get_image_size(img_path)
        if img_w == 0 or img_h == 0:
            print(f'[WARNING] 无法获取图片尺寸 {img_id}，跳过')
            stats['skipped'] += 1
            continue

        # 读取 YOLO 标注并转换
        dota_lines = []
        with open(yolo_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                result = yolo_obb_to_poly_abs(line, img_w, img_h)
                if result is None:
                    continue
                cls_id, abs_coords = result

                if cls_id < 0 or cls_id >= len(class_names):
                    print(f'[WARNING] {img_id}: class_id={cls_id} 超出范围 (0~{len(class_names)-1})，跳过')
                    continue

                class_name = class_names[cls_id]
                coords_str = ' '.join(f'{c:.4f}' for c in abs_coords)
                dota_line = f'{coords_str} {class_name} {difficulty}'
                dota_lines.append(dota_line)

        # 写入输出文件
        if dota_lines:
            with open(out_path, 'w') as f:
                f.write('\n'.join(dota_lines) + '\n')
            stats['converted'] += 1
        else:
            # 空标注：写入空文件（保留文件以匹配图片列表）
            open(out_path, 'w').close()
            stats['empty'] += 1

    return stats


def print_stats(stats: Dict[str, int], label: str = ''):
    """打印转换统计信息。"""
    prefix = f'[{label}] ' if label else ''
    print(f'{prefix}总计: {stats["total"]} | '
          f'已转换: {stats["converted"]} | '
          f'空标注: {stats["empty"]} | '
          f'跳过: {stats["skipped"]}')


def main():
    args = parse_args()

    # --- 加载类别名称 ---
    class_names = load_class_names(args.classes, args.class_names)
    print(f'[INFO] 共 {len(class_names)} 个类别:')
    for i, name in enumerate(class_names):
        print(f'  {i}: {name}')

    # --- 解析图片扩展名 ---
    img_exts = args.img_ext.split(',')
    img_exts = [e if e.startswith('.') else f'.{e}' for e in img_exts]

    # --- 批量模式 ---
    if args.batch:
        root = args.batch
        print(f'\n{"="*60}')
        print(f'批量转换模式: {root}')
        print(f'子集: {args.splits}')
        print(f'{"="*60}')

        total_stats = {'total': 0, 'skipped': 0, 'converted': 0, 'empty': 0}
        for split in args.splits:
            yolo_dir = osp.join(root, split, args.yolo_subdir)
            image_dir = osp.join(root, split, args.image_subdir)
            output_dir = osp.join(root, split, args.output_subdir)

            print(f'\n--- [{split}] ---')
            print(f'  YOLO 标注: {yolo_dir}')
            print(f'  图片目录:  {image_dir}')
            print(f'  输出目录:  {output_dir}')

            if not osp.isdir(yolo_dir):
                print(f'  [SKIP] YOLO 标注目录不存在')
                continue

            stats = convert_single(
                yolo_dir=yolo_dir,
                image_dir=image_dir,
                output_dir=output_dir,
                class_names=class_names,
                img_exts=img_exts,
                difficulty=args.difficulty,
                overwrite=args.overwrite,
            )
            print_stats(stats, split)
            for k in total_stats:
                total_stats[k] += stats[k]

        print(f'\n{"="*60}')
        print('全部转换完成')
        print_stats(total_stats, '总计')
        return

    # --- 单集模式 ---
    if not args.yolo_dir or not args.image_dir or not args.output_dir:
        print('[ERROR] 单集模式需要 --yolo_dir, --image_dir, --output_dir')
        print('        或使用 --batch 进行批量转换')
        sys.exit(1)

    print(f'\n{"="*60}')
    print(f'YOLO 标注: {args.yolo_dir}')
    print(f'图片目录:  {args.image_dir}')
    print(f'输出目录:  {args.output_dir}')
    print(f'{"="*60}')

    stats = convert_single(
        yolo_dir=args.yolo_dir,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        class_names=class_names,
        img_exts=img_exts,
        difficulty=args.difficulty,
        overwrite=args.overwrite,
    )
    print_stats(stats)
    print('转换完成')


if __name__ == '__main__':
    main()
