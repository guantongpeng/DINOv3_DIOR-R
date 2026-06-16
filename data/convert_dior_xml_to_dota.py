#!/usr/bin/env python3
"""
Convert DIOR-R XML annotations (oriented bounding boxes) to DOTA format,
and organize dataset into mmrotate-compatible directory structure.

Input structure:
    data/DIOR-R/
    ├── Annotations/Oriented Bounding Boxes/*.xml   # 23,463 XML files
    ├── JPEGImages-trainval/*.jpg                    # 11,725 train+val images
    ├── JPEGImages-test/*.jpg                        # 11,738 test images
    └── Main/
        ├── train.txt    # 5,862 IDs
        ├── val.txt      # 5,863 IDs
        └── test.txt     # 11,738 IDs

Output structure:
    data/DIOR-R/
    ├── train/
    │   ├── images/      # symlinks to JPEGImages-trainval
    │   └── labelTxt/    # DOTA-format .txt annotations
    ├── val/
    │   ├── images/
    │   └── labelTxt/
    └── test/
        ├── images/      # symlinks to JPEGImages-test
        └── labelTxt/

DOTA format per line:
    x1 y1 x2 y2 x3 y3 x4 y4 category difficult

Usage:
    python data/convert_dior_xml_to_dota.py
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

# DIOR-R 20 categories
DIOR_CATEGORIES = [
    'airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
    'chimney', 'dam', 'Expressway-Service-area', 'Expressway-toll-station',
    'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship',
    'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle',
    'windmill',
]

DATA_ROOT = Path(__file__).resolve().parent / 'DIOR-R'
XML_DIR = DATA_ROOT / 'Annotations' / 'Oriented Bounding Boxes'
TRAINVAL_IMG_DIR = DATA_ROOT / 'JPEGImages-trainval'
TEST_IMG_DIR = DATA_ROOT / 'JPEGImages-test'
MAIN_DIR = DATA_ROOT / 'Main'


def parse_xml_annotation(xml_path: Path) -> List[Tuple[str, str, str]]:
    """Parse a DIOR-R XML annotation file and extract oriented bounding boxes.

    Args:
        xml_path: Path to XML annotation file.

    Returns:
        List of DOTA format lines: [(poly_str, category, difficult), ...]
        where poly_str is 'x1 y1 x2 y2 x3 y3 x4 y4'
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annotations = []

    for obj in root.findall('object'):
        name = obj.find('name').text.strip()

        # Skip if not in our category list
        if name not in DIOR_CATEGORIES:
            print(f'  Warning: unknown category "{name}" in {xml_path.name}')
            continue

        difficult = obj.find('difficult').text.strip() if obj.find('difficult') is not None else '0'

        # Get oriented bounding box corners
        robndbox = obj.find('robndbox')
        if robndbox is None:
            continue

        x1 = robndbox.find('x_left_top').text.strip()
        y1 = robndbox.find('y_left_top').text.strip()
        x2 = robndbox.find('x_right_top').text.strip()
        y2 = robndbox.find('y_right_top').text.strip()
        x3 = robndbox.find('x_right_bottom').text.strip()
        y3 = robndbox.find('y_right_bottom').text.strip()
        x4 = robndbox.find('x_left_bottom').text.strip()
        y4 = robndbox.find('y_left_bottom').text.strip()

        poly_str = f'{x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4}'
        annotations.append((poly_str, name, difficult))

    return annotations


def load_split(split_name: str) -> List[str]:
    """Load image IDs from a split file.

    Args:
        split_name: 'train', 'val', or 'test'.

    Returns:
        List of image IDs (without extension).
    """
    split_file = MAIN_DIR / f'{split_name}.txt'
    if not split_file.exists():
        print(f'Warning: split file {split_file} not found')
        return []

    with open(split_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def convert_and_write(
    image_ids: List[str],
    img_src_dir: Path,
    out_img_dir: Path,
    out_label_dir: Path,
    split_name: str,
):
    """Convert XML annotations to DOTA txt format and symlink images.

    Args:
        image_ids: List of image IDs.
        img_src_dir: Source directory for images.
        out_img_dir: Output images directory.
        out_label_dir: Output labelTxt directory.
        split_name: Name of the split (for logging).
    """
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    missing_xml = 0
    missing_img = 0
    total_boxes = 0
    class_counts = {cat: 0 for cat in DIOR_CATEGORIES}

    for img_id in image_ids:
        xml_path = XML_DIR / f'{img_id}.xml'

        # Find image file (try .jpg and .png)
        img_path = None
        for ext in ['.jpg', '.png', '.JPEG', '.JPG']:
            candidate = img_src_dir / f'{img_id}{ext}'
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            missing_img += 1
            continue

        if not xml_path.exists():
            missing_xml += 1
            continue

        # Parse XML
        annotations = parse_xml_annotation(xml_path)

        # Write DOTA format annotation
        label_path = out_label_dir / f'{img_id}.txt'
        with open(label_path, 'w') as f:
            for poly_str, category, difficult in annotations:
                f.write(f'{poly_str} {category} {difficult}\n')
                total_boxes += 1
                class_counts[category] += 1

        # Symlink image (or copy if symlink fails)
        dst_img = out_img_dir / img_path.name
        if not dst_img.exists():
            try:
                os.symlink(os.path.relpath(img_path, out_img_dir), dst_img)
            except OSError:
                import shutil
                shutil.copy2(img_path, dst_img)

    print(f'  [{split_name}] Images: {len(image_ids)}, Missing XML: {missing_xml}, '
          f'Missing Img: {missing_img}, Total Boxes: {total_boxes}')

    # Print class distribution
    print(f'  [{split_name}] Class distribution:')
    for cat, count in sorted(class_counts.items()):
        if count > 0:
            print(f'    {cat}: {count}')


def main():
    print('=' * 60)
    print('DIOR-R XML → DOTA Format Converter')
    print('=' * 60)

    # Load splits
    train_ids = load_split('train')
    val_ids = load_split('val')
    test_ids = load_split('test')

    print(f'\nSplit sizes:')
    print(f'  train: {len(train_ids)}')
    print(f'  val:   {len(val_ids)}')
    print(f'  test:  {len(test_ids)}')
    print(f'  total: {len(train_ids) + len(val_ids) + len(test_ids)}')

    # Train split
    print(f'\n--- Converting train split ---')
    convert_and_write(
        train_ids,
        TRAINVAL_IMG_DIR,
        DATA_ROOT / 'train' / 'images',
        DATA_ROOT / 'train' / 'labelTxt',
        'train',
    )

    # Val split
    print(f'\n--- Converting val split ---')
    convert_and_write(
        val_ids,
        TRAINVAL_IMG_DIR,
        DATA_ROOT / 'val' / 'images',
        DATA_ROOT / 'val' / 'labelTxt',
        'val',
    )

    # Test split
    print(f'\n--- Converting test split ---')
    convert_and_write(
        test_ids,
        TEST_IMG_DIR,
        DATA_ROOT / 'test' / 'images',
        DATA_ROOT / 'test' / 'labelTxt',
        'test',
    )

    # Verify output
    print(f'\n--- Verification ---')
    for split in ['train', 'val', 'test']:
        img_dir = DATA_ROOT / split / 'images'
        label_dir = DATA_ROOT / split / 'labelTxt'
        n_imgs = len(list(img_dir.glob('*'))) if img_dir.exists() else 0
        n_labels = len(list(label_dir.glob('*.txt'))) if label_dir.exists() else 0
        print(f'  {split}: {n_imgs} images, {n_labels} labels')

    print(f'\nDone! Dataset is ready for training.')


if __name__ == '__main__':
    main()
