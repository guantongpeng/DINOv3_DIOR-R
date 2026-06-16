#!/usr/bin/env python3
"""
Prepare DIOR-R dataset for mmrotate training.

DIOR-R is a large-scale benchmark dataset for oriented object detection
in aerial/remote sensing images. It contains 23,463 images and 192,472
oriented bounding box instances across 20 object categories.

Dataset Download:
    The DIOR-R dataset can be downloaded from:
    - Official site: https://gcheng-nwpu.github.io/
    - Or from OpenDataLab: https://opendatalab.com/DIOR

Directory Structure After Preparation:
    data/DIOR-R/
    ├── trainval/
    │   ├── images/          # Training/validation images
    │   │   ├── 00001.jpg
    │   │   ├── 00002.jpg
    │   │   └── ...
    │   └── labelTxt/        # Annotations in DOTA format
    │       ├── 00001.txt
    │       ├── 00002.txt
    │       └── ...
    ├── test/
    │   ├── images/          # Test images
    │   │   ├── 00001.jpg
    │   │   ├── 00002.jpg
    │   │   └── ...
    │   └── labelTxt/        # Test annotations in DOTA format
    │       ├── 00001.txt
    │       ├── 00002.txt
    │       └── ...
    └── ImageSets/           # Train/val/test split files (optional)
        ├── train.txt
        ├── val.txt
        └── test.txt

Usage:
    python data/prepare_dior.py --data_root ./data/DIOR-R
    python data/prepare_dior.py --data_root ./data/DIOR-R --download  # attempt download

Annotation Format (DOTA format):
    Each line in labelTxt/*.txt:
        x1 y1 x2 y2 x3 y3 x4 y4 category difficult

    The 4 corners (x1,y1) ... (x4,y4) define the oriented bounding box
    in clockwise order starting from the top-left corner.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple


# DIOR-R 20 object categories
DIOR_CATEGORIES = [
    'airplane',
    'airport',
    'baseballfield',
    'basketballcourt',
    'bridge',
    'chimney',
    'dam',
    'Expressway-Service-area',
    'Expressway-toll-station',
    'golffield',
    'groundtrackfield',
    'harbor',
    'overpass',
    'ship',
    'stadium',
    'storagetank',
    'tenniscourt',
    'trainstation',
    'vehicle',
    'windmill',
]

# Class name mapping for DIOR-R (some names differ from DIOR to DOTA convention)
DIOR_CLASS_MAP = {
    'airplane': 'airplane',
    'airport': 'airport',
    'baseballfield': 'baseballfield',
    'basketballcourt': 'basketballcourt',
    'bridge': 'bridge',
    'chimney': 'chimney',
    'dam': 'dam',
    'Expressway-Service-area': 'Expressway-Service-area',
    'Expressway-toll-station': 'Expressway-toll-station',
    'golffield': 'golffield',
    'groundtrackfield': 'groundtrackfield',
    'harbor': 'harbor',
    'overpass': 'overpass',
    'ship': 'ship',
    'stadium': 'stadium',
    'storagetank': 'storagetank',
    'tenniscourt': 'tenniscourt',
    'trainstation': 'trainstation',
    'vehicle': 'vehicle',
    'windmill': 'windmill',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare DIOR-R dataset for mmrotate training'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default='./data/DIOR-R',
        help='Root directory for DIOR-R dataset',
    )
    parser.add_argument(
        '--download',
        action='store_true',
        help='Attempt to download the dataset (requires wget/gdown)',
    )
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.1,
        help='Validation split ratio from training set',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for train/val split',
    )
    return parser.parse_args()


def create_directory_structure(data_root: str) -> dict:
    """Create the required directory structure for DIOR-R.

    Args:
        data_root: Root directory for the dataset.

    Returns:
        Dict mapping split names to their paths.
    """
    paths = {
        'root': Path(data_root),
        'trainval_images': Path(data_root) / 'trainval' / 'images',
        'trainval_labels': Path(data_root) / 'trainval' / 'labelTxt',
        'test_images': Path(data_root) / 'test' / 'images',
        'test_labels': Path(data_root) / 'test' / 'labelTxt',
        'imagesets': Path(data_root) / 'ImageSets',
    }

    for key, path in paths.items():
        if key != 'root':
            path.mkdir(parents=True, exist_ok=True)
            print(f'Created directory: {path}')

    return paths


def download_dataset(data_root: str):
    """Attempt to download DIOR-R dataset.

    Note: DIOR-R requires manual download from the official website.
    This function provides instructions and attempts automated download
    where possible.
    """
    print('=' * 60)
    print('DIOR-R Dataset Download')
    print('=' * 60)
    print()
    print('The DIOR-R dataset is available from multiple sources:')
    print()
    print('1. Official Website (Google Drive / Baidu Cloud):')
    print('   https://gcheng-nwpu.github.io/')
    print()
    print('2. OpenDataLab:')
    print('   pip install openxlab')
    print('   openxlab dataset info --dataset-repo OpenDataLab/DIOR')
    print()
    print('3. PapersWithCode:')
    print('   https://paperswithcode.com/dataset/dior')
    print()
    print('Please download the following files:')
    print('  - DIOR-R.zip (or DIOR.zip)')
    print('  - The archive contains both images and DOTA-format labels')
    print()
    print(f'After downloading, extract to: {data_root}')
    print()
    print('Expected structure:')
    print(f'  {data_root}/')
    print(f'  ├── JPEGImages/  (all images)')
    print(f'  └── labelTxt/    (DOTA-format annotations)')
    print()
    print('=' * 60)

    # Try using openxlab if available
    try:
        import openxlab
        print('Attempting download via openxlab...')
        print('Run: openxlab dataset download --dataset-repo OpenDataLab/DIOR')
    except ImportError:
        print('openxlab not installed. Install with: pip install openxlab')


def convert_dota_to_mmrotate_format(
    label_file: Path,
    class_map: dict,
) -> Tuple[List[str], dict]:
    """Parse a DOTA-format annotation file.

    Args:
        label_file: Path to the DOTA annotation .txt file.
        class_map: Dictionary mapping class names to standard names.

    Returns:
        Tuple of (annotations_list, stats_dict).
        stats_dict contains counts per class.
    """
    annotations = []
    stats = {cat: 0 for cat in class_map.values()}

    if not label_file.exists():
        return annotations, stats

    with open(label_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 10:
                continue

            # DOTA format: x1 y1 x2 y2 x3 y3 x4 y4 category difficult
            coords = [float(p) for p in parts[:8]]
            category = parts[8]
            difficult = int(parts[9]) if len(parts) >= 10 else 0

            # Map category name
            mapped_cat = class_map.get(category, category)
            if mapped_cat in stats:
                stats[mapped_cat] += 1

            annotations.append({
                'poly': coords,
                'category': mapped_cat,
                'difficult': difficult,
            })

    return annotations, stats


def create_imageset_files(
    paths: dict,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    """Create train/val/test split files.

    Args:
        paths: Dict of dataset paths.
        val_ratio: Fraction of training data to use for validation.
        seed: Random seed for reproducibility.
    """
    import random
    random.seed(seed)

    # Collect all image files from trainval
    image_dir = paths['trainval_images']
    if not image_dir.exists():
        print(f'Warning: {image_dir} does not exist, skipping split creation')
        return

    image_files = sorted([
        f.stem for f in image_dir.glob('*.jpg')
    ] + [f.stem for f in image_dir.glob('*.png')])

    if not image_files:
        print(f'Warning: No images found in {image_dir}')
        return

    print(f'Found {len(image_files)} images in trainval')

    # Shuffle and split
    random.shuffle(image_files)
    n_val = max(1, int(len(image_files) * val_ratio))
    val_files = sorted(image_files[:n_val])
    train_files = sorted(image_files[n_val:])

    # Test files
    test_dir = paths['test_images']
    test_files = []
    if test_dir.exists():
        test_files = sorted([
            f.stem for f in test_dir.glob('*.jpg')
        ] + [f.stem for f in test_dir.glob('*.png')])

    # Write split files
    imageset_dir = paths['imagesets']
    imageset_dir.mkdir(parents=True, exist_ok=True)

    for name, files in [('train.txt', train_files),
                         ('val.txt', val_files),
                         ('test.txt', test_files)]:
        filepath = imageset_dir / name
        with open(filepath, 'w') as f:
            for img_name in files:
                f.write(f'{img_name}\n')
        print(f'  Created {filepath} with {len(files)} entries')


def validate_dataset(paths: dict) -> bool:
    """Validate the dataset structure and annotations.

    Args:
        paths: Dict of dataset paths.

    Returns:
        True if dataset is valid, False otherwise.
    """
    print('\n' + '=' * 60)
    print('Dataset Validation')
    print('=' * 60)

    valid = True

    # Check trainval
    trainval_imgs = list(paths['trainval_images'].glob('*.[jp][pn][g]'))
    trainval_labels = list(paths['trainval_labels'].glob('*.txt'))

    print(f'\nTraining/Validation Set:')
    print(f'  Images: {len(trainval_imgs)}')
    print(f'  Labels: {len(trainval_labels)}')

    if len(trainval_imgs) == 0:
        print('  WARNING: No training images found!')
        valid = False

    # Check test
    test_imgs = list(paths['test_images'].glob('*.[jp][pn][g]'))
    test_labels = list(paths['test_labels'].glob('*.txt'))

    print(f'\nTest Set:')
    print(f'  Images: {len(test_imgs)}')
    print(f'  Labels: {len(test_labels)}')

    # Validate annotations
    if trainval_labels:
        print('\nValidating annotations...')
        total_boxes = 0
        class_counts = {cat: 0 for cat in DIOR_CATEGORIES}

        for label_file in trainval_labels[:10]:  # Sample first 10
            annotations, _ = convert_dota_to_mmrotate_format(
                label_file, DIOR_CLASS_MAP
            )
            for ann in annotations:
                class_counts[ann['category']] += 1
                total_boxes += 1

        print(f'  Sampled {min(10, len(trainval_labels))} annotation files')
        print(f'  Total boxes in sample: {total_boxes}')

        # Report class distribution
        print('\n  Class distribution in sample:')
        for cat, count in sorted(class_counts.items()):
            if count > 0:
                print(f'    {cat}: {count}')

    if valid:
        print('\nDataset validation PASSED!')
    else:
        print('\nDataset validation FAILED - see warnings above.')

    return valid


def main():
    args = parse_args()

    print('=' * 60)
    print('DIOR-R Dataset Preparation for mmrotate')
    print('=' * 60)

    # Download instructions / attempt
    if args.download:
        download_dataset(args.data_root)

    # Create directory structure
    paths = create_directory_structure(args.data_root)

    # Create train/val/test split files
    create_imageset_files(paths, args.val_ratio, args.seed)

    # Validate dataset
    validate_dataset(paths)

    # Print summary
    print('\n' + '=' * 60)
    print('Summary')
    print('=' * 60)
    print(f'Dataset root: {os.path.abspath(args.data_root)}')
    print(f'Categories: {len(DIOR_CATEGORIES)}')
    print(f'Categories list: {", ".join(DIOR_CATEGORIES)}')
    print()
    print('To start training:')
    print(f'  python tools/train.py configs/oriented_rcnn/'
          f'oriented_rcnn_dinov3_fpn_dior.py')
    print()
    print('To evaluate:')
    print(f'  python tools/test.py configs/oriented_rcnn/'
          f'oriented_rcnn_dinov3_fpn_dior.py '
          f'<checkpoint_path> --eval mAP')


if __name__ == '__main__':
    main()
