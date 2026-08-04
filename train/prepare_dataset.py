#!/usr/bin/env python3
"""
Prepare Tianjin Mahjong training dataset.
Downloads Camerash mahjong-dataset and converts to YOLOv8 classification format.

Tianjin mahjong uses 34 classes (no bonus/flowers):
  dots-1..9    → classes 0-8
  bamboo-1..9  → classes 9-17
  char-1..9    → classes 18-26
  east, south, west, north → classes 27-30
  red, green, white        → classes 31-33

Camerash dataset has 42 classes; we filter out bonus tiles (35-41).
"""

import os
import sys
import csv
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import requests
from tqdm import tqdm
from PIL import Image

# Config
DATASET_URL = "https://github.com/Camerash/mahjong-dataset/raw/master/train.zip"
WORK_DIR = Path("datasets")
TRAIN_ZIP = WORK_DIR / "train.zip"
YOLO_DIR = WORK_DIR / "tianjin_yolo"
IMAGES_DIR = YOLO_DIR / "images"
LABELS_DIR = YOLO_DIR / "labels"

# Tianjin classes (34) — mapping from Camerash class names
TIANJIN_CLASSES = {
    "dots-1": 0, "dots-2": 1, "dots-3": 2, "dots-4": 3, "dots-5": 4,
    "dots-6": 5, "dots-7": 6, "dots-8": 7, "dots-9": 8,
    "bamboo-1": 9, "bamboo-2": 10, "bamboo-3": 11, "bamboo-4": 12,
    "bamboo-5": 13, "bamboo-6": 14, "bamboo-7": 15, "bamboo-8": 16,
    "bamboo-9": 17,
    "characters-1": 18, "characters-2": 19, "characters-3": 20,
    "characters-4": 21, "characters-5": 22, "characters-6": 23,
    "characters-7": 24, "characters-8": 25, "characters-9": 26,
    "honors-east": 27, "honors-south": 28, "honors-west": 29,
    "honors-north": 30,
    "honors-red": 31, "honors-green": 32, "honors-white": 33,
}

# Tile display names
TILE_NAMES = [
    "1万","2万","3万","4万","5万","6万","7万","8万","9万",
    "1筒","2筒","3筒","4筒","5筒","6筒","7筒","8筒","9筒",
    "1条","2条","3条","4条","5条","6条","7条","8条","9条",
    "东","南","西","北","中","发","白",
]


def download_dataset():
    """Download Camerash dataset if not cached."""
    if TRAIN_ZIP.exists():
        print(f"✓ Dataset already downloaded: {TRAIN_ZIP}")
        return

    print(f"Downloading {DATASET_URL} ...")
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    response = requests.get(DATASET_URL, stream=True)
    total = int(response.headers.get("content-length", 0))

    with open(TRAIN_ZIP, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc="Downloading"
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))

    print(f"✓ Downloaded: {TRAIN_ZIP}")


def extract_and_convert():
    """Extract dataset and reorganize for YOLO classification fine-tuning."""
    extract_dir = WORK_DIR / "camerash_extracted"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)

    print("Extracting...")
    with zipfile.ZipFile(TRAIN_ZIP, "r") as zf:
        zf.extractall(extract_dir)

    # Read labels
    csv_path = extract_dir / "data.csv"
    if not csv_path.exists():
        print(f"❌ data.csv not found in {extract_dir}")
        return

    labels = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row.get("image-name", row.get("image_name", "")).strip()
            label_name = row.get("label-name", row.get("label_name", "")).strip()
            labels[fname] = label_name

    # Copy images to YOLO directories, organized by class
    YOLO_DIR.mkdir(parents=True, exist_ok=True)

    # YOLO classification structure: train/<class_name>/image.jpg
    images_src = extract_dir / "images"
    if not images_src.exists():
        print(f"❌ images directory not found in {extract_dir}")
        return

    # Create train/val/test splits
    splits = {"train": 0.7, "val": 0.15, "test": 0.15}
    import random
    random.seed(42)

    # Group images by class
    class_images = {name: [] for name in TIANJIN_CLASSES}
    skipped = 0

    for img_file in sorted(images_src.iterdir()):
        if not img_file.suffix.lower() in (".jpg", ".jpeg", ".png"):
            continue
        label_name = labels.get(img_file.name, "")
        if label_name in TIANJIN_CLASSES:
            class_images[label_name].append(img_file)
        else:
            skipped += 1

    print(f"Found {sum(len(v) for v in class_images.values())} tile images across "
          f"{len([v for v in class_images.values() if v])} Tianjin classes")
    print(f"Skipped {skipped} non-Tianjin images (bonus/flowers)")

    # Create YOLO classification structure
    for split_name, split_ratio in splits.items():
        for class_name, class_id in TIANJIN_CLASSES.items():
            split_dir = YOLO_DIR / split_name / str(class_id)
            split_dir.mkdir(parents=True, exist_ok=True)

    # Assign images to splits
    count = 0
    for class_name, images in class_images.items():
        class_id = TIANJIN_CLASSES[class_name]
        random.shuffle(images)
        n = len(images)
        n_train = int(n * splits["train"])
        n_val = int(n * splits["val"])

        for i, img_path in enumerate(images):
            if i < n_train:
                split = "train"
            elif i < n_train + n_val:
                split = "val"
            else:
                split = "test"
            dest = YOLO_DIR / split / str(class_id) / f"{class_id}_{count:05d}.jpg"
            shutil.copy2(img_path, dest)
            count += 1

    print(f"✓ Converted {count} images to YOLO classification format")
    print(f"  train: {int(count * splits['train'])} images")
    print(f"  val:   {int(count * splits['val'])} images")
    print(f"  test:  {count - int(count * splits['train']) - int(count * splits['val'])} images")

    # Cleanup
    shutil.rmtree(extract_dir)


def create_labels_file():
    """Create YOLO class names file."""
    labels_path = YOLO_DIR / "labels.txt"
    with open(labels_path, "w") as f:
        for name in TILE_NAMES:
            f.write(f"{name}\n")
    print(f"✓ Class labels written: {labels_path}")


def create_data_yaml():
    """Create YOLO data.yaml config."""
    import yaml

    data_config = {
        "path": str(YOLO_DIR.resolve()),
        "train": "train",
        "val": "val",
        "test": "test",
        "nc": 34,
        "names": TILE_NAMES,
    }

    yaml_path = YOLO_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_config, f, allow_unicode=True, default_flow_style=False)
    print(f"✓ YOLO config written: {yaml_path}")


def main():
    print("=" * 60)
    print("Tianjin Mahjong — Dataset Preparation")
    print("=" * 60)

    download_dataset()
    extract_and_convert()
    create_labels_file()
    create_data_yaml()

    print("\n✅ Dataset ready for training!")
    print(f"   Run: python train/train.py --data {YOLO_DIR / 'data.yaml'}")


if __name__ == "__main__":
    main()
