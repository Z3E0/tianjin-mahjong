#!/usr/bin/env python3
"""
Generate synthetic training data for mahjong tile detection.
Places tile images on random table-like backgrounds with rotations,
perspective transforms, lighting variations, and occlusions.

Creates YOLO-format detection dataset with bounding box annotations.
"""

import os
import sys
import random
import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from tqdm import tqdm

# Config
WORK_DIR = Path("datasets")
TILE_DATASET = WORK_DIR / "tianjin_yolo"  # From prepare_dataset.py
DETECTION_DIR = WORK_DIR / "tianjin_detection"
OUTPUT_IMAGES = DETECTION_DIR / "images"
OUTPUT_LABELS = DETECTION_DIR / "labels"
BG_TEXTURES_DIR = WORK_DIR / "backgrounds"

# Augmentation parameters
NUM_IMAGES = 5000          # Total synthetic images
TILES_PER_IMAGE = (4, 16)  # Range of tiles per image
IMG_SIZE = 640             # YOLOv8n input size
TILE_SCALE = (0.3, 0.8)    # Tile size range relative to image
ROTATION_RANGE = (-30, 30) # Degrees
PERSPECTIVE_SCALE = 0.05   # Random perspective warp
JITTER_BRIGHTNESS = 0.3    # Brightness variation
JITTER_CONTRAST = 0.3      # Contrast variation
OVERLAP_PROB = 0.15        # Probability of partial overlap

# Tile aspect ratio (width/height) — mahjong tiles are tall
TILE_ASPECT = 0.62

# Class index to name
CLASS_NAMES = [
    "1万","2万","3万","4万","5万","6万","7万","8万","9万",
    "1筒","2筒","3筒","4筒","5筒","6筒","7筒","8筒","9筒",
    "1条","2条","3条","4条","5条","6条","7条","8条","9条",
    "东","南","西","北","中","发","白",
]


def generate_table_backgrounds():
    """Generate synthetic table-like backgrounds."""
    BG_TEXTURES_DIR.mkdir(parents=True, exist_ok=True)

    colors = [
        (34, 100, 50),    # Green felt
        (20, 80, 40),     # Dark green
        (30, 60, 80),     # Blue-grey
        (40, 40, 60),     # Dark blue
        (60, 50, 40),     # Brown wood
        (80, 70, 50),     # Light wood
    ]

    for i, color in enumerate(colors):
        # Create base color with noise
        bg = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        bg[:] = color

        # Add texture noise
        noise = np.random.randint(-15, 15, (IMG_SIZE, IMG_SIZE, 3), dtype=np.int16)
        bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Add subtle horizontal lines (fabric texture)
        for y in range(0, IMG_SIZE, random.randint(4, 8)):
            alpha = random.randint(2, 8)
            bg[y:y+1, :] = np.clip(bg[y:y+1, :].astype(np.int16) + alpha, 0, 255).astype(np.uint8)

        # Gaussian blur for softer look
        bg = cv2.GaussianBlur(bg, (5, 5), 2)

        cv2.imwrite(str(BG_TEXTURES_DIR / f"bg_{i:02d}.jpg"), bg)

    print(f"✓ Generated {len(colors)} table backgrounds")


def load_tile_images() -> dict:
    """Load all tile images grouped by class."""
    tiles = {}
    # Use validation images as source tiles (they're clean individual tile photos)
    for class_dir in sorted((TILE_DATASET / "train").iterdir()):
        if not class_dir.is_dir():
            continue
        class_id = int(class_dir.name)
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg"))
        if images:
            tiles[class_id] = images
    return tiles


def random_tile_image(tile_images: dict, class_id: int) -> np.ndarray:
    """Load a random tile image for given class."""
    path = random.choice(tile_images[class_id])
    img = cv2.imread(str(path))
    if img is None:
        return np.zeros((100, 62, 3), dtype=np.uint8)
    return img


def apply_augmentation(
    tile_img: np.ndarray,
    rotation: float,
    scale: float,
    brightness: float,
    contrast: float,
) -> np.ndarray:
    """Apply rotation, scale, brightness, and contrast to a tile."""
    h, w = tile_img.shape[:2]

    # Rotate
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, rotation, scale)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    matrix[0, 2] += new_w / 2 - center[0]
    matrix[1, 2] += new_h / 2 - center[1]
    rotated = cv2.warpAffine(
        tile_img, matrix, (new_w, new_h),
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
    )

    # Convert to PIL for color adjustments
    pil_img = Image.fromarray(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))

    # Brightness
    enhancer = ImageEnhance.Brightness(pil_img)
    pil_img = enhancer.enhance(brightness)

    # Contrast
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(contrast)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def place_tile_on_background(
    bg: np.ndarray,
    tile: np.ndarray,
    x: int,
    y: int,
) -> Tuple[int, int, int, int]:
    """Place a tile on background at position (x, y). Returns (x, y, w, h)."""
    th, tw = tile.shape[:2]
    bh, bw = bg.shape[:2]

    # Clip to background bounds
    x = max(0, min(x, bw - 1))
    y = max(0, min(y, bh - 1))
    place_w = min(tw, bw - x)
    place_h = min(th, bh - y)

    if place_w <= 0 or place_h <= 0:
        return x, y, 0, 0

    # Create alpha mask from non-black pixels
    tile_region = tile[:place_h, :place_w]
    bg_region = bg[y:y+place_h, x:x+place_w]

    # Simple alpha: black pixels are transparent
    mask = np.any(tile_region > 10, axis=2).astype(np.uint8) * 255
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0

    # Blend
    blended = (tile_region * mask_3ch + bg_region * (1 - mask_3ch)).astype(np.uint8)
    bg[y:y+place_h, x:x+place_w] = blended

    return x, y, place_w, place_h


def generate_synthetic_image(
    tile_images: dict,
    bg_img: np.ndarray,
    img_idx: int,
) -> Tuple[str, str, list]:
    """Generate one synthetic image with random tiles. Returns (img_path, label_path, annotations)."""
    bh, bw = bg_img.shape[:2]
    bg = bg_img.copy()

    num_tiles = random.randint(*TILES_PER_IMAGE)
    annotations = []

    # Class distribution: weighted toward common classes
    if random.random() < 0.3:
        # Mostly one suit (simulating suit-focused hands)
        suit_offset = random.choice([0, 9, 18])  # 万, 筒, or 条
        class_ids = [
            suit_offset + random.randint(0, 8)
            for _ in range(num_tiles)
        ]
    else:
        class_ids = [
            random.randint(0, 33)
            for _ in range(num_tiles)
        ]

    # Place tiles
    placed_boxes = []  # (x, y, w, h)

    for class_id in class_ids:
        if class_id not in tile_images:
            continue

        tile = random_tile_image(tile_images, class_id)

        # Random augmentation
        rotation = random.uniform(*ROTATION_RANGE)
        scale = random.uniform(*TILE_SCALE)
        brightness = random.uniform(1 - JITTER_BRIGHTNESS, 1 + JITTER_BRIGHTNESS)
        contrast = random.uniform(1 - JITTER_CONTRAST, 1 + JITTER_CONTRAST)

        tile = apply_augmentation(tile, rotation, scale, brightness, contrast)

        th, tw = tile.shape[:2]

        # Find non-overlapping position (with some overlap allowed)
        max_attempts = 20
        for attempt in range(max_attempts):
            if random.random() < OVERLAP_PROB and placed_boxes:
                # Place near existing tile (partial overlap)
                ref_box = random.choice(placed_boxes)
                x = ref_box[0] + random.randint(-tw//2, ref_box[2]//2)
                y = ref_box[1] + random.randint(-th//2, ref_box[3]//2)
            else:
                x = random.randint(0, max(1, bw - tw))
                y = random.randint(0, max(1, bh - th))

            # Check overlap
            new_box = (x, y, tw, th)
            overlap_ok = True
            for existing in placed_boxes:
                iou = compute_iou(new_box, existing)
                if iou > 0.3:  # Max 30% overlap
                    overlap_ok = False
                    break

            if overlap_ok or attempt == max_attempts - 1:
                break

        # Place tile
        px, py, pw, ph = place_tile_on_background(bg, tile, x, y)

        if pw > 5 and ph > 5:
            # YOLO format: class_id, x_center, y_center, width, height (normalized)
            x_center = (px + pw / 2) / bw
            y_center = (py + ph / 2) / bh
            norm_w = pw / bw
            norm_h = ph / bh

            annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
            placed_boxes.append((px, py, pw, ph))

    # Add global lighting variation
    brightness = random.uniform(0.7, 1.3)
    bg = np.clip(bg.astype(np.float32) * brightness, 0, 255).astype(np.uint8)

    # Save
    img_name = f"synthetic_{img_idx:06d}"
    img_path = str(OUTPUT_IMAGES / f"{img_name}.jpg")
    label_path = str(OUTPUT_LABELS / f"{img_name}.txt")

    cv2.imwrite(img_path, bg)
    with open(label_path, "w") as f:
        f.write("\n".join(annotations))

    return img_path, label_path, annotations


def compute_iou(box1: tuple, box2: tuple) -> float:
    """Compute Intersection over Union between two boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0

    inter_area = (xi2 - xi1) * (yi2 - yi1)
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def create_detection_yaml():
    """Create YOLO detection config."""
    import yaml

    data_config = {
        "path": str(DETECTION_DIR.resolve()),
        "train": "images",
        "val": "images",
        "nc": 34,
        "names": CLASS_NAMES,
    }

    yaml_path = DETECTION_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_config, f, allow_unicode=True, default_flow_style=False)
    print(f"✓ Detection config: {yaml_path}")


def main():
    print("=" * 60)
    print("Tianjin Mahjong — Synthetic Data Generation")
    print("=" * 60)

    # Setup
    OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)
    OUTPUT_LABELS.mkdir(parents=True, exist_ok=True)

    # Generate backgrounds
    generate_table_backgrounds()

    # Load source tile images
    print("Loading tile images...")
    tile_images = load_tile_images()
    if not tile_images:
        print("❌ No tile images found! Run prepare_dataset.py first.")
        sys.exit(1)
    print(f"✓ Loaded tiles for {len(tile_images)} classes "
          f"({sum(len(v) for v in tile_images.values())} images)")

    # Load backgrounds
    bg_paths = list(BG_TEXTURES_DIR.glob("bg_*.jpg"))
    if not bg_paths:
        print("❌ No backgrounds generated!")
        sys.exit(1)
    backgrounds = [cv2.imread(str(p)) for p in bg_paths]

    print(f"\nGenerating {NUM_IMAGES} synthetic detection images...")
    total_annotations = 0

    for i in tqdm(range(NUM_IMAGES), desc="Synthetic images"):
        bg = backgrounds[i % len(backgrounds)].copy()
        _, _, annotations = generate_synthetic_image(tile_images, bg, i)
        total_annotations += len(annotations)

    create_detection_yaml()

    print(f"\n✅ Generated {NUM_IMAGES} synthetic images with {total_annotations} tile annotations")
    print(f"   Images: {OUTPUT_IMAGES}")
    print(f"   Labels: {OUTPUT_LABELS}")


if __name__ == "__main__":
    main()
