#!/usr/bin/env python3
"""
Train YOLOv8n on Tianjin mahjong tile detection.
Uses synthetic data + Camerash classification data for fine-tuning.

Usage:
    python train/train.py --data datasets/tianjin_detection/data.yaml --epochs 100
"""

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 for Tianjin mahjong tiles")
    parser.add_argument("--data", default="datasets/tianjin_detection/data.yaml",
                        help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs (default: 100)")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size (default: 640)")
    parser.add_argument("--device", default="0",
                        help="Device: 0 for GPU, cpu for CPU")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    args = parser.parse_args()

    print("=" * 60)
    print("Tianjin Mahjong — YOLOv8n Training")
    print("=" * 60)
    print(f"Data:     {args.data}")
    print(f"Epochs:   {args.epochs}")
    print(f"Batch:    {args.batch}")
    print(f"Image sz: {args.imgsz}")
    print(f"Device:   {args.device}")
    print("=" * 60)

    # Load pretrained YOLOv8n
    model = YOLO("yolov8n.pt")

    # Train
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        resume=args.resume,

        # Optimization
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,

        # Augmentation (applied during training)
        hsv_h=0.015,   # Hue
        hsv_s=0.4,     # Saturation
        hsv_v=0.3,     # Value
        degrees=10.0,  # Rotation
        translate=0.1, # Translation
        scale=0.3,     # Scale
        shear=2.0,     # Shear
        perspective=0.0,
        flipud=0.0,
        fliplr=0.3,

        # Save
        save=True,
        save_period=10,
        project="runs/tianjin_mahjong",
        name="yolov8n",
        exist_ok=True,

        # Patience
        patience=20,
    )

    print("\n✅ Training complete!")
    print(f"   Best model: {results.save_dir / 'weights' / 'best.pt'}")
    print(f"   Run: python train/export_onnx.py --weights {results.save_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
