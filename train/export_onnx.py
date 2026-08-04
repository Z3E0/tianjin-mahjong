#!/usr/bin/env python3
"""
Export trained YOLOv8 model to ONNX format for browser inference.
ONNX Runtime Web can run this in the browser.

Usage:
    python train/export_onnx.py --weights runs/tianjin_mahjong/yolov8n/weights/best.pt
"""

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Export YOLOv8 to ONNX")
    parser.add_argument("--weights", required=True,
                        help="Path to trained .pt weights")
    parser.add_argument("--output", default="model.onnx",
                        help="Output ONNX path (default: model.onnx)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size")
    parser.add_argument("--simplify", action="store_true", default=True,
                        help="Simplify ONNX graph")
    parser.add_argument("--opset", type=int, default=12,
                        help="ONNX opset version (12 for browser compat)")
    args = parser.parse_args()

    if not Path(args.weights).exists():
        print(f"❌ Weights not found: {args.weights}")
        sys.exit(1)

    print("=" * 60)
    print("Tianjin Mahjong — ONNX Export")
    print("=" * 60)
    print(f"Weights:  {args.weights}")
    print(f"Output:   {args.output}")
    print(f"Image sz: {args.imgsz}")
    print(f"Opset:    {args.opset}")
    print("=" * 60)

    # Load model
    model = YOLO(args.weights)

    # Export to ONNX
    success = model.export(
        format="onnx",
        imgsz=args.imgsz,
        simplify=args.simplify,
        opset=args.opset,
        dynamic=False,
        half=False,  # FP32 for browser compatibility
    )

    if success:
        output_path = Path(args.weights).parent / f"{Path(args.weights).stem}.onnx"
        if Path(args.output) != output_path:
            import shutil
            shutil.copy2(output_path, args.output)

        size_mb = Path(args.output).stat().st_size / (1024 * 1024)
        print(f"\n✅ ONNX model exported: {args.output}")
        print(f"   Size: {size_mb:.1f} MB")
        print(f"\nNext: Upload to GitHub Release")
        print(f"   gh release create v1 {args.output} --title 'v1 - YOLOv8n Tianjin Tiles'")
    else:
        print("❌ Export failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
