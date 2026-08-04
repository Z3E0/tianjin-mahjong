# Tianjin Mahjong AI 🀄

Real-time camera-based mahjong assistant for Tianjin rules. Point your phone at your tiles, get spoken strategy recommendations.

## Architecture

```
iPhone camera → snapshot every 3s → ONNX Runtime Web (YOLOv8n)
    → tile detections → Tianjin strategy engine → voice + display
```

## Project Structure

```
tianjin-mahjong/
├── train/              # YOLOv8 training pipeline (runs on RunPod GPU)
│   ├── prepare_dataset.py
│   ├── augment_dataset.py
│   ├── train.py
│   └── export_onnx.py
├── web/                # Static web app (hosted on kit.nightrelay.dev)
│   ├── index.html
│   ├── style.css
│   ├── app.js
│   └── strategy/
│       └── tianjin_engine.js
└── README.md
```

## Quick Start

### 1. Train the model (on RunPod GPU)

```bash
# On RunPod RTX 3090:
pip install -r train/requirements.txt
python train/prepare_dataset.py      # Downloads + prepares data
python train/augment_dataset.py       # Generates synthetic training data
python train/train.py                 # Trains YOLOv8n (~30 min)
python train/export_onnx.py           # Exports model.onnx
```

### 2. Deploy the web app

```bash
# Copy to nightrelay.dev:
cp -r web/* /var/www/kit.nightrelay.dev/mahjong/

# Upload ONNX model to GitHub Release:
export GH_TOKEN="..."
gh release create v1 model.onnx --title "v1 - YOLOv8n Tianjin Tiles"

# Open on iPhone:
# https://kit.nightrelay.dev/mahjong/
```

## Tianjin Mahjong Rules

- **136 tiles** (万/筒/条 + 东南西北 + 中发白), no flowers
- **碰 allowed, 吃 NOT allowed**
- **No 放炮** — self-draw only (自摸)
- **混儿** — wild card system (the tile after the indicator becomes wild)
- **Minimum fan requirement** — must have meaningful hand (混儿吊 alone insufficient)
- **Multiplicative scoring** with key patterns: 龙, 捉伍儿, 混儿吊, 素的, 本混儿龙

## License

MIT
