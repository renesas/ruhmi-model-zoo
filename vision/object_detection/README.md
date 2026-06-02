# Object Detection

This directory contains models for the **object detection** task, targeting the Renesas RA8P1 MCU (Cortex-M85 + Ethos-U55 NPU). Each subfolder holds a complete model package — pretrained weights, conversion scripts, inference code, compile configuration, and embedded C-code artifacts.

## Available Models

| Model | Dataset | Input Shape | Classes | README |
|-------|---------|:-----------:|:-------:|:--------|
| YOLO-Fastest 1.1 | COCO 2017 | (1, 320, 320, 3) | 80 | [Link](yolo_fastest_1_1/README.md) |
| YOLOX-Tiny | COCO 2017 | (1, 224, 224, 3) | 80 | [Link](yolox_tiny/README.md) |

## Directory Structure

```
object_detection/
├── README.md              ← This file
├── yolo_fastest_1_1/      ← YOLO-Fastest 1.1 (CPU-only)
│   ├── README.md
│   ├── python/
│   └── embedded_c/
└── yolox_tiny/            ← YOLOX-Tiny (CPU + NPU)
    ├── README.md
    ├── python/
    └── embedded_c/
```

## Getting Started

1. Pick a model from the table above and open its README.
2. Follow the per-model instructions to set up a virtual environment, download weights, and run inference.
3. For compilation to MCU C-code, see the model's README (Step 4) and the [top-level README](../../README.md) for RUHMI/MERA compiler setup.
