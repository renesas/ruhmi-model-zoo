# Face Detection

This directory contains models for the **face detection** task, targeting the Renesas RA8P1 MCU (Cortex-M85 + Ethos-U55 NPU). Each subfolder holds a complete model package — pretrained weights, conversion scripts, inference code, compile configuration, and embedded C-code artifacts.

## Available Models

| Model | Dataset | Input Shape | Anchors | Source | README |
|-------|---------|:-----------:|:-------:|--------|--------|
| BlazeFace Front | WIDER FACE | (1, 128, 128, 3) | 896 | PINTO model zoo (Google MediaPipe) | [Link](blazeface/README.md) |

## Directory Structure

```
face_detection/
├── README.md              ← This file
└── blazeface/             ← BlazeFace Front (128×128)
    ├── README.md
    ├── python/            ← Inference scripts, conversion tools, config
    └── embedded_c/        ← Compiled C-code for MCU (CPU & NPU)
```

## Getting Started

1. Pick a model from the table above and open its README.
2. Follow the per-model instructions to set up a virtual environment, download weights, and run inference.
3. For compilation to MCU C-code, see the model's README (Step 4) and the [top-level README](../../README.md) for RUHMI/MERA compiler setup.
