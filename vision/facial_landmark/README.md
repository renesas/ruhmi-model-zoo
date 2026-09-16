# Facial Landmark

This directory contains models for the **facial landmark** task, targeting the Renesas RA8P1 MCU (Cortex-M85 + Ethos-U55 NPU). Each subfolder holds a complete model package — pretrained weights, conversion scripts, inference code, compile configuration, and embedded C-code artifacts.

## Available Models

| Model | Dataset | Input Shape | Landmarks | Source | README |
|-------|---------|:-----------:|:---------:|--------|--------|
| MediaPipe Face Landmark | iBUG 300W | (1, 192, 192, 3) | 468 | patlevin/face-detection-tflite (Google MediaPipe) | [Link](mediapipe/README.md) |

## Directory Structure

```
facial_landmark/
├── README.md              ← This file
└── mediapipe/             ← MediaPipe Face Landmark (192×192, 468 pts)
```

## Getting Started

1. Pick a model from the table above and open its README.
2. Follow the per-model instructions to set up a virtual environment, download weights, and run inference.
3. For compilation to MCU C-code, see the model's README (Step 4) and the [top-level README](../../README.md) for RUHMI/MERA compiler setup.
