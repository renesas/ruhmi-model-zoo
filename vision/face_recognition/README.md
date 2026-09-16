# Face Recognition

This directory contains models for the **face recognition** task, targeting the Renesas RA8P1 MCU (Cortex-M85 + Ethos-U55 NPU). Each subfolder holds a complete model package — pretrained weights, conversion scripts, inference code, compile configuration, and embedded C-code artifacts.

## Available Models

| Model | Training Dataset | Evaluation Dataset | Input Shape | Embedding Dim | README |
|-------|:----------------:|:------------------:|:-----------:|:-------------:|--------|
| MobileFaceNet | MS-Celeb-1M | LFW (10-fold) | (1, 112, 112, 3) | 128 | [Link](mobilefacenet/README.md) |

## Directory Structure

```
face_recognition/
├── README.md              ← This file
└── mobilefacenet/         ← MobileFaceNet 128-D face embedding
```

## Getting Started

1. Pick a model from the table above and open its README.
2. Follow the per-model instructions to set up a virtual environment, download weights, and run inference.
3. For compilation to MCU C-code, see the model's README (Step 4) and the [top-level README](../../README.md) for RUHMI/MERA compiler setup.
