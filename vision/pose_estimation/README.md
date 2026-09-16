# Pose Estimation

This directory contains models for the **pose estimation** task, targeting the Renesas RA8P1 MCU (Cortex-M85 + Ethos-U55 NPU). Each subfolder holds a complete model package including model files, inference scripts, and embedded C artifacts.

## Available Models

| Model | Dataset | Input Shape | Keypoints | README |
|-------|---------|:-----------:|:---------:|:--------|
| PoseNet (MobileNetV1-0.5) | COCO 2017 Keypoints | (1, 257, 257, 3) | 17 | [Link](posenet/README.md) |

## Directory Structure

```
pose_estimation/
├── README.md              ← This file
└── posenet/               ← PoseNet (MobileNetV1-0.5, 257×257)
```

## Getting Started

1. Open [posenet/README.md](posenet/README.md).
2. Create a Python 3.10 virtual environment in `posenet/python/`.
3. Install dependencies from `requirements.txt` and run inference.
4. For MCU compilation, use the embedded artifacts in `posenet/embedded_c/` and follow RUHMI compiler setup from the repository [README](../../README.md).
