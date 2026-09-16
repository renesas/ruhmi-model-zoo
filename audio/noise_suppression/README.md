# Noise Suppression

This directory contains models for the **noise suppression** task, targeting the Renesas RA8P1 MCU (Cortex-M85 + Ethos-U55 NPU). Each subfolder holds a complete model package — pretrained weights, conversion scripts, inference code, and embedded C-code artifacts.

## Available Models

| Model | Dataset | Input | Metric (PESQ-wb) | README |
|-------|---------|-------|:----------------:|:--------|
| RNNoise INT8 (Arm ML-zoo) | Edinburgh Noisy Speech DB | Feature frames (42 × float) | 2.44 | [Link](rnnoise/README.md) |

## Directory Structure

```
noise_suppression/
├── README.md              ← This file
└── rnnoise/               ← RNNoise INT8 (Arm ML-zoo)
```

## Getting Started

1. Open [rnnoise/README.md](rnnoise/README.md).
2. Create a Python 3.10 virtual environment inside `rnnoise/python/`.
3. Install dependencies from `requirements.txt` and run inference or validation.
4. For MCU C-code artifacts, see `rnnoise/embedded_c/` and the [top-level README](../../README.md) for RUHMI compiler setup.
