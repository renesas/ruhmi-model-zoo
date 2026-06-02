# Audio Classification

This directory contains models for the **audio classification** task, targeting the Renesas RA8P1 MCU (Cortex-M85 + Ethos-U55 NPU). Each subfolder holds a complete model package — pretrained weights, conversion scripts, inference code, compile configuration, and embedded C-code artifacts.

## Available Models

| Model | Dataset | Input Shape | Classes | README |
|-------|---------|:-----------:|:-------:|:--------|
| Keyword Spotting (DS-CNN) | Google Speech Commands v0.02 | (1, 49, 10) | 12 | [Link](key_word_spotting/README.md) |

## Directory Structure

```
audio_classification/
├── README.md              ← This file
└── key_word_spotting/     ← DS-CNN Keyword Spotting
    ├── README.md
    ├── python/            ← Inference scripts, conversion tools, config
    └── embedded_c/        ← Compiled C-code for MCU (CPU & NPU)
```

## Getting Started

1. Pick a model from the table above and open its README.
2. Follow the per-model instructions to set up a virtual environment, download weights, and run inference.
3. For compilation to MCU C-code, see the model's README (Step 4) and the [top-level README](../../README.md) for RUHMI/MERA compiler setup.
