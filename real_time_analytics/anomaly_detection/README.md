# Anomaly Detection

This directory contains models for the **anomaly detection** task, targeting the Renesas RA8P1 MCU (Cortex-M85 + Ethos-U55 NPU). Each subfolder holds a complete model package — pretrained weights, conversion scripts, inference code, compile configuration, and embedded C-code artifacts.

## Available Models

| Model | Dataset | Input Shape | Metric | README |
|-------|---------|:-----------:|:------:|:--------|
| AD Dense Autoencoder (ad01) | DCASE 2020 Task 2 — ToyCar | (1, 640) | AUC / pAUC | [Link](auto_encoder/README.md) |

## Directory Structure

```
anomaly_detection/
├── README.md              ← This file
└── auto_encoder/          ← AD Dense Autoencoder
```

## Getting Started

1. Pick a model from the table above and open its README.
2. Follow the per-model instructions to set up a virtual environment, download weights, and run inference.
3. For compilation to MCU C-code, see the model's README (Step 4) and the [top-level README](../../README.md) for RUHMI/MERA compiler setup.
