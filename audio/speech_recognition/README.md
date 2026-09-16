# Speech Recognition

This directory contains models for the **speech recognition** task, targeting the Renesas RA8P1 MCU (Cortex-M85 CPU). Each subfolder holds a complete model package — pretrained weights, conversion scripts, inference code, and embedded C-code artifacts.

## Available Models

| Model | Dataset | Input Shape | Output | README |
|-------|---------|:-----------:|--------|:-------|
| TinyWav2letter (Arm ML-zoo, pruned) | Fluent Speech Commands | `[1, 296, 39]` MFCC window | CTC character logits `[1, 1, 148, 29]` | [Link](tinywav2letter/README.md) |

## Directory Structure

```
speech_recognition/
├── README.md              ← This file
└── tinywav2letter/        ← TinyWav2letter pruned INT8 speech recognition
```

## Getting Started

Each model directory contains its own `README.md` with step-by-step setup, inference, and compilation instructions.
