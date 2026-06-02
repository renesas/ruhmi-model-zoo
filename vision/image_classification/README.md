# Image Classification

This directory contains models for the **image classification** task, targeting the Renesas RA8P1 MCU (Cortex-M85 + Ethos-U55 NPU). Each subfolder holds a complete model package — pretrained weights, conversion scripts, inference code, compile configuration, and embedded C-code artifacts.

## Available Models

| Model | Dataset | Input Shape | Classes | README |
|-------|---------|:-----------:|:-------:|:--------|
| MobileNetV1 (0.25x) | ImageNet | (1, 224, 224, 3) | 1000 | [Link](mobilenetv1/README.md) |
| MobileNetV2 (1.0x) | ImageNet | (1, 224, 224, 3) | 1000 | [Link](mobilenetv2/README.md) |
| MobileNetV3-Small | ImageNet | (1, 192, 192, 3) | 1000 | [Link](mobilenetv3/README.md) |
| ShuffleNetV2 (x0.5) | ImageNet | (1, 224, 224, 3) | 1000 | [Link](shufflenetv2/README.md) |
| SqueezeNet 1.1 | ImageNet | (1, 224, 224, 3) | 1000 | [Link](squeezenet_1_1/README.md) |
| ResNet8 | CIFAR-10 | (1, 32, 32, 3) | 10 | [Link](resnet8/README.md) |
| Visual Wake Words | COCO VWW | (1, 96, 96, 3) | 2 | [Link](visualwakeword/README.md) |

## Directory Structure

```
image_classification/
├── README.md              ← This file
├── mobilenetv1/           ← MobileNetV1 (alpha=0.25, ImageNet)
├── mobilenetv2/           ← MobileNetV2 (1.0x, ImageNet)
├── mobilenetv3/           ← MobileNetV3-Small (192×192, ImageNet)
├── shufflenetv2/          ← ShuffleNetV2 x0.5 (ImageNet)
├── squeezenet_1_1/        ← SqueezeNet 1.1 (ImageNet)
├── resnet8/               ← ResNet8 (CIFAR-10)
└── visualwakeword/        ← Visual Wake Words (person/not-person)
```

## Getting Started

1. Pick a model from the table above and open its README.
2. Follow the per-model instructions to set up a virtual environment, download weights, and run inference.
3. For compilation to MCU C-code, see the model's README (Step 4) and the [top-level README](../../README.md) for RUHMI/MERA compiler setup.
