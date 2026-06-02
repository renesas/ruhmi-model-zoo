# Tutorials

This folder contains end-to-end Jupyter Notebook tutorials that walk through the full deployment pipeline for the Renesas RA8P1 MCU (Cortex-M85 + Ethos-U55 NPU) using the MERA (RUHMI) compiler.

Each tutorial covers the complete workflow: model acquisition → conversion → quantization → inference validation → MERA compilation → embedded C generation.

---

## Available Tutorials

| # | Notebook | Task | Models Covered |
|---|----------|------|----------------|
| 01 | [`image_classification.ipynb`](image_classification.ipynb) | Image Classification | MobileNetV1 (0.25×), MobileNetV2 (1.0×) |
| 02 | [`object_detection.ipynb`](object_detection.ipynb) | Object Detection | YOLOX-Tiny (224×224) |
| 03 | [`audio_classification.ipynb`](audio_classification.ipynb) | Audio Classification (KWS) | DS-CNN (12-class Keyword Spotting) |

---

## Prerequisites

1. **Python 3.10** — see the [top-level README](../README.md#install-python-310) for installation steps.
2. **Jupyter** — install via `pip install jupyter` or use VS Code's built-in notebook support.
3. **MERA compiler wheel** — required for compilation sections. The wheel file (`mera-*.whl`) is included in this folder for convenience. Update the path in each notebook's install cell if needed.

---

## Quick Start

1. Create and activate a virtual environment:

    **Windows PowerShell**

    ```powershell
    cd tutorials
    python -m venv .venv_tutorials
    .\.venv_tutorials\Scripts\Activate.ps1
    pip install --upgrade pip
    pip install jupyter ipykernel
    pip install --user --upgrade ipywidgets
    python -m ipykernel install --user --name=venv_tutorials --display-name "Python (venv_tutorials)"
    ```

    **Ubuntu / bash**

    ```bash
    cd tutorials
    python3.10 -m venv .venv_tutorials
    source .venv_tutorials/bin/activate
    pip install --upgrade pip
    pip install jupyter ipykernel
    pip install --user --upgrade ipywidgets
    python -m ipykernel install --user --name=venv_tutorials --display-name "Python (venv_tutorials)"
    ```

2. **Select the kernel in VS Code** — click the kernel picker (top-right of the notebook) and choose **"Python (venv_tutorials)"**.

3. Launch Jupyter or open the notebook in VS Code:

    ```bash
    jupyter notebook
    ```

3. Each notebook installs its own dependencies in the first code cell — just run all cells sequentially.

---

## Tutorial Pipelines

### Tutorial 01 — Image Classification

```
tf.keras.applications (ImageNet weights)
    → TFLite FP32
    → TFLite INT8 (post-training quantization)
    → MERA compiler → C source for RA8P1
```

### Tutorial 02 — Object Detection

```
YOLOX-Tiny .pth (Megvii, COCO 80 classes)
    → ONNX FP32 (torch.onnx.export)
    → TFLite FP32 / INT8 (onnx2tf)
    → MERA compiler → C source for RA8P1
```

### Tutorial 03 — Audio Classification (KWS)

```
DS-CNN SavedModel (MLCommons Tiny, Speech Commands v0.02)
    → TFLite FP32
    → TFLite INT8 (MFCC calibration data)
    → MERA compiler → C source for RA8P1
```

---

## Folder Contents

```
tutorials/
├── README.md                          ← This file
├── image_classification.ipynb         ← Tutorial 01
├── object_detection.ipynb             ← Tutorial 02
└── audio_classification.ipynb         ← Tutorial 03
```
---
