# PoseNet (MobileNetV1-0.5) — Pose Estimation (COCO Keypoints)

PoseNet is a lightweight multi-person pose estimation model that predicts 17 COCO keypoints per person. This package includes ready-to-run Python inference and model-conversion scripts, TFLite model artifacts, and precompiled embedded C outputs for RA8P1 workflows.

## Model Details

| Field | Value |
|-------|-------|
| **Model name** | PoseNet MobileNetV1-0.5 (257x257) |
| **Task** | Multi-person pose estimation |
| **Dataset** | COCO 2017 Keypoints |
| **Input shape** | `(1, 257, 257, 3)` |
| **Outputs** | Heatmaps, offsets, forward and backward displacement maps |
| **Keypoints** | 17 (COCO format) |
| **Formats included** | TFLite FP32, TFLite INT8 |
| **Source** | [tensorflow/tfjs-models PoseNet](https://github.com/tensorflow/tfjs-models/tree/master/posenet) |

## Model Report Card

The source validation logs include full COCO keypoint metrics across five runtime backends.

| Runtime Mode | AP (OKS) | AP50 | AP75 | AR |
|--------------|:--------:|:----:|:----:|:--:|
| TFLite FP32 | 7.71 | 21.87 | 3.69 | 13.45 |
| TFLite INT8 | 7.31 | 21.57 | 3.43 | 13.17 |
| MERA FP32 | 7.71 | 21.87 | 3.69 | 13.45 |
| MERA INT8 | 7.21 | 21.28 | 3.56 | 13.10 |
| MERA TFLite INT8 | 7.24 | 21.65 | 3.52 | 13.11 |

> [!NOTE]
> The model was compiled using `mera-2.6.0+pkg.4815` and `FSP 6.5.1` was used for building and testing the project.

### Inference Performance (RA8P1)

Measured on-target (Cortex-M85 @ 1 GHz, Ethos-U55 NPU @ 500 MHz). AI-only latency.

| Memory Configuration | CPU (ms) | NPU (ms) |
|----------------------|:--------:|:--------:|
| ‡ OSPI + External SDRAM | 372 | 5  |

## Folder Structure

```
posenet/
├── README.md                    ← This file
├── python/                      ← Inference scripts and model files
│   ├── download_model.py        ← Model download and conversion pipeline
│   ├── inference.py             ← TFLite inference for FP32/INT8 models
│   ├── config.yaml              ← RUHMI compiler configuration
│   ├── requirements.txt         ← Python dependencies
│   ├── model/                   ← Generated TFLite model artifacts
│   ├── sample_images/           ← Sample input images
│   ├── posenet/                 ← PoseNet model and decoder modules
│   └── utils/                   ← Shared helpers (progress/logging)
└── embedded_c/                  ← MCU integration files
    ├── model_metadata.h
    ├── preprocessing.h
    ├── preprocessing.c
    ├── postprocessing.h
    ├── postprocessing.c
    ├── src_mcu/                 ← CPU-target model artifacts
    └── src_mcu_npu/             ← NPU-target model artifacts
```

## Prerequisites

> [!IMPORTANT]
> **COCO 2017 keypoint validation images are for academic research only and are NOT redistributable under a commercial licence.**
> To convert to INT8 or validate accuracy, supply your own images from the [COCO 2017 dataset](https://cocodataset.org/#download). Pass the directory with `--calib-dir`.

1. **Python 3.10** installed.
2. **Inference venv** — navigate to `python/` and create a dedicated virtual environment:

    **Windows PowerShell**
    > **Note:** If venv activation is blocked by PowerShell execution policy, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in the same PowerShell window, then run the activation command again.

    ```powershell
    cd vision\pose_estimation\posenet\python
    py -3.10 -m venv .venv_posenet
    .\.venv_posenet\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    ```

    **Ubuntu / bash**

    ```bash
    cd vision/pose_estimation/posenet/python
    python3.10 -m venv .venv_posenet
    source .venv_posenet/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

3. **Compiler venv** (`.mera_venv`) — required only for [Step 4](#step-4--compile-for-ra8p1-ruhmi). See the [top-level README](../../../README.md) for setup instructions.

---

## Step 1 — Obtain the Model

The TFLite models are already provided in `python/model/`. To regenerate them from scratch, proceed to [Step 2](#step-2--build-the-models).

---

## Step 2 — Build the Models

Use `download_model.py` to fetch PoseNet weights, export ONNX, and convert to TFLite. Activate the **inference venv** and navigate to `python/`.

**Both FP32 + INT8 (default)**

```bash
python download_model.py
```

Use `--mode fp32` for FP32 only or `--mode int8` for INT8 only. Use `--calib-num N` to control calibration sample count for INT8 (default: 1000).

Output files are written to `python/model/`.

## Step 3 — Run Inference (Python)

```bash
python inference.py --image sample_images/000000017905.jpg --model model/posenet_mbv1_050_257_INT8.tflite
```

Generated outputs are written to `outputs/` under `python/`.

Sample output (captured):

```text
============================================================
  PoseNet MobileNetV1-050 @ 257x257  (output_stride=16)
  Model : posenet_mbv1_050_257_INT8.tflite (INT8)
  Image : 000000017905.jpg
============================================================

  Inference time : 0.8 ms
  Heatmap range  : [0.000, 0.996]
  Pose #0  score=0.937
```

## Step 4 — Compile for RA8P1 (RUHMI)

This step converts the TFLite INT8 model into C-code for the RA8P1 MCU. Activate the **compiler venv** (`.mera_venv`).

### 4.1 — Edit the compile configuration

Open `python/config.yaml` and verify the paths match your system:

```yaml
model_path: vision/pose_estimation/posenet/python/model/posenet_mbv1_050_257_INT8.tflite
output_dir: vision/pose_estimation/posenet/embedded_c/src_mcu
target: cpu
quantize: false
external: false
```

> [!IMPORTANT]
> Use absolute paths if running the compiler from outside the repo root.

> [!TIP]
> For NPU deployment set `target: npu` and point `output_dir` to `embedded_c/src_mcu_npu`.

### 4.2 — Run the compiler

Navigate back to the **repository root** and run the compiler with `.mera_venv` active:

**Windows PowerShell**

```powershell
cd C:\Users\<you>\Model-zoo
python ruhmi_tools\mcu_compile.py vision\pose_estimation\posenet\python\config.yaml
```

**Ubuntu / bash**

```bash
cd ~/Model-zoo
python ruhmi_tools/mcu_compile.py vision/pose_estimation/posenet/python/config.yaml
```

The compiled C-code will be written to the `output_dir` specified in `config.yaml`.

## Step 5 — Embedded C Integration

The `embedded_c/` folder contains portable, board-independent files for bare-metal or RTOS firmware.

| File | Purpose |
|------|---------|
| `model_metadata.h` | Compile-time constants: tensor geometry, quantization parameters, decoder thresholds |
| `preprocessing.h` / `preprocessing.c` | Input preprocessing (bilinear resize + normalize + INT8 quantize) |
| `postprocessing.h` / `postprocessing.c` | Multi-person PersonLab pose decoding from 4 output heads |

> [!NOTE]
> `preprocessing.c` and `postprocessing.c` depend on `<math.h>`. The interface headers are otherwise C99-friendly and board-independent.

### 5.1 — Add files to your project

Copy the following files into your firmware project:

```text
embedded_c/
├── model_metadata.h
├── preprocessing.h
├── preprocessing.c
├── postprocessing.h
└── postprocessing.c
```

Also copy the compiled model artifacts from the appropriate subdirectory:

- **CPU-only (CMSIS-NN):** `embedded_c/src_mcu/`
- **NPU-accelerated (Ethos-U55):** `embedded_c/src_mcu_npu/`

### 5.2 — `model_metadata.h` — Key constants

```c
#include "model_metadata.h"
```

| Constant | Value | Description |
|----------|-------|-------------|
| `MODEL_INPUT_H` | `257` | Input image height (pixels) |
| `MODEL_INPUT_W` | `257` | Input image width (pixels) |
| `MODEL_INPUT_C` | `3` | RGB channels |
| `MODEL_NUM_KEYPOINTS` | `17` | COCO keypoints per pose |
| `MODEL_OUTPUT_STRIDE` | `16` | Output stride from input space to 17x17 grid |
| `MODEL_OUTPUT_GH` / `MODEL_OUTPUT_GW` | `17` / `17` | Output grid height and width |
| `INPUT_QUANT_SCALE` | `0.0078431377f` | INT8 input quantization scale |
| `INPUT_QUANT_ZP` | `-1` | INT8 input zero-point |
| `HEATMAP_QUANT_SCALE` | `0.00390625f` | Heatmap dequantization scale |
| `HEATMAP_QUANT_ZP` | `-128` | Heatmap zero-point |
| `OFFSET_QUANT_SCALE` | `0.58911788f` | Offset dequantization scale |
| `OFFSET_QUANT_ZP` | `-1` | Offset zero-point |
| `MAX_POSE_DETECTIONS` | `10` | Maximum poses returned by `postprocess()` |
| `POSE_MIN_SCORE` | `0.5f` | Minimum instance score to keep a decoded pose |

### 5.3 — Preprocessing — `preprocess()`

**API:**

```c
#include "preprocessing.h"

void preprocess(const uint8_t *p_source_image_hwc,
                int _source_width,
                int _source_height,
                int8_t *p_input_tensor);
```

Pipeline summary:

1. Bilinear resize source RGB image to `257x257` (half-pixel-centre mapping).
2. Normalize to `[-1, 1]` with `pixel * (2/255) - 1`.
3. Quantize to INT8 using `INPUT_QUANT_SCALE` and `INPUT_QUANT_ZP`.

**Typical call:**

```c
#include "model_metadata.h"
#include "preprocessing.h"

/* Camera frame: interleaved RGB888 */
static uint8_t src_rgb[640 * 480 * 3];

/* PoseNet INT8 input tensor */
static int8_t input_int8[MODEL_INPUT_H * MODEL_INPUT_W * MODEL_INPUT_C];

preprocess(src_rgb, 640, 480, input_int8);
```

### 5.4 — Postprocessing — `postprocess()`

**API:**

```c
#include "postprocessing.h"

int postprocess(const int8_t   *p_heatmap_q,
                const int8_t   *p_offset_q,
                const int8_t   *p_disp_fwd_q,
                const int8_t   *p_disp_bwd_q,
                int              _source_width,
                int              _source_height,
                pose_keypoint_t (*p_keypoints_out)[MODEL_NUM_KEYPOINTS],
                float           *p_pose_scores,
                int              _max_poses);
```

The decoder consumes all 4 output heads (heatmap, offset, forward displacement, backward displacement), performs local-maximum root detection, NMS, keypoint chain walking, and returns accepted poses in source-image coordinates.

**Typical call:**

```c
#include "model_metadata.h"
#include "postprocessing.h"

/* Raw INT8 outputs from the model */
int8_t heatmap_q[MODEL_HEATMAP_SIZE];
int8_t offset_q[MODEL_OFFSET_SIZE];
int8_t disp_fwd_q[MODEL_DISP_FWD_SIZE];
int8_t disp_bwd_q[MODEL_DISP_BWD_SIZE];

pose_keypoint_t keypoints_out[MAX_POSE_DETECTIONS][MODEL_NUM_KEYPOINTS];
float pose_scores[MAX_POSE_DETECTIONS];

int poses = postprocess(
    heatmap_q, offset_q, disp_fwd_q, disp_bwd_q,
    640, 480,
    keypoints_out, pose_scores,
    MAX_POSE_DETECTIONS);
```

> [!NOTE]
> `postprocess()` handles tensor dequantization internally using constants from `model_metadata.h`; no host-side dequantization is required.

---

## License / Provenance

| Component | Source | Licence |
|-----------|--------|---------|
| PoseNet weights | [tensorflow/tfjs-models](https://github.com/tensorflow/tfjs-models/tree/master/posenet) | Apache-2.0 |
| COCO 2017 dataset (calibration / validation) | [cocodataset.org](https://cocodataset.org) | CC-BY 4.0 |