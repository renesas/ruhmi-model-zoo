# YOLO-Fastest 1.1 — Object Detection (COCO)

YOLO-Fastest is an ultra-lightweight anchor-based object detector ([dog-qiuqiu, 2020](https://github.com/dog-qiuqiu/Yolo-Fastest)) optimised for embedded and mobile deployment.It achieves real-time detection on resource-constrained MCUs. This folder contains everything needed to obtain the model, convert it to TFLite, compile it for the Renesas RA8P1 (Cortex-M85) using the RUHMI toolchain, and run inference.

> [!IMPORTANT]
> This model supports **CPU-only** deployment (CMSIS-NN). NPU (Ethos-U55) is **not supported**.

---

## Model Details

| Field | Value |
|-------|-------|
| **Model name** | YOLO-Fastest 1.1 |
| **Task** | Object Detection |
| **Framework** | Darknet → ONNX → TFLite |
| **Dataset** | COCO (80 classes) |
| **Input shape** | `(1, 320, 320, 3)` — NHWC, RGB |
| **Output heads** | Head 0: `(1, 10, 10, 255)` stride-32; Head 1: `(1, 20, 20, 255)` stride-16 |
| **Anchors** | Head 0: (115,73), (119,199), (242,238); Head 1: (12,18), (37,49), (52,132) |
| **Source** | [dog-qiuqiu/Yolo-Fastest](https://github.com/dog-qiuqiu/Yolo-Fastest) |

## Model Report Card

Evaluated on COCO val2017 (4952 images), mAP computed at IoU=0.5 and IoU=0.5:0.95.

| Model Variant | Format | mAP@0.5 (%) | mAP@0.5:0.95 (%) |
|---------------|--------|:-----------:|:----------------:|
| TFLite FP32 | `.tflite` | 24.80 | 11.75 |
| TFLite INT8 | `.tflite` | 21.77 | 10.19 |
| MERA FP32 | `.mera` | 24.80 | 11.75 |
| MERA INT8 (TFLite Quantized) | `.mera` | 20.40 | 8.80 |

> [!NOTE]
> The model was compiled using `mera-2.6.0+pkg.4293` and `FSP 6.2` was used for building and testing the project.

### Inference Performance (RA8P1)

Measured on-target (Cortex-M85 @ 1 GHz). AI-only latency. **CPU-only — NPU not supported.**

| Memory Configuration | CPU (ms) |
|----------------------|:--------:|
| † Internal Flash + Internal SRAM | 368 ms |
---

## Folder Structure

```
yolo_fastest_1_1/
├── README.md                   ← This file
├── python                      ← Inference scripts, conversion tools, config
└── embedded_c                  ← Compiled C-code for MCU (CPU only)
```

---

## Prerequisites

1. **Python 3.10** installed (see [Install Python 3.10](../../../README.md#install-python-310) in the top-level README for platform-specific steps).
2. **Inference venv** — navigate to the `python/` directory and create a dedicated virtual environment:

    **Windows PowerShell**

    ```powershell
    cd vision\object_detection\yolo_fastest\python
    python -m venv .venv_yolof
    .\.venv_yolof\Scripts\Activate.ps1
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

    **Ubuntu / bash**

    ```bash
    cd vision/object_detection/yolo_fastest_1_1/python
    python3.10 -m venv .venv_yolof
    source .venv_yolof/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

3. **Compiler venv** (`.mera_venv`) — required only for [Step 4](#step-4--compile-for-ra8p1-ruhmi). See the [top-level README](../../../README.md) for setup instructions.

---

## Step 1 — Obtain the Model

The TFLite models are already provided in `python/model/`. To regenerate them from scratch, proceed to [Step 2](#step-2--convert-to-tflite).

---

## Step 2 — Convert to TFLite

Use `download_model.py` to download Darknet weights and convert to TFLite. Activate the **inference venv** and navigate to `python/`.

**Both FP32 + INT8 (default)**

```bash
python download_model.py
```

The script will automatically download the Darknet `.cfg` / `.weights` from GitHub, convert to ONNX, then to TFLite FP32 and INT8 (calibrated on COCO val2017, auto-downloaded). Use `--mode fp32` to skip INT8 or `--mode int8` for INT8 only. Output files are written to `python/model/`.

---

## Step 3 — Run Inference (Python)

Activate the **inference venv** and navigate to `python/`:

**Windows PowerShell**

```powershell
cd vision\object_detection\yolo_fastest\python
.\.venv_yolof\Scripts\Activate.ps1
python inference.py --image sample_images\000000000139.jpg
```

**Ubuntu / bash**

```bash
cd vision/object_detection/yolo_fastest_1_1/python
source .venv_yolof/bin/activate
python inference.py --image sample_images/000000000139.jpg
```

Optional flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `model/yolo_fastest_1.1.tflite` | Path to TFLite model |
| `--image` | — | Input image path |
| `--score` | `0.25` | Confidence threshold |
| `--nms` | `0.45` | NMS IoU threshold |
| `--output` | `outputs/` | Output directory or file path |
| `--display` | off | Open a display window |

**Example output:**

```
Model   : model/yolo_fastest_1.1.tflite
Input   : 320 x 320
Latency : 45.2 ms
Detected: 3 object(s)
  -> person           score=0.872  box=[120,45,280,410]
  -> car              score=0.654  box=[10,200,150,320]
  -> dog              score=0.523  box=[200,180,310,300]
Saved   : outputs/sample_result.jpg
```

> By default `inference.py` loads `model/yolo_fastest_1.1.tflite`. Pass `--model model/yolo_fastest_1.1_int8.tflite` to use the INT8 variant.

---

## Step 4 — Compile for RA8P1 (RUHMI)

This step converts the TFLite model into C-code for the RA8P1 MCU. Activate the **compiler venv** (`.mera_venv`).

> [!IMPORTANT]
> YOLO-Fastest supports **CPU target only** (`target: cpu`). NPU deployment is not available for this model.

### 4.1 — Edit the compile configuration

Open `python/config.yaml` and verify the paths match your system:

```yaml
model_path: vision/object_detection/yolo_fastest_1_1/python/model/yolo_fastest_1.1_int8.tflite
output_dir: vision/object_detection/yolo_fastest_1_1/embedded_c/src_mcu
target: cpu        # CPU only — NPU not supported
quantize: false    # model is already INT8
```

> [!IMPORTANT]
> Use **absolute paths** if running the compiler from outside the repo root.

### 4.2 — Run the compiler

Navigate back to the **repository root** and run the compiler with `.mera_venv` active:

**Windows PowerShell**

```powershell
cd C:\Users\<you>\Model-zoo
python ruhmi_tools\mcu_compile.py vision\object_detection\yolo_fastest\python\config.yaml
```

**Ubuntu / bash**

```bash
cd ~/Model-zoo
python ruhmi_tools/mcu_compile.py vision/object_detection/yolo_fastest_1_1/python/config.yaml
```

The compiled C-code will be written to the `output_dir` specified in `config.yaml`.

---

## Step 5 — Embedded C Integration

The `embedded_c/` folder contains portable, board-independent files you can drop directly into any bare-metal or RTOS project.

| File | Purpose |
|------|---------|
| `model_metadata.h` | All compile-time constants — input shape, output heads, anchors, quant params, thresholds |
| `preprocessing.h` / `preprocessing.c` | Letterbox resize + quantize uint8 RGB → int8 input tensor |
| `postprocessing.h` / `postprocessing.c` | Decode anchor-based boxes, score filter, per-class NMS |

> [!NOTE]
> These files have **no external dependencies** beyond `<stdint.h>`, `<math.h>`, and `<stddef.h>`. They compile cleanly with any C99-compatible toolchain (GCC, Clang, IAR, AC6).

---

### 5.1 — Add files to your project

Copy the following files into your firmware project (or add them as include paths):

```
embedded_c/
├── model_metadata.h
├── preprocessing.h
├── preprocessing.c
├── postprocessing.h
└── postprocessing.c
```

Also copy the compiled model artifacts:

- **CPU-only (CMSIS-NN):** `embedded_c/src_mcu/`

---

### 5.2 — `model_metadata.h` — Key constants

```c
#include "model_metadata.h"
```

| Constant | Value | Description |
|----------|-------|-------------|
| `MODEL_INPUT_H` | `320` | Input height (pixels) |
| `MODEL_INPUT_W` | `320` | Input width (pixels) |
| `MODEL_INPUT_C` | `3` | RGB channels |
| `MODEL_NUM_CLASSES` | `80` | COCO classes |
| `MODEL_INPUT_SCALE` | `0.00392157f` | INT8 input scale (≈ 1/255) |
| `MODEL_INPUT_ZERO_POINT` | `-128` | INT8 input zero-point |
| `MODEL_HEAD0_GRID_H/W` | `10 / 10` | Head 0 grid (stride-32, large objects) |
| `MODEL_HEAD1_GRID_H/W` | `20 / 20` | Head 1 grid (stride-16, small objects) |
| `MODEL_HEAD0_SCALE` | `0.21769431f` | Head 0 output dequant scale |
| `MODEL_HEAD1_SCALE` | `0.20347583f` | Head 1 output dequant scale |
| `POSTPROC_SCORE_THRESH` | `0.25f` | Minimum detection confidence |
| `POSTPROC_NMS_THRESH` | `0.45f` | NMS IoU threshold |
| `POSTPROC_MAX_DETS` | `100` | Max detections per image |

---

### 5.3 — Preprocessing

```c
#include "preprocessing.h"

void preprocess(const uint8_t *p_source_image,
                int source_width, int source_height,
                int8_t *p_destination_tensor);
```

Performs letterbox resize to 320×320 (gray=114 padding), then quantizes using `MODEL_INPUT_SCALE` and `MODEL_INPUT_ZERO_POINT`.

---

### 5.4 — Postprocessing

```c
#include "postprocessing.h"

int postprocess(const int8_t *p_head0_output,
                const int8_t *p_head1_output,
                Detection    *p_detections,
                int           max_detections);
```

Decodes both anchor-based YOLO heads, applies score filtering and per-class NMS. Returns the number of valid detections.

Each `Detection`:
```c
typedef struct {
    float x1, y1, x2, y2;   /* bounding box in original image coords */
    float score;             /* objectness × class confidence */
    int   class_id;          /* COCO class index (0–79) */
} Detection;
```

---

## License / Provenance

| Component | Source | Licence |
|-----------|--------|---------|
| YOLO-Fastest weights & architecture | [dog-qiuqiu/Yolo-Fastest](https://github.com/dog-qiuqiu/Yolo-Fastest) | MIT |
| COCO dataset (calibration & evaluation) | [cocodataset.org](https://cocodataset.org) | CC-BY 4.0 |
