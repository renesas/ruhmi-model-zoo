# YOLOX-Tiny — Object Detection (COCO)

YOLOX-Tiny is a compact anchor-free object detection network ([Ge et al., 2021](https://arxiv.org/abs/2107.08430)) from Megvii. It uses a CSPDarknet backbone with a PAFPN neck and a decoupled detection head. This folder contains everything needed to obtain the model, compile it for the Renesas RA8P1 (Cortex-M85 + Ethos-U55 NPU) using the RUHMI toolchain, and run inference.

---

## Model Details

| Field | Value |
|-------|-------|
| **Model name** | YOLOX-Tiny (224×224) |
| **Task** | Object Detection (anchor-free) |
| **Framework** | PyTorch → ONNX → TFLite (via onnx2tf) |
| **Dataset** | COCO 2017 — 80 classes |
| **Input shape** | `(1, 224, 224, 3)` — NHWC, RGB, [0–255] float (no /255 normalisation) |
| **Output shape** | `(1, 1029, 85)` — 1029 predictions × (4 bbox + 1 obj + 80 cls) |
| **FPN grids** | 28×28 (stride 8) + 14×14 (stride 16) + 7×7 (stride 32) |
| **Source** | [Megvii YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) |.

## Model Report Card

Accuracy measured on the COCO 2017 validation set (5000 images, 224×224 input).

| Model Variant | Format | COCO AP@[.5:.95] | AP@0.50 |
|---------------|--------|:----------------:|:-------:|
| TFLite FP32 | `.tflite` | 22.65 | 35.95 |
| TFLite INT8 | `.tflite` | 21.72 | 35.13 |
| MERA FP32 | `.mera` |  22.65 | 35.95 |
| MERA INT8 (TFLite Quantized) | `.mera` | 21.69 | 35.21 |

> [!NOTE]
> The model was compiled using `mera-2.6.0+pkg.4293` and `FSP 6.2` was used for building and testing the project.

### Inference Performance (RA8P1)

Measured on-target (Cortex-M85 @ 1 GHz, Ethos-U55 NPU @ 500 MHz). AI-only latency.

| Memory Configuration | CPU (ms) | NPU (ms) |
|----------------------|:--------:|:--------:|
| ‡ OSPI + External SDRAM | 1914 | 73 |

---

## Folder Structure

```
yolox_tiny/
├── README.md                   ← This file
├── python                      ← Inference scripts, conversion tools, config
└── embedded_c                  ← Compiled C-code for MCU (CPU & NPU)
```

---

## Prerequisites

1. **Python 3.10** installed (see [Install Python 3.10](../../../README.md#install-python-310) in the top-level README for platform-specific steps).
2. **Inference venv** — navigate to the `python/` directory and create a dedicated virtual environment:

    **Windows PowerShell**

    ```powershell
    cd vision\object_detection\yolox_tiny\python
    python -m venv .venv_yolox
    .\.venv_yolox\Scripts\Activate.ps1
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

    **Ubuntu / bash**

    ```bash
    cd vision/object_detection/yolox_tiny/python
    python3.10 -m venv .venv_yolox
    source .venv_yolox/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

3. **Compiler venv** (`.mera_venv`) — required only for [Step 3](#step-3--compile-for-ra8p1-ruhmi). See the [top-level README](../../../README.md) for setup instructions.

---

## Step 1 — Obtain the Model

The ONNX and TFLite models (FP32 + INT8) for YOLOX-Tiny 224×224 are **already included in the repository** under `python/model/`.

To regenerate them from the original PyTorch checkpoint, use `download_model.py`. Make sure the **inference venv** (`.venv_yolox`) is active and you are in the `python/` directory:

**Windows PowerShell**

```powershell
python download_model.py
```

**Ubuntu / bash**

```bash
python download_model.py
```

This will download the pretrained `.pth` from the [Megvii YOLOX GitHub releases](https://github.com/Megvii-BaseDetection/YOLOX/releases), export to ONNX, and convert to TFLite (FP32 + INT8). INT8 conversion requires COCO val2017 calibration images — see the script for details.

---

## Step 2 — Run Inference (Python)

Run single-image inference using the ONNX or TFLite model. Make sure the **inference venv** (`.venv_yolox`) is active and you are in the `python/` directory:

**Windows PowerShell**

```powershell
python inference.py --image sample_images/000000000139.jpg
```

**Ubuntu / bash**

```bash
python inference.py --image sample_images/000000000139.jpg
```

`<image_path>` can be any `.png` / `.jpg` image file (will be letterbox-resized to 224×224 automatically).

Optional flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `model/yolox_tiny.onnx` | Path to ONNX or TFLite model |
| `--score` | `0.3` | Confidence threshold |
| `--nms` | `0.45` | NMS IoU threshold |
| `--output` | `output/` | Save annotated image to a specific path |
| `--display` | off | Open a window to view the annotated result |
| `--verbose` | off | Print per-detection table and model I/O details |

**Example output:**

```
Detected: 3 object(s)
  person        score=0.87  box=[45,  12, 198, 210]
  bicycle       score=0.73  box=[102, 85, 180, 190]
  car           score=0.61  box=[0,   40, 100, 165]
```

> [!NOTE]
> By default `inference.py` loads the ONNX model from `model/`. To use a different model variant, pass the `--model` flag.

---

## Step 3 — Compile for RA8P1 (RUHMI)

This step converts the TFLite model into C-code that runs on the RA8P1 MCU. Activate the **compiler venv** (`.mera_venv`).

### 3.1 — Edit the compile configuration

Open `python/config.yaml` and set the `model_path` and `output_dir` to **absolute paths** on your system:

```yaml
model_path: /path/to/vision/object_detection/yolox_tiny/python/model/yolox_tiny_224_INT8.tflite
output_dir: /path/to/vision/object_detection/yolox_tiny/embedded_c/src_mcu
target: cpu        # 'cpu' → CMSIS-NN  |  'npu' → Ethos-U55 NPU
quantize: false    # model is already INT8
```

> [!IMPORTANT]
> The `model_path` and `output_dir` fields must be **absolute paths**.

> [!TIP]
> For NPU deployment, set `target: npu` and change `output_dir` to point to `embedded_c/src_mcu_npu`. For FP32 models with MERA quantization, set `quantize: true` and provide a `calib_data` path.

### 3.2 — Run the compiler

Navigate to the **repository root** and run the compiler with `.mera_venv` active:

**Windows PowerShell**

```powershell
cd C:\Users\<you>\ruhmi-model-zoo
python ruhmi_tools\mcu_compile.py vision\object_detection\yolox_tiny\python\config.yaml
```

**Ubuntu / bash**

```bash
cd ~/ruhmi-model-zoo
python ruhmi_tools/mcu_compile.py vision/object_detection/yolox_tiny/python/config.yaml
```

The compiled C-code will be written to the `output_dir` specified in `config.yaml` (e.g. `embedded_c/src_mcu/`).

---

## Step 4 — Embedded C Integration

The `embedded_c/` folder contains three portable, board-independent files you can drop directly into any bare-metal or RTOS project.

| File | Purpose |
|------|---------|
| `model_metadata.h` | All compile-time constants — quant params, FPN grid dimensions, class count, arena sizes |
| `preprocessing.h` / `preprocessing.c` | Prepare model input |
| `postprocessing.h` / `postprocessing.c` | Process model output |

> [!NOTE]
> These files have **no external dependencies** beyond `<stdint.h>` and `<stddef.h>`. They compile cleanly with any C99-compatible toolchain (GCC, Clang, IAR, AC6).

---

### 4.1 — Add files to your project

Copy the following files into your firmware project (or add them as include paths):

```
embedded_c/
├── model_metadata.h
├── preprocessing.h
├── preprocessing.c
├── postprocessing.h
└── postprocessing.c
```

Also copy the compiled model artifacts from the appropriate subdirectory into your project:

- **CPU-only (CMSIS-NN):** copy `embedded_c/src_mcu/` into your e² studio project
- **NPU-accelerated (Ethos-U55):** copy `embedded_c/src_mcu_npu/`

---

### 4.2 — `model_metadata.h` — Key constants

Include this header anywhere you need model-specific values:

```c
#include "model_metadata.h"
```

| Constant | Value | Description |
|----------|-------|-------------|
| `MODEL_INPUT_H` | `224` | Input image height (pixels) |
| `MODEL_INPUT_W` | `224` | Input image width (pixels) |
| `MODEL_INPUT_C` | `3` | Number of colour channels (RGB) |
| `MODEL_INPUT_SIZE` | `150528` | Flat input buffer size (H × W × C) |
| `MODEL_NUM_CLASSES` | `80` | COCO object classes |
| `MODEL_NUM_ANCHORS` | `1029` | Total anchor-free predictions (28²+14²+7²) |
| `MODEL_OUTPUT_CHANNELS` | `85` | Values per anchor (4 bbox + 1 obj + 80 cls) |
| `MODEL_INPUT_SCALE` | `1.0f` | INT8 input quantization scale |
| `MODEL_INPUT_ZERO_POINT` | `-128` | INT8 input zero-point — `q = pixel - 128` |
| `INPUT_LETTERBOX_PAD` | `114` | Letterbox padding fill value (grey) |
| `MODEL_OUTPUT_SCALE` | `0.0267025f` | INT8 output dequantization scale |
| `MODEL_OUTPUT_ZERO_POINT` | `-30` | INT8 output zero-point |
| `POSTPROC_SCORE_THRESH` | `0.25f` | Default confidence threshold |
| `POSTPROC_NMS_THRESH` | `0.45f` | Default NMS IoU threshold |
| `MODEL_STRIDE_S8` | `8` | FPN level 0 stride (28×28 grid) |
| `MODEL_STRIDE_S16` | `16` | FPN level 1 stride (14×14 grid) |
| `MODEL_STRIDE_S32` | `32` | FPN level 2 stride (7×7 grid) |

---

### 4.3 — Preprocessing — `preprocess()`

```c
#include "preprocessing.h"

void preprocess(const uint8_t   *source_image,
                uint16_t         source_width,
                uint16_t         source_height,
                int8_t          *p_destination_image,
                uint16_t         destination_width,
                uint16_t         destination_height,
                letterbox_params_t *p_params);
```

**Typical call:**

```c
#include "model_metadata.h"
#include "preprocessing.h"

/* Camera frame: any resolution BGR, stored as interleaved uint8 */
static uint8_t camera_buf[320 * 240 * 3];

/* Quantized int8 buffer fed to the model input tensor */
static int8_t nn_input[MODEL_INPUT_SIZE];   /* 150528 bytes */

/* Letterbox params needed by postprocess to undo padding */
static letterbox_params_t lb_params;

/* --- inside your inference loop --- */
preprocess(camera_buf, 320, 240,
           nn_input, MODEL_INPUT_W, MODEL_INPUT_H, &lb_params);
```

> [!TIP]
> `preprocess()` performs letterbox resize internally — the source image is scaled to fit within 224×224 while preserving aspect ratio, padded with grey (114), BGR→RGB swapped, and quantized to INT8 (`q = pixel - 128`).

---

### 4.4 — Postprocessing — `postprocess()`

Decode the raw INT8 output into filtered detections with letterbox undo:

```c
#include "model_metadata.h"
#include "postprocessing.h"

/* Raw INT8 output from inference */
int8_t nn_output[MODEL_OUTPUT_SIZE];   /* 87465 bytes */

/* Detection buffer */
Detection_t detections[POSTPROC_MAX_DETS];

/* Decode: dequantize → grid decode → score filter → NMS → undo letterbox */
int32_t num_det = postprocess(nn_output, &lb_params,
                              320, 240,           /* original image w, h */
                              POSTPROC_SCORE_THRESH,
                              POSTPROC_NMS_THRESH,
                              detections);

for (int i = 0; i < num_det; i++)
    printf("Class %u  score=%.2f  box=[%.0f,%.0f,%.0f,%.0f]\n",
           detections[i].cls_id, detections[i].score,
           detections[i].x1, detections[i].y1,
           detections[i].x2, detections[i].y2);
```

> [!TIP]
> `postprocess()` handles dequantization, FPN grid decoding, score filtering, per-class NMS, and letterbox coordinate unscaling internally. Pass the `letterbox_params_t` from `preprocess()` to correctly map boxes back to original image coordinates.

---

## License / Provenance

| Component | Source | Licence |
|-----------|--------|---------|
| YOLOX-Tiny weights | [Megvii YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) | Apache 2.0 |
| COCO 2017 dataset (calibration / validation) | [cocodataset.org](https://cocodataset.org) | CC-BY 4.0 |
