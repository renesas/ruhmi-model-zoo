# BlazeFace Front — Face Detection (WIDER FACE)

BlazeFace is a lightweight anchor-based face detector from Google MediaPipe ([Bazarevsky et al., 2019](https://arxiv.org/abs/1907.05047)), optimised for mobile real-time inference. This implementation uses the PINTO model zoo conversion of the front-face variant (128×128 input). The model detects faces and regresses 6 facial keypoints per detection. This folder contains everything needed to obtain the model, convert it to TFLite, compile it for the Renesas RA8P1 (Cortex-M85 + Ethos-U55 NPU) using the RUHMI toolchain, and run inference.

---

## Model Details

| Field | Value |
|-------|-------|
| **Model name** | BlazeFace Front |
| **Task** | Face Detection (bounding box + 6 keypoints) |
| **Framework** | TensorFlow SavedModel → TFLite (PINTO model zoo) |
| **Dataset** | WIDER FACE (validation) |
| **Input shape** | `(1, 128, 128, 3)` — NHWC, RGB, normalised to [0, 1] |
| **Output tensors** | 4 tensors: boxes-s8 `(1,512,16)`, scores-s8 `(1,512,1)`, scores-s16 `(1,384,1)`, boxes-s16 `(1,384,16)` |
| **Anchors** | 896 total (512 stride-8 + 384 stride-16) |
| **Source** | [PINTO model zoo — 030_BlazeFace](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/030_BlazeFace) |

## Model Report Card

Accuracy measured on the WIDER FACE validation set (frontal-face subset, easy difficulty, 1326 images).

| Model Variant | Format | Precision (%) | Recall (%) | F1 (%) | AP@IoU=0.5 (%) |
|---------------|--------|:-------------:|:----------:|:------:|:--------------:|
| TFLite FP32 | `.tflite` | 74.36 | 38.21 | 50.48 | 35.47 |
| TFLite INT8 | `.tflite` | 73.93 | 37.86 | 50.07 | 34.69 |
| MERA FP32 | `.mera` | 74.36 | 38.21 | 50.48 | 35.47 |
| MERA INT8 (TFLite Quantized) | `.mera` | 73.93 | 37.86 | 50.07 | 34.71 |

> [!NOTE]
> The model was compiled using `mera-2.5.0+pkg.3577` and `FSP 6.2` was used for building and testing the project.

### Inference Performance (RA8P1)

Measured on-target (Cortex-M85 @ 1 GHz, Ethos-U55 NPU @ 500 MHz). AI-only latency.

| Memory Configuration | CPU (ms) | NPU (ms) |
|----------------------|:--------:|:--------:|
| † Internal Flash + Internal SRAM | 36 | 0.033 |

---

## Folder Structure

```
blazeface/
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
    cd vision\face_detection\blazeface\python
    python -m venv .venv_blazeface
    .\.venv_blazeface\Scripts\Activate.ps1
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

    **Ubuntu / bash**

    ```bash
    cd vision/face_detection/blazeface/python
    python3.10 -m venv .venv_blazeface
    source .venv_blazeface/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

3. **Compiler venv** (`.mera_venv`) — required only for [Step 3](#step-3--compile-for-ra8p1-ruhmi). See the [top-level README](../../../README.md) for setup instructions.

---

## Step 1 — Obtain the Model

The TFLite models (FP32 + INT8) and anchors are **already included in the repository** under `python/model/`.

To regenerate them from the PINTO model zoo source, use `download_model.py`:

**Ubuntu / bash**

```bash
cd vision/face_detection/blazeface/python
source .venv_blazeface/bin/activate
python download_model.py                   # FP32 + INT8
python download_model.py --mode fp32       # FP32 only
python download_model.py --mode int8       # INT8 only (requires WIDER FACE val images for calibration)
```

**Windows PowerShell**

```powershell
cd vision\face_detection\blazeface\python
.\.venv_blazeface\Scripts\Activate.ps1
python download_model.py
```

The script downloads the PINTO SavedModel archive, converts to TFLite (FP32 + INT8), and saves the 896 front-face anchors to `model/anchors.npy`.

---

## Step 2 — Run Inference (Python)

Activate the **inference venv** and navigate to `python/`:

**Ubuntu / bash**

```bash
cd vision/face_detection/blazeface/python
source .venv_blazeface/bin/activate
python inference.py --image sample_images/000000001296.jpg
```

**Windows PowerShell**

```powershell
cd vision\face_detection\blazeface\python
.\.venv_blazeface\Scripts\Activate.ps1
python inference.py --image sample_images/000000001296.jpg
```

Optional flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `model/blazeface_front_fp32.tflite` | Path to TFLite model |
| `--thresh` | `0.5` | Minimum face confidence score |
| `--nms` | `0.3` | Weighted-NMS IoU threshold |
| `--display` | off | Open a window to view the annotated result |
| `--verbose` | off | Print per-detection coordinates |

**Example output:**

```
Model   : model/blazeface_front_int8.tflite
Image   : face.jpg (640×480)
Thresh  : 0.50  NMS IoU: 0.30

Detected 2 face(s):
  #1  score=0.94  box=[102, 45, 245, 210]  keypoints: 6
  #2  score=0.87  box=[350, 60, 480, 230]  keypoints: 6
```

> By default `inference.py` loads `model/blazeface_front_fp32.tflite`. Pass `--model model/blazeface_front_int8.tflite` to use the INT8 variant.

---

## Step 3 — Compile for RA8P1 (RUHMI)

This step converts the TFLite model into C-code that runs on the RA8P1 MCU. Activate the **compiler venv** (`.mera_venv`) — see the [top-level README](../../../README.md) for setup and installation instructions.

### 3.1 — Edit the compile configuration

Open `python/config.yaml` and set the `model_path` and `output_dir` to **absolute paths** on your system:

```yaml
model_path: /path/to/vision/face_detection/blazeface/python/model/blazeface_front_int8.tflite
output_dir: /path/to/vision/face_detection/blazeface/embedded_c/src_mcu
target: cpu        # 'cpu' → CMSIS-NN  |  'npu' → Ethos-U55 NPU
quantize: false    # model is already INT8
```

> [!IMPORTANT]
> The `model_path` and `output_dir` fields must be **absolute paths**.

> [!TIP]
> For NPU deployment, set `target: npu` and change `output_dir` to point to `embedded_c/src_mcu_npu`.

### 3.2 — Run the compiler

Navigate to the **repository root** and run the compiler with `.mera_venv` active:

**Ubuntu / bash**

```bash
cd ~/ruhmi-model-zoo
python ruhmi_tools/mcu_compile.py vision/face_detection/blazeface/python/config.yaml
```

**Windows PowerShell**

```powershell
cd C:\Users\<you>\ruhmi-model-zoo
python ruhmi_tools\mcu_compile.py vision\face_detection\blazeface\python\config.yaml
```

The compiled C-code will be written to the `output_dir` specified in `config.yaml`.

---

## Step 4 — Embedded C Integration

The `embedded_c/` folder contains portable, board-independent files you can drop directly into any bare-metal or RTOS project.

| File | Purpose |
|------|---------|
| `model_metadata.h` | All compile-time constants — input/output shape, quantization parameters, anchor layout, post-processing thresholds |
| `preprocessing.h` / `preprocessing.c` | Resize and quantize input image |
| `postprocessing.h` / `postprocessing.c` | Dequantize, decode anchors, weighted NMS |

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

- **CPU-only (CMSIS-NN):** `embedded_c/src_mcu/`
- **NPU-accelerated (Ethos-U55):** `embedded_c/src_mcu_npu/`

---

### 4.2 — `model_metadata.h` — Key constants

Include this header anywhere you need model-specific values:

```c
#include "model_metadata.h"
```

| Constant | Value | Description |
|----------|-------|-------------|
| `MODEL_INPUT_H` | `128` | Input image height (pixels) |
| `MODEL_INPUT_W` | `128` | Input image width (pixels) |
| `MODEL_INPUT_C` | `3` | RGB channels |
| `MODEL_INPUT_SIZE` | `49152` | Flat input buffer size (H × W × C bytes) |
| `INPUT_ZP` | `-128` | INT8 input zero-point — `q = pixel - 128` |
| `NUM_KEYPOINTS` | `6` | Facial keypoints per detection |
| `TOTAL_ANCHORS` | `896` | Total anchors (512 stride-8 + 384 stride-16) |
| `OUTPUT_BOX_COORDS` | `16` | Values per box (4 bbox + 6×2 keypoints) |
| `OUTPUT_SCORES_S8_SCALE` | `0.03629f` | Stride-8 scores dequant scale |
| `OUTPUT_SCORES_S8_ZP` | `46` | Stride-8 scores zero-point |
| `OUTPUT_SCORES_S16_SCALE` | `1.3571f` | Stride-16 scores dequant scale |
| `OUTPUT_SCORES_S16_ZP` | `126` | Stride-16 scores zero-point |
| `OUTPUT_BOXES_S8_SCALE` | `0.3178f` | Stride-8 boxes dequant scale |
| `OUTPUT_BOXES_S8_ZP` | `-40` | Stride-8 boxes zero-point |
| `OUTPUT_BOXES_S16_SCALE` | `1.2865f` | Stride-16 boxes dequant scale |
| `OUTPUT_BOXES_S16_ZP` | `-56` | Stride-16 boxes zero-point |
| `POSTPROC_SCORE_THRESH` | `0.50f` | Default confidence threshold |
| `POSTPROC_NMS_THRESH` | `0.30f` | Default weighted-NMS IoU threshold |
| `POSTPROC_MAX_DETS` | `64` | Maximum detections post-NMS |

---

### 4.3 — Preprocessing — `preprocess()`

```c
#include "preprocessing.h"

void preprocess(const uint8_t *p_source_rgb,
                int32_t        source_width,
                int32_t        source_height,
                int8_t         p_output_tensor[MODEL_INPUT_SIZE]);
```

**Typical call:**

```c
#include "model_metadata.h"
#include "preprocessing.h"

/* Camera frame: any resolution RGB, stored as interleaved uint8 */
static uint8_t camera_buf[320 * 240 * 3];

/* Quantized int8 buffer fed to the model input tensor */
static int8_t nn_input[MODEL_INPUT_SIZE];   /* 49152 bytes */

/* --- inside your inference loop --- */
preprocess(camera_buf, 320, 240, nn_input);
```

> [!TIP]
> `preprocess()` performs bilinear resize to 128×128 and quantizes: `q = (int8_t)(pixel - 128)`.

---

### 4.4 — Postprocessing — `postprocess()`

Decode the 4 INT8 output tensors into face detections with keypoints:

```c
#include "model_metadata.h"
#include "postprocessing.h"

/* Raw INT8 outputs from inference (4 tensors) */
int8_t scores_s8[ANCHORS_S8];           /* (512,)  */
int8_t scores_s16[ANCHORS_S16];         /* (384,)  */
int8_t boxes_s8[ANCHORS_S8 * 16];      /* (512, 16) */
int8_t boxes_s16[ANCHORS_S16 * 16];    /* (384, 16) */

/* Anchor array (896×4) — load from anchors.npy or embed as const */
extern const float anchors[896][4];

/* Detection output */
BfDetections results;

postprocess(scores_s8, scores_s16, boxes_s8, boxes_s16,
            anchors, POSTPROC_SCORE_THRESH, POSTPROC_NMS_THRESH,
            &results);

for (int i = 0; i < results.n; i++)
    printf("Face %d  score=%.2f  box=[%.3f,%.3f,%.3f,%.3f]\n",
           i, results.dets[i].score,
           results.dets[i].xmin, results.dets[i].ymin,
           results.dets[i].xmax, results.dets[i].ymax);
```

> [!TIP]
> `postprocess()` handles dequantization (each tensor has its own scale/zp), sigmoid activation, anchor decoding, and weighted NMS internally. Output coordinates are normalised to [0, 1] — multiply by image dimensions to get pixel values.

---

## License / Provenance

| Component | Source | Licence |
|-----------|--------|---------|
| BlazeFace model | [Google MediaPipe](https://github.com/google/mediapipe) | Apache 2.0 |
| PINTO TFLite conversion | [PINTO model zoo](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/030_BlazeFace) | Apache 2.0 |
| WIDER FACE dataset (validation) | [WIDER FACE](http://shuoyang1213.me/WIDERFACE/) | Creative Common License |
