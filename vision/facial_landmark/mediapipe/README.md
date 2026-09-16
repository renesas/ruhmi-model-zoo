# MediaPipe Face Landmark — Facial Landmark (iBUG 300W)

MediaPipe Face Landmark predicts 468 3D facial landmarks and a face-presence score from a 192x192 RGB face crop. This package includes Python inference/conversion scripts, TFLite model artifacts, and embedded C integration files for RA8P1 workflows.

---

## Model Details

| Field | Value |
|-------|-------|
| **Model name** | MediaPipe Face Landmark  |
| **Task** | Facial Landmark Detection (468 3D landmarks + face flag) |
| **Framework** | TensorFlow Lite (from `patlevin/face-detection-tflite`) |
| **Input shape** | `(1, 192, 192, 3)` — NHWC, RGB, float32 normalised to `[0, 1]` |
| **Output tensors** | 2 tensors: `conv2d_20` `(1, 1, 1, 1404)` — 468 × (x, y, z) landmarks; `conv2d_30` `(1, 1, 1, 1)` — face flag logit (sigmoid gate at 0.5) |
| **Pre-stage** | MTCNN face detector (eye-aligned square ROI, `ROI_SCALE = 1.5`) |
| **Landmarks** | 468 |
| **Source** | [patlevin/face-detection-tflite](https://github.com/patlevin/face-detection-tflite) (Google MediaPipe conversion) |

## Model Report Card

Accuracy measured on the iBUG 300W testset (68-point landmarks). Metric: Interocular Normalized Mean Error (**NME %**, lower is better). MediaPipe 468-point predictions are mapped to the 68-point iBUG scheme via a fixed index table. 

| Model Variant | Format | NME (%) | 
|---------------|--------|:-------:|
| TFLite FP32 | `.tflite` | 6.660 | 
| TFLite INT8 | `.tflite` | 6.707 | 
| MERA FP32 | `.mera` | 6.660 | 
| MERA INT8 (TFLite Quantized) | `.mera` | 6.709 | 

> [!NOTE]
> The model was compiled using `mera-2.6.0+pkg.4815` and `FSP 6.5.1` was used for building and testing the pro


## Inference Performance (RA8P1)

Measured on-target (Cortex-M85 @ 1 GHz, Ethos-U55 NPU @ 500 MHz). AI-only latency per 192×192 face crop (single forward pass of the landmark model; MTCNN pre-stage runs on the host).

| Memory Configuration | CPU (ms) | NPU (ms) |
|----------------------|:--------:|:--------:|
| † Internal Flash + Internal SRAM | 104 | - |
| ‡ OSPI + External SDRAM | - | 125 |

> [!Note]
> NPU execution was slower due to the partitioning to multiple subgraphs. This issue is fixed in latest MCU compiler release with Vela 5.1.0
---

## Folder Structure

```
facial_landmark
└── mediapipe/
    ├── README.md                   ← This file
    ├── python                      ← Inference scripts, conversion tools, validation, config
    └── embedded_c                  ← Compiled C-code for MCU (CPU & NPU)

```

---

## Prerequisites

1. **Python 3.10** installed.
2. **Inference venv** — navigate to the `python/` directory and create a dedicated virtual environment:

    **Windows PowerShell**
    > **Note:** If venv activation is blocked by PowerShell execution policy ("running scripts is disabled"), run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in the same PowerShell window, then run the activation command again.

    ```powershell
    cd vision\facial_landmark\mediapipe\python
    py -3.10 -m venv .venv_facelandmark
    .\.venv_facelandmark\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    ```

    **Ubuntu / bash**

    ```bash
    cd vision/facial_landmark/mediapipe/python
    python3.10 -m venv .venv_facelandmark
    source .venv_facelandmark/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

3. **Compiler venv** (`.mera_venv`) — required only for [Step 3](#step-3--compile-for-ra8p1-ruhmi). See the [top-level README](../../../README.md) for setup instructions.

---

## Step 1 — Obtain the Model

The TFLite models (FP32 + INT8) are **already included in the repository** under `python/model/`.

To regenerate them from the `patlevin/face-detection-tflite` source, use `download_model.py`:

**Ubuntu / bash**

```bash
cd vision/facial_landmark/mediapipe/python
source .venv_facelandmark/bin/activate
python download_model.py --mode fp32
python download_model.py --mode int8 --calib-dir /path/to/face_images --calib-num 200
```

**Windows PowerShell**

```powershell
cd vision\facial_landmark\mediapipe\python
.\.venv_facelandmark\Scripts\Activate.ps1
python download_model.py --mode fp32
python download_model.py --mode int8 --calib-dir C:\path\to\face_images --calib-num 200
```

The script fetches the FP32 TFLite from the `patlevin/face-detection-tflite` release, then produces a full-integer INT8 build (int8 in / int8 out, INT8 activations, INT32 bias) via the low-level TFLite `Calibrator` API. Calibration images default to random `[-1, 1]` if `--calib-dir` is omitted; for best INT8 accuracy point `--calib-dir` at any folder of representative face crops.

`download_model.py` CLI flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `all` | `fp32` (download only), `int8` (download + quantize), `all` (same as `int8`) |
| `--calib-num` | `100` | Representative sample count (`int` or `all`) |
| `--calib-dir` | — | Folder of calibration images (omit → random `[-1, 1]`) |

---

## Step 2 — Run Inference (Python)

Activate the **inference venv** and navigate to `python/`. `inference.py` runs MTCNN face detection first, then applies the landmark model on each detected ROI and renders a mesh + bounding-box overlay.

**Ubuntu / bash**

```bash
cd vision/facial_landmark/mediapipe/python
source .venv_facelandmark/bin/activate
python inference.py --input sample_images/000000001296.jpg --model model/face_landmark_FP32.tflite
```

**Windows PowerShell**

```powershell
cd vision\facial_landmark\mediapipe\python
.\.venv_facelandmark\Scripts\Activate.ps1
python inference.py --input sample_images\000000001296.jpg --model model\face_landmark_FP32.tflite
```

Optional flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | — | Path to image or directory (required) |
| `--model` | `model/face_landmark_FP32.tflite` | Path to TFLite model |
| `--output-dir` | `outputs/` | Directory for rendered overlays |

**Example output:**

```
[ok] 000000001296.jpg: mode=auto(full) faces=1 -> outputs/000000001296.jpg
```

> By default `inference.py` loads `model/face_landmark_INT8.tflite`. Pass `--model model/face_landmark_FP32.tflite` to use the FP32 variant.

---

## Step 3 — Compile for RA8P1 (RUHMI)

This step converts the TFLite model into C-code that runs on the RA8P1 MCU. Activate the **compiler venv** (`.mera_venv`) — see the [top-level README](../../../README.md) for setup and installation instructions.

### 3.1 — Edit the compile configuration

Open `python/config.yaml` and set the `model_path` and `output_dir` to **absolute paths** on your system:

```yaml
model_path: /path/to/vision/facial_landmark/mediapipe/python/model/face_landmark_INT8.tflite
output_dir: /path/to/vision/facial_landmark/mediapipe/embedded_c/src_mcu
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
cd ~/Model-zoo
python ruhmi_tools/mcu_compile.py vision/facial_landmark/mediapipe/python/config.yaml
```

**Windows PowerShell**

```powershell
cd C:\Users\<you>\Model-zoo
python ruhmi_tools\mcu_compile.py vision\facial_landmark\mediapipe\python\config.yaml
```

The compiled C-code will be written to the `output_dir` specified in `config.yaml`.

---

## Step 4 — Embedded C Integration

The `embedded_c/` folder contains three portable, board-independent files you can drop directly into any bare-metal or RTOS project.

| File | Purpose |
|------|---------|
| `model_metadata.h` | All compile-time constants — input/output shape, quantization parameters, landmark count, detection threshold |
| `preprocessing.h` / `preprocessing.c` | Prepare model input |
| `postprocessing.h` / `postprocessing.c` | Process model output |

> [!NOTE]
> `preprocessing.c` and `postprocessing.c` depend on `<math.h>`, and `postprocessing.c` also uses `<stdbool.h>`. The interfaces are otherwise board-independent and compile with any C99-compatible toolchain (GCC, Clang, IAR, AC6).

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

```c
#include "model_metadata.h"
```

| Constant | Value | Description |
|----------|-------|-------------|
| `MODEL_INPUT_H` | `192` | Input image height (pixels) |
| `MODEL_INPUT_W` | `192` | Input image width (pixels) |
| `MODEL_INPUT_C` | `3` | RGB channels |
| `MODEL_INPUT_SIZE` | `110592` | Flat input tensor size (192 × 192 × 3) |
| `MODEL_NUM_LANDMARKS` | `468` | Number of predicted 3D landmarks |
| `MODEL_LANDMARK_DIMS` | `3` | Coordinates per landmark (x, y, z) |
| `MODEL_LANDMARK_SIZE` | `1404` | Flat landmark tensor size (468 × 3) |
| `MODEL_FACE_FLAG_SIZE` | `1` | Face-presence logit tensor size |
| `INPUT_QUANT_SCALE` | `0.007843137718737125f` | INT8 input quantisation scale (2/255) |
| `INPUT_QUANT_ZERO_POINT` | `-1` | INT8 input zero-point |
| `LANDMARK_QUANT_SCALE` | `0.9023275971412659f` | INT8 landmark output dequant scale |
| `LANDMARK_QUANT_ZERO_POINT` | `-84` | INT8 landmark output zero-point |
| `FACE_FLAG_QUANT_SCALE` | `0.31730183959007263f` | INT8 face-flag output dequant scale |
| `FACE_FLAG_QUANT_ZERO_POINT` | `-107` | INT8 face-flag output zero-point |
| `FACE_DETECTION_THRESHOLD` | `0.5f` | `sigmoid(face_flag)` gate for a valid detection |

> [!NOTE]
> The quantisation scale/zero-point values above are taken from `python/model/face_landmark_INT8.tflite` (`interpreter.get_input_details()` / `get_output_details()`). If you regenerate the INT8 model, refresh these constants from the new flatbuffer.

---

### 4.3 — Preprocessing — `preprocess()`

```c
#include "preprocessing.h"

void preprocess(const uint8_t *p_source_image_hwc,
                int32_t        _source_width,
                int32_t        _source_height,
                int8_t        *p_input_tensor);
```

Performs a 2-tap separable bilinear resize (half-pixel-centre, edge replicate) of the input face ROI from `(_source_width × _source_height)` to `[MODEL_INPUT_H × MODEL_INPUT_W]` (192 × 192), applies `pixel * (1.0f / 255.0f)` normalisation to `[0, 1]`, then quantises to INT8 using `INPUT_QUANT_SCALE` and `INPUT_QUANT_ZERO_POINT`.

> [!IMPORTANT]
> `preprocess()` assumes `p_source_image_hwc` is already a **cropped face ROI** in RGB888 HWC layout. 

**Typical call:**

```c
#include "model_metadata.h"
#include "preprocessing.h"

/* Cropped face ROI: any resolution RGB888, stored as interleaved uint8 */
static uint8_t face_roi[192 * 192 * 3];

/* Quantized int8 buffer fed to the model input tensor */
static int8_t nn_input[MODEL_INPUT_SIZE];

/* --- inside your inference loop --- */
preprocess(face_roi, 192, 192, nn_input);
```

---

### 4.4 — Postprocessing — `postprocess()`

The postprocessing pipeline dequantises both output tensors, evaluates the face-presence sigmoid, and rescales the 468 landmarks from the 192 × 192 model-input frame back to the original source-image pixel frame. All results are written into a single `face_result_t` struct:

```c
#include "model_metadata.h"
#include "postprocessing.h"

/* Raw INT8 outputs from the model */
int8_t raw_landmarks[MODEL_LANDMARK_SIZE];   /* 1404 = 468 × 3 */
int8_t raw_face_flag[MODEL_FACE_FLAG_SIZE];  /* 1 */

/* Decoded result (landmark coordinates in source-image pixels) */
face_result_t result;

postprocess(raw_landmarks,
            raw_face_flag,
            source_width,   /* original image width  in pixels */
            source_height,  /* original image height in pixels */
            &result);

if (result.face_detected) {
    printf("Face detected  score=%.3f\n", result.face_probability);
    for (int i = 0; i < MODEL_NUM_LANDMARKS; i++) {
        const face_landmark_t *lm = &result.landmarks[i];
        printf("  lm[%d] = (%.1f, %.1f, %.1f)\n",
               lm->part_id, lm->x, lm->y, lm->z);
    }
} else {
    printf("No face  score=%.3f (below %.2f threshold)\n",
           result.face_probability, FACE_DETECTION_THRESHOLD);
}
```

**API:**

```c
void postprocess(const int8_t   *p_raw_landmarks,
                 const int8_t   *p_raw_face_flag,
                 int32_t         _source_width,
                 int32_t         _source_height,
                 face_result_t  *p_result);
```

On return, `p_result->face_probability` holds `sigmoid(face_flag)`, `p_result->face_detected` is `true` when the probability meets `FACE_DETECTION_THRESHOLD`, and `p_result->landmarks[k]` holds the 468 (x, y, z) landmarks already **rescaled into source-image pixel coordinates** (`x` and `z` use the width scale, `y` uses the height scale).

> [!NOTE]
> `postprocess()` handles dequantisation, sigmoid, and rescaling internally — you do **not** need to dequantise, apply sigmoid, or map back from the 192 × 192 frame manually.
> The face-flag output is a raw logit; `postprocess()` applies sigmoid before threshold gating.

---

## License / Provenance

| Component | Source | Licence |
|-----------|--------|---------|
| MediaPipe Face Landmark model | [Google MediaPipe](https://github.com/google/mediapipe) | Apache 2.0 |
| TFLite conversion | [patlevin/face-detection-tflite](https://github.com/patlevin/face-detection-tflite) | MIT |
| MTCNN (pre-stage) | [ipazc/mtcnn](https://github.com/ipazc/mtcnn) | MIT |
| iBUG 300W dataset | [iBUG 300 Faces In-The-Wild Challenge](https://ibug.doc.ic.ac.uk/resources/300-W/) | Research use only |
