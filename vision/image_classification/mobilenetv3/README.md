# MobileNetV3-Small — Image Classification (ImageNet)

MobileNetV3-Small (minimalistic) is a lightweight convolutional network ([Howard et al., 2019](https://arxiv.org/abs/1905.02244)) optimized for mobile and embedded vision applications. This folder contains everything needed to obtain the model, convert it to TFLite, compile it for the Renesas RA8P1 (Cortex-M85 + Ethos-U55 NPU) using the RUHMI toolchain, and run inference.

---

## Model Details

| Field | Value |
|-------|-------|
| **Model name** | MobileNetV3-Small (minimalistic) |
| **Task** | Image Classification |
| **Framework** | Keras / TensorFlow 2.x |
| **Dataset** | ImageNet-1k (1000 classes) |
| **Input shape** | `(1, 192, 192, 3)` — HWC, RGB |
| **Output shape** | `(1, 1000)` — 1000-class probabilities (softmax in-graph) |
| **Source** | `tf.keras.applications.MobileNetV3Small(weights="imagenet", minimalistic=True)` |

## Model Report Card

Accuracy measured on the ImageNet-1k validation set (50 000 images).

| Model Variant | Format | Top-1 (%) | Top-5 (%) |
|---------------|--------|:---------:|:---------:|
| TFLite FP32 | `.tflite` | 52.83 | 76.54 |
| TFLite INT8 | `.tflite` | 35.89 | 60.40 |
| MERA FP32 | `.mera` | 52.83 | 76.54 |
| MERA INT8 (TFLite Quantized) | `.mera` | 35.98 | 60.25 |

> [!NOTE]
> The model was compiled using `mera-2.6.0+pkg.4293` and `FSP 6.2` was used for building and testing the project.

### Inference Performance (RA8P1)

Measured on-target (Cortex-M85 @ 1 GHz, Ethos-U55 NPU @ 500 MHz). AI-only latency.

| Memory Configuration | CPU (ms) | NPU (ms) |
|----------------------|:--------:|:--------:|
| ‡ OSPI + External SDRAM | 87 | 10 |
| * OSPI + Internal SRAM | — | 24 |

---

> [!NOTE]
> **Performance figures may differ slightly on your machine.**
> Running `download_model.py` generates a fresh INT8 model calibrated on your local dataset with your installed TensorFlow version. The architecture and weights are identical, but per-channel quantization constants will vary, resulting in:
> - **Inference speed:** no difference — same graph structure and operation count.
> - **Accuracy:** within ~0.5 % Top-1 of the reported figures.
> - **C-source / `.tflite` files:** will not match the pre-built artifacts in `embedded_c/` byte-for-byte — this is expected.
>
> To reproduce results exactly, use the pre-built files in `embedded_c/` instead of regenerating them.

## Folder Structure

```
mobilenetv3/
├── README.md                   ← This file
├── python                      ← Inference scripts, conversion tools, config
└── embedded_c                  ← Compiled C-code for MCU (CPU & NPU)
```

## Prerequisites

1. **Python 3.10** installed (see [Install Python 3.10](../../../README.md#install-python-310) in the top-level README for platform-specific steps).
2. **Inference venv** — navigate to the `python/` directory and create a dedicated virtual environment:

    **Windows PowerShell**

    ```powershell
    cd vision\image_classification\mobilenetv3\python
    python -m venv .venv_mobilenetv3
    .\.venv_mobilenetv3\Scripts\Activate.ps1
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

    **Ubuntu / bash**

    ```bash
    cd vision/image_classification/mobilenetv3/python
    python3.10 -m venv .venv_mobilenetv3
    source .venv_mobilenetv3/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

3. **Compiler venv** (`.mera_venv`) — required only for [Step 4](#step-4--compile-for-ra8p1-ruhmi). See the [top-level README](../../../README.md) for setup instructions.

---

## Step 1 — Obtain the Model

The TFLite models are already provided in `python/model/`. To regenerate them from scratch, proceed to [Step 2](#step-2--convert-to-tflite).

---

## Step 2 — Convert to TFLite

Use `download_model.py` to build and convert the model. Activate the **inference venv** and navigate to `python/`.

**Both FP32 + INT8 (default)** — requires ImageNet validation images for INT8 calibration

```bash
python download_model.py --calib-dir /path/to/ILSVRC2012_img_val
```

Use `--mode FP32` to skip INT8 (no calibration images needed) or `--mode INT8` for INT8 only. Use `--input-size 224` or `--input-size 128` to generate models at a different resolution. Output files are written to `python/model/`.

---

## Step 3 — Run Inference (Python)

Activate the **inference venv** and navigate to `python/`:

**Windows PowerShell**

```powershell
cd vision\image_classification\mobilenetv3\python
.\.venv_mobilenetv3\Scripts\Activate.ps1
python inference.py --image sample_images/coco_cat.jpg
```

**Ubuntu / bash**

```bash
cd vision/image_classification/mobilenetv3/python
source .venv_mobilenetv3/bin/activate
python inference.py --image sample_images/coco_cat.jpg
```

Optional flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `model/mobilenet_v3_small_192_FP32.tflite` | Path to TFLite model |
| `--labels` | `utils/imagenet_labels.txt` | Path to labels file (one class per line) |
| `--top-k` | `5` | Number of top predictions to display |
| `--input-size` | `192` | Input image size H=W. Supported: 128, 192, 224 |

**Example output:**

```
══════════════════════════════════════
Image              : dog.jpg
Predicted class ID : 207
Predicted label    : golden_retriever
Confidence (score) : 0.8124
══════════════════════════════════════
```

> By default `inference.py` loads `model/mobilenet_v3_small_192_FP32.tflite` and `utils/imagenet_labels.txt`. Pass `--model model/mobilenet_v3_small_192_INT8.tflite` to use the INT8 variant. Use `--input-size` if your model was trained at a different resolution.

---

## Step 4 — Compile for RA8P1 (RUHMI)

This step converts the TFLite model into C-code for the RA8P1 MCU. Activate the **compiler venv** (`.mera_venv`).

### 4.1 — Edit the compile configuration

Open `python/config.yaml` and verify the paths match your system:

```yaml
model_path: vision/image_classification/mobilenetv3/python/model/mobilenet_v3_small_192_INT8.tflite
output_dir: vision/image_classification/mobilenetv3/embedded_c/src_mcu
target: cpu        # 'cpu' → CMSIS-NN  |  'npu' → Ethos-U55 NPU
quantize: false    # model is already INT8
```

> [!IMPORTANT]
> Use **absolute paths** if running the compiler from outside the repo root.

> [!TIP]
> For NPU deployment set `target: npu` and point `output_dir` to `embedded_c/src_mcu_npu`.

### 4.2 — Run the compiler

Navigate back to the **repository root** and run the compiler with `.mera_venv` active:

**Windows PowerShell**

```powershell
cd C:\Users\<you>\Model-zoo
python ruhmi_tools\mcu_compile.py vision\image_classification\mobilenetv3\python\config.yaml
```

**Ubuntu / bash**

```bash
cd ~/Model-zoo
python ruhmi_tools/mcu_compile.py vision/image_classification/mobilenetv3/python/config.yaml
```

The compiled C-code will be written to the `output_dir` specified in `config.yaml`.

---

## Step 5 — Embedded C Integration

The `embedded_c/` folder contains three portable, board-independent files you can drop directly into any bare-metal or RTOS project.

| File | Purpose |
|------|---------|
| `model_metadata.h` | All compile-time constants — input/output shape, quantization parameters, tensor-arena sizes, class count |
| `preprocessing.h` / `preprocessing.c` | Prepare model input |
| `postprocessing.h` / `postprocessing.c` | Process model output |

> [!NOTE]
> These files have **no external dependencies** beyond `<stdint.h>` and `<stddef.h>`. They compile cleanly with any C99-compatible toolchain (GCC, Clang, IAR, AC6).

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

Also copy the compiled model artifacts from the appropriate subdirectory into your project:

- **CPU-only (CMSIS-NN):** `embedded_c/src_mcu/`
- **NPU-accelerated (Ethos-U55):** `embedded_c/src_mcu_npu/`

---

### 5.2 — `model_metadata.h` — Key constants

Include this header anywhere you need model-specific values:

```c
#include "model_metadata.h"
```

| Constant | Value | Description |
|----------|-------|-------------|
| `MODEL_INPUT_H` | `192` | Input image height (pixels) |
| `MODEL_INPUT_W` | `192` | Input image width (pixels) |
| `MODEL_INPUT_C` | `3` | RGB channels |
| `MODEL_INPUT_SIZE` | `110592` | Flat input buffer size (H × W × C) |
| `MODEL_OUTPUT_SIZE` | `1000` | Number of output classes |
| `INPUT_SCALE` | `1.0f` | INT8 input quantisation scale |
| `INPUT_ZP` | `-128` | INT8 input zero-point — `q = pixel - 128` |
| `OUTPUT_SCALE` | `0.00390625f` | INT8 output dequantisation scale (= 1/256) |
| `OUTPUT_ZP` | `-128` | INT8 output zero-point |

---

### 5.3 — Preprocessing — `preprocess()`

```c
#include "preprocessing.h"

void preprocess(const uint8_t *p_source_image_hwc,
                int            _source_width,
                int            _source_height,
                int8_t        *p_quantized_input_tensor);
```

Performs bilinear resize to `[192 × 192]`, then quantizes to INT8: `q = clamp(pixel - 128, -128, 127)`. No external normalization is needed — the model has an internal Rescaling(1/255) layer.

**Typical call:**

```c
#include "model_metadata.h"
#include "preprocessing.h"

/* Camera frame: 320×240 RGB, stored as interleaved uint8 */
static uint8_t camera_buf[320 * 240 * 3];

/* Quantized int8 buffer fed to the model input tensor */
static int8_t nn_input[MODEL_INPUT_SIZE];

/* --- inside your inference loop --- */
preprocess(camera_buf, 320, 240, nn_input);
```

---

### 5.4 — Postprocessing — `postprocess()`

The postprocessing pipeline dequantizes the INT8 output, then selects the top-k highest-scoring classes:

```c
#include "model_metadata.h"
#include "postprocessing.h"

/* Raw INT8 output from the model */
int8_t raw_out[MODEL_OUTPUT_SIZE];

/* Top-k results */
int   top_indices[TOP_K];
float top_scores[TOP_K];

postprocess(raw_out, MODEL_OUTPUT_SIZE,
            OUTPUT_SCALE, OUTPUT_ZP,
            TOP_K, top_indices, top_scores);

for (int i = 0; i < TOP_K; i++)
    printf("#%d  class=%d  score=%.4f\n", i + 1, top_indices[i], top_scores[i]);
```

**API:**

```c
void postprocess(const int8_t *p_quantized_output,
                 int           _class_count,
                 float         _output_scale,
                 int           _output_zero_point,
                 int           _top_k_count,
                 int          *p_top_indices,
                 float        *p_top_scores);
```

> [!NOTE]
> `postprocess()` handles dequantization internally — you do **not** need to dequantize manually.
> A separate `dequantize_output_to_scores()` helper is also available if you need the full 1000-class score array.

---

## License / Provenance

MobileNetV3 weights are provided by the TensorFlow / Keras team under the [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0) via `tf.keras.applications`.

The ImageNet dataset used for calibration and evaluation is subject to the [ImageNet Terms of Access](https://image-net.org/download.php).
