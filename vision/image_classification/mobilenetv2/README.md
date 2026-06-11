# MobileNetV2 — Image Classification (ImageNet)

MobileNetV2 is an efficient inverted-residual convolutional network ([Sandler et al., 2018](https://arxiv.org/abs/1801.04381)) designed for mobile and embedded vision applications. This folder contains everything needed to obtain the model, convert it to TFLite, compile it for the Renesas RA8P1 (Cortex-M85 + Ethos-U55 NPU) using the RUHMI toolchain, and run inference.

---

## Model Details

| Field | Value |
|-------|-------|
| **Model name** | MobileNetV2 (alpha=1.0) |
| **Task** | Image Classification |
| **Framework** | Keras / TensorFlow 2.x |
| **Dataset** | ImageNet-1k (1000 classes) |
| **Input shape** | `(1, 224, 224, 3)` — HWC, RGB |
| **Output shape** | `(1, 1000)` — 1000-class probabilities (softmax in-graph) |
| **Source** | `tf.keras.applications.MobileNetV2(weights="imagenet")` |

## Model Report Card

Accuracy measured on the ImageNet-1k validation set (50 000 images).

| Model Variant | Format | Top-1 (%) | Top-5 (%) |
|---------------|--------|:---------:|:---------:|
| TFLite FP32 | `.tflite` | 57.67 | 80.78 |
| TFLite INT8 | `.tflite` | 57.07 | 80.42 |
| MERA FP32 | `.mera` | 57.67 | 80.78 |
| MERA INT8 (TFLite Quantized) | `.mera` | 57.05 | 80.30 |

> [!NOTE]
> The model was compiled using `mera-2.5.0+pkg.3577` and `FSP 6.2` was used for building and testing the project.

### Inference Performance (RA8P1)

Measured on-target (Cortex-M85 @ 1 GHz, Ethos-U55 NPU @ 500 MHz). AI-only latency.

| Memory Configuration | CPU (ms) | NPU (ms) |
|----------------------|:--------:|:--------:|
| ‡ OSPI + External SDRAM | 859 | 108 |

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
mobilenetv2/
├── README.md                   ← This file
├── python                      ← Inference scripts, conversion tools, config
└── embedded_c                  ← Compiled C-code for MCU (CPU & NPU)
```

## Prerequisites

1. **Python 3.10** installed (see [Install Python 3.10](../../../README.md#install-python-310) in the top-level README for platform-specific steps).
2. **Inference venv** — navigate to the `python/` directory and create a dedicated virtual environment:

    **Windows PowerShell**

    ```powershell
    cd vision\image_classification\mobilenetv2\python
    python -m venv .venv_mobilenetv2
    .\.venv_mobilenetv2\Scripts\Activate.ps1
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

    **Ubuntu / bash**

    ```bash
    cd vision/image_classification/mobilenetv2/python
    python3.10 -m venv .venv_mobilenetv2
    source .venv_mobilenetv2/bin/activate
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

Use `--mode FP32` to skip INT8 (no calibration images needed) or `--mode INT8` for INT8 only. Output files are written to `python/model/`.

---

## Step 3 — Run Inference (Python)

Activate the **inference venv** and navigate to `python/`:

**Windows PowerShell**

```powershell
cd vision\image_classification\mobilenetv2\python
.\.venv_mobilenetv2\Scripts\Activate.ps1
python inference.py --image sample_images/coco_cat.jpg
```

**Ubuntu / bash**

```bash
cd vision/image_classification/mobilenetv2/python
source .venv_mobilenetv2/bin/activate
python inference.py --image sample_images/coco_cat.jpg
```

Optional flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `model/mobilenet_v2_FP32.tflite` | Path to TFLite model |
| `--labels` | `utils/imagenet_labels.txt` | Path to labels file (one class per line) |
| `--top-k` | `5` | Number of top predictions to display |

**Example output:**

```
========================================
Image              : dog.jpg
Predicted class ID : 207
Predicted label    : golden_retriever
Confidence (score) : 0.8124
========================================
```

> By default `inference.py` loads `model/mobilenet_v2_FP32.tflite` and `utils/imagenet_labels.txt`. Pass `--model model/mobilenet_v2_INT8.tflite` to use the INT8 variant.

---

## Step 4 — Compile for RA8P1 (RUHMI)

This step converts the TFLite model into C-code for the RA8P1 MCU. Activate the **compiler venv** (`.mera_venv`).

### 4.1 — Edit the compile configuration

Open `python/config.yaml` and verify the paths match your system:

```yaml
model_path: vision/image_classification/mobilenetv2/python/model/mobilenet_v2_INT8.tflite
output_dir: vision/image_classification/mobilenetv2/embedded_c/src_mcu
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
cd C:\Users\<you>\ruhmi-model-zoo
python ruhmi_tools\mcu_compile.py vision\image_classification\mobilenetv2\python\config.yaml
```

**Ubuntu / bash**

```bash
cd ~/ruhmi-model-zoo
python ruhmi_tools/mcu_compile.py vision/image_classification/mobilenetv2/python/config.yaml
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

```c
#include "model_metadata.h"
```

| Constant | Value | Description |
|----------|-------|-------------|
| `MODEL_INPUT_H` | `224` | Input image height (pixels) |
| `MODEL_INPUT_W` | `224` | Input image width (pixels) |
| `MODEL_INPUT_C` | `3` | RGB channels |
| `MODEL_OUTPUT_SIZE` | `1000` | Number of output classes |
| `INPUT_SCALE` | `0.007843f` | INT8 input quantisation scale (≈ 1/127.5) |
| `INPUT_ZP` | `-1` | INT8 input zero-point |
| `OUTPUT_SCALE` | `0.00390625f` | INT8 output dequantisation scale (= 1/256) |
| `OUTPUT_ZP` | `-128` | INT8 output zero-point |

---

### 5.3 — Preprocessing — `preprocess()`

```c
#include "preprocessing.h"

void preprocess(const uint8_t *p_source_image,
                int source_width,
                int source_height,
                int8_t *p_destination_tensor);
```

Performs bilinear resize to `[MODEL_INPUT_H × MODEL_INPUT_W]`, applies `pixel / 127.5 − 1.0` normalisation, then quantizes to INT8 using `INPUT_SCALE` and `INPUT_ZP`.

**Typical call:**

```c
#include "model_metadata.h"
#include "preprocessing.h"

/* Camera frame: 320×240 RGB, stored as interleaved uint8 */
static uint8_t camera_buf[320 * 240 * 3];

/* Quantized int8 buffer fed to the model input tensor */
static int8_t nn_input[MODEL_INPUT_H * MODEL_INPUT_W * MODEL_INPUT_C];

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

/* Working buffer + top-k results */
float dequantized_scores[MODEL_OUTPUT_SIZE];
#define TOP_K 5
int   top_indices[TOP_K];
float top_scores[TOP_K];

postprocess(raw_out, MODEL_OUTPUT_SIZE, TOP_K,
            dequantized_scores, top_indices, top_scores);

for (int i = 0; i < TOP_K; i++)
    printf("#%d  class=%d  score=%.4f\n", i + 1, top_indices[i], top_scores[i]);
```

**API:**

```c
void postprocess(const int8_t *p_quantized_output,
                 int class_count,
                 int top_count,
                 float *p_dequantized_scores,   /* working buffer [class_count] */
                 int *p_top_indices,
                 float *p_top_scores);
```

> [!NOTE]
> `postprocess()` handles dequantization internally via the `p_dequantized_scores` working buffer.
> The model already includes softmax (`OUTPUT_HAS_SOFTMAX = 1`), so scores are valid probabilities.

---

## License / Provenance

MobileNetV2 weights are provided by the TensorFlow / Keras team under the [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0) via `tf.keras.applications`.

The ImageNet dataset used for calibration and evaluation is subject to the [ImageNet Terms of Access](https://image-net.org/download.php).
