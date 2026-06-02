# SqueezeNet 1.1 — Image Classification (ImageNet)

SqueezeNet is a compact DNN network ([Iandola et al., 2016](https://arxiv.org/abs/1602.07360)) that achieves AlexNet-level accuracy with ~50x fewer parameters. While maintaining competitive accuracy Compared to SqueezeNet 1.0, version 1.1 achieves 2.4x less computation and slightly fewer parameters. This folder contains everything needed to obtain the SqueezeNet 1.1 model, convert it to TFLite, compile it for the Renesas RA8P1 (Cortex-M85 + Ethos-U55 NPU) using the RUHMI toolchain, and run inference.

---

## Model Details

| Field | Value |
|-------|-------|
| **Model name** | SqueezeNet 1.1 |
| **Task** | Image Classification |
| **Framework** | ONNX → TFLite (via onnx2tf) |
| **Dataset** | ImageNet-1k (1000 classes) |
| **Input shape** | `(1, 224, 224, 3)` — HWC, RGB |
| **Output shape** | `(1, 1000)` — raw logits (no softmax in graph) |
| **Original datatype** | float32 |
| **Quantised datatype** | int8 (full-integer, TFLite / MERA) |
| **Source** | [`torchvision.models.squeezenet1_1`](https://pytorch.org/hub/pytorch_vision_squeezenet/) |

## Model Report Card

Accuracy measured on the ImageNet-1k validation set (50 000 images).

| Model Variant | Format | Top-1 (%) | Top-5 (%) |
|---------------|--------|:---------:|:---------:|
| TFLite FP32 | `.tflite` | 52.08 | 76.20 |
| TFLite INT8 | `.tflite` | 51.90 | 75.93 |
| MERA FP32   | `.mera` | 52.08 | 76.20 |
| MERA INT8(TFLite Quantized) | `.mera` | 51.91 | 75.90 |

> [!NOTE]
> The model was compiled using `mera-2.5.0+pkg.3577` and `FSP 6.2` was used for building and testing the project.

### Inference Performance (RA8P1)

Measured on-target (Cortex-M85 @ 1 GHz, Ethos-U55 NPU @ 500 MHz). AI-only latency.

| Memory Configuration | CPU (ms) | NPU (ms) |
|----------------------|:--------:|:--------:|
| ‡ OSPI + External SDRAM | 565 | 24 |

---

## Folder Structure

```
squeezenet_1_1/
├── README.md                   ← This file
├── python                      ← Inference scripts, conversion tools, config
└── embedded_c                  ← Compiled C-code for MCU (CPU & NPU)
```

> [!NOTE]
> SqueezeNet's output is **raw logits** — the graph does **not** include a Softmax layer.
> You **must** apply softmax after dequantisation before computing Top-K predictions.

---

## Prerequisites

1. **Python 3.10** installed (see [Install Python 3.10](../../../README.md#install-python-310) in the top-level README for platform-specific steps).
2. **Inference venv** — navigate to the `python/` directory and create a dedicated virtual environment:

    **Windows PowerShell**

    ```powershell
    cd vision\image_classification\squeezenet\python
    python -m venv .venv_squeezenet
    .\.venv_squeezenet\Scripts\Activate.ps1
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

    **Ubuntu / bash**

    ```bash
    cd vision/image_classification/squeezenet_1_1/python
    python3.10 -m venv .venv_squeezenet
    source .venv_squeezenet/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

3. **Compiler venv** (`.mera_venv`) — required only for [Step 4](#step-4--compile-for-ra8p1-ruhmi). See the [top-level README](../../../README.md) for setup instructions.

---

## Step 1 — Obtain the Model

The TFLite models are already provided in `python/model/`. To regenerate them from scratch, proceed to [Step 2](#step-2--convert-to-tflite).

---

## Step 2 — Convert to TFLite

Use `download_model.py` to download the ONNX model and convert it to TFLite. Activate the **inference venv** and navigate to `python/`.

**Both FP32 + INT8 (default)** — requires ImageNet validation images for INT8 calibration

```bash
python download_model.py --calib-dir /path/to/ILSVRC2012_img_val
```

Use `--mode FP32` to skip INT8 (no calibration images needed) or `--mode INT8` for INT8 only. Output files are written to `python/model/`.

---
Output files are written to `python/model/`.

---

## Step 3 — Run Inference (Python)

Activate the **inference venv** and navigate to `python/`:

**Windows PowerShell**

```powershell
cd vision\image_classification\squeezenet\python
.\.venv_squeezenet\Scripts\Activate.ps1
python inference.py --image sample_images/coco_cat.jpg
```

**Ubuntu / bash**

```bash
cd vision/image_classification/squeezenet_1_1/python
source .venv_squeezenet/bin/activate
python inference.py --image sample_images/coco_cat.jpg
```

Optional flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `model/squeezenet1_1_FP32.tflite` | Path to TFLite model |
| `--labels` | `utils/imagenet_labels.txt` | Path to labels file (one class per line) |
| `--top-k` | `5` | Number of top predictions to display |

**Example output:**

```
========================================
Image              : bird.jpg
Predicted class ID : 94
Predicted label    : hummingbird
Confidence (score) : 0.5217
========================================
```

> By default `inference.py` loads `model/squeezenet1_1_FP32.tflite` and `utils/imagenet_labels.txt`. Pass `--model model/squeezenet1_1_INT8.tflite` to use the INT8 variant.

---

## Step 4 — Compile for RA8P1 (RUHMI)

This step converts the TFLite model into C-code for the RA8P1 MCU. Activate the **compiler venv** (`.mera_venv`).

### 4.1 — Edit the compile configuration

Open `python/config.yaml` and verify the paths match your system:

```yaml
model_path: vision/image_classification/squeezenet_1_1/python/model/squeezenet1_1_INT8.tflite
output_dir: vision/image_classification/squeezenet_1_1/embedded_c/src_mcu
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
python ruhmi_tools\mcu_compile.py vision\image_classification\squeezenet\python\config.yaml
```

**Ubuntu / bash**

```bash
cd ~/Model-zoo
python ruhmi_tools/mcu_compile.py vision/image_classification/squeezenet_1_1/python/config.yaml
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
| `INPUT_SCALE` | `0.01865845f` | INT8 input quantisation scale |
| `INPUT_ZP` | `-14` | INT8 input zero-point |
| `INPUT_NORM_DIV` | `255.0f` | Pixel normalisation divisor |
| `INPUT_MEAN[3]` | `{0.485, 0.456, 0.406}` | ImageNet per-channel mean (R, G, B) |
| `INPUT_STD[3]` | `{0.229, 0.224, 0.225}` | ImageNet per-channel std (R, G, B) |
| `OUTPUT_SCALE` | `0.18399939f` | INT8 output dequantisation scale |
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

Performs bilinear resize to `[MODEL_INPUT_H × MODEL_INPUT_W]`, applies ImageNet per-channel mean/std normalisation (`(pixel/255 − mean) / std`), then quantizes to INT8 using `INPUT_SCALE` and `INPUT_ZP`.

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

The postprocessing pipeline dequantizes the INT8 output, applies softmax (since `OUTPUT_HAS_SOFTMAX = 0`), then selects the top-k highest-scoring classes:

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
void postprocess(const int8_t *p_out_q,
                 int class_count,
                 int top_count,
                 float *p_scores,            /* working buffer [class_count] */
                 int *p_top_indices,
                 float *p_top_scores);
```

> [!IMPORTANT]
> SqueezeNet outputs raw logits (`OUTPUT_HAS_SOFTMAX = 0`). `postprocess()` handles dequantization **and** softmax internally before top-k selection.

Additional public helpers are also available:

| Function | Description |
|----------|-------------|
| `dequantize_output_to_scores()` | Bulk INT8 → float dequantization |
| `softmax_inplace()` | In-place softmax (used when `OUTPUT_HAS_SOFTMAX = 0`) |
---

## License / Provenance

SqueezeNet 1.1 weights are provided by PyTorch / torchvision under the [BSD 3-Clause License](https://github.com/pytorch/vision/blob/main/LICENSE).

The ImageNet dataset used for calibration and evaluation is subject to the [ImageNet Terms of Access](https://image-net.org/download.php).
