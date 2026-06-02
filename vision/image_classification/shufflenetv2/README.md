# ShuffleNetV2 x0.5 — Image Classification (ImageNet)

ShuffleNetV2 is an efficient convolutional network designed for mobile devices using channel-shuffle operations ([Ma et al., 2018](https://arxiv.org/abs/1807.11164)). The x0.5 variant has ~1.4 M parameters and provides a strong accuracy-efficiency tradeoff. This folder contains everything needed to obtain the model, convert it to TFLite, compile it for the Renesas RA8P1 (Cortex-M85 + Ethos-U55 NPU) using the RUHMI toolchain, and run inference.

---

## Model Details

| Field | Value |
|-------|-------|
| **Model name** | ShuffleNetV2 x0.5 |
| **Task** | Image Classification |
| **Framework** | PyTorch (torchvision) → ONNX → TFLite |
| **Dataset** | ImageNet-1k (1000 classes) |
| **Input shape** | `(1, 224, 224, 3)` — HWC, RGB |
| **Output shape** | `(1, 1000)` — 1000 class logits |
| **Source** | [`torchvision.models.shufflenet_v2_x0_5(weights="IMAGENET1K_V1")`](https://pytorch.org/hub/pytorch_vision_shufflenet_v2/)|

## Model Report Card

Accuracy measured on the ImageNet-1k validation set (50 000 images).

| Model Variant | Format | Top-1 (%) | Top-5 (%) |
|---------------|--------|:---------:|:---------:|
| TFLite FP32 | `.tflite` | 60.52 | 81.75 |
| TFLite INT8 | `.tflite` | 59.82 | 81.19 |
| MERA FP32 | `.mera` | 60.52 | 81.75 |
| MERA INT8 (TFLite Quantized) | `.mera` | 59.79 | 81.22 |

> [!NOTE]
> The model was compiled using `mera-2.6.0+pkg.4513` and `FSP 6.2` was used for building and testing the project.

### Inference Performance (RA8P1)

Measured on-target (Cortex-M85 @ 1 GHz, Ethos-U55 NPU @ 500 MHz). AI-only latency.

| Memory Configuration | CPU (ms) | NPU (ms) |
|----------------------|:--------:|:--------:|
|  ‡ OSPI + External SDRAM/PSRAM | 172 | _  |
| * OSPI + Internal SRAM | - | 147 |

---

## Folder Structure

```
shufflenetv2/
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
    cd vision\image_classification\shufflenetv2\python
    python -m venv .venv_shufflenet
    .\.venv_shufflenet\Scripts\Activate.ps1
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

    **Ubuntu / bash**

    ```bash
    cd vision/image_classification/shufflenetv2/python
    python3.10 -m venv .venv_shufflenet
    source .venv_shufflenet/bin/activate
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

Use `--mode fp32` to skip INT8 (no calibration images needed) or `--mode int8` for INT8 only. Output files are written to `python/model/`.

> [!NOTE]
> INT8 calibration requires ImageNet validation images. Due to ImageNet licensing, the dataset cannot be auto-downloaded. See [ImageNet download page](https://image-net.org/download.php) for access.

---

## Step 3 — Run Inference (Python)

Activate the **inference venv** and navigate to `python/`:

**Windows PowerShell**

```powershell
cd vision\image_classification\shufflenetv2\python
.\.venv_shufflenet\Scripts\Activate.ps1
python inference.py sample_images\coco_cat.jpg
```

**Ubuntu / bash**

```bash
cd vision/image_classification/shufflenetv2/python
source .venv_shufflenet/bin/activate
python inference.py sample_images/coco_cat.jpg
```

Optional flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `model/shufflenet_v2_x0_5_FP32.tflite` | Path to TFLite model |
| `--labels` | `utils/imagenet_labels.txt` | Path to labels file |
| `--top-k` | `5` | Number of top predictions to display |

**Example output:**

```
────────────────────────────────────────────────────────────
Rank   Class   Label                               Probability
────────────────────────────────────────────────────────────
  1      65   sea_snake                              0.412345
  2     390   eel                                    0.187654
  3      58   water_snake                            0.098765
  4      62   rock_python                            0.054321
  5      56   king_snake                             0.032109
────────────────────────────────────────────────────────────
```

> By default `inference.py` loads `model/shufflenet_v2_x0_5_FP32.tflite`. Pass `--model model/shufflenet_v2_x0_5_INT8.tflite` to use the INT8 variant.

---

## Step 4 — Compile for RA8P1 (RUHMI)

This step converts the TFLite model into C-code for the RA8P1 MCU. Activate the **compiler venv** (`.mera_venv`).

### 4.1 — Edit the compile configuration

Open `python/config.yaml` and verify the paths match your system:

```yaml
model_path: vision/image_classification/shufflenetv2/python/model/shufflenet_v2_x0_5_INT8.tflite
output_dir: vision/image_classification/shufflenetv2/embedded_c/src_mcu
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
python ruhmi_tools\mcu_compile.py vision\image_classification\shufflenetv2\python\config.yaml
```

**Ubuntu / bash**

```bash
cd ~/Model-zoo
python ruhmi_tools/mcu_compile.py vision/image_classification/shufflenetv2/python/config.yaml
```

The compiled C-code will be written to the `output_dir` specified in `config.yaml`.

---

## Step 5 — Embedded C Integration

The `embedded_c/` folder contains portable, board-independent files you can drop directly into any bare-metal or RTOS project.

| File | Purpose |
|------|---------|
| `model_metadata.h` | All compile-time constants — input/output shape, quantization parameters, class count |
| `preprocessing.h` / `preprocessing.c` | Resize, center-crop, normalize, and quantize image to INT8 |
| `postprocessing.h` / `postprocessing.c` | Dequantize output, apply softmax, return Top-K predictions |

> [!NOTE]
> These files have **no external dependencies** beyond `<stdint.h>`, `<stddef.h>`, and `<math.h>`. They compile cleanly with any C99-compatible toolchain (GCC, Clang, IAR, AC6).

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
| `MODEL_NUM_CLASSES` | `1000` | Number of output classes |
| `INPUT_SCALE` | `0.01865845f` | INT8 input quantization scale |
| `INPUT_ZP` | `-14` | INT8 input zero-point |
| `INPUT_NORM_MEAN_R/G/B` | `0.485/0.456/0.406` | ImageNet channel means |
| `INPUT_NORM_STD_R/G/B` | `0.229/0.224/0.225` | ImageNet channel stds |
| `OUTPUT_SCALE` | `0.18189311f` | INT8 output dequantization scale |
| `OUTPUT_ZP` | `-57` | INT8 output zero-point |
| `OUTPUT_HAS_SOFTMAX` | `1` | Softmax baked in — **do not apply again** |

---

### 5.3 — Preprocessing — `preprocess()`

```c
#include "preprocessing.h"

void preprocess(const uint8_t *p_source_image,
                int source_width,
                int source_height,
                int8_t *p_destination_tensor);
```

Performs: resize shortest edge to 256 → center-crop to 224×224 → normalize (ImageNet mean/std) → quantize to INT8.

**Typical call:**

```c
#include "model_metadata.h"
#include "preprocessing.h"

/* Camera frame: 320×240 RGB */
static uint8_t camera_buf[320 * 240 * 3];

/* Quantized int8 buffer fed to model input */
static int8_t nn_input[MODEL_INPUT_SIZE];

preprocess(camera_buf, 320, 240, nn_input);
```

---

### 5.4 — Postprocessing — `postprocess()`

```c
#include "model_metadata.h"
#include "postprocessing.h"

/* Raw INT8 output from model */
int8_t raw_out[MODEL_NUM_CLASSES];

/* Top-K results */
int   top_indices[TOP_K];
float top_scores[TOP_K];

postprocess(raw_out, MODEL_NUM_CLASSES, TOP_K, top_indices, top_scores);

for (int i = 0; i < TOP_K; i++)
    printf("#%d  class=%d  score=%.4f\n", i + 1, top_indices[i], top_scores[i]);
```

> [!NOTE]
> `postprocess()` handles dequantization internally. Since `OUTPUT_HAS_SOFTMAX = 1`, scores are valid probabilities.

---

## License / Provenance

| Component | Source | Licence |
|-----------|--------|---------|
| ShuffleNetV2 weights | [torchvision](https://pytorch.org/vision/stable/models/shufflenetv2.html) | BSD-3-Clause |
| ImageNet-1k (calibration) | [image-net.org](https://image-net.org) | Restricted (research use) |
