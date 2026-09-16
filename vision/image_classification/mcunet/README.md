# MCUNet — Image Classification (ImageNet)

MCUNet is an ultra-compact convolutional network ([Lin et al., NeurIPS 2020](https://arxiv.org/abs/2007.10319)) designed specifically for microcontroller-class hardware. This folder contains everything needed to obtain the model, convert it to TFLite, compile it for the Renesas RA8P1 (Cortex-M85 + Ethos-U55 NPU) using the RUHMI toolchain, and run inference.

---

## Model Details

| Field | Value |
|-------|-------|
| **Model name** | MCUNet (mcunet-10fps_imagenet) |
| **Task** | Image Classification |
| **Framework** | PyTorch → ONNX → TFLite |
| **Dataset** | ImageNet-1k (1000 classes) |
| **Input shape** | `(1, 48, 48, 3)` — HWC, RGB |
| **Output shape** | `(1, 1000)` — 1000 class logits |
| **Source** | [MIT HAN Lab MCUNet](https://github.com/mit-han-lab/mcunet) |

## Model Report Card

Accuracy measured on the full ImageNet-1k validation set (50 000 images).

| Model Variant | Format | Top-1 (%) | Top-5 (%) |
|---------------|--------|:---------:|:---------:|
| TFLite FP32 | `.tflite` | 41.68 | 66.72 |
| TFLite INT8 | `.tflite` | 40.46 | 65.27 |
| MERA FP32 | C-code | 41.68 | 66.72 |
| MERA INT8 (TFLite INT8 deployed) | C-code | 40.44 | 65.29 |

> [!NOTE]
> The model was compiled using `mera-2.6.0+pkg.4815` and `FSP 6.5.1` was used for building and testing the project.

### Inference Performance (RA8P1)

Measured on-target (Cortex-M85 @ 1 GHz, Ethos-U55 NPU @ 500 MHz). AI-only latency.

| Memory Configuration | CPU (ms) | NPU (ms) |
|----------------------|:--------:|:--------:|
| † Internal Flash + Internal SRAM | 9 | 0.655  |
| ¶ Internal SRAM only | 9 | 0.655 |
---

## Folder Structure

```
mcunet/
├── README.md                   ← This file
├── python/                     ← Inference scripts, conversion tools, config
│   ├── download_model.py       ← Download pretrained weights and convert to TFLite
│   ├── inference.py            ← Single-image TFLite inference (FP32 or INT8)
│   ├── config.yaml             ← RUHMI compiler configuration
│   ├── requirements.txt        ← Python dependencies
│   ├── model/                  ← Pre-built TFLite models
│   └── utils/
│       └── imagenet_labels.txt ← 1000 class names (one per line)
└── embedded_c/                 ← Compiled C-code for MCU (CPU & NPU)
    ├── model_metadata.h        ← Compile-time constants (shape, quant params)
    ├── preprocessing.h / .c    ← Input preprocessing
    ├── postprocessing.h / .c   ← Output postprocessing
    ├── src_mcu/                ← CPU-only CMSIS-NN build
    └── src_mcu_npu/            ← Ethos-U55 NPU-accelerated build
```

---

## Prerequisites

> [!IMPORTANT]
> **No ImageNet images are bundled in this repo** — ImageNet's terms of access prohibit
> redistributing the dataset. To run inference, calibrate INT8, or validate accuracy, you
> must supply your own images from the ILSVRC 2012 validation set, obtained via
> [https://image-net.org/download.php](https://image-net.org/download.php) (registration
> required). Pass the directory with `--calib-dir`, or place individual images anywhere
> and pass the path to `inference.py`.

1. **Python 3.10** installed.
2. **Inference venv** — navigate to `python/` and create a dedicated virtual environment:

    **Windows PowerShell**
    > **Note:** If venv activation is blocked by PowerShell execution policy ("running scripts is disabled"), run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in the same PowerShell window, then run the activation command again.

    ```powershell
    cd vision\image_classification\mcunet\python
    py -3.10 -m venv .venv_mcunet
    .\.venv_mcunet\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    ```

    **Ubuntu / bash**

    ```bash
    cd vision/image_classification/mcunet/python
    python3.10 -m venv .venv_mcunet
    source .venv_mcunet/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

3. **Compiler venv** (`.mera_venv`) — required only for [Step 4](#step-4--compile-for-ra8p1-ruhmi). See the [top-level README](../../../README.md) for setup instructions.

---

## Step 1 — Obtain the Model

The TFLite models are already provided in `python/model/`. To regenerate them from scratch, proceed to [Step 2](#step-2--convert-to-tflite).

---

## Step 2 — Build the Models

Use `download_model.py` to clone the MIT HAN Lab MCUNet package, export pretrained weights, and convert to TFLite. Activate the **inference venv** and navigate to `python/`.

**FP32 only** — no calibration images required:

```bash
python download_model.py --mode fp32
```

**INT8 only** — requires ImageNet validation images:

```bash
python download_model.py --mode int8 --calib-dir /path/to/ILSVRC2012_img_val
```

**Both FP32 + INT8** (default):

```bash
python download_model.py --calib-dir /path/to/ILSVRC2012_img_val
```

Use `--calib-num N` to limit calibration to N images (default: 1000). Output files are written to `python/model/`.

> [!NOTE]
> `download_model.py` monkey-patches the MCUNet `forward()` method before ONNX export to fuse the chained `x.mean(3).mean(2)` global-average-pool into a single `mean(dim=(2, 3))`. This is required for MERA MCU compiler compatibility. Numerics are unchanged — only the ONNX graph layout differs.

---

## Step 3 — Run Inference (Python)

Activate the **inference venv** and navigate to `python/`:

**Windows PowerShell**

```powershell
cd vision\image_classification\mcunet\python
.\.venv_mcunet\Scripts\Activate.ps1
python inference.py \path\to\your_own_image.JPEG
```

**Ubuntu / bash**

```bash
cd vision/image_classification/mcunet/python
source .venv_mcunet/bin/activate
python inference.py /path/to/your_own_image.JPEG
```

> No sample images are bundled — supply your own ImageNet-1k validation image (see the
> ImageNet note in Prerequisites above).

Optional flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `model/mcunet_in0_FP32.tflite` | Path to TFLite model |
| `--labels` | `utils/imagenet_labels.txt` | Path to labels file (one class per line) |
| `--top-k` | `5` | Number of top predictions to display |

**Example output:**

```text
==================================================
Image : your_own_image.JPEG
Model : mcunet_in0_FP32.tflite
--------------------------------------------------
    # 1  [  84]  peacock                         1.0000
    # 2  [  14]  indigo_bunting                  0.0000
    # 3  [  88]  macaw                           0.0000
    # 4  [  17]  jay                             0.0000
    # 5  [ 136]  European_gallinule              0.0000
==================================================
```

> By default `inference.py` loads `model/mcunet_in0_FP32.tflite` and `utils/imagenet_labels.txt`. Pass `--model model/mcunet_in0_INT8.tflite` to use the INT8 variant.

---

## Step 4 — Compile for RA8P1 (RUHMI)

This step converts the TFLite INT8 model into C-code for the RA8P1 MCU. Activate the **compiler venv** (`.mera_venv`).

### 4.1 — Edit the compile configuration

Open `python/config.yaml` and verify the paths match your system:

```yaml
model_path: vision/image_classification/mcunet/python/model/mcunet_in0_INT8.tflite
output_dir: vision/image_classification/mcunet/embedded_c/src_mcu
target: cpu        # 'cpu' → CMSIS-NN  |  'npu' → Ethos-U55 NPU
quantize: false    # model is already INT8
external: true
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
python ruhmi_tools\mcu_compile.py vision\image_classification\mcunet\python\config.yaml
```

**Ubuntu / bash**

```bash
cd ~/Model-zoo
python ruhmi_tools/mcu_compile.py vision/image_classification/mcunet/python/config.yaml
```

The compiled C-code will be written to the `output_dir` specified in `config.yaml`.

---

## Step 5 — Embedded C Integration

The `embedded_c/` folder contains portable, board-independent files for bare-metal or RTOS firmware.

| File | Purpose |
|------|---------|
| `model_metadata.h` | All compile-time constants — input/output shape, quantization parameters, class count, memory footprint |
| `preprocessing.h` / `preprocessing.c` | Input preprocessing |
| `postprocessing.h` / `postprocessing.c` | Output postprocessing |

> [!NOTE]
> `preprocessing.c` depends on `<math.h>` (for `floorf` / `fabsf`). All other files compile cleanly with any C99-compatible toolchain (GCC, Clang, IAR, AC6) with no additional dependencies beyond `<stdint.h>` and `<stddef.h>`.

### 5.1 — Add files to your project

Copy the following files into your firmware project:

```
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
| `MODEL_INPUT_H` | `48` | Input image height (pixels) |
| `MODEL_INPUT_W` | `48` | Input image width (pixels) |
| `MODEL_INPUT_C` | `3` | RGB channels |
| `MODEL_OUTPUT_SIZE` | `1000` | Number of output classes |
| `INPUT_SCALE` | `0.007843f` | INT8 input quantisation scale (≈ 1/127.5) |
| `INPUT_ZP` | `-1` | INT8 input zero-point |
| `INPUT_NORM_SCALE` | `127.5f` | Pixel normalisation divisor |
| `INPUT_NORM_BIAS` | `-1.0f` | Bias after division (maps to [-1, 1]) |
| `OUTPUT_SCALE` | `0.17176612f` | INT8 output dequantisation scale |
| `OUTPUT_ZP` | `-31` | INT8 output zero-point |
| `OUTPUT_HAS_SOFTMAX` | `1` | Metadata flag in `model_metadata.h`; runtime `postprocess()` still applies softmax before top-k |

---

### 5.3 — Preprocessing — `preprocess()`

**API:**

```c
#include "preprocessing.h"

void preprocess(const uint8_t *p_source_image_hwc,
                int _source_width,
                int _source_height,
                int8_t *p_input_tensor);
```

Performs a short-edge bilinear resize (separable triangle filter, half-pixel centres), centre-crops to `MODEL_INPUT_H × MODEL_INPUT_W` (48 × 48), applies `pixel / 127.5 − 1.0` normalisation, then quantizes to INT8 using `INPUT_SCALE` and `INPUT_ZP`.

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

**API:**

```c
void postprocess(const int8_t *p_output_q,
                 int _class_count,
                 int _top_k_count,
                 int *p_top_indices,
                 float *p_top_scores);
```

The postprocessing pipeline dequantizes the INT8 output, applies softmax for calibrated probabilities, then selects the top-k highest-scoring classes.

**Typical call:**

```c
#include "model_metadata.h"
#include "postprocessing.h"

/* Raw INT8 output from the model */
int8_t raw_out[MODEL_OUTPUT_SIZE];

/* Top-k results */
#define TOP_K 5
int   top_indices[TOP_K];
float top_scores[TOP_K];

postprocess(raw_out, MODEL_OUTPUT_SIZE, TOP_K, top_indices, top_scores);

for (int i = 0; i < TOP_K; i++)
    printf("#%d  class=%d  score=%.4f\n", i + 1, top_indices[i], top_scores[i]);
```

> [!NOTE]
> `postprocess()` handles dequantization and softmax internally using `OUTPUT_SCALE` and `OUTPUT_ZP` — you do **not** need to dequantize manually or apply softmax yourself.

---

## References

- Paper: [MCUNet: Tiny Deep Learning on IoT Devices (NeurIPS 2020)](https://arxiv.org/abs/2007.10319)
- Code: [https://github.com/mit-han-lab/mcunet](https://github.com/mit-han-lab/mcunet)