# Visual Wake Words (VWW) — Binary Image Classification

Visual Wake Words (VWW) is a binary image classification model that detects whether a **person** is present in an image. It is part of the [MLCommons Tiny](https://github.com/mlcommons/tiny) benchmark suite and is optimised for ultra-low-power MCU deployment. This folder contains everything needed to obtain the model, convert it to TFLite, compile it for the Renesas RA8P1 (Cortex-M85 + Ethos-U55 NPU) using the RUHMI toolchain, and run inference.

---

## Model Details

| Field | Value |
|-------|-------|
| **Model name** | Visual Wake Word |
| **Task** | Binary Image Classification (person / notperson) |
| **Framework** | Keras / TensorFlow 2.x |
| **Dataset** | COCO val2014 (person/notperson subset) |
| **Input shape** | `(1, 96, 96, 3)` — HWC, RGB, 96×96 pixels |
| **Output shape** | `(1, 2)` — 2 class logits [notperson, person] |
| **Source** | [MLCommons Tiny — vww_96.h5](https://github.com/mlcommons/tiny/tree/master/benchmark/training/visual_wake_words) |

> [!NOTE]
> The model output is **raw logits** — the graph does **not** include a Softmax layer.
> You **must** apply softmax after dequantisation before computing a binary decision.

## Model Report Card

Accuracy measured on the VWW COCO val2014 test subset.

| Model Variant | Format | Accuracy (%) |
|---------------|--------|:------------:|
| TFLite FP32 | `.tflite` | 84.20 |
| TFLite INT8 | `.tflite` | 83.50 |
| MERA FP32 | `.mera` |  83.18 |
| MERA INT8(TFLite Quantized) | `.mera` | 83.05 |

> [!NOTE]
> The model was compiled using `mera-2.5.0+pkg.3577` and `FSP 6.2` was used for building and testing the project.

### Inference Performance (RA8P1)

Measured on-target (Cortex-M85 @ 1 GHz, Ethos-U55 NPU @ 500 MHz). AI-only latency.

| Memory Configuration | CPU (ms) | NPU (ms) |
|----------------------|:--------:|:--------:|
| † Internal Flash + Internal SRAM | 6 | 0.364 |
| ¶ Internal SRAM only | 6 | 0.406 |

---

> [!NOTE]
> **Performance figures may differ slightly on your machine.**
> Running `download_model.py` generates a fresh INT8 model calibrated on your local dataset with your installed TensorFlow version. The architecture and weights are identical, but per-channel quantization constants will vary,also the TFLite models are generated from the pre-trained .h5 are downloaded from [MLCommons Tiny](https://github.com/mlcommons/tiny) which might be updated by them time to time resulting in:
> - **Inference speed:** slight difference — due to graph optimizations of the possible updated .h5 model.
> - **Accuracy:** within ~0.5 % Top-1 of the reported figures.
> - **C-source / `.tflite` files:** will not match the pre-built artifacts in `embedded_c/` byte-for-byte — this is expected.
>
> To reproduce results exactly, use the pre-built files in `embedded_c/` instead of regenerating them.

## Folder Structure

```
visualwakeword/
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
    cd vision\image_classification\visualwakeword\python
    python -m venv .venv_vww
    .\.venv_vww\Scripts\Activate.ps1
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

    **Ubuntu / bash**

    ```bash
    cd vision/image_classification/visualwakeword/python
    python3.10 -m venv .venv_vww
    source .venv_vww/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

3. **Compiler venv** (`.mera_venv`) — required only for [Step 4](#step-4--compile-for-ra8p1-ruhmi). See the [top-level README](../../../README.md) for setup instructions.

---

## Step 1 — Obtain the Model

The TFLite models are already provided in `python/model/`. To regenerate them from scratch, proceed to [Step 2](#step-2--convert-to-tflite).

---

## Step 2 — Convert to TFLite

Use `download_model.py` to download `vww_96.h5` from MLCommons Tiny and convert it to TFLite. Make sure the **inference venv** (`.venv_vww`) is active and you are in the `python/` directory.

> [!NOTE]
> COCO val2014 images are released under a [Creative Commons Attribution 4.0 License](https://cocodataset.org/#termsofuse). The script downloads them automatically — no manual registration required.

**FP32 conversion only** (no dataset download):

```bash
python download_model.py --mode fp32
```

**INT8 conversion** (downloads COCO val2014 automatically, ~13 GB):

```bash
python download_model.py --mode int8
```

**Both FP32 + INT8:**

```bash
python download_model.py
```

**Use a pre-existing VWW calibration directory:**

```bash
python download_model.py --mode int8 --calib-dir /path/to/vww_coco_val2014
```
---

## Step 3 — Run Inference (Python)

Activate the **inference venv** and navigate to `python/`:

**Ubuntu / bash**

```bash
cd vision/image_classification/visualwakeword/python
source .venv_vww/bin/activate
python inference.py --image sample_images/COCO_val2014_000000000042.jpg
```

**Windows PowerShell**

```powershell
cd vision\image_classification\visualwakeword\python
.\.venv_vww\Scripts\Activate.ps1
python inference.py --image sample_images/COCO_val2014_000000000042.jpg
```

Optional flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `model/vww_96_FP32.tflite` | Path to TFLite model |
| `--labels` | `utils/labels.txt` | Path to labels file (one class per line) |
| `--top-k` | `2` | Number of top predictions to display |

**Example output:**

```
Model   : model/vww_96_FP32.tflite
Image   : person.jpg
Input   : (96, 96, 3)  normalised to [0, 1]
Classes : ['notperson', 'person']

─────────────────────────────────────────────
Rank   Class   Label           Probability
─────────────────────────────────────────────
  1        1   person             0.912453
  2        0   notperson          0.087547
─────────────────────────────────────────────

Decision : PERSON  (91.2% confidence)
```

> By default `inference.py` loads `model/vww_96_FP32.tflite`. Pass `--model model/vww_96_INT8.tflite` to use the INT8 variant.

> [!IMPORTANT]
> The model output is **raw logits** — `inference.py` applies softmax automatically. On the MCU, you must apply softmax after dequantising the INT8 output tensor.

---

## Step 4 — Compile for RA8P1 (RUHMI)

This step converts the TFLite model into C-code that runs on the RA8P1 MCU. Activate the **compiler venv** (`.mera_venv`) — see the [top-level README](../../../README.md) for setup and installation instructions.

### 4.1 — Edit the compile configuration

Open `python/config.yaml` and set the `model_path` and `output_dir` to **absolute paths** on your system:

```yaml
model_path: /path/to/vision/image_classification/visualwakeword/python/model/vww_96_INT8.tflite
output_dir: /path/to/vision/image_classification/visualwakeword/embedded_c/src_mcu
target: cpu        # 'cpu' → CMSIS-NN  |  'npu' → Ethos-U55 NPU
quantize: false    # model is already INT8
```

> [!IMPORTANT]
> The `model_path` and `output_dir` fields must be **absolute paths**.

> [!TIP]
> For NPU deployment, set `target: npu` and change `output_dir` to point to `embedded_c/src_mcu_npu`.

### 4.2 — Run the compiler

Navigate back to the **repository root** and run the compiler with `.mera_venv` active:

**Ubuntu / bash**

```bash
cd ~/Model-zoo
python ruhmi_tools/mcu_compile.py vision/image_classification/visualwakeword/python/config.yaml
```

**Windows PowerShell**

```powershell
cd C:\Users\<you>\Model-zoo
python ruhmi_tools\mcu_compile.py vision\image_classification\visualwakeword\python\config.yaml
```

The compiled C-code will be written to the `output_dir` specified in `config.yaml`.

---

## Step 5 — Embedded C Integration

The `embedded_c/` folder contains portable, board-independent files you can drop directly into any bare-metal or RTOS project.

| File | Purpose |
|------|---------|
| `model_metadata.h` | All compile-time constants — input/output shape, quantization parameters, tensor-arena sizes |
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
| `MODEL_INPUT_H` | `96` | Input image height (pixels) |
| `MODEL_INPUT_W` | `96` | Input image width (pixels) |
| `MODEL_INPUT_C` | `3` | Number of colour channels (RGB) |
| `MODEL_INPUT_SIZE` | `27648` | Flat input buffer size (96 × 96 × 3 bytes) |
| `MODEL_NUM_CLASSES` | `2` | Number of output classes |
| `MODEL_OUTPUT_SIZE` | `2` | Flat output buffer size |
| `INPUT_SCALE` | `~0.003922f` | INT8 input quantization scale (≈ 1/255) |
| `INPUT_ZP` | `-128` | INT8 input zero-point |
| `OUTPUT_HAS_SOFTMAX` | `0` | Softmax is **not** in graph — apply it after dequantization |

---

### 5.3 — Preprocessing — `preprocess()`

```c
#include "preprocessing.h"

void preprocess(const uint8_t *p_source_image,
                int source_width,
                int source_height,
                int8_t *p_destination_tensor);
```

Performs bilinear resize to `[MODEL_INPUT_H × MODEL_INPUT_W]` (96×96), applies `pixel / 255.0` normalisation, then quantizes to INT8 using `INPUT_SCALE` and `INPUT_ZP`. Because `INPUT_SCALE ≈ 1/255` and `INPUT_ZP = -128`, this simplifies to `q = pixel − 128`.

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

The postprocessing pipeline dequantizes the INT8 output, applies softmax (since `OUTPUT_HAS_SOFTMAX = 0`), and returns the top-1 class index:

```c
#include "model_metadata.h"
#include "postprocessing.h"

/* Raw INT8 output from the model */
int8_t raw_out[MODEL_OUTPUT_SIZE];   /* 2 elements: [notperson, person] */

/* Score buffer + predicted class */
float scores[MODEL_NUM_CLASSES];
int   predicted_class;

postprocess(raw_out, MODEL_NUM_CLASSES, scores, &predicted_class);

static const char *vww_labels[] = { "notperson", "person" };
printf("Detected: %s  Score: %.4f\n",
       vww_labels[predicted_class], scores[predicted_class]);
```

**API:**

```c
void postprocess(const int8_t *p_out_q,
                 int class_count,
                 float *p_scores,            /* output float scores [class_count] */
                 int *p_predicted_class);     /* top-1 class index */
```

Additional public helpers are also available:

| Function | Description |
|----------|-------------|
| `dequantize_output_to_scores()` | Bulk INT8 → float dequantization |
| `softmax_inplace()` | In-place softmax (used when `OUTPUT_HAS_SOFTMAX = 0`) |
| `argmax()` | Returns index of highest score |

---

## License / Provenance

| Component | Source | Licence |
|-----------|--------|---------|
| `vww_96.h5` pre-trained weights | [MLCommons Tiny](https://github.com/mlcommons/tiny/tree/master/benchmark/training/visual_wake_words) | Apache 2.0 |
| COCO val2014 images (calibration) | [cocodataset.org](https://cocodataset.org) | CC-BY 4.0 |
| COCO annotations (calibration) | [cocodataset.org](https://cocodataset.org) | CC-BY 4.0 |
