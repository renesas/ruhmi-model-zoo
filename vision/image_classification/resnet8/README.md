# ResNet8 — Image Classification (CIFAR-10)

ResNet8 is a compact residual network (ResNet v1 variant) used in the [MLCommons Tiny (TinyMLPerf)](https://github.com/mlcommons/tiny) image-classification benchmark. The architecture is based on the ResNet family ([He et al., 2015](https://arxiv.org/abs/1512.03385)) and adapted by [EEMBC](https://www.eembc.org/) for the TinyMLPerf benchmark suite (`resnet_v1_eembc`). It uses three residual stacks with a starting filter width of 16, totalling ~79 K parameters. It fits comfortably on resource-constrained MCUs while still achieving strong accuracy on CIFAR-10. This folder contains everything needed to obtain the model, convert it to TFLite, compile it for the Renesas RA8P1 (Cortex-M85 + Ethos-U55 NPU) using the RUHMI toolchain, and run inference.

---

## Model Details

| Field | Value |
|-------|-------|
| **Model name** | ResNet8 |
| **Task** | Image Classification |
| **Framework** | Keras / TensorFlow 2.x |
| **Dataset** | CIFAR-10 (10 classes) |
| **Input shape** | `(1, 32, 32, 3)` — HWC, RGB, 32×32 pixels |
| **Output shape** | `(1, 10)` — 10 class logits |
| **Source** | [MLCommons Tiny — pretrainedResnet.h5](https://github.com/mlcommons/tiny/blob/master/benchmark/training/image_classification/trained_models/pretrainedResnet.h5) |

## Model Report Card

Accuracy measured on the CIFAR-10 test set (10 000 images).

| Model Variant | Format | Accuracy (%) |
|---------------|--------|:------------:|
| TFLite FP32 | `.tflite` | 87.19 |
| TFLite INT8 | `.tflite` | 86.82 |
| MERA FP32 | `.mera` | 87.19 |
| MERA Quantised INT8 | `.mera` | 86.02 |

> [!NOTE]
> The model was compiled using `mera-2.5.0+pkg.3577` and `FSP 6.2` was used for building and testing the project.

### Inference Performance (RA8P1)

Measured on-target (Cortex-M85 @ 1 GHz, Ethos-U55 NPU @ 500 MHz). AI-only latency.

| Memory Configuration | CPU (ms) | NPU (ms) |
|----------------------|:--------:|:--------:|
| † Internal Flash + Internal SRAM | 12 | 0.640 |
| ¶ Internal SRAM only | 12 | 0.640 |

---

## Folder Structure

```
resnet8/
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
    cd vision\image_classification\resnet\python
    python -m venv .venv_resnet
    .\.venv_resnet\Scripts\Activate.ps1
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

    **Ubuntu / bash**

    ```bash
    cd vision/image_classification/resnet8/python
    python3.10 -m venv .venv_resnet
    source .venv_resnet/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

3. **Compiler venv** (`.mera_venv`) — required only for [Step 4](#step-4--compile-for-ra8p1-ruhmi). See the [top-level README](../../../README.md) for setup instructions.

---

## Step 1 — Obtain the Model

The pretrained Keras `.h5` weights are hosted in the MLCommons Tiny repository:

```
https://github.com/mlcommons/tiny/blob/master/benchmark/training/image_classification/trained_models/pretrainedResnet.h5
```

Download the file and place it in `model/`. If the `.h5` model is not present in the `model/` directory, `download_model.py` will download it automatically. Ensure you are in the `python/` directory as described in [Prerequisites](#prerequisites):

**Windows PowerShell**

```powershell
Invoke-WebRequest -Uri "https://github.com/mlcommons/tiny/raw/master/benchmark/training/image_classification/trained_models/pretrainedResnet.h5" -OutFile "model/pretrainedResnet.h5"
```

**Ubuntu / bash**

```bash
wget -O model/pretrainedResnet.h5 \
  "https://github.com/mlcommons/tiny/raw/master/benchmark/training/image_classification/trained_models/pretrainedResnet.h5"
```

---

## Step 2 — Convert to TFLite

Use `download_model.py` to convert the Keras `.h5` model to TFLite format. Make sure the **inference venv** (`.venv_resnet`) is active and you are in the `python/` directory.

**FP32 conversion**

```bash
python download_model.py --mode fp32 --h5 model/pretrainedResnet.h5
```

**INT8 conversion** (requires CIFAR-10 calibration data)

```bash
python download_model.py --mode int8 --h5 model/pretrainedResnet.h5 --calib-dir /path/to/cifar-10-batches-py
```

The `--calib-dir` flag should point to the unpacked [CIFAR-10 python batch folder](https://www.cs.toronto.edu/~kriz/cifar.html). If omitted, the script looks for `cifar-10-batches-py` next to itself.


## Step 3 — Run Inference (Python)

Run single-image inference using the TFLite model. in the `python/` directory:

**Windows PowerShell**

```powershell
python inference.py sample_images/0000_cat_domestic_cat_s_000907.png
```

**Ubuntu / bash**

```bash
python inference.py sample_images/0000_cat_domestic_cat_s_000907.png
```

`<image_path>` can be:
- A `.png` / `.jpg` image file (will be resized to 32×32 and normalised automatically)
- A `.npy` file containing a preprocessed `(1, 32, 32, 3)` float32 array

**Example output:**

```
========================================
Image              : cat.png
Predicted class ID : 3
Predicted label    : cat
Confidence (score) : 0.9142
========================================
```

> [!NOTE]
> By default `inference.py` loads `model/CIFAR10ResNet8_fp32.tflite`. To use a different model variant, edit the `MODEL_PATH` variable at the top of the script.

---

## Step 4 — Compile for RA8P1 (RUHMI)

This step converts the TFLite model into C-code that runs on the RA8P1 MCU. Activate the **compiler venv** (`.mera_venv`) — see the [top-level README](../../../README.md) for setup and installation instructions.

### 4.1 — Edit the compile configuration

Open `python/config.yaml` and set the `model_path` and `output_dir` to **absolute paths** on your system:

```yaml
model_path: vision/image_classification/resnet8/python/model/Resnet_fp32.tflite   # FP32 model — RUHMI quantizes to INT8
output_dir: vision/image_classification/resnet8/embedded_c/src_mcu                # output directory for C-code
target: cpu                                             # 'cpu' (CMSIS-NN) or 'npu' (Ethos-U55)
quantize: true                                          # RUHMI will quantize FP32 → INT8 (better accuracy)
```

> [!IMPORTANT]
> The `model_path` and `output_dir` fields must be **absolute paths**. Replace `<you>` with your actual username or adjust to match your system layout.

> [!TIP]
> For NPU deployment, set `target: npu` and change `output_dir` to point to `embedded_c/src_mcu_npu`. If the model exceeds the on-chip SRAM budget, set `external: true` to enable OSPI external memory.

See `python/config.yaml` for the full list of available options (calibration, memory mode, optimisation level, etc.).

### 4.2 — Run the compiler

Navigate back to the **repository root** and run the compiler with `.mera_venv` active:

**Windows PowerShell**

```powershell
cd C:\Users\<you>\Model-zoo
python ruhmi_tools\mcu_compile.py vision\image_classification\resnet\python\config.yaml
```

**Ubuntu / bash**

```bash
cd ~/Model-zoo
python ruhmi_tools/mcu_compile.py vision/image_classification/resnet8/python/config.yaml
```

The compiled C-code will be written to the `output_dir` specified in `config.yaml` (e.g. `embedded_c/src_mcu/`).

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
| `MODEL_INPUT_H` | `32` | Input image height (pixels) |
| `MODEL_INPUT_W` | `32` | Input image width (pixels) |
| `MODEL_INPUT_C` | `3` | Number of colour channels (RGB) |
| `MODEL_INPUT_SIZE` | `3072` | Flat input buffer size (H × W × C bytes) |
| `MODEL_NUM_CLASSES` | `10` | Number of output classes |
| `MODEL_OUTPUT_SIZE` | `10` | Flat output buffer size |
| `INPUT_SCALE` | `1.0` | INT8 input quantization scale |
| `INPUT_ZP` | `-128` | INT8 input zero-point — maps `pixel → pixel - 128` |
| `OUTPUT_SCALE` | `0.00390625` | INT8 output dequantization scale (= 1/256) |
| `OUTPUT_ZP` | `-128` | INT8 output zero-point |
| `OUTPUT_HAS_SOFTMAX` | `1` | Softmax is already baked into the graph — **do not apply again** |

---

### 5.3 — Preprocessing — `preprocess()`

```c
#include "preprocessing.h"

void preprocess(const uint8_t *source_image,
                uint16_t source_width, uint16_t source_height,
                float    *destination_image,
                uint16_t destination_width, uint16_t destination_height);
```

**Typical call:**

```c
#include "model_metadata.h"
#include "preprocessing.h"

/* Camera frame: 320×240 RGB, stored as interleaved uint8 */
static uint8_t camera_buf[320 * 240 * 3];

/* Resized float buffer fed to the model input tensor */
static float   nn_input_float[MODEL_INPUT_SIZE];   /* 3072 floats */

/* --- inside your inference loop --- */
preprocess(camera_buf, 320, 240,
           nn_input_float, MODEL_INPUT_W, MODEL_INPUT_H);
```

> [!TIP]
> If your camera already delivers 32×32 frames, `preprocess()` still works correctly — it degenerates to a simple `uint8 → float` cast per pixel.

---

### 5.4 — Postprocessing — `postprocess()`

Dequantize the INT8 output tensor, then call `postprocess()` to get the Top-1 (or Top-k) prediction:

```c
#include "model_metadata.h"
#include "postprocessing.h"

static const char *cifar10_labels[MODEL_NUM_CLASSES] = {
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
};

/* Dequantize output */
float class_scores[MODEL_OUTPUT_SIZE];
for (int i = 0; i < MODEL_OUTPUT_SIZE; i++)
    class_scores[i] = ((float)raw_out[i] - OUTPUT_ZP) * OUTPUT_SCALE;

/* Top-1 */
NN_Postprocess_Result_t top1;
postprocess(class_scores, MODEL_NUM_CLASSES, cifar10_labels, &top1, 0, NULL);
printf("Class: %s  Score: %.4f\n", top1.top1_class_name, top1.top1_score);
```

> [!TIP]
> For Top-k results, pass a `NN_Postprocess_Result_t` array and `k` as the last two arguments:
> `postprocess(class_scores, MODEL_NUM_CLASSES, cifar10_labels, NULL, 3, top3);`

---

## License / Provenance

The pretrained ResNet8 weights and training code originate from the [MLCommons Tiny Benchmark](https://github.com/mlcommons/tiny), licensed under the [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0).

> Licensed under the Apache License, Version 2.0.
> You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

The CIFAR-10 dataset used for training and evaluation is available from the [University of Toronto](https://www.cs.toronto.edu/~kriz/cifar.html) (Krizhevsky, 2009).

If you retrain or fine-tune the model, update this section with your training recipe, dataset, and any applicable license information.
