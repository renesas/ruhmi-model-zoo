# KWS (DS-CNN) — Keyword Spotting (Google Speech Commands)

KWS (Keyword Spotting) is a compact audio classification network based on a Depthwise Separable CNN (DS-CNN) architecture ([Zhang et al., 2017](https://arxiv.org/abs/1711.07971)). It classifies 1-second audio clips into 12 categories using MFCC (Mel-Frequency Cepstral Coefficient) features. This folder contains everything needed to obtain the model, convert it to TFLite, compile it for the Renesas RA8P1 (Cortex-M85 + Ethos-U55 NPU) using the RUHMI toolchain, and run inference.

---

## Model Details

| Field | Value |
|-------|-------|
| **Model name** | KWS DS-CNN (Keyword Spotting) |
| **Task** | Audio classification — keyword spotting |
| **Framework** | TensorFlow → TFLite |
| **Dataset** | Google Speech Commands v0.02 |
| **Input shape** | `(1, 49, 10, 1)` — NHWC MFCC feature map (frames × coefficients) |
| **Output shape** | `(1, 12)` — softmax probabilities over 12 classes |
| **Classes** | `down, go, left, no, off, on, right, stop, up, yes, silence, unknown` |
| **Source** | [MLCommons Tiny](https://github.com/mlcommons/tiny) |

> [!NOTE]
> The input to the model is a **49×10 MFCC feature map** computed from a 1-second, 16 kHz mono WAV file. The raw audio is never passed directly — MFCC preprocessing must be performed before calling the model.

## Model Report Card

Accuracy measured on the Google Speech Commands v0.02 test set (4890 samples).

| Model Variant | Format | Top-1 Accuracy |
|---------------|--------|:--------------:|
| TFLite FP32 | `.tflite` | 92.17 % |
| TFLite INT8 | `.tflite` | 92.25 % |
| MERA FP32 | `.mera` | 92.17 % |
| MERA INT8 (TFLite Quantized) | `.mera` | 92.25 % |

> [!NOTE]
> The model was compiled using `mera-2.5.0+pkg.3577` and `FSP 6.2` was used for building and testing the project.

### Inference Performance (RA8P1)

Measured on-target (Cortex-M85 @ 1 GHz, Ethos-U55 NPU @ 500 MHz). AI-only latency.

| Memory Configuration | CPU (ms) | NPU (ms) |
|----------------------|:--------:|:--------:|
| † Internal Flash + Internal SRAM | 1 | 0.108 |
| ¶ Internal SRAM only | 1 | 0.141 |

---

## Folder Structure

```
key_word_spotting/
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
    cd audio\audio_classification\key_word_spotting\python
    python -m venv .venv_kws
    .\.venv_kws\Scripts\Activate.ps1
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

    **Ubuntu / bash**

    ```bash
    cd audio/audio_classification/key_word_spotting/python
    python3.10 -m venv .venv_kws
    source .venv_kws/bin/activate
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

**Both FP32 + INT8 (default)**

```bash
python download_model.py
```

The script will automatically download the Google Speech Commands v0.02 dataset for INT8 calibration. Use `--mode fp32` to skip INT8 (no download needed) or `--mode int8` for INT8 only. Output files are written to `python/model/`.

---

## Step 3 — Run Inference (Python)

Activate the **inference venv** and navigate to `python/`:

**Windows PowerShell**

```powershell
cd audio\audio_classification\key_word_spotting\python
.\.venv_kws\Scripts\Activate.ps1
python inference.py --audio sample_audio\test_yes.wav
```

**Ubuntu / bash**

```bash
cd audio/audio_classification/key_word_spotting/python
source .venv_kws/bin/activate
python inference.py --audio sample_audio/test_yes.wav
```

Optional flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `model/kws_ref_model_FP32.tflite` | Path to TFLite model |
| `--audio` | — | Path to 16 kHz mono WAV file (1 second) |
| `--verbose` | off | Print model I/O details and MFCC feature stats |

**Example output:**

```
Loading audio: sample_audio/test_yes.wav

Results:
  down      : 0.0000
  go        : 0.0000
  left      : 0.0000
  no        : 0.0000
  off       : 0.0000
  on        : 0.0000
  right     : 0.0000
  stop      : 0.0000
  up        : 0.0039
  yes       : 0.9766  ████████████████████████████████████████
  silence   : 0.0156
  unknown   : 0.0039

Predicted: yes (index 9, confidence 0.9766)
```

> By default `inference.py` loads `model/kws_ref_model_FP32.tflite`. Pass `--model model/kws_ref_model_INT8.tflite` to use the INT8 variant.

---

## Step 4 — Compile for RA8P1 (RUHMI)

This step converts the TFLite model into C-code for the RA8P1 MCU. Activate the **compiler venv** (`.mera_venv`).

### 4.1 — Edit the compile configuration

Open `python/config.yaml` and verify the paths match your system:

```yaml
model_path: audio/audio_classification/key_word_spotting/python/model/kws_ref_model_INT8.tflite
output_dir: audio/audio_classification/key_word_spotting/embedded_c/src_mcu
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
python ruhmi_tools\mcu_compile.py audio\audio_classification\key_word_spotting\python\config.yaml
```

**Ubuntu / bash**

```bash
cd ~/ruhmi-model-zoo
python ruhmi_tools/mcu_compile.py audio/audio_classification/key_word_spotting/python/config.yaml
```

The compiled C-code will be written to the `output_dir` specified in `config.yaml`.

---

## Step 5 — Embedded C Integration

The `embedded_c/` folder contains portable, board-independent files you can drop directly into any bare-metal or RTOS project.

| File | Purpose |
|------|---------|
| `model_metadata.h` | All compile-time constants — MFCC dims, quant params, class count, arena sizes |
| `preprocessing.h` / `preprocessing.c` | Convert PCM audio → quantized MFCC features (model input) |
| `postprocessing.h` / `postprocessing.c` | Dequantize output and return top-1 class index |

> [!NOTE]
> These files have **no external dependencies** beyond CMSIS-DSP (for MFCC), `<stdint.h>`, and `<stddef.h>`. They compile cleanly with any C99-compatible toolchain (GCC, Clang, IAR, AC6).

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
| `MODEL_INPUT_FRAMES` | `49` | Number of MFCC time frames |
| `MODEL_INPUT_COEFFS` | `10` | Number of DCT coefficients per frame |
| `MODEL_INPUT_CHANNELS` | `1` | Mono |
| `MODEL_NUM_CLASSES` | `12` | Total output classes |
| `INPUT_SCALE` | `0.5821f` | INT8 input quantization scale |
| `INPUT_ZP` | `84` | INT8 input zero-point |
| `OUTPUT_SCALE` | `0.00390625f` | INT8 output dequantization scale (1/256) |
| `OUTPUT_ZP` | `-128` | INT8 output zero-point |
| `OUTPUT_HAS_SOFTMAX` | `1` | Softmax baked in — do NOT re-apply |
---

### 5.3 — Preprocessing — `preprocess()`

```c
#include "preprocessing.h"

void preprocess(const int16_t *p_pcm_input,
                int32_t        sample_count,
                int8_t        *p_mfcc_output);
```

**Typical call:**

```c
#include "model_metadata.h"
#include "preprocessing.h"

/* 1-second PCM audio: 16 kHz, 16-bit mono */
static int16_t pcm_buf[DESIRED_SAMPLES];  /* 16000 samples */

/* Quantized int8 MFCC feature map fed to the model input tensor */
static int8_t nn_input[MODEL_INPUT_SIZE];  /* 490 bytes */

/* --- inside your inference loop --- */
preprocess(pcm_buf, DESIRED_SAMPLES, nn_input);
```

> [!TIP]
> `preprocess()` internally performs: PCM → float normalisation → STFT (480 window, 320 hop) → 40-bin mel filterbank → log-mel → DCT (10 coeffs) → INT8 quantization. Ensure CMSIS-DSP v1.15+ is included in your project.

---

### 5.4 — Postprocessing — `postprocess()`

```c
#include "model_metadata.h"
#include "postprocessing.h"

/* Raw INT8 output from model */
static int8_t raw_out[MODEL_NUM_CLASSES];  /* 12 bytes */

/* Get top-1 class index */
int32_t predicted_class = postprocess(raw_out, MODEL_NUM_CLASSES);

/* Map to label */
static const char *kws_labels[] = KWS_CLASS_NAMES;
printf("Keyword: %s\n", kws_labels[predicted_class]);
```

> [!TIP]
> `postprocess()` dequantizes the INT8 output using `OUTPUT_SCALE` and `OUTPUT_ZP`, then returns the argmax class index. Since `OUTPUT_HAS_SOFTMAX = 1`, no additional softmax is needed.

---

## License / Provenance

| Component | Source | Licence |
|-----------|--------|---------|
| DS-CNN model weights | [MLCommons Tiny](https://github.com/mlcommons/tiny) | Apache 2.0 |
| Google Speech Commands v0.02 (test data) | [research.google.com/audioset/download](https://research.google.com/audioset/download.html) | CC-BY 4.0 |
