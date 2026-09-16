# YAMNet — Audio Classification (AudioSet)

YAMNet is a deep-net that predicts 521 audio event classes from the [AudioSet](https://research.google.com/audioset/) taxonomy, based on the MobileNet v1 depthwise-separable convolution architecture ([Howard et al., 2017](https://arxiv.org/abs/1704.04861)). This folder contains everything needed to obtain the model, compile it for the Renesas RA8P1 (Cortex-M85 + Ethos-U55 NPU) using the RUHMI toolchain, and run inference.

---

## Model Details

| Field | Value |
|-------|-------|
| **Model name** | YAMNet |
| **Task** | Audio Classification — 521-class AudioSet taxonomy |
| **Framework** | TensorFlow → TFLite |
| **Dataset** | AudioSet |
| **Input shape** | `(1, 96, 64)` — log-mel spectrogram patch (frames × mel bands) |
| **Output shape** | `(1, 521)` — per-class sigmoid scores |
| **Classes** | 521 AudioSet sound event classes |
| **Source** | [Google YAMNet — TensorFlow Hub](https://tfhub.dev/google/yamnet/1) |

> [!NOTE]
> YAMNet uses a **split-input** pipeline: each audio clip is converted to a sequence of log-mel spectrogram patches, and the model runs on each patch individually with shape `(1, 96, 64)`. The final class score is the maximum score across all patches for that clip. `inference.py` handles the waveform-to-patch conversion internally via `utils/features.py`, and the INT16 model replaces the original Dense classifier head with a mathematically equivalent 1×1 Conv2D during conversion for TFLite compatibility (see `download_model.py`).

> [!NOTE]
> **Why INT16?** Audio spectral features have a wide dynamic range and are more sensitive to quantization noise than most vision models. Plain INT8 quantization causes a measurable mAP drop; the **16×8 mixed-precision scheme** (INT16 activations and I/O, INT8 weights, INT64 biases) preserves accuracy close to FP32 while still compressing the model to ~28 % of the FP32 size (3.8 MB vs 14.9 MB). The Dense classifier head was replaced with a mathematically equivalent 1×1 Conv2D before quantization because the CMSIS-NN TFLM FullyConnected kernel does not support per-channel INT8 weights when activations are INT16.

## Model Report Card

Evaluated on the AudioSet HF 20k test set (18,877 clips, clip-score pooling: mean).

| Model Variant | Format | Clip-level mAP |
|---------------|--------|:--------------:|
| TFLite FP32 | `.tflite` | 30.99 % |
| TFLite INT16 | `.tflite` | 30.86 % |

> [!NOTE]
> Metric is clip-level mean Average Precision (mAP) across 521 AudioSet classes. Clips with no labels mapping to the 521-class YAMNet taxonomy are excluded (354 unmapped label names filtered).

> [!NOTE]
> The model was compiled using `mera-2.6.0+pkg.4815` and `FSP 6.5.1` was used for building and testing the project.

### Inference Performance (RA8P1)

Measured on-target (Cortex-M85 @ 1 GHz, Ethos-U55 NPU @ 500 MHz). AI-only latency per inference call.

| Memory Configuration | CPU (ms) | NPU (ms) |
|----------------------|:--------:|:--------:|
| ‡ OSPI + External SDRAM | 386 | 8 |
---

## Folder Structure

```
yamnet/
├── README.md                    ← This file
├── python/                      ← Inference scripts and model files
│   ├── download_model.py
│   ├── inference.py
│   ├── requirements.txt
│   ├── model/                   ← TFLite models and class map CSV
│   ├── sample_audio/            ← Sample WAV files
│   └── utils/                   ← YAMNet DSP frontend (feature extraction)
└── embedded_c/                  ← MCU integration files (TFLM deployment)
    ├── model_metadata.h
    ├── preprocessing.h
    ├── preprocessing.c
    ├── postprocessing.h
    ├── postprocessing.c
    ├── audio_feature_extraction/ ← STFT, Hann window, mel filterbank
    ├── src_mcu/                  ← CPU-only TFLM model artifacts
    └── src_mcu_npu/              ← NPU-accelerated model artifacts
```

---

## Prerequisites

1. **Python 3.10** installed.
2. **Inference venv** — navigate to the `python/` directory and create a dedicated virtual environment:

    **Windows PowerShell**
    > **Note:** If venv activation is blocked by PowerShell execution policy, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in the same window first.

    ```powershell
    cd audio\audio_classification\yamnet\python
    py -3.10 -m venv .venv_yamnet
    .\.venv_yamnet\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    ```

    **Ubuntu / bash**

    ```bash
    cd audio/audio_classification/yamnet/python
    python3.10 -m venv .venv_yamnet
    source .venv_yamnet/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

---

## Step 1 — Obtain the Model

The TFLite models are already provided in `python/model/`. To regenerate them from scratch, proceed to [Step 2](#step-2--build-the-models).

---

## Step 2 — Build the Models

Use `download_model.py` to fetch YAMNet weights from TensorFlow Hub and export the split-input TFLite models. Activate the **inference venv** and navigate to `python/`.

**Both FP32 + INT16 (default)**

```bash
python download_model.py
```

**One variant only**

```bash
python download_model.py --mode fp32
python download_model.py --mode int16
```

**INT16 with local calibration audio (recommended)**

```bash
python download_model.py --mode int16 --audio-dir <path/to/16k_wavs> --calib-count 3000
```

Output files are written to `python/model/`:
- `yamnet_FP32.tflite`
- `yamnet_INT16.tflite`
- `yamnet_class_map.csv`

---

## Step 3 — Run Inference (Python)

Activate the **inference venv** and navigate to `python/`:

**Windows PowerShell**

```powershell
cd audio\audio_classification\yamnet\python
.\.venv_yamnet\Scripts\Activate.ps1
python inference.py sample_audio\Y-eK9VHuiH74.wav
```

**Ubuntu / bash**

```bash
cd audio/audio_classification/yamnet/python
source .venv_yamnet/bin/activate
python inference.py sample_audio/Y-eK9VHuiH74.wav
```

Optional flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `model/yamnet_FP32.tflite` | Path to TFLite model |
| `--class-map` | `model/yamnet_class_map.csv` | Path to class label CSV |
| `--top-k` | `5` | Number of top classes to print |
| `--audio-dir` | — | Run on all WAVs in a directory |
| `--limit` | — | Cap number of files in `--audio-dir` mode |

**Example output:**

```
Model   : .../model/yamnet_FP32.tflite
Classes : .../model/yamnet_class_map.csv
Input   : float32  q=(0.0, 0)
Output  : float32  q=(0.0, 0)

sample_audio/Y-eK9VHuiH74.wav
------------------------------------------------------------
Rank Idx   Label                                    Score
------------------------------------------------------------
1    195   Bell                                    1.0000
2    196   Church bell                             0.8446
3    133   Musical instrument                      0.5548
4    202   Change ringing (campanology)            0.4817
5    132   Music                                   0.4546
------------------------------------------------------------
```

> By default `inference.py` loads `model/yamnet_FP32.tflite`. Pass `--model model/yamnet_INT16.tflite` to use the INT16 variant.

---

## Step 4 — Embedded C Integration

> [!NOTE]
> This model is deployed with the TensorFlow Lite for Microcontrollers (TFLM) runtime. The MERA compiler does not currently support INT16 activations, so the standard RUHMI compile flow does not apply. The INT16 model uses **INT8 weights, INT16 activations and I/O** (TFLite 16×8 quantization).

The `embedded_c/` folder contains portable files for bare-metal or RTOS firmware integration.

| File | Purpose |
|------|--------|
| `model_metadata.h` | Compile-time constants — tensor shapes, quantisation params, audio frontend geometry |
| `preprocessing.h` / `preprocessing.c` | Full waveform-to-patch pipeline: PCM → quantised INT16 log-mel patch batch |
| `postprocessing.h` / `postprocessing.c` | Patch output pipeline: INT16 model outputs → top-K class indices and scores |
| `audio_feature_extraction/` | STFT frontend support library (FFT twiddles, Hann window, mel filterbank) |

---

### 4.1 — Add files to your project

Copy the following into your firmware project:

```
embedded_c/
├── model_metadata.h
├── preprocessing.h
├── preprocessing.c
├── postprocessing.h
├── postprocessing.c
└── audio_feature_extraction/
```

Also copy the compiled model artifacts from the appropriate subdirectory:

- **CPU-only (TFLM):** `embedded_c/src_mcu/`
- **NPU-accelerated:** `embedded_c/src_mcu_npu/`

---

### 4.2 — `model_metadata.h` — Key constants

```c
#include "model_metadata.h"
```

| Constant | Value | Description |
|----------|-------|-------------|
| `MODEL_SAMPLE_RATE_HZ` | `16000` | Operating sample rate (Hz) |
| `MODEL_STFT_WINDOW_SAMPLES` | `400` | STFT window size (25 ms) |
| `MODEL_STFT_HOP_SAMPLES` | `160` | STFT hop size (10 ms) |
| `MODEL_INPUT_FRAMES` | `96` | Log-mel rows per patch |
| `MODEL_INPUT_BANDS` | `64` | Mel filterbank bands |
| `MODEL_INPUT_ELEMENTS` | `6144` | Patch elements (96 × 64) |
| `MODEL_OUTPUT_CLASSES` | `521` | AudioSet output classes |
| `TOP_K_COUNT` | `5` | Default top-K results |
| `MODEL_INPUT_SCALE` | `0.0002108144f` | INT16 input quantisation scale |
| `MODEL_INPUT_ZERO_POINT` | `0` | INT16 input zero-point |
| `MODEL_OUTPUT_SCALE` | `3.051757812e-05f` | INT16 output dequantisation scale |
| `MODEL_OUTPUT_ZERO_POINT` | `0` | INT16 output zero-point |

---

### 4.3 — Preprocessing — `preprocess()`

**API:**

```c
#include "preprocessing.h"

size_t preprocess(const int16_t *p_waveform,
                  size_t         _num_samples,
                  float         *p_log_mel_scratch,
                  size_t         _max_rows,
                  int16_t       *p_quantized_patches,
                  size_t         _max_patches);
```

**Typical call:**

```c
#include "model_metadata.h"
#include "preprocessing.h"

#define AUDIO_SAMPLES    (16000U)   /* 1 s @ 16 kHz */
#define MAX_LOG_MEL_ROWS (1000U)    /* scratch capacity in rows */
#define MAX_PATCHES      (10U)      /* output patch buffer capacity */

int16_t pcm_in[AUDIO_SAMPLES];
float   log_mel_scratch[MAX_LOG_MEL_ROWS * MODEL_INPUT_BANDS];
int16_t quantized_patches[MAX_PATCHES * MODEL_INPUT_ELEMENTS];

size_t num_patches = preprocess(pcm_in, AUDIO_SAMPLES,
                                log_mel_scratch, MAX_LOG_MEL_ROWS,
                                quantized_patches, MAX_PATCHES);
/* quantized_patches[] now contains num_patches INT16 patches ready for TFLM inference */
```

---

### 4.4 — Postprocessing — `postprocess()`

**API:**

```c
#include "postprocessing.h"

void postprocess(const int16_t *p_quantized_outputs,
                 size_t         _num_patches,
                 size_t         _class_count,
                 int            _top_k,
                 int           *p_top_indices,
                 float         *p_top_scores);
```

**Typical call:**

```c
#include "model_metadata.h"
#include "postprocessing.h"

int16_t model_outputs[MAX_PATCHES * MODEL_OUTPUT_CLASSES];
int     top_indices[TOP_K_COUNT];
float   top_scores[TOP_K_COUNT];

/* Populate model_outputs[] from num_patches TFLM inference calls first */
postprocess(model_outputs, num_patches, MODEL_OUTPUT_CLASSES, TOP_K_COUNT,
            top_indices, top_scores);
/* top_indices[0] is the highest-scoring class index; top_scores[0] is its score */
```

---

## License / Provenance

| Component | Source | Licence |
|-----------|--------|---------|
| YAMNet model | [Google YAMNet — TensorFlow Hub](https://tfhub.dev/google/yamnet/1) | Apache-2.0 |
| AudioSet dataset | [research.google.com/audioset](https://research.google.com/audioset/) | CC-BY 4.0 |
