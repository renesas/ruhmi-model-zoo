# RNNoise INT8 (Arm ML-zoo) — Noise Suppression

RNNoise is a recurrent neural network speech denoiser originally developed by Mozilla, adapted and quantised here from the [Arm ML Model Zoo](https://github.com/Arm-Examples/ML-zoo/tree/master/models/noise_suppression/RNNoise/tflite_int8) INT8 variant. It takes raw noisy audio, extracts Opus-style feature frames, and predicts per-band gain masks that suppress background noise.

---

## Model Details

| Field | Value |
|-------|-------|
| **Model name** | RNNoise INT8 (Arm ML-zoo) |
| **Task** | Noise Suppression |
| **Framework** | Arm ML-zoo (INT8 TFLite, quantised from [xiph/rnnoise](https://github.com/xiph/rnnoise)) |
| **Dataset** | Edinburgh Noisy Speech Database |
| **Input shape** | `(1, 1, 42)` — main features; plus 3 GRU state inputs `(1, 24)`, `(1, 48)`, `(1, 96)` |
| **Output shape** | `(1, 1, 22)` band gains + `(1, 1, 1)` VAD probability + 3 GRU next states |
| **Formats included** | TFLite INT8 |
| **Source** | [Arm ML Model Zoo](https://github.com/ARM-software/ML-zoo) |

---

## Model Report Card

Validation performed on the Edinburgh Noisy Speech Database (824 clean/noisy WAV pairs), averaged PESQ wide-band.

| Variant | avg PESQ-wb | Δ vs noisy baseline |
|---------|:-----------:|:-------------------:|
| Noisy baseline (no denoising) | 1.969 | — |
| TFLite INT8 | 2.435 | +0.466 |
| MERA TFLite INT8 | 2.438 | +0.469 |

> [!NOTE]
> The model was compiled using `mera-2.6.0+pkg.4815` and `FSP 6.5.1` was used for building and testing the project.

### Inference Performance (RA8P1)

Measured on-target (Cortex-M85 @ 1 GHz, Ethos-U55 NPU @ 500 MHz). AI-only latency per 10 ms frame.

| Memory Configuration | CPU (ms) | NPU (ms) |
|----------------------|:--------:|:--------:|
| † Internal Flash + Internal SRAM | 0.293 | 0.068 |
| ¶ Internal SRAM only | 0.290 | 0.068 |

---

## Folder Structure

```
rnnoise/
├── README.md                    ← This file
├── python/                      ← Host-side inference and model files
│   ├── download_model.py
│   ├── inference.py
│   ├── config.yaml              ← RUHMI compiler configuration
│   ├── requirements.txt
│   ├── model/                   ← Downloaded TFLite model (rnnoise_INT8.tflite)
│   ├── sample_audio/            ← Sample noisy/clean WAV pairs
│   └── utils/                   ← DSP preprocessing helpers
└── embedded_c/                  ← Precompiled C-code artifacts (MERA 2.6.0+pkg.4815)
    ├── model_metadata.h         ← Compile-time tensor and quantisation metadata
    ├── preprocessing.h / .c     ← Pre-NN DSP wrapper (PCM → 42 features)
    ├── postprocessing.h / .c    ← Post-NN DSP wrapper (gains → denoised PCM)
    ├── rnnoise_dsp/             ← Upstream DSP support library
    ├── src_mcu/             ← CPU-only CMSIS-NN code-gen (rnnoise_INT8_CPU)
    └── src_mcu_npu/         ← NPU-accelerated code-gen (rnnoise_INT8_NPU)
```

---

## Prerequisites

**Required tools:**

| Tool | Version | Required for |
|------|---------|--------------|
| Python | 3.10 | All steps |
| pip | ≥ 22 | All steps |
| CMake + gcc | recent | `mera_tflite_int8` mode only |
| MERA SDK | 2.6.0+pkg.4815 | `mera_tflite_int8` mode only |

If you only need TFLite INT8, CMake and MERA are **not** required.

1. **Python 3.10** installed.
2. **Inference venv** — navigate to `python/` and create a dedicated virtual environment:

    **Windows PowerShell**
    > **Note:** If venv activation is blocked by PowerShell execution policy, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in the same PowerShell window, then run the activation command again.

    ```powershell
    cd audio\noise_suppression\rnnoise\python
    py -3.10 -m venv .venv_rnnoise
    .\.venv_rnnoise\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    ```

    **Ubuntu / bash**

    ```bash
    cd audio/noise_suppression/rnnoise/python
    python3.10 -m venv .venv_rnnoise
    source .venv_rnnoise/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

3. **Compiler venv** (`.mera_venv`) — required only for [Step 4](#step-4--compile-for-ra8p1-ruhmi). See the [top-level README](../../../README.md) for setup instructions.

---

## Step 1 — Obtain the Model

The TFLite model is already provided in `python/model/`. To re-download it from scratch, proceed to [Step 2](#step-2--build-the-models).

---

## Step 2 — Build the Models

```bash
python download_model.py
```

This downloads `model/rnnoise_INT8.tflite` from the Arm ML Model Zoo.

---

## Step 3 — Run Inference (Python)

Denoise a single noisy WAV file:

```bash
python inference.py --input sample_audio/p257_036_noisy.wav
```


Output is written alongside the input as `p257_036_noisy_denoised.wav`.

To also compute the wide-band PESQ score against a clean reference:

```bash
python inference.py --input sample_audio/p257_036_noisy.wav \
                    --reference sample_audio/p257_036_clean.wav
```

Expected output:

```
========================================================================
  RNNoise TFLite  -  single-WAV denoise
========================================================================
  Input   : .../sample_audio/p257_036_noisy.wav
  Output  : .../sample_audio/p257_036_noisy_denoised.wav
  Model   : rnnoise_INT8.tflite  (113472 B)
  Ref     : sample_audio/p257_036_clean.wav

  Loaded 85725 samples = 1.79 s @ 48000 Hz
  Wrote  85440 samples -> p257_036_noisy_denoised.wav

  PESQ-wb (wide-band, 16 kHz):
    noisy    vs clean : 1.413
    denoised vs clean : 3.064   (delta = +1.652)

Done.
```

Optional flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--output <path>` | `<input_stem>_denoised.wav` | Custom output path |
| `--reference <path>` | — | Clean reference WAV; enables PESQ-wb scoring |
| `--model <path>` | `model/rnnoise_INT8.tflite` | Explicit TFLite model path |
| `--verbose` | off | Print per-stage frame counts and wall-time |

---

## Step 4 — Compile for RA8P1 (RUHMI)

This step converts the TFLite INT8 model into C-code for the RA8P1 MCU. Activate the **compiler venv** (`.mera_venv`).

### 4.1 — Edit the compile configuration

Open `python/config.yaml` and verify the paths match your system:

```yaml
model_path: audio/noise_suppression/rnnoise/python/model/rnnoise_INT8.tflite
output_dir: audio/noise_suppression/rnnoise/embedded_c/src_mcu
target: cpu
quantize: false
external: false
```

> [!IMPORTANT]
> Use absolute paths if running the compiler from outside the repo root.

> [!TIP]
> For NPU deployment set `target: npu` and point `output_dir` to `embedded_c/src_mcu_npu`.

### 4.2 — Run the compiler

Navigate back to the **repository root** and run the compiler with `.mera_venv` active:

**Windows PowerShell**

```powershell
cd C:\Users\<you>\Model-zoo
python ruhmi_tools\mcu_compile.py audio\noise_suppression\rnnoise\python\config.yaml
```

**Ubuntu / bash**

```bash
cd ~/Model-zoo
python ruhmi_tools/mcu_compile.py audio/noise_suppression/rnnoise/python/config.yaml
```

The compiled C-code will be written to the `output_dir` specified in `config.yaml`.

---

## Step 5 — Embedded C Integration

The `embedded_c/` folder contains portable files for bare-metal or RTOS firmware integration.

| File | Purpose |
|------|--------|
| `model_metadata.h` | Compile-time constants — tensor shapes, quantisation params, memory footprint |
| `preprocessing.h` / `preprocessing.c` | Pre-NN DSP wrapper: PCM → 42 INT8 features + per-frame scratch |
| `postprocessing.h` / `postprocessing.c` | Post-NN DSP wrapper: INT8 band gains → denoised PCM |
| `rnnoise_dsp/` | Upstream DSP support library (required by pre/postprocessing) |

---

### 5.1 — Add files to your project

Copy the following into your firmware project:

```
embedded_c/
├── model_metadata.h
├── preprocessing.h
├── preprocessing.c
├── postprocessing.h
├── postprocessing.c
└── rnnoise_dsp/
```

Also copy the compiled model artifacts from the appropriate subdirectory:

- **CPU-only (CMSIS-NN):** `embedded_c/src_mcu/`
- **NPU-accelerated (Ethos-U55):** `embedded_c/src_mcu_npu/`

---

### 5.2 — `model_metadata.h` — Key constants

```c
#include "model_metadata.h"
```

| Constant | Value | Description |
|----------|-------|-------------|
| `MODEL_SAMPLE_RATE_HZ` | `48000` | Operating sample rate |
| `MODEL_FRAME_SIZE` | `480` | Samples per 10 ms frame |
| `MODEL_NB_BANDS` | `22` | Bark-scale critical bands |
| `MODEL_NB_FEATURES` | `42` | NN input feature vector size |
| `MODEL_INPUT_FEATURES_SCALE` | `0.2215006053f` | INT8 input quantisation scale |
| `MODEL_INPUT_FEATURES_ZP` | `+14` | INT8 input zero-point |
| `MODEL_OUTPUT_GAINS_SCALE` | `0.0039062500f` | Band gains dequantisation scale |
| `MODEL_OUTPUT_GAINS_ZP` | `-128` | Band gains zero-point |
| `MODEL_GRU_STATE_TOTAL_BYTES` | `168` | Total GRU state storage (3 INT8 buffers) |
| `MODEL_MERA_BUFFER_BYTES` | `2209` | MERA inference scratch buffer size (bytes) |

---

### 5.3 — Preprocessing — `preprocess_frame()`

**API:**

```c
#include "preprocessing.h"

void preprocess_frame(DenoiseState            *p_st,
                      const int16_t            p_pcm_in[MODEL_FRAME_SIZE],
                      float                    p_features[MODEL_NB_FEATURES],
                      preprocess_scratch_t    *p_scratch);
```

**Typical call:**

```c
#include "model_metadata.h"
#include "preprocessing.h"
#include "rnnoise_dsp/rnnoise.h"

/* Long-lived state — allocate once at startup, never reset between frames */
DenoiseState         *p_st  = rnnoise_create(NULL);
preprocess_scratch_t  scratch;                        /* re-used every frame */

int16_t pcm_in  [MODEL_FRAME_SIZE];    /* 480 int16 PCM samples @ 48 kHz */
float   features[MODEL_NB_FEATURES];   /* 42 float features               */

preprocess_frame(p_st, pcm_in, features, &scratch);
/* scratch.features_int8[] is now ready for compute_sub_0000() */
```

---

### 5.4 — Postprocessing — `postprocess_frame()`

**API:**

```c
#include "postprocessing.h"

void postprocess_frame(DenoiseState         *p_st,
                       preprocess_scratch_t *p_scratch,
                       int16_t               p_pcm_out[MODEL_FRAME_SIZE],
                       float                *p_vad_prob_out);
```

**Typical call:**

```c
#include "model_metadata.h"
#include "postprocessing.h"

int16_t pcm_out [MODEL_FRAME_SIZE];
float   vad_prob;

/* Populate scratch.gains_int8 and scratch.vad_prob_int8 from compute_sub_0000() first */
postprocess_frame(p_st, &scratch, pcm_out, &vad_prob);
/* pcm_out holds the denoised 480-sample frame; vad_prob in [0.0, 1.0] */
```

---

## License / Provenance

| Component | Source | Licence |
|-----------|--------|---------|
| RNNoise model | [Arm ML Model Zoo](https://github.com/Arm-Examples/ML-zoo/tree/master/models/noise_suppression/RNNoise/tflite_int8) | Apache-2.0 |