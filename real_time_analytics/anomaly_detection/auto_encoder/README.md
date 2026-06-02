# Anomaly Detection — Dense Autoencoder (DCASE 2020 ToyCar)

A fully-connected autoencoder for unsupervised anomaly detection on machine audio ([Koizumi et al., 2020](https://arxiv.org/abs/2006.05822)). The model reconstructs log-mel spectrogram features; the anomaly score is the mean reconstruction error (MSE) across all sliding-window vectors. This folder contains everything needed to obtain the model, convert it to TFLite, compile it for the Renesas RA8P1 (Cortex-M85 + Ethos-U55 NPU) using the RUHMI toolchain, and run inference.

---

## Model Details

| Field | Value |
|-------|-------|
| **Model name** | AD Dense Autoencoder (ad01) |
| **Task** | Unsupervised anomaly detection — machine audio |
| **Framework** | Keras / TensorFlow 2.x |
| **Dataset** | DCASE 2020 Task 2 — ToyCar |
| **Input shape** | `(1, 640)` — sliding-window log-mel vector (128 mels × 5 frames) |
| **Output shape** | `(1, 640)` — reconstructed vector |
| **Metric** | AUC / pAUC (max_fpr=0.1) |
| **Source** | [MLCommons Tiny — Anomaly Detection](https://github.com/mlcommons/tiny/tree/master/benchmark/training/anomaly_detection) |

## Model Report Card

Evaluated on the DCASE 2020 Task 2 ToyCar test set (all machine IDs).

| Model Variant | Format | AUC | pAUC (fpr=0.1) |
|---------------|--------|:---:|:--------------:|
| TFLite FP32 | `.tflite` | 0.8760 | 0.7641 |
| TFLite INT8 | `.tflite` | 0.8403 | 0.7201 |
| MERA FP32 | `.mera` | 0.8760 | 0.7641 |
| MERA INT8 (TFLite Quantized) | `.mera` | 0.8448 | 0.7253 |

> [!NOTE]
> The model was compiled using `mera-2.5.0+pkg.3577` and `FSP 6.2` was used for building and testing the project.

### Inference Performance (RA8P1)

Measured on-target (Cortex-M85 @ 1 GHz, Ethos-U55 NPU @ 500 MHz). AI-only latency per feature vector (single forward pass of 640→640 autoencoder).

| Memory Configuration | CPU (ms) | NPU (ms) |
|----------------------|:--------:|:--------:|
| † Internal Flash + Internal SRAM | 54 | 28 |
| ¶ Internal SRAM only | 44 | 29 |

---

## Folder Structure

```
anomaly_detection/
└── auto_encoder/
    ├── README.md                   ← This file
    ├── python                      ← Inference scripts, conversion tools, config
    └── embedded_c                  ← Compiled C-code for MCU (CPU & NPU)
```

---

## Prerequisites

1. **Python 3.10** installed (see [Install Python 3.10](../../README.md#install-python-310) in the top-level README for platform-specific steps).
2. **Inference venv** — navigate to the `python/` directory and create a dedicated virtual environment:

    **Windows PowerShell**

    ```powershell
    cd audio\anomaly_detection\python
    python -m venv .venv_ad
    .\.venv_ad\Scripts\Activate.ps1
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

    **Ubuntu / bash**

    ```bash
    cd audio/anomaly_detection/auto_encoder/python
    python3.10 -m venv .venv_ad
    source .venv_ad/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

3. **Compiler venv** (`.mera_venv`) — required only for [Step 4](#step-4--compile-for-ra8p1-ruhmi). See the [top-level README](../../README.md) for setup instructions.

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

The script will automatically download the Keras `.h5` model from MLCommons Tiny and the DCASE 2020 ToyCar training dataset (from Zenodo) for INT8 calibration. Use `--mode fp32` to skip INT8 or `--mode int8` for INT8 only. Output files are written to `python/model/`.

---

## Step 3 — Run Inference (Python)

Activate the **inference venv** and navigate to `python/`:

**Windows PowerShell**

```powershell
cd audio\anomaly_detection\python
.\.venv_ad\Scripts\Activate.ps1
python inference.py --audio sample_audio\anomaly_id_01_00000000.wav
```

**Ubuntu / bash**

```bash
cd audio/anomaly_detection/auto_encoder/python
source .venv_ad/bin/activate
python inference.py --audio sample_audio/anomaly_id_01_00000000.wav
```

Optional flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `model/ad01_FP32.tflite` | Path to TFLite model |
| `--audio` | — | Path to WAV file for anomaly detection |
| `--verbose` | off | Print detailed model IO and feature stats |

**Example output:**

```
==================================================
  ANOMALY DETECTION RESULT
==================================================
  Audio file     : anomaly_id_01_00000000.wav
  Model          : ad01_FP32.tflite (FP32)
  Feature vectors: 196
  Anomaly score  : 14.283195
  Per-vector MSE : min=3.271842, max=42.108253
  Threshold      : 10.70
  Predicted      : ANOMALY
  Ground truth   : ANOMALY
  Result         : ✅ CORRECT
==================================================
```

> By default `inference.py` loads `model/ad01_FP32.tflite`. Pass `--model model/ad01_INT8.tflite` to use the INT8 variant.

---

## Step 4 — Compile for RA8P1 (RUHMI)

This step converts the TFLite model into C-code for the RA8P1 MCU. Activate the **compiler venv** (`.mera_venv`).

### 4.1 — Edit the compile configuration

Open `python/config.yaml` and verify the paths match your system:

```yaml
model_path: audio/anomaly_detection/auto_encoder/python/model/ad01_FP32.tflite
output_dir: audio/anomaly_detection/auto_encoder/embedded_c/src_mcu
target: cpu        # 'cpu' → CMSIS-NN  |  'npu' → Ethos-U55 NPU
quantize: true     # RUHMI will quantize FP32 → INT8 (better accuracy than pre-quantized)
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
python ruhmi_tools\mcu_compile.py audio\anomaly_detection\python\config.yaml
```

**Ubuntu / bash**

```bash
cd ~/Model-zoo
python ruhmi_tools/mcu_compile.py audio/anomaly_detection/auto_encoder/python/config.yaml
```

The compiled C-code will be written to the `output_dir` specified in `config.yaml`.

---

## Step 5 — Embedded C Integration

The `embedded_c/` folder contains portable, board-independent files you can drop directly into any bare-metal or RTOS project.

| File | Purpose |
|------|---------|
| `preprocessing.h` / `preprocessing.c` | Extract log-mel spectrogram sliding-window vectors from raw PCM |
| `postprocessing.h` / `postprocessing.c` | Compute MSE anomaly score from input vs reconstruction |

> [!NOTE]
> These files have **no external dependencies** beyond `<stdint.h>`, `<math.h>`, and `<stddef.h>`. They compile cleanly with any C99-compatible toolchain (GCC, Clang, IAR, AC6).

---

### 5.1 — Add files to your project

Copy the following files into your firmware project (or add them as include paths):

```
embedded_c/
├── preprocessing.h
├── preprocessing.c
├── postprocessing.h
└── postprocessing.c
```

Also copy the compiled model artifacts from the appropriate subdirectory into your project:

- **CPU-only (CMSIS-NN):** `embedded_c/src_mcu/`
- **NPU-accelerated (Ethos-U55):** `embedded_c/src_mcu_npu/`

---

### 5.2 — Preprocessing — `preprocess()`

```c
#include "preprocessing.h"

int preprocess(const int16_t *p_pcm,
               int            num_samples,
               int            sample_rate,
               float         *p_output_vectors,
               int           *p_output_count);
```

Performs the full DSP pipeline: PCM → STFT (n_fft=1024, hop=512) → 128-bin mel filterbank → log-dB → central crop [50:250) → sliding window (width=5) → 196 × 640 float32 vectors.

**Typical call:**

```c
#include "preprocessing.h"

/* ~10 seconds of 16 kHz mono PCM */
static int16_t pcm_buf[160000];

/* Output buffer: up to 196 vectors × 640 floats */
static float feature_vectors[196 * 640];
int num_vectors = 0;

preprocess(pcm_buf, 160000, 16000, feature_vectors, &num_vectors);
```

---

### 5.3 — Postprocessing — `postprocess()`

```c
#include "postprocessing.h"

float postprocess(const float *p_input_vectors,
                  const float *p_output_vectors,
                  int          num_vectors,
                  int          vector_dim);
```

Computes the clip-level anomaly score: mean MSE between the original input vectors and the model's reconstruction output across all sliding-window vectors.

**Typical call:**

```c
#include "postprocessing.h"

/* input_vectors: 196×640 from preprocess() */
/* recon_vectors: 196×640 from model output  */
float anomaly_score = postprocess(input_vectors, recon_vectors, 196, 640);

float threshold = 10.7f;  /* tune per machine ID */
if (anomaly_score >= threshold) {
    printf("ANOMALY detected! score=%.4f\n", anomaly_score);
} else {
    printf("Normal. score=%.4f\n", anomaly_score);
}
```

> [!NOTE]
> `postprocess()` returns a single float anomaly score. There is no argmax or class label — you must compare against a threshold to decide normal vs. anomalous.

---

## License / Provenance

| Component | Source | Licence |
|-----------|--------|---------|
| AD Autoencoder weights (ad01.h5) | [MLCommons Tiny](https://github.com/mlcommons/tiny) | Apache 2.0 |
| DCASE 2020 Task 2 ToyCar dataset | [Zenodo #3678171](https://zenodo.org/record/3678171) | CC-BY-SA 4.0 |
