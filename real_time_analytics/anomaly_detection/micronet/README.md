# Anomaly Detection — ARM MicroNet (DCASE 2020 Slider)

A compact convolutional classifier from the ARM ML-zoo (`ad_medium_int8.tflite`) used for unsupervised anomaly detection on machine audio. The model consumes 32×32 log-mel patches and emits 8 logits (one per training machine-id bucket); the per-window anomaly score is `-logit[machine_class]` and the clip score is the mean over sliding windows. This folder contains everything needed to obtain the pinned INT8 model, run inference, compile it for the Renesas RA8P1 (Cortex-M85 + Ethos-U55 NPU) using the RUHMI toolchain, and integrate it in embedded C.

---

## Model Details

| Field | Value |
|-------|-------|
| **Model name** | AD MicroNet Medium (INT8) |
| **Task** | Unsupervised anomaly detection — machine audio |
| **Framework** | TensorFlow Lite (pre-quantized INT8 upstream) |
| **Dataset** | DCASE 2020 Task 2 — Slider (dev-test, machine IDs 0/2/4/6) |
| **Input shape** | `(1, 32, 32, 1)` — 32×32 log-mel patch, INT8 |
| **Output shape** | `(1, 8)` — 8 class logits, INT8 |
| **Metric** | AUC / pAUC (max_fpr=0.1) — macro-averaged across machine IDs |
| **Source** | [ARM ML-zoo — Anomaly Detection (commit 7c32b09)](https://github.com/Arm-Examples/ML-zoo/tree/7c32b097f7d94aae2cd0b98a8ed5a3ba81e66b18/models/anomaly_detection) |

## Model Report Card

Evaluated on the DCASE 2020 Task 2 slider dev-test (four machine IDs, macro-averaged). Only an INT8 variant is published upstream — no FP32 reference exists.

| Model Variant | Format | AUC | pAUC (fpr=0.1) |
|---------------|--------|:---:|:--------------:|
| TFLite INT8  | `.tflite` | 0.958 | 87.95 |
| MERA INT8  | `.mera` | 0.958 | 87.94 |

> [!NOTE]
> The reference macro AUC of **0.963** is the value published by ARM ML-zoo in `definition.yaml` for this model. 

> [!NOTE]
> The model was compiled using `mera-2.6.0+pkg.4815` and `FSP 6.5.1` was used for building and testing the project.

### Inference Performance (RA8P1)

Measured on-target (Cortex-M85 @ 1 GHz, Ethos-U55 NPU @ 500 MHz). AI-only latency per 32×32 patch (single forward pass).

| Memory Configuration | CPU (ms) | NPU (ms) |
|----------------------|:--------:|:--------:|
| † Internal Flash + Internal SRAM | 33.975 | 1.261 |
---

## Folder Structure

```
anomaly_detection/
└── micronet/
    ├── README.md                   ← This file
    ├── python                      ← Inference scripts, download tool, config
    └── embedded_c                  ← Compiled C-code for MCU (CPU & NPU)
```

---

## Prerequisites

1. **Python 3.10** installed.
2. **Inference venv** — navigate to the `python/` directory and create a dedicated virtual environment:

    **Windows PowerShell**
    > **Note:** If venv activation is blocked by PowerShell execution policy ("running scripts is disabled"), run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in the same PowerShell window, then run the activation command again.

    ```powershell
    cd real_time_analytics\anomaly_detection\micronet\python
    py -3.10 -m venv .venv_micronet
    .\.venv_micronet\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    ```

    **Ubuntu / bash**

    ```bash
    cd real_time_analytics/anomaly_detection/micronet/python
    python3.10 -m venv .venv_micronet
    source .venv_micronet/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

3. **Compiler venv** (`.mera_venv`) — required only for [Step 4](#step-4--compile-for-ra8p1-ruhmi). See the [top-level README](../../../README.md) for setup instructions.

---

## Step 1 — Obtain the Model

The pre-quantized INT8 TFLite model is already provided in `python/model/`. To re-download it from scratch, proceed to [Step 2](#step-2--download-tflite).

---

## Step 2 — Download TFLite

Use `download_model.py` to fetch the pre-quantized INT8 TFLite model from the pinned ARM ML-zoo commit. Activate the **inference venv** and navigate to `python/`.

```bash
python download_model.py
```
---
The script:

- Downloads `ad_medium_int8.tflite` into `model/` and verifies its SHA-256
  against the hash pinned in `download_model.py` (`EXPECTED_INT8_SHA256`). There is no FP32
  model and no separate quantization step.


|Optional flags | Description |
|---------------|--------|
| --force-download	 | Re-download even if the file already exists on disk.| 
| --allow-unsupported-model	 | Skip the SHA-256 integrity check (only use when intentionally swapping in a non-pinned model) | 


---

## Step 3 — Run Inference (Python)

Activate the **inference venv** and navigate to `python/`:

  Downloads `dev_data_slider.zip` into `dataset/`, extracts it to
  `dataset/slider/{train,test}/`for checking the inference.


**Windows PowerShell**

```powershell
cd real_time_analytics\anomaly_detection\micronet\python
.\.venv_micronet\Scripts\Activate.ps1
python inference.py --model model\ad_medium_int8.tflite --wav sample.wav
```

**Ubuntu / bash**

```bash
cd real_time_analytics/anomaly_detection/micronet/python
source .venv_micronet/bin/activate
python inference.py --model model/ad_medium_int8.tflite --wav sample.wav
```

Optional flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | — | Path to `ad_medium_int8.tflite` (required) |
| `--wav` | — | Path to input WAV file (required) |
| `--machine-id` | auto-detect from filename | DCASE slider machine ID: `0`, `2`, `4`, or `6` |
| `--threshold` | `-7.2` | Anomaly threshold on the mean `-logit` score |
| `--quiet` | off | Print only `<score>  <verdict>` on a single line |

**Example output:**

```
Model      : model/ad_medium_int8.tflite
WAV        : sample.wav
Machine ID : 0  ->  output index 0
  window  1/13  logit[0]=+7.72859  score=-7.7286  top3_classes=[0, 4, 2]
  window  2/13  logit[0]=+7.72859  score=-7.7286  top3_classes=[0, 4, 2]
  ...
Windows scored : 13
Mean score     : -7.6759
Threshold      : -7.2000
Verdict        : normal
```

Quiet mode:

```
-7.6759  normal
```

> [!NOTE]
> `inference.py` uses `--machine-id` to select which of the 8 output logits is treated as the "expected class". If the WAV filename follows the DCASE convention (`..._id_XX_...`) the ID is auto-detected; otherwise pass `--machine-id` explicitly.

---

## Step 4 — Compile for RA8P1 (RUHMI)

This step converts the TFLite model into C-code for the RA8P1 MCU. Activate the **compiler venv** (`.mera_venv`).

### 4.1 — Edit the compile configuration

Open `python/config.yaml` and verify the paths match your system:

```yaml
model_path: real_time_analytics/anomaly_detection/micronet/python/model/ad_medium_int8.tflite
output_dir: real_time_analytics/anomaly_detection/micronet/embedded_c/src_mcu
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
python ruhmi_tools\mcu_compile.py real_time_analytics\anomaly_detection\micronet\python\config.yaml
```

**Ubuntu / bash**

```bash
cd ~/Model-zoo
python ruhmi_tools/mcu_compile.py real_time_analytics/anomaly_detection/micronet/python/config.yaml
```

The compiled C-code will be written to the `output_dir` specified in `config.yaml`.

---

## Step 5 — Embedded C Integration

The `embedded_c/` folder contains portable, board-independent files you can drop directly into any bare-metal or RTOS project.

| File | Purpose |
|------|---------|
| `model_metadata.h` | Compile-time constants — patch dimensions, quant params, machine-ID map |
| `preprocessing.h` / `preprocessing.c` | Extract 32×32 log-mel patches (sliding window + 2×2 mean pool + INT8 quantize) from raw PCM |
| `postprocessing.h` / `postprocessing.c` | Compute per-window and clip-level anomaly scores from the 8 INT8 logits |

> [!NOTE]
> These files have **no external dependencies** beyond `<stdint.h>`, `<math.h>`, and `<stddef.h>`. They compile cleanly with any C99-compatible toolchain (GCC, Clang, IAR, AC6).

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

### 5.2 — Preprocessing — `preprocess()`

```c
#include "preprocessing.h"

int32_t preprocess(const int16_t *p_pcm,
                   int32_t        _num_samples,
                   int32_t        _sample_rate,
                   int8_t        *p_patches_int8,
                   int32_t        _patches_capacity,
                   int32_t       *p_num_windows);
```

Performs the full DSP pipeline: PCM → STFT (`n_fft = PREP_N_FFT` = 1024, `hop = PREP_HOP` = 512, Hann periodic window, `center=False`) → `PREP_N_MELS` = 64-bin mel filterbank (Slaney norm, `fmin = 0`, `fmax = 8000`) → `10 * log10(max(spec, tiny))` with training-mean shift (`PREP_TRAINING_MEAN = -30.0`) → sliding window over the log-mel matrix (width `PREP_PATCH_RAW_FRAMES` = 64 frames, stride `PREP_OUTER_STRIDE` = 20) → 2×2 mean-pool to `PATCH_MODEL_DIM × PATCH_MODEL_DIM` (32×32) → INT8 quantise using `MODEL_INPUT_SCALE` / `MODEL_INPUT_ZP`. Returns `0` on success and `-1` on error (null pointer, buffer too small, clip too short, or clip exceeds `PREP_MAX_PCM_SAMPLES`); the number of windows written is returned via `p_num_windows`.

**Typical call:**

```c
#include "preprocessing.h"
#include "model_metadata.h"

/* Up to 10 seconds of 16 kHz mono PCM. */
static int16_t pcm_buf[PREP_MAX_PCM_SAMPLES];  /* 160000 */

/* Output buffer: up to PREP_MAX_WINDOWS patches of INPUT_DIM int8 samples each. */
static int8_t  patch_buf[PREP_MAX_WINDOWS * INPUT_DIM];
int32_t        num_windows = 0;

int32_t rc = preprocess(pcm_buf,
                        /*_num_samples      =*/ PREP_MAX_PCM_SAMPLES,
                        /*_sample_rate      =*/ PREP_SAMPLE_RATE,
                        patch_buf,
                        /*_patches_capacity =*/ (int32_t)sizeof(patch_buf),
                        &num_windows);
if (0 != rc) {
    /* handle preprocessing error */
}
```

---

### 5.3 — Postprocessing — `postprocess()`

```c
#include "postprocessing.h"

int32_t postprocess(const int8_t *p_outputs_int8,
                    int32_t       _num_windows,
                    int32_t       _machine_id,
                    float        *p_clip_score);
```

Dequantises the `OUTPUT_DIM` = 8 output logits per window using `MODEL_OUTPUT_SCALE` / `MODEL_OUTPUT_ZP`, maps the DCASE slider machine ID (`0`, `2`, `4`, or `6`) to its target output index internally (`0 → 0`, `2 → 1`, `4 → 2`, `6 → 3`), computes each per-window anomaly score as `-logit[target_index]`, and writes the mean over all sliding windows into `*p_clip_score` as the clip-level score. Higher score = more anomalous. Returns `0` on success and `-1` on error (null pointer, no windows, or unsupported machine ID).

**Typical call:**

```c
#include "postprocessing.h"
#include "model_metadata.h"

/* logits: num_windows × OUTPUT_DIM int8 output from the model. */
float   clip_score  = 0.0f;
int32_t machine_id  = 0;    /* DCASE slider machine ID: 0, 2, 4, or 6 */

int32_t rc = postprocess(logits, num_windows, machine_id, &clip_score);
if (0 != rc) {
    /* handle postprocessing error */
}

if (ANOMALY_THRESHOLD < clip_score) {
    printf("ANOMALY detected! score=%.4f\n", clip_score);
} else {
    printf("Normal. score=%.4f\n", clip_score);
}
```

> [!NOTE]
> `postprocess()` writes a single float clip-level anomaly score to `*p_clip_score`. There is no argmax or class label — compare against `ANOMALY_THRESHOLD` (default **-7.2**, defined in `model_metadata.h`) to decide normal vs. anomalous. The threshold and per-ID score distributions are documented at the top of `python/inference.py`.

---

## License / Provenance

| Component | Source | Licence |
|-----------|--------|---------|
| AD MicroNet Medium weights (`ad_medium_int8.tflite`) | [ARM ML-zoo](https://github.com/Arm-Examples/ML-zoo) | Apache 2.0 |
| DCASE 2020 Task 2 Slider dataset | [Zenodo #3678171](https://zenodo.org/record/3678171) | CC-BY-SA 4.0 |