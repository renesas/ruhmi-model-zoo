# TinyWav2letter — Speech Recognition (Fluent Speech Commands)

TinyWav2letter is a pruned, lightweight speech-to-text network derived from the Wav2Letter architecture ([Collobert et al., 2016](https://arxiv.org/abs/1609.03193)). It operates on 39-dimensional MFCC+delta+delta-delta features and emits CTC character logits over a 29-class alphabet (`a–z`, `'`, space, blank). This folder contains everything needed to obtain the model, run inference, compile it for the Renesas RA8P1 (Cortex-M85 CPU) using the RUHMI toolchain, and integrate it into embedded firmware.

> [!NOTE]
> Both **CPU-only** (`embedded_c/src_mcu/`) and **CPU + Ethos-U55 NPU** (`embedded_c/src_mcu_npu/`) MERA-generated kernels are provided.

---

## Model Details

| Field | Value |
|-------|-------|
| **Model name** | TinyWav2letter (Arm ML-zoo, pruned) |
| **Task** | Speech recognition — streaming CTC character-level transcription |
| **Framework** | TensorFlow SavedModel → TFLite |
| **Dataset** | Fluent Speech (trianed on LibriSpeech,Mini LibrySpeech,Fluent Speech) |
| **Input shape** | `[1, 296, 39]` — one sliding window of MFCC+delta+delta-delta features |
| **Output shape** | `[1, 1, 148, 29]` — CTC logits (148 frames × 29 classes) |
| **Source** | [Arm ML-zoo](https://github.com/ARM-software/ML-zoo) (Apache-2.0) |

> [!NOTE]
> The model input is a **[296, 39] quantized MFCC feature window**, not raw audio. MFCC extraction (13 coefficients + delta + delta-delta, 16 kHz, n_fft=512, hop=160) must be applied before calling the model.

---

## Model Report Card

Accuracy measured on the Fluent Speech Commands **validation split** (3,118 utterances).

| Model Variant | Format | Mean LER | Mean WER | Exact Match |
|---------------|--------|:--------:|:--------:|:-----------:|
| TFLite FP32 | `.tflite` | 9.55 % | 24.46 % | 53.78 % |
| TFLite INT8 | `.tflite` | 9.78 % | 24.96 % | 52.47 % |
| MERA FP32 | `.mera` | 9.55 % | 24.46 % | 53.78 % |
| MERA TFLite INT8 | `.mera` | 9.71 % | 25.01 % | 52.79 % |

LER = Letter Error Rate; WER = Word Error Rate; Exact Match = fraction of utterances decoded without any character error.

> [!NOTE]
> The model was compiled using `mera-2.6.0+pkg.4815` and `FSP 6.5.1` was used for building and testing the project.

### Inference Performance (RA8P1)

Measured on-target (Cortex-M85 @ 1 GHz, Ethos-U55 NPU @ 500 MHz). AI-only latency per 10 ms frame.

| Memory Configuration | CPU (ms) | NPU (ms) |
|----------------------|:--------:|:--------:|
| † Internal Flash + Internal SRAM | 943.08  | 43.663 |

---

## Folder Structure

```
tinywav2letter/
├── README.md                      ← This file
├── python/                        ← Inference scripts and model files
│   ├── download_model.py          ← Downloads FP32 SavedModel + INT8 from Arm ML-zoo, generates FP32 TFLite
│   ├── inference.py               ← Single-file transcription + optional LER/WER
│   ├── config.yaml                ← RUHMI compiler configuration
│   ├── requirements.txt           ← Pinned Python dependencies
│   └── model/
│       ├── tiny_wav2letter_pruned_fp32.tflite
│       └── tiny_wav2letter_pruned_int8.tflite
└── embedded_c/                    ← Embedded C code for MCU deployment
    ├── model_metadata.h           ← Compile-time tensor / quantisation constants
    ├── mfcc_librosa.h / .c        ← Librosa-compatible MFCC front end (C99)
    ├── preprocessing.h / .c       ← PCM → [296, 39] int8 feature tensor
    ├── postprocessing.h / .c      ← [1, 1, 148, 29] int8 → transcript (greedy CTC)
    ├── src_mcu/                   ← CPU-only CMSIS-NN code-gen (rnnoise_INT8_CPU)
    └── src_mcu_npu/               ← NPU-accelerated code-gen (rnnoise_INT8_NPU)
```

---

## Prerequisites

> [!IMPORTANT]
> **Fluent Speech Commands** is released under a non-commercial research licence and is **not bundled with this repository**. To validate accuracy, obtain it directly from the dataset authors: <https://fluent.ai/fluent-speech-commands-a-dataset-for-spoken-language-understanding-research/>. Do not redistribute.

1. **Python 3.10** installed.
2. **Git LFS** — required to pull `.tflite` model files (if cloned without LFS, run `git lfs pull`).
3. **Inference venv** — create a dedicated virtual environment from `python/`:

   **Ubuntu / bash**

   ```bash
   cd audio/speech_recognition/tinywav2letter/python
   python3.10 -m venv .venv_tinywav2letter
   source .venv_tinywav2letter/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   **Windows PowerShell**

   ```powershell
   cd audio\speech_recognition\tinywav2letter\python
   py -3.10 -m venv .venv_tinywav2letter
   .\.venv_tinywav2letter\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

4. **Compiler venv** (`.mera_venv`) — required only for [Step 4](#step-4--compile-for-ra8p1-ruhmi). See the [top-level README](../../../README.md) for setup instructions.

---

## Step 1 — Obtain the Model

The TFLite models are already provided in `python/model/`. To regenerate them from scratch, proceed to [Step 2](#step-2--build-the-models).

---

## Step 2 — Build the Models

`download_model.py` performs a sparse clone of the [Arm ML-zoo](https://github.com/ARM-software/ML-zoo), extracts the pruned TinyWav2letter artifacts, and converts the SavedModel to FP32 TFLite.

Activate the **inference venv** and navigate to `python/`:

**Both FP32 + INT8 (default)**

```bash
python download_model.py
```

**FP32 only** (no INT8 copy):

```bash
python download_model.py --mode fp32
```

**INT8 only** (copy pre-quantized INT8 TFLite, skip SavedModel conversion):

```bash
python download_model.py --mode int8
```

> [!NOTE]
> `download_model.py` requires `git`, `git sparse-checkout`, and optionally `git-lfs` to fetch LFS-tracked binaries from the Arm ML-zoo. The clone is deleted after copying.

---

## Step 3 — Run Inference (Python)

Activate the **inference venv** and navigate to `python/`:

> [!NOTE]
> No sample audio is bundled because [Fluent Speech Commands](https://fluent.ai/fluent-speech-commands-a-dataset-for-spoken-language-understanding-research/) is a non-commercial research dataset. Provide your own 16 kHz mono WAV/FLAC file from the official dataset release or another licensed source.

**Windows PowerShell**

```powershell
cd audio\speech_recognition\tinywav2letter\python
.\.venv_tinywav2letter\Scripts\Activate.ps1
```

**Ubuntu / bash**

```bash
cd audio/speech_recognition/tinywav2letter/python
source .venv_tinywav2letter/bin/activate
```

**Transcribe a WAV file**

```bash
python inference.py \
  --model model/tiny_wav2letter_pruned_int8.tflite \
  --wav /path/to/your_audio.wav
```

**Run FP32 model**

```bash
python inference.py \
  --model model/tiny_wav2letter_pruned_fp32.tflite \
  --wav /path/to/your_audio.wav
```

**With optional LER/WER evaluation**

```bash
python inference.py \
  --model model/tiny_wav2letter_pruned_int8.tflite \
  --wav /path/to/your_audio.wav \
  --transcript "TURN SOUND UP"
```

Flags:

| Flag | Required | Description |
|------|:--------:|-------------|
| `--model` | ✓ | Path to `.tflite` model file |
| `--wav` | ✓ | Input audio file (`.wav` or `.flac`) at 16 kHz mono |
| `--transcript` | — | Ground-truth transcript for LER/WER (case-insensitive) |

**Example output (INT8 model, sample 1):**

```
Input quantization: scale=0.17129258811473846, zero_point=4
Output quantization: scale=0.4950784146785736, zero_point=5

Transcription : pause the music
```

**Example output (INT8 model, sample 2):**

```
Input quantization: scale=0.17129258811473846, zero_point=4
Output quantization: scale=0.4950784146785736, zero_point=5

Transcription : turn the heat down in the kitchen
```

---

## Step 4 — Compile for RA8P1 (RUHMI)

This step converts the TFLite INT8 model into C-code for the RA8P1 MCU. Activate the **compiler venv** (`.mera_venv`).

### 4.1 — Edit the compile configuration

Open `python/config.yaml` and verify the paths match your system:

```yaml
model_path: audio/speech_recognition/tinywav2letter/python/model/tiny_wav2letter_pruned_int8.tflite
output_dir: audio/speech_recognition/tinywav2letter/embedded_c/src_mcu
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
python ruhmi_tools\mcu_compile.py audio\speech_recognition\tinywav2letter\python\config.yaml
```

**Ubuntu / bash**

```bash
cd ~/Model-zoo
python ruhmi_tools/mcu_compile.py audio/speech_recognition/tinywav2letter/python/config.yaml
```

The compiled C-code will be written to the `output_dir` specified in `config.yaml`.

---

## Step 5 — Embedded C Integration

The `embedded_c/` folder contains portable, board-independent files you can drop into any bare-metal or RTOS project.

| File | Purpose |
|------|--------|
| `model_metadata.h` | Compile-time constants — tensor shapes, quant params, CTC alphabet, MERA scratch buffer sizes |
| `mfcc_librosa.h` / `mfcc_librosa.c` | Librosa-compatible MFCC + delta + delta-delta front end (C99) |
| `preprocessing.h` / `preprocessing.c` | Convert int16 mono PCM → quantized `[296, 39]` int8 tensor |
| `postprocessing.h` / `postprocessing.c` | Greedy CTC decode: int8 logits `[1, 1, 148, 29]` → null-terminated transcript |

> [!NOTE]
> All files compile with any C99-compatible toolchain (GCC, Clang, IAR, AC6). Dependencies: `<stdint.h>`, `<math.h>`.

---

### 5.1 — Add files to your project

Copy the following files into your firmware project (or add them to your include paths):

```
embedded_c/
├── model_metadata.h
├── mfcc_librosa.h
├── mfcc_librosa.c
├── preprocessing.h
├── preprocessing.c
├── postprocessing.h
└── postprocessing.c
```

Also copy the compiled model artifacts from the appropriate subdirectory:

- **CPU-only (CMSIS-NN):** `embedded_c/src_mcu/`
- **NPU-accelerated (Ethos-U55):** `embedded_c/src_mcu_npu/`

---

### 5.2 — `model_metadata.h` — Key constants

```c
#include "model_metadata.h"
```

| Macro | Value | Description |
|-------|-------|-------------|
| `MODEL_INPUT_SIZE` | `11544` | Flat size of input tensor (1 × 296 × 39) |
| `MODEL_INPUT_TIME` | `296` | MFCC frames per inference window |
| `MODEL_INPUT_FEATURES` | `39` | Feature dimensions (13 MFCC + 13 Δ + 13 ΔΔ) |
| `MODEL_OUTPUT_SIZE` | `4292` | Flat size of output tensor (1 × 1 × 148 × 29) |
| `MODEL_OUTPUT_TIME` | `148` | CTC output frames per window |
| `MODEL_OUTPUT_CLASSES` | `29` | Alphabet size incl. blank |
| `MODEL_CTC_BLANK_INDEX` | `28` | CTC blank token index |
| `MODEL_INPUT_SCALE` | `0.17129258811473846f` | Input quantisation scale |
| `MODEL_INPUT_ZERO_POINT` | `4` | Input quantisation zero-point |
| `MODEL_OUTPUT_SCALE` | `0.4950784146785736f` | Output dequantisation scale |
| `MODEL_OUTPUT_ZERO_POINT` | `5` | Output dequantisation zero-point |
---

### 5.3 — Preprocessing — `preprocess()`

**API:**

```c
#include "preprocessing.h"

preprocess_status_t preprocess(const int16_t *p_pcm_in,
                               uint32_t _pcm_sample_count,
                               int8_t p_out_q[MODEL_INPUT_SIZE],
                               uint32_t *p_out_time_frames);
```

Accepts up to `TW2L_PCM_MAX_SAMPLES` (48,000 samples / ~3 s) of int16 mono PCM at 16 kHz. Extracts MFCC + delta + delta-delta (39 features), right-pads to `MODEL_INPUT_TIME` (296) frames, and quantizes to int8. Returns `PREPROCESS_ERR_TOO_LONG` when input exceeds the limit.

**Typical call:**

```c
#include "model_metadata.h"
#include "preprocessing.h"

static int8_t s_input[MODEL_INPUT_SIZE];   /* 11544 bytes */

uint32_t frames;
preprocess_status_t ret = preprocess(pcm_buf, n_samples, s_input, &frames);
/* s_input is ready to pass to compute_sub_0000() */
```

---

### 5.4 — Postprocessing — `postprocess()`
**API:**


```c
#include "postprocessing.h"

postprocess_status_t postprocess(const int8_t p_out_q[MODEL_OUTPUT_SIZE],
                                 char *p_transcript_out,
                                 uint32_t _transcript_capacity,
                                 uint32_t *p_out_length);
```

Performs greedy CTC decode (argmax per frame → collapse repeats → drop blank index `MODEL_CTC_BLANK_INDEX`) and writes a null-terminated transcript string.

**Typical call:**

```c
#include "model_metadata.h"
#include "postprocessing.h"

static int8_t s_output[MODEL_OUTPUT_SIZE];   /* 4292 bytes */
static char   s_transcript[128];

/* After calling compute_sub_0000(s_input, s_output, scratch, scratch_bytes): */
uint32_t len;
postprocess_status_t ret = postprocess(s_output, s_transcript, sizeof(s_transcript), &len);
/* s_transcript now contains the decoded text, e.g. "pause the music" */
```

---

## License / Provenance

| Component | Source | Licence |
|-----------|--------|---------|
| TinyWav2letter model weights | [Arm ML-zoo](https://github.com/ARM-software/ML-zoo) | Apache 2.0 |
| Fluent Speech Commands dataset | [fluent.ai](https://fluent.ai/fluent-speech-commands-a-dataset-for-spoken-language-understanding-research/) | Non-commercial research only |
