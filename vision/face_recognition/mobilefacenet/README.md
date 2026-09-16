# MobileFaceNet — Face Recognition (LFW)

MobileFaceNet ([Chen et al., 2018](https://arxiv.org/abs/1804.07573)) is a compact convolutional network that produces a 128-D face embedding from a 112×112 MTCNN-aligned RGB face crop. Two faces are compared by cosine similarity of their L2-normalised embeddings. This folder contains everything needed to obtain the model, convert it to TFLite, compile it for the Renesas RA8P1 (Cortex-M85 + Ethos-U55 NPU) using the RUHMI toolchain, and run inference.

---

## Model Details

| Field | Value |
|-------|-------|
| **Model name** | MobileFaceNet (foamliu/MobileFaceNet) |
| **Task** | Face Recognition (1:1 verification) |
| **Framework** | PyTorch TorchScript → ONNX → TFLite |
| **Training dataset** | MS-Celeb-1M |
| **Evaluation dataset** | LFW (6 000-pair 10-fold verification protocol) |
| **Input shape** | `(1, 112, 112, 3)` — NHWC, MTCNN-aligned RGB, normalised to [−1, 1] |
| **Output shape** | `(1, 128)` — raw face embedding (L2-normalise before cosine comparison) |
| **Source** | [foamliu/MobileFaceNet](https://github.com/foamliu/MobileFaceNet) |

## Model Report Card

10-fold LFW verification accuracy (6 000 pairs, official protocol).

| Model Variant | Format | 10-fold Acc (%) | ROC AUC | TAR @ FAR=0.001 (%) |
|---------------|--------|:---------------:|:-------:|:-------------------:|
| TFLite FP32 | `.tflite` | 97.03 ± 0.46 | 0.9750 | 92.96 |
| TFLite INT8 | `.tflite` | 96.90 ± 0.41 | 0.9750 | 93.00 |
| MERA FP32 | C-code | 97.03 ± 0.46 | 0.9750 | 92.96 |
| MERA INT8 (TFLite INT8 deployed) | C-code | 96.97 ± 0.46 | 0.9750 | 93.10 |

> [!NOTE]
> The model was compiled using `mera-2.6.0+pkg.4815` and `FSP 6.5.1` was used for building and testing the project.

### Inference Performance (RA8P1)

Measured on-target (Cortex-M85 @ 1 GHz, Ethos-U55 NPU @ 500 MHz). AI-only latency per face embedding inference.

| Memory Configuration | CPU (ms) | NPU (ms) |
|----------------------|:--------:|:--------:|
|  OSPI + SRAM | 411  | 20 |
---

## Folder Structure

```
mobilefacenet/
├── README.md                    ← This file
├── python/                      ← Inference scripts, conversion tools, config
│   ├── download_model.py        ← Download pretrained weights and convert to TFLite
│   ├── inference.py             ← Single/pair TFLite inference (FP32 or INT8)
│   ├── config.yaml              ← RUHMI compiler configuration
│   ├── requirements.txt         ← Python dependencies
│   ├── model/                   ← Pre-built TFLite models
│   └── utils/
│       ├── align.py             ← MTCNN face alignment (112×112, ArcFace template)
│       └── progress.py          ← Spinner helper
└── embedded_c/                  ← Compiled C-code for MCU (CPU & NPU)
    ├── model_metadata.h         ← Compile-time constants (shape, quant params, threshold)
    ├── preprocessing.h / .c     ← Normalize uint8 face crop + quantize to INT8
    ├── postprocessing.h / .c    ← INT8 dequantize + L2 normalise + cosine similarity
    ├── src_mcu/                 ← CPU-only CMSIS-NN build
    └── src_mcu_npu/             ← Ethos-U55 NPU-accelerated build
```

---

## Prerequisites

> [!IMPORTANT]
> **LFW (Labeled Faces in the Wild) images are for academic research only and are NOT redistributable under a commercial licence.**
> To convert to INT8, you must supply your own face images (e.g. from LFW deep-funneled obtained from [TensorFlow Datasets (TFDS) builder](https://ndownloader.figshare.com/files/5976018)).
> Pass the directory with `--calib-dir`.

1. **Python 3.10** installed.
2. **Inference venv** — navigate to `python/` and create a dedicated virtual environment:

    **Windows PowerShell**
    > **Note:** If venv activation is blocked by PowerShell execution policy ("running scripts is disabled"), run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in the same PowerShell window, then run the activation command again.

    ```powershell
    cd vision\face_recognition\mobilefacenet\python
    py -3.10 -m venv .venv_mobilefacenet
    .\.venv_mobilefacenet\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    ```

    **Ubuntu / bash**

    ```bash
    cd vision/face_recognition/mobilefacenet/python
    python3.10 -m venv .venv_mobilefacenet
    source .venv_mobilefacenet/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

3. **Compiler venv** (`.mera_venv`) — required only for [Step 4](#step-4--compile-for-ra8p1-ruhmi). See the [top-level README](../../../README.md) for setup instructions.

---

## Step 1 — Obtain the Model

The TFLite models are already provided in `python/model/`. To regenerate them from scratch, proceed to [Step 2](#step-2--convert-to-tflite).

---

## Step 2 — Build the Models

Use `download_model.py` to clone the foamliu/MobileFaceNet repository, export pretrained weights, and convert to TFLite. Activate the **inference venv** and navigate to `python/`.

**FP32 only** — no calibration images required:

```bash
python download_model.py --mode fp32
```

**INT8 only** — requires a directory of face images:

```bash
python download_model.py --mode int8 --calib-dir /path/to/lfw-deepfunneled
```

**Both FP32 + INT8** (default):

```bash
python download_model.py --calib-dir /path/to/lfw-deepfunneled
```

Use `--calib-num N` to limit calibration to N face images (default: 1000). Output files are written to `python/model/`.

> [!NOTE]
> `download_model.py` loads the upstream TorchScript checkpoint via `torch.jit.load` (no model source code needed), exports to ONNX, then converts to TFLite via `onnx2tf`. MTCNN is used to detect and align each calibration face to the canonical ArcFace 112×112 template before quantization.

---

## Step 3 — Run Inference (Python)

Activate the **inference venv** and navigate to `python/`:

**Windows PowerShell**

```powershell
cd vision\face_recognition\mobilefacenet\python
.\.venv_mobilefacenet\Scripts\Activate.ps1
python inference.py --image C:\path\to\face.jpg
```

**Ubuntu / bash**

```bash
cd vision/face_recognition/mobilefacenet/python
source .venv_mobilefacenet/bin/activate
python inference.py --image /path/to/face.jpg
```

**1:1 Verification** — compare two face images:

```bash
python inference.py --image /path/to/face_a.jpg \
                    --reference /path/to/face_b.jpg
```

Optional flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `model/mobilefacenet_INT8.tflite` | Path to TFLite model |
| `--threshold` | `0.28` | Cosine-similarity match threshold |
| `--verbose` | off | Print first 10 embedding values |

Pass `--model model/mobilefacenet_FP32.tflite` to use the FP32 variant.

**Example output (INT8 default model):**

```
  Model  : mobilefacenet_INT8.tflite
  Input  : [  1 112 112   3] int8
  Output : [  1 128] int8

Embedding extracted in 83.8 ms (shape=(128,), ||x||=1.0000)

1:1 verification : MATCH
  cosine similarity : 0.5759
  threshold         : 0.28
```

---

## Step 4 — Compile for RA8P1 (RUHMI)

This step converts the TFLite INT8 model into C-code for the RA8P1 MCU. Activate the **compiler venv** (`.mera_venv`).

### 4.1 — Edit the compile configuration

Open `python/config.yaml` and verify the paths match your system:

```yaml
model_path: vision/face_recognition/mobilefacenet/python/model/mobilefacenet_INT8.tflite
output_dir: vision/face_recognition/mobilefacenet/embedded_c/src_mcu
target: cpu        # 'cpu' → CMSIS-NN  |  'npu' → Ethos-U55 NPU
quantize: false    # model is already INT8
external: false
```

> [!IMPORTANT]
> Use **absolute paths** if running the compiler from outside the repo root.

> [!TIP]
> For NPU deployment set `target: npu`.

### 4.2 — Run the compiler

Navigate back to the **repository root** and run the compiler with `.mera_venv` active:

**Windows PowerShell**

```powershell
cd C:\Users\<you>\Model-zoo
python ruhmi_tools\mcu_compile.py vision\face_recognition\mobilefacenet\python\config.yaml
```

**Ubuntu / bash**

```bash
cd ~/Model-zoo
python ruhmi_tools/mcu_compile.py vision/face_recognition/mobilefacenet/python/config.yaml
```

The compiled C-code will be written to the `output_dir` specified in `config.yaml`.

---

## Step 5 — Embedded C Integration

The `embedded_c/` folder contains portable, board-independent files for bare-metal or RTOS firmware.

| File | Purpose |
|------|---------|
| `model_metadata.h` | All compile-time constants — input/output shape, quantization parameters, embedding dimension, match threshold, memory footprint |
| `preprocessing.h` / `preprocessing.c` | Normalize uint8 face crop and quantize to INT8 |
| `postprocessing.h` / `postprocessing.c` | INT8 dequantize, L2 normalize, cosine similarity + match decision |

> [!NOTE]
> All files compile cleanly with any C99-compatible toolchain (GCC, Clang, IAR, AC6). Dependencies: `<math.h>` (for `lroundf`, `sqrtf`), `<stdint.h>`, `<stdbool.h>`, `<stddef.h>`.

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
| `MODEL_INPUT_H` | `112` | Input face height (pixels) |
| `MODEL_INPUT_W` | `112` | Input face width (pixels) |
| `MODEL_INPUT_C` | `3` | RGB channels |
| `MODEL_INPUT_SIZE` | `37632` | Total input elements (112 × 112 × 3) |
| `MODEL_OUTPUT_SIZE` | `128` | Embedding dimension |
| `EMBEDDING_DIM` | `128` | Alias for `MODEL_OUTPUT_SIZE` |
| `INPUT_QUANT_SCALE` | `0.007843f` | INT8 input quantisation scale (≈ 2/255) |
| `INPUT_QUANT_ZP` | `-1` | INT8 input zero-point |
| `OUTPUT_QUANT_SCALE` | `0.054182f` | INT8 output dequantisation scale |
| `OUTPUT_QUANT_ZP` | `5` | INT8 output zero-point |
| `OUTPUT_HAS_L2_NORM` | `0` | L2 norm **not** baked in — apply in postprocess() |
| `MATCH_THRESHOLD` | `0.28f` | Cosine-similarity match threshold (tuned on LFW) |

---

### 5.3 — Preprocessing — `preprocess()`

**API:**

```c
#include "preprocessing.h"

void preprocess(const uint8_t *p_source_image_hwc,
                int8_t        *p_input_tensor);
```

Accepts a **pre-aligned 112×112 RGB face** (stored as interleaved `uint8_t` in HWC order, exactly `MODEL_INPUT_SIZE` bytes). The input must already be MTCNN-aligned to the ArcFace canonical template — on-device alignment is not included.

Two-step pipeline per pixel:
1. Normalise: `float_norm = pixel / 127.5f − 1.0f` → [−1.0, 1.0]
2. Quantize: `q = clamp(round(float_norm / INPUT_QUANT_SCALE) + INPUT_QUANT_ZP, −128, 127)`

**Typical call:**

```c
#include "model_metadata.h"
#include "preprocessing.h"

/* Pre-aligned 112×112 RGB face (from host-side MTCNN or fixed-crop pipeline) */
static uint8_t aligned_face[MODEL_INPUT_H * MODEL_INPUT_W * MODEL_INPUT_C];

/* Quantized int8 buffer fed to the model input tensor */
static int8_t nn_input[MODEL_INPUT_SIZE];

/* --- inside your inference loop --- */
preprocess(aligned_face, nn_input);
```

---

### 5.4 — Postprocessing — `postprocess()` and `postprocess_verify()`

**API:**

```c
#include "postprocessing.h"

void postprocess(const int8_t *p_raw_q,
                 float        *p_embedding,
                 int32_t       _dim);

bool postprocess_verify(const float *p_a,
                        const float *p_b,
                        int32_t      _dim,
                        float        _threshold,
                        float       *p_out_sim);
```

**`postprocess()`** — per-face: dequantizes the raw INT8 model output to `float[128]`, then L2-normalises in-place.

1. Dequantize: `f[i] = ((float)q[i] − OUTPUT_QUANT_ZP) × OUTPUT_QUANT_SCALE`
2. L2 normalise: `f[i] /= sqrt(sum(f[j]²))`

**`postprocess_verify()`** — per-pair: computes the cosine similarity of two L2-normalised embeddings and returns `true` if the similarity meets or exceeds `_threshold`.

Cosine similarity = dot product of unit-norm embeddings.

**Typical call — 1:1 face verification:**

```c
#include "model_metadata.h"
#include "postprocessing.h"

/* Raw INT8 outputs from two model runs */
int8_t  raw_emb_a[MODEL_OUTPUT_SIZE];
int8_t  raw_emb_b[MODEL_OUTPUT_SIZE];

/* Float buffers for dequantized + L2-normalised embeddings */
float   emb_a[MODEL_OUTPUT_SIZE];
float   emb_b[MODEL_OUTPUT_SIZE];

/* --- after running the model for face A --- */
postprocess(raw_emb_a, emb_a, EMBEDDING_DIM);

/* --- after running the model for face B --- */
postprocess(raw_emb_b, emb_b, EMBEDDING_DIM);

/* --- 1:1 verification --- */
float sim;
bool  is_match = postprocess_verify(emb_a, emb_b, EMBEDDING_DIM,
                                    MATCH_THRESHOLD, &sim);
/* is_match == true  → same identity   (sim >= 0.28) */
/* is_match == false → different identity             */
```

---

## License / Provenance

| Component | Source | Licence |
|-----------|--------|---------|
| MobileFaceNet weights | [foamliu/MobileFaceNet](https://github.com/foamliu/MobileFaceNet) | Apache-2.0 |
| MTCNN (calibration preprocessing) | [facenet-pytorch](https://github.com/timesler/facenet-pytorch) | MIT |
| LFW dataset (validation) | [vis-www.cs.umass.edu/lfw](http://vis-www.cs.umass.edu/lfw/) | Research use only |
