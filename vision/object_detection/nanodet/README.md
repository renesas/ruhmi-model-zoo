# NanoDet-Plus-m — Object Detection (COCO)

NanoDet-Plus-m is a lightweight anchor-free object detector from the [NanoDet project](https://github.com/RangiLyu/nanodet) by RangiLyu, using a Generalized Focal Loss (GFL) head over an FPN backbone. This implementation runs at 320×320 (416×416 also supported) on 80 COCO classes and produces detections through per-class NMS on a 2125-anchor multi-scale grid (strides 8/16/32/64). This folder contains everything needed to obtain the model, convert it to ONNX / TFLite, compile it for the Renesas RA8P1 (Cortex-M85 + Ethos-U55 NPU) using the RUHMI toolchain, and run inference.

---

## Model Details

| Field | Value |
|-------|-------|
| **Model name** | NanoDet-Plus-m |
| **Task** | Object Detection (bounding box + class) |
| **Framework** | PyTorch → ONNX → TFLite (via `onnx2tf`) |
| **Dataset** | COCO val2017 (80 classes) |
| **Input shape** | `(1, 320, 320, 3)` NHWC (TFLite) / `(1, 3, 320, 320)` NCHW (ONNX); 416×416 also supported |
| **Output shape** | `(1, 2125, 112)` — 80 class logits + 32 regression bins (8 bins × 4 sides) per anchor |
| **Anchors / Strides** | 2125 anchors on strides 8 / 16 / 32 / 64 (40² + 20² + 10² + 5²) |
| **Preprocessing** | Warp resize (no letterbox), BGR mean `[103.53, 116.28, 123.675]` / std `[57.375, 57.12, 58.395]` |
| **Source** | [NanoDet by RangiLyu](https://github.com/RangiLyu/nanodet) |

## Model Report Card

Accuracy measured on the COCO val2017 set (5,000 images, 80 classes). Metric: COCO AP@[.5:.95] and AP@0.50.

| Model Variant | Format | AP@[.5:.95] | AP@0.50 |
|---------------|--------|:-----------:|:-------:|
| TFLite FP32 | `.tflite` | 33.71 | 49.65 |
| TFLite INT8 | `.tflite` | 29.01 | 43.74 |
| MERA FP32 | `.mera` | 33.71 | 49.65 |
| MERA INT8 (TFLite Quantized) | `.mera` | 28.79 | 43.13 |

## Inference Performance (RA8P1)

> [!NOTE]
> The model was compiled using `mera-2.6.0+pkg.4815` and `FSP 6.5.1` was used for building and testing the project.

Measured on-target (Cortex-M85 @ 1 GHz, Ethos-U55 NPU @ 500 MHz). AI-only latency per 320×320 frame.

| Memory Configuration | CPU (ms) | NPU (ms) |
|----------------------|:--------:|:--------:|
| ‡ OSPI + External SDRAM | 3520  | 1000 |

---

## Folder Structure

```
object_detection
└── nanodet/
    ├── README.md                   ← This file
    ├── python                      ← Inference scripts, conversion tools, config
    └── embedded_c                  ← Compiled C-code for MCU (CPU & NPU)
```

---

## Prerequisites

1. **Python 3.10** installed.
2. **Inference venv** — navigate to the `python/` directory and create a dedicated virtual environment:

    **Windows PowerShell**
    > **Note:** If venv activation is blocked by PowerShell execution policy ("running scripts is disabled"), run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in the same PowerShell window, then run the activation command again.

    ```powershell
    cd vision\object_detection\nanodet\python
    py -3.10 -m venv .venv_nanodet
    .\.venv_nanodet\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    ```

    **Ubuntu / bash**

    ```bash
    cd vision/object_detection/nanodet/python
    python3.10 -m venv .venv_nanodet
    source .venv_nanodet/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

3. **Compiler venv** (`.mera_venv`) — required only for [Step 4](#step-4--compile-for-ra8p1-ruhmi). See the [top-level README](../../../README.md) for setup instructions.

---

## Step 1 — Obtain the Model

The ONNX (FP32 + INT8) and TFLite (FP32 + INT8) models are already provided in `python/model/`. To regenerate them from scratch, proceed to [Step 2](#step-2--convert-to-tflite).

---

## Step 2 — Convert to TFLite

Use `download_model.py` to build and convert the model. Activate the **inference venv** and navigate to `python/`.

**Full pipeline (ONNX FP32 + ONNX INT8 + TFLite FP32 + TFLite INT8, default)**

```bash
python download_model.py --mode all
```

**Selected outputs**

```bash
python download_model.py --mode fp32           # ONNX FP32 only
python download_model.py --mode int8           # ONNX FP32 + ONNX INT8
python download_model.py --mode tflite         # ONNX FP32 + TFLite FP32 / INT8
```

**Custom input size (416 also supported)**

```bash
python download_model.py --mode all --input-size 416
```

**Custom calibration images (defaults to COCO val2017 sample)**

```bash
python download_model.py --mode all --calib-dir /path/to/images --calib-num 200
```

Output files are written to `python/model/` — e.g. `nanodet-plus-m_320.onnx`, `nanodet-plus-m_320_int8.onnx`, `nanodet-plus-m_320_FP32.tflite`, `nanodet-plus-m_320_INT8.tflite`.

---

## Step 3 — Run Inference (Python)

Activate the **inference venv** and navigate to `python/`:

**Windows PowerShell**

```powershell
cd vision\object_detection\nanodet\python
.\.venv_nanodet\Scripts\Activate.ps1
python inference.py --image sample_images\test_data_1_000000000139_person.jpg
```

**Ubuntu / bash**

```bash
cd vision/object_detection/nanodet/python
source .venv_nanodet/bin/activate
python inference.py --image sample_images/test_data_1_000000000139_person.jpg
```

Optional flags:

| Flag | Default | Description |
|------|---------|-------------|
| `-i`, `--image` | — | Input image path (required) |
| `-m`, `--model` | `model/nanodet-plus-m_320.onnx` | ONNX or TFLite model path |
| `-o`, `--output` | `outputs/` | Output file or directory |
| `--score` | `0.35` | Score threshold |
| `--nms` | `0.60` | NMS IoU threshold |
| `-v`, `--verbose` | off | Print model I/O details |

**Example output:**

```
Model   : model/nanodet-plus-m_320.onnx
Type    : ONNX
Input   : 320 x 320 (auto-detected)
Latency : 10.7 ms
Detected: 3 object(s)
Saved   : outputs/test_data_1_000000000139_person_result.jpg
```

> By default `inference.py` runs the ONNX FP32 model. Pass `--model model/nanodet-plus-m_320_INT8.tflite` (or any of the other three variants) to switch runtime.

---

## Step 4 — Compile for RA8P1 (RUHMI)

This step converts the TFLite model into C-code for the RA8P1 MCU. Activate the **compiler venv** (`.mera_venv`).

### 4.1 — Edit the compile configuration

Open `python/config.yaml` and verify the paths match your system:

```yaml
model_path: vision/object_detection/nanodet/python/model/nanodet-plus-m_320_INT8.tflite
output_dir: vision/object_detection/nanodet/embedded_c/src_mcu
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
python ruhmi_tools\mcu_compile.py vision\object_detection\nanodet\python\config.yaml
```

**Ubuntu / bash**

```bash
cd ~/Model-zoo
python ruhmi_tools/mcu_compile.py vision/object_detection/nanodet/python/config.yaml
```

The compiled C-code will be written to the `output_dir` specified in `config.yaml`.

---

## Step 5 — Embedded C Integration

The `embedded_c/` folder contains three portable, board-independent files you can drop directly into any bare-metal or RTOS project.

| File | Purpose |
|------|---------|
| `model_metadata.h` | All compile-time constants — input/output shape, quantization parameters, FPN strides, post-processing thresholds |
| `preprocessing.h` / `preprocessing.c` | Warp-resize the source frame, apply BGR mean/std normalisation, and quantize to INT8 |
| `postprocessing.h` / `postprocessing.c` | Dequantize the 2125×112 head, decode GFL bins into boxes, and run per-class NMS |

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
| `MODEL_INPUT_H` | `320` | Input image height (pixels) |
| `MODEL_INPUT_W` | `320` | Input image width (pixels) |
| `MODEL_INPUT_C` | `3` | RGB channels |
| `MODEL_INPUT_SIZE` | `307200` | Flat input buffer size (H × W × C bytes) |
| `MODEL_INPUT_SCALE` | `0.018658448f` | INT8 input quantisation scale |
| `MODEL_INPUT_ZERO_POINT` | `-14` | INT8 input zero-point |
| `MODEL_NUM_ANCHORS` | `2125` | Total anchors (40² + 20² + 10² + 5²) |
| `MODEL_OUTPUT_CHANNELS` | `112` | Per-anchor channels (80 class + 32 reg logits) |
| `MODEL_OUTPUT_SIZE` | `238000` | Flat output buffer size (`2125 × 112`) |
| `MODEL_OUTPUT_SCALE` | `0.166149095f` | INT8 output dequantisation scale |
| `MODEL_OUTPUT_ZERO_POINT` | `76` | INT8 output zero-point |
| `MODEL_STRIDE_S8/S16/S32/S64` | `8` / `16` / `32` / `64` | FPN strides |
| `MODEL_GRID_S8/S16/S32/S64` | `40` / `20` / `10` / `5` | FPN grid sizes |
| `MODEL_REG_MAX` | `7` | GFL distribution regression max index |
| `MODEL_NUM_FPN_LEVELS` | `4` | FPN pyramid depth |
| `MODEL_NUM_CLASSES` | `80` | COCO class count |
| `MODEL_MEAN_B/G/R` | `103.530f` / `116.280f` / `123.675f` | BGR mean (applied after RGB→BGR swap) |
| `MODEL_STD_B/G/R` | `57.375f` / `57.120f` / `58.395f` | BGR std (applied after RGB→BGR swap) |
| `POSTPROC_SCORE_THRESH` | `0.35f` | Default score threshold |
| `POSTPROC_NMS_THRESH` | `0.60f` | Default NMS IoU threshold |
| `POSTPROC_MAX_DETS` | `100` | Maximum detections post-NMS |

---

### 5.3 — Preprocessing — `preprocess()`

```c
#include "preprocessing.h"

void preprocess(const uint8_t      *p_source_image,
                uint16_t            _source_width,
                uint16_t            _source_height,
                int8_t             *p_destination_image,
                uint16_t            _destination_width,
                uint16_t            _destination_height,
                letterbox_params_t *p_letterbox_params);
```



**Typical call:**

```c
#include "model_metadata.h"
#include "preprocessing.h"

/* Camera frame: any resolution RGB888, stored as interleaved uint8 */
static uint8_t camera_buf[640 * 480 * 3];

/* Quantized int8 buffer fed to the model input tensor */
static int8_t nn_input[MODEL_INPUT_SIZE];   /* 307200 bytes */

/* Warp-resize metadata reused by postprocess() to undo the scale */
static letterbox_params_t lb_params;

/* --- inside your inference loop --- */
preprocess(camera_buf,
           640, 480,
           nn_input,
           MODEL_INPUT_W, MODEL_INPUT_H,
           &lb_params);
```

> [!TIP]
> `preprocess()` performs a **warp resize** (unequal scaling), not letterboxing. The `_pad_x`/`_pad_y` fields of `letterbox_params_t` are kept for API compatibility and are always `0` in warp mode; `postprocess()` uses only `_scale_w` / `_scale_h` to map boxes back to the original frame.
---

### 5.4 — Postprocessing — `postprocess()`



**API:**

```c
int32_t postprocess(const int8_t             *p_raw_output,
                    const letterbox_params_t *p_letterbox_params,
                    uint32_t                  _original_width_pixels,
                    uint32_t                  _original_height_pixels,
                    float                     _score_threshold,
                    float                     _nms_iou_threshold,
                    Detection_t              *p_output_detections);
```

Returns the number of detections written into `p_output_detections` (0 on empty result or invalid arguments; capped at `POSTPROC_MAX_DETS`). Each `Detection_t` holds `_x1`, `_y1`, `_x2`, `_y2` (top-left / bottom-right corners), `_score` (best class confidence), and `_cls_id` (COCO class index in `[0, MODEL_NUM_CLASSES - 1]`).
Decode the INT8 head into class-labelled bounding boxes in **original source-image pixel coordinates**:

```c
#include "model_metadata.h"
#include "postprocessing.h"

/* Raw INT8 output from the model: MODEL_NUM_ANCHORS × MODEL_OUTPUT_CHANNELS */
int8_t raw_out[MODEL_OUTPUT_SIZE];   /* 2125 * 112 = 238000 */

/* Detection output buffer sized for the worst case (POSTPROC_MAX_DETS = 100) */
Detection_t detections[POSTPROC_MAX_DETS];

int32_t n = postprocess(raw_out,
                        &lb_params,                 /* filled by preprocess() */
                        /* _original_width_pixels  =*/ 640,
                        /* _original_height_pixels =*/ 480,
                        POSTPROC_SCORE_THRESH,
                        POSTPROC_NMS_THRESH,
                        detections);

for (int32_t i = 0; i < n; i++) {
    printf("Det %ld  cls=%u  score=%.2f  box=[%.1f,%.1f,%.1f,%.1f]\n",
           (long)i,
           detections[i]._cls_id, detections[i]._score,
           detections[i]._x1, detections[i]._y1,
           detections[i]._x2, detections[i]._y2);
}
```


> [!NOTE]
> `postprocess()` handles dequantisation internally (via `MODEL_OUTPUT_SCALE` / `MODEL_OUTPUT_ZERO_POINT`), decodes the 4×8 GFL regression bins (`MODEL_REG_MAX = 7`) into `(l, t, r, b)` distances at each anchor stride (`MODEL_STRIDE_S8/S16/S32/S64`), projects to `(x1, y1, x2, y2)`, **divides by `p_letterbox_params->_scale_w` / `_scale_h` so output coordinates are already in the original source-image pixel frame**, and runs per-class NMS with `_nms_iou_threshold`. The output buffer must hold at least `POSTPROC_MAX_DETS` entries.

---

## License / Provenance

| Component | Source | Licence |
|-----------|--------|---------|
| NanoDet-Plus-m weights | [NanoDet by RangiLyu](https://github.com/RangiLyu/nanodet) | Apache 2.0 |
| COCO val2017 dataset | [COCO Dataset](https://cocodataset.org/) | CC BY 4.0 |
