/**
 * @file    model_metadata.h
 * @brief   Compile-time metadata for YOLOX-Tiny INT8 model (224×224 input).
 *
 * Generated from:
 *   Model  : yolox_tiny_224_INT8 (EdgeCortix / Renesas)
 *   Task   : Object Detection (COCO 80-class)
 *   Date   : 2026-04-28
 *
 * ──────────────────────────────────────────────────────────────────────
 * QUICK-START (bare-metal)
 * ──────────────────────────────────────────────────────────────────────
 *
 *  1. Feed  : raw BGR uint8 image (any resolution)
 *  2. Letterbox resize → [224 × 224] → quantize to int8
 *  3. Run inference via compute_sub_0000()
 *  4. Dequantize output → decode 1029 anchors × 85 channels → boxes
 *  5. Apply score filtering + per-class NMS
 *  6. Output: list of (x1,y1,x2,y2,score,class_id) in original coords
 *
 * ──────────────────────────────────────────────────────────────────────
 */

#ifndef MODEL_METADATA_H
#define MODEL_METADATA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif


/* ══════════════════════════════════════════════════════════════════════
 * 1.  MODEL IDENTITY
 * ══════════════════════════════════════════════════════════════════════ */

/** Human-readable model name. */
#define MODEL_NAME              "YOLOX-Tiny"

/** Dataset the model was trained on. */
#define MODEL_DATASET           "COCO"

/** Number of output classes. */
#define MODEL_NUM_CLASSES       (80U)

/** Model data type. */
#define MODEL_DTYPE             "int8"


/* ══════════════════════════════════════════════════════════════════════
 * 2.  INPUT TENSOR
 *     Shape : [1, 224, 224, 3]
 *     DType : int8
 *     Quantization: scale=1.0, zero_point=-128
 *       int8_value = (float_value / scale) + zero_point
 *       For pixel [0,255]: int8 = pixel - 128  (maps 0→-128, 255→127)
 * ══════════════════════════════════════════════════════════════════════ */

/** Expected input height in pixels. */
#define MODEL_INPUT_H           (224)

/** Expected input width in pixels. */
#define MODEL_INPUT_W           (224)

/** Number of colour channels (BGR→RGB). */
#define MODEL_INPUT_C           (3)

/** Flat size of one input tensor in elements (H × W × C). */
#define MODEL_INPUT_SIZE        (MODEL_INPUT_H * MODEL_INPUT_W * MODEL_INPUT_C)
/* = 224 × 224 × 3 = 150528 */

/** Letterbox padding fill value (gray). */
#define INPUT_LETTERBOX_PAD     (114)

/** Input quantization: scale */
#define MODEL_INPUT_SCALE       (1.0f)

/** Input quantization: zero point */
#define MODEL_INPUT_ZERO_POINT  (-128)


/* ══════════════════════════════════════════════════════════════════════
 * 3.  OUTPUT TENSOR
 *     Shape : [1, 1029, 85]
 *     DType : int8
 *     Quantization: scale=0.0267025213688612, zero_point=-30
 *       float_value = (int8_value - zero_point) * scale
 *
 *     1029 anchors = 28×28 + 14×14 + 7×7  (FPN strides 8/16/32)
 *     85 channels  = 4 (bbox reg) + 1 (objectness) + 80 (class logits)
 * ══════════════════════════════════════════════════════════════════════ */

/** Number of anchor boxes (grid cells across all FPN levels). */
#define MODEL_NUM_ANCHORS       (1029U)   /* 28*28 + 14*14 + 7*7 */

/** Channels per anchor: 4 (box) + 1 (obj) + 80 (class). */
#define MODEL_OUTPUT_CHANNELS   (85U)

/** Flat size of one output tensor in elements (int8). */
#define MODEL_OUTPUT_SIZE       (MODEL_NUM_ANCHORS * MODEL_OUTPUT_CHANNELS)
/* = 1029 × 85 = 87465 */

/** Output quantization: scale */
#define MODEL_OUTPUT_SCALE      (0.0267025213688612f)

/** Output quantization: zero point */
#define MODEL_OUTPUT_ZERO_POINT (-30)

/** FPN stride / grid sizes. */
#define MODEL_STRIDE_S8         (8U)
#define MODEL_STRIDE_S16        (16U)
#define MODEL_STRIDE_S32        (32U)
#define MODEL_GRID_S8           (28U)     /* = 224 / 8  */
#define MODEL_GRID_S16          (14U)     /* = 224 / 16 */
#define MODEL_GRID_S32          (7U)      /* = 224 / 32 */


/* ══════════════════════════════════════════════════════════════════════
 * 4.  POST-PROCESSING THRESHOLDS
 * ══════════════════════════════════════════════════════════════════════ */

/** Minimum score (objectness × class_conf) to keep a detection. */
#define POSTPROC_SCORE_THRESH   (0.25f)

/** IoU threshold for per-class Non-Maximum Suppression. */
#define POSTPROC_NMS_THRESH     (0.45f)

/** Maximum number of detections to retain per image (post-NMS). */
#define POSTPROC_MAX_DETS       (100U)

/** Maximum number of detections to log per image via RTT. */
#define LOG_MAX_DETS            (20U)


/* ══════════════════════════════════════════════════════════════════════
 * 5.  TEST CONFIGURATION
 * ══════════════════════════════════════════════════════════════════════ */

/** Number of test images for evaluation. */
#define NUM_TEST_IMAGES         (3U)

/** Maximum output image buffer size for bbox visualization (640×480×3). */
#define OUTPUT_IMAGE_MAX_SIZE   (921600U)


#ifdef __cplusplus
}
#endif

#endif /* MODEL_METADATA_H */
