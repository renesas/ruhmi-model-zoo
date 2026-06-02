/**
 * @file    model_metadata.h
 * @brief   Compile-time metadata for YOLO-Fastest 1.1 INT8 model (320×320 input).
 *
 * Generated from:
 *   Model  : yolo_fastest_1_1_int8 (EdgeCortix / Renesas)
 *   Task   : Object Detection (COCO 80-class)
 *   Date   : 2026-05-13
 *
 * ──────────────────────────────────────────────────────────────────────
 * QUICK-START (bare-metal)
 * ──────────────────────────────────────────────────────────────────────
 *
 *  1. Feed  : raw RGB uint8 image (any resolution)
 *  2. Letterbox resize → [320 × 320] → quantize to int8
 *  3. Run inference via compute_sub_0000()
 *  4. Dequantize two output heads → decode anchor-based boxes
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
#define MODEL_NAME              "YOLO-Fastest-1.1"

/** Dataset the model was trained on. */
#define MODEL_DATASET           "COCO"

/** Number of output classes. */
#define MODEL_NUM_CLASSES       (80U)

/** Model data type. */
#define MODEL_DTYPE             "int8"


/* ══════════════════════════════════════════════════════════════════════
 * 2.  INPUT TENSOR
 *     Shape : [1, 320, 320, 3]
 *     DType : int8
 *     Quantization: scale=0.00392157, zero_point=-128
 *       int8_value = round(pixel_float / scale) + zero_point
 *       For pixel [0,255] normalised to [0,1]:
 *         int8 = round((pixel/255.0) / 0.00392157) + (-128)
 *              = pixel - 128
 * ══════════════════════════════════════════════════════════════════════ */

/** Expected input height in pixels. */
#define MODEL_INPUT_H           (320)

/** Expected input width in pixels. */
#define MODEL_INPUT_W           (320)

/** Number of colour channels (RGB). */
#define MODEL_INPUT_C           (3)

/** Flat size of one input tensor in elements (H × W × C). */
#define MODEL_INPUT_SIZE        (MODEL_INPUT_H * MODEL_INPUT_W * MODEL_INPUT_C)
/* = 320 × 320 × 3 = 307200 */

/** Letterbox padding fill value (gray). */
#define INPUT_LETTERBOX_PAD     (114)

/** Input quantization: scale */
#define MODEL_INPUT_SCALE       (0.00392157f)

/** Input quantization: zero point */
#define MODEL_INPUT_ZERO_POINT  (-128)


/* ══════════════════════════════════════════════════════════════════════
 * 3.  OUTPUT TENSORS  (two anchor-based YOLO heads)
 *
 *     Head 0 (large objects): [1, 10, 10, 255]
 *       stride=32, anchors: (115,73), (119,199), (242,238)
 *       Quant: scale=0.21769431, zero_point=73
 *
 *     Head 1 (small objects): [1, 20, 20, 255]
 *       stride=16, anchors: (12,18), (37,49), (52,132)
 *       Quant: scale=0.20347583, zero_point=72
 *
 *     255 channels = 3 anchors × (4 bbox + 1 obj + 80 classes)
 * ══════════════════════════════════════════════════════════════════════ */

/** Number of anchors per grid cell. */
#define MODEL_NUM_ANCHORS_PER_CELL  (3U)

/** Channels per anchor: 4 (box) + 1 (obj) + 80 (class). */
#define MODEL_FIELDS_PER_ANCHOR     (85U)

/** Total channels per output cell: 3 × 85 = 255. */
#define MODEL_OUTPUT_CHANNELS       (255U)

/* ── Head 0: stride-32, 10×10 grid ───────────────────────────────── */
#define MODEL_HEAD0_GRID_H      (10U)
#define MODEL_HEAD0_GRID_W      (10U)
#define MODEL_HEAD0_STRIDE      (32U)
#define MODEL_HEAD0_SIZE        (MODEL_HEAD0_GRID_H * MODEL_HEAD0_GRID_W * MODEL_OUTPUT_CHANNELS)
/* = 10 × 10 × 255 = 25500 */
#define MODEL_HEAD0_SCALE       (0.21769431f)
#define MODEL_HEAD0_ZERO_POINT  (73)

/* Head 0 anchors (w, h) in input-image pixels */
#define MODEL_HEAD0_AW0  (115U)
#define MODEL_HEAD0_AH0  (73U)
#define MODEL_HEAD0_AW1  (119U)
#define MODEL_HEAD0_AH1  (199U)
#define MODEL_HEAD0_AW2  (242U)
#define MODEL_HEAD0_AH2  (238U)

/* ── Head 1: stride-16, 20×20 grid ───────────────────────────────── */
#define MODEL_HEAD1_GRID_H      (20U)
#define MODEL_HEAD1_GRID_W      (20U)
#define MODEL_HEAD1_STRIDE      (16U)
#define MODEL_HEAD1_SIZE        (MODEL_HEAD1_GRID_H * MODEL_HEAD1_GRID_W * MODEL_OUTPUT_CHANNELS)
/* = 20 × 20 × 255 = 102000 */
#define MODEL_HEAD1_SCALE       (0.20347583f)
#define MODEL_HEAD1_ZERO_POINT  (72)

/* Head 1 anchors (w, h) in input-image pixels */
#define MODEL_HEAD1_AW0  (12U)
#define MODEL_HEAD1_AH0  (18U)
#define MODEL_HEAD1_AW1  (37U)
#define MODEL_HEAD1_AH1  (49U)
#define MODEL_HEAD1_AW2  (52U)
#define MODEL_HEAD1_AH2  (132U)

/** Total anchors across both heads = 10*10*3 + 20*20*3 = 1500. */
#define MODEL_TOTAL_ANCHORS     (MODEL_HEAD0_GRID_H * MODEL_HEAD0_GRID_W * MODEL_NUM_ANCHORS_PER_CELL \
                                + MODEL_HEAD1_GRID_H * MODEL_HEAD1_GRID_W * MODEL_NUM_ANCHORS_PER_CELL)


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

/** Maximum output image buffer size for bbox visualization (586×640×3). */
#define OUTPUT_IMAGE_MAX_SIZE   (1125120U)


#ifdef __cplusplus
}
#endif

#endif /* MODEL_METADATA_H */
