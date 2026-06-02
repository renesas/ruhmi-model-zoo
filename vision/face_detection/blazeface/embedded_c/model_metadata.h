/**
 * @file    model_metadata.h
 * @brief   Compile-time metadata for BlazeFace Front-Face INT8 model (CPU-only).
 *
 *
 * Generated from:
 *   Model  : blazeface_front_int8.tflite  (PINTO model zoo — 030_BlazeFace)
 *   Task   : Face Detection (bounding box + 6 keypoints)
 *   Target : RA8P1 CPU (MERA C-codegen)
 *
 * ──────────────────────────────────────────────────────────────────────
 * QUICK-START (bare-metal)
 * ──────────────────────────────────────────────────────────────────────
 *
 *  1. Capture  : raw uint8 RGB image (any resolution)
 *  2. Resize   : bilinear → [128 × 128]
 *  3. Quantize : q = (int8_t)(pixel - 128)
 *  4. Run inference via compute_sub_0000()
 *  5. Dequantize 4 INT8 output tensors
 *  6. Decode 896 anchors → face bounding boxes + 6 keypoints
 *  7. Apply sigmoid + score filtering + Weighted NMS
 *  8. Output: list of (ymin,xmin,ymax,xmax, 6 keypoints, score)
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
#define MODEL_NAME              "BlazeFace-Front"

/** Model data type. */
#define MODEL_DTYPE             "int8"

/** Number of facial keypoints per detection. */
#define NUM_KEYPOINTS           (6)


/* ══════════════════════════════════════════════════════════════════════
 * 2.  INPUT TENSOR
 *     Shape : [1, 128, 128, 3]
 *     DType : int8
 * ══════════════════════════════════════════════════════════════════════ */

/** Expected input height in pixels. */
#define MODEL_INPUT_H           (128)

/** Expected input width in pixels. */
#define MODEL_INPUT_W           (128)

/** Number of colour channels (RGB). */
#define MODEL_INPUT_C           (3)

/** Flat size of one input tensor in elements (H × W × C). */
#define MODEL_INPUT_SIZE        (MODEL_INPUT_H * MODEL_INPUT_W * MODEL_INPUT_C)
/* = 128 × 128 × 3 = 49152 bytes */

/**
 * INT8 input quantization zero-point.
 * Source: interpreter.get_input_details()[0]['quantization']
 * Quantise: q = clamp(pixel + INPUT_ZP, -128, 127)
 */
#define INPUT_ZP                (-128)


/* ══════════════════════════════════════════════════════════════════════
 * 3.  OUTPUT TENSORS  (4 tensors, all INT8)
 *
 *  compute_sub_0000() produces 4 outputs:
 *    [0] scores_s8  : [1, 512, 1]   — stride-8 score logits
 *    [1] boxes_s8   : [1, 512, 16]  — stride-8 box regressions
 *    [2] scores_s16 : [1, 384, 1]   — stride-16 score logits
 *    [3] boxes_s16  : [1, 384, 16]  — stride-16 box regressions
 *
 *  896 anchors = 512 (stride-8) + 384 (stride-16)
 * ══════════════════════════════════════════════════════════════════════ */

/** Number of stride-8 anchors. */
#define ANCHORS_S8              (512)

/** Number of stride-16 anchors. */
#define ANCHORS_S16             (384)

/** Total anchors. */
#define TOTAL_ANCHORS           (ANCHORS_S8 + ANCHORS_S16)   /* 896 */

/** Box regression values per anchor. */
#define OUTPUT_BOX_COORDS       (16)

/* ── Dequantization parameters ──────────────────────────────────────── */
#define OUTPUT_SCORES_S8_SCALE  0.03629210218787193f
#define OUTPUT_SCORES_S8_ZP     46

#define OUTPUT_SCORES_S16_SCALE 1.3571407794952393f
#define OUTPUT_SCORES_S16_ZP    126

#define OUTPUT_BOXES_S8_SCALE   0.3177529275417328f
#define OUTPUT_BOXES_S8_ZP      (-40)

#define OUTPUT_BOXES_S16_SCALE  1.2865246534347534f
#define OUTPUT_BOXES_S16_ZP     (-56)

/* ══════════════════════════════════════════════════════════════════════
 * 4.  POST-PROCESSING PARAMETERS
 * ══════════════════════════════════════════════════════════════════════ */

/** Minimum face confidence (after sigmoid) to keep a detection. */
#define POSTPROC_SCORE_THRESH   (0.50f)

/** Weighted-NMS IoU threshold. */
#define POSTPROC_NMS_THRESH     (0.30f)

/** Maximum number of detections to retain (post-NMS). */
#define POSTPROC_MAX_DETS       (64U)


#ifdef __cplusplus
}
#endif

#endif /* MODEL_METADATA_H */
