/**
 * @file    model_metadata.h
 * @brief   Compile-time metadata for Visual Wake Words (VWW) INT8 TFLite model.
 *
 * Generated from:
 *   Model  : vww_96_INT8.tflite  /  vww_96_FP32.tflite
 *   Source : MLCommons Tiny benchmark (vww_96.h5 → TFLite export)
 *   Script : inference.py  (VWW, binary: person vs notperson, INT8)
 *   Date   : 2026-04-08
 *
 * ──────────────────────────────────────────────────────────────────────
 * QUICK-START (bare-metal / TFLite Micro)
 * ──────────────────────────────────────────────────────────────────────
 *
 *  1. Capture : raw uint8 RGB image (any resolution)
 *  2. Resize  : bilinear → [96 × 96]
 *  3. Pre-process each pixel:
 *       float f = pixel / 255.0f;                         // [0, 1]
 *       int8  q = clamp(round(f / INPUT_SCALE) + INPUT_ZP, -128, 127);
 *     Fast C:  q = (int8_t)((int16_t)pixel - 128)
 *  4. Copy into INT8 input tensor  [1×96×96×3]
 *  5. Run inference
 *  6. Read int8 output tensor  [1×2]  → [notperson_q, person_q]
 *  7. Dequantize:  score[i] = (out_q[i] - OUTPUT_ZP) * OUTPUT_SCALE
 *  8. Apply softmax on score[0..1] → probabilities
 *  9. argmax → 0=notperson, 1=person
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

#define MODEL_NAME          "VisualWakeWord"
#define MODEL_FILE_INT8     "model/vww_96_INT8.tflite"
#define MODEL_FILE_FP32     "model/vww_96_FP32.tflite"
#define MODEL_FILE_H5       "model/vww_96.h5"
#define MODEL_TASK          "Binary Image Classification"
#define MODEL_DATASET       "COCO 2014 (person vs notperson subset)"
#define MODEL_NUM_CLASSES   2


/* ══════════════════════════════════════════════════════════════════════
 * 2.  INPUT TENSOR
 *     Verified with: interpreter.get_input_details()[0]
 *     Name  : "serving_default_input_1:0"
 *     Shape : [1, 96, 96, 3]   DType : int8
 * ══════════════════════════════════════════════════════════════════════ */

#define MODEL_INPUT_H       96
#define MODEL_INPUT_W       96

#ifndef MODEL_CHANNELS
#define MODEL_INPUT_C       3
#else
#define MODEL_INPUT_C       MODEL_CHANNELS
#endif

#ifndef MODEL_INPUT_SIZE
#define MODEL_INPUT_SIZE    (MODEL_INPUT_H * MODEL_INPUT_W * MODEL_INPUT_C)
#endif
/* = 96 × 96 × 3 = 27 648 bytes */

#define MODEL_INPUT_LAYOUT  "NHWC"

/**
 * INT8 quantization parameters (from interpreter.get_input_details()[0]):
 *   scale      = 0.003921568859368563  (≈ 1/255)
 *   zero_point = -128
 *
 * Normalisation used at training: pixel / 255.0  → [0.0, 1.0]
 *
 * Quantize: q = clamp(round(f / INPUT_SCALE) + INPUT_ZP, -128, 127)
 *   where f = pixel / 255.0f
 * Fast C:   q = (int8_t)((int16_t)pixel - 128)
 *   (exact because INPUT_SCALE ≈ 1/255 and INPUT_ZP = -128)
 */
#define INPUT_SCALE         0.003921568859368563f   /* ≈ 1/255 */
#define INPUT_ZP            (-128)
#define INPUT_NORM_SCALE    255.0f                  /* divisor */
#define INPUT_NORM_BIAS     0.0f                    /* no bias after division */
#define INPUT_PIXEL_MIN     0
#define INPUT_PIXEL_MAX     255
#define INPUT_RESIZE_METHOD "Bilinear (half-pixel centre convention)"


/* ══════════════════════════════════════════════════════════════════════
 * 3.  OUTPUT TENSOR
 *     Verified with: interpreter.get_output_details()[0]
 *     Name  : "StatefulPartitionedCall_1:0"
 *     Shape : [1, 2]   DType : int8
 *       index 0 → notperson score
 *       index 1 → person    score
 * ══════════════════════════════════════════════════════════════════════ */

#define MODEL_OUTPUT_SIZE   2

/**
 * INT8 output dequantization parameters:
 *   scale      = 0.00390625  (= 1/256)
 *   zero_point = -128
 *
 * Dequantize: score[i] = (out_q[i] - OUTPUT_ZP) * OUTPUT_SCALE
 *
 * NOTE: Output is RAW LOGITS — softmax is NOT applied in the graph.
 *       You MUST apply softmax after dequantization to get probabilities.
 */
#define OUTPUT_SCALE        0.00390625f   /* = 1/256 */
#define OUTPUT_ZP           (-128)
#define OUTPUT_HAS_SOFTMAX  0    /* 0 = raw logits — MUST apply softmax */
#define OUTPUT_FLOAT_MIN    (-0.5f)       /* approximate range after dequant */
#define OUTPUT_FLOAT_MAX    0.99609375f

/** Class index → label mapping. */
#define VWW_CLASS_0         "notperson"   /* output index 0 */
#define VWW_CLASS_1         "person"      /* output index 1 */

/**
 * Minimum softmax probability to report a positive "person" detection.
 * Tune on your target hardware / use-case power budget.
 */
#define PERSON_THRESHOLD    0.5f


/* ══════════════════════════════════════════════════════════════════════
 * 4.  CLASS LABELS
 * ══════════════════════════════════════════════════════════════════════ */

/**
 * C string-array initialiser — use as:
 *   static const char * const CLASS_NAMES[] = VWW_CLASS_NAMES_INIT;
 *   printf("%s\n", CLASS_NAMES[predicted_class]);
 */
#define VWW_CLASS_NAMES_INIT { "notperson", "person" }


/* ══════════════════════════════════════════════════════════════════════
 * 5.  MODEL FILE SIZES
 * ══════════════════════════════════════════════════════════════════════ */
/**
 * TFLite Micro tensor arena size (bytes).
 * Measured value from TFLite Micro profiling on this model.
 */
#define CPU_RAM_TENSOR_ARENA_BYTES   78193    /* ~ 78 KB  */
#define NPU_RAM_TENSOR_ARENA_BYTES   73728    /* ~ 72 KB  */


/* ══════════════════════════════════════════════════════════════════════
 * 6.  MEMORY FOOTPRINT  (approximate)
 * ══════════════════════════════════════════════════════════════════════ */

/** Input tensor buffer size in bytes  (H × W × C × 1 byte/int8). */
#define MODEL_INPUT_BUF_BYTES    MODEL_INPUT_SIZE   /* 27 648 B ≈ 27 KB */

/** Output tensor buffer size in bytes. */
#define MODEL_OUTPUT_BUF_BYTES   MODEL_OUTPUT_SIZE  /* 2 B */


/* ══════════════════════════════════════════════════════════════════════
 * 7.  MERA COMPILATION TARGETS
 *     Source directories produced by the RUHMI MERA compiler
 * ══════════════════════════════════════════════════════════════════════ */

/** CPU-only INT8 compiled output.  embedded_c/src_mcu/vww_96_INT8_CPU/ */
#define MERA_CPU_SRC_DIR    "embedded_c/src_mcu/vww_96_INT8_CPU"

/** NPU INT8 compiled output.  embedded_c/src_mcu_npu/vww_96_INT8_NPU/ */
#define MERA_NPU_SRC_DIR    "embedded_c/src_mcu_npu/vww_96_INT8_NPU"

#ifdef __cplusplus
}
#endif

#endif /* MODEL_METADATA_H */