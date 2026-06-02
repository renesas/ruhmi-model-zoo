/**
 * @file    model_metadata.h
 * @brief   Compile-time metadata for ResNet8 INT8 TFLite model (CIFAR-10).
 *
 * Generated from:
 *   Model  : Resnet_int8.tflite  /  Resnet_fp32.tflite
 *   Script : validate.py  (ResNet8, CIFAR-10 10-class, INT8 + FP32)
 *   Date   : 2026-04-08
 *
 * ──────────────────────────────────────────────────────────────────────
 * QUICK-START (bare-metal / TFLite Micro)
 * ──────────────────────────────────────────────────────────────────────
 *
 *  1. Capture : 32×32 RGB image (CIFAR-10 resolution)
 *  2. Pre-process (NO float normalisation — raw uint8 [0,255] fed directly):
 *       int8_t q = (int8_t)((int16_t)pixel - 128);
 *       // Equivalent to: q = pixel / INPUT_SCALE + INPUT_ZP
 *       // (INPUT_SCALE=1.0, INPUT_ZP=-128  ⟹  q = pixel - 128)
 *  3. Copy into the INT8 input tensor  [1×32×32×3]
 *  4. Run inference
 *  5. Read int8 output tensor  [1×10]
 *  6. De-quantize:  prob[i] = (out_q[i] - OUTPUT_ZP) * OUTPUT_SCALE
 *  7. argmax → CIFAR-10 class index (0–9) → CLASS_NAMES[]
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
#define MODEL_NAME          "ResNet8"

/** Relative path to the INT8 TFLite flatbuffer (from project root). */
#define MODEL_FILE_INT8     "model/Resnet_int8.tflite"

/** Relative path to the FP32 TFLite flatbuffer (for reference / validation). */
#define MODEL_FILE_FP32     "model/Resnet_fp32.tflite"

/** Keras/H5 pre-trained weights file (used to export the TFLite models). */
#define MODEL_FILE_H5       "model/pretrainedResnet.h5"

/** Task performed by the model. */
#define MODEL_TASK          "Image Classification"

/** Dataset the model was trained on. */
#define MODEL_DATASET       "CIFAR-10"

/** Number of output classes. */
#define MODEL_NUM_CLASSES   10


/* ══════════════════════════════════════════════════════════════════════
 * 2.  INPUT TENSOR
 *     Verified with: interpreter.get_input_details()[0]
 *     Name  : "serving_default_input_1:0"
 *     Index : 0
 *     Shape : [1, 32, 32, 3]
 *     DType : int8
 * ══════════════════════════════════════════════════════════════════════ */

/** Expected input height in pixels (CIFAR-10 native resolution). */
#define MODEL_INPUT_H       32

/** Expected input width in pixels (CIFAR-10 native resolution). */
#define MODEL_INPUT_W       32

/** Number of colour channels (RGB). */
#ifndef MODEL_CHANNELS
#define MODEL_INPUT_C       3
#else
#define MODEL_INPUT_C       MODEL_CHANNELS
#endif

/** Flat size of one input tensor (H × W × C). */
#ifndef MODEL_INPUT_SIZE
#define MODEL_INPUT_SIZE    (MODEL_INPUT_H * MODEL_INPUT_W * MODEL_INPUT_C)
#endif
/* = 32 × 32 × 3 = 3072 bytes */

/** Memory layout of the input tensor. */
#define MODEL_INPUT_LAYOUT  "NHWC"   /* batch=1, Height, Width, Channels */

/**
 * INT8 input quantization scale.
 * Source: interpreter.get_input_details()[0]['quantization'][0]
 *   → 1.0
 *
 * Quantize: q = clamp(round(f / INPUT_SCALE) + INPUT_ZP, -128, 127)
 *
 * Because INPUT_SCALE=1.0 and INPUT_ZP=-128, the FP32 model is fed raw
 * float pixel values in [0, 255], and the INT8 model receives those same
 * values shifted to [-128, 127]:
 *   q = pixel - 128
 * No normalisation (no division by 127.5, no mean subtraction) is needed.
 */
#define INPUT_SCALE         1.0f
#define INPUT_ZP            (-128)

/** Model expects raw pixel values in [0, 255] — no float normalisation. */
#define INPUT_PIXEL_MIN     0
#define INPUT_PIXEL_MAX     255

/** Source data format: CIFAR-10 stored as uint8, channels-first, then HWC. */
#define INPUT_RESIZE_METHOD "No resize needed — model is 32×32 (CIFAR-10 native)"


/* ══════════════════════════════════════════════════════════════════════
 * 3.  OUTPUT TENSOR
 *     Verified with: interpreter.get_output_details()[0]
 *     Name  : "StatefulPartitionedCall_1:0"
 *     Index : (model internal)
 *     Shape : [1, 10]
 *     DType : int8
 * ══════════════════════════════════════════════════════════════════════ */

/** Flat size of one output tensor (= MODEL_NUM_CLASSES). */
#define MODEL_OUTPUT_SIZE   10

/**
 * INT8 output dequantization parameters.
 * Source: interpreter.get_output_details()[0]['quantization']
 *   scale      = 0.00390625  (= 1/256)
 *   zero_point = -128
 *
 * Dequantize: score[i] = (out_q[i] - OUTPUT_ZP) * OUTPUT_SCALE
 * Then: predicted_class = argmax(score[0..9])
 *
 * NOTE: Softmax is ALREADY applied in the graph.
 *       DO NOT apply softmax again after dequantization.
 */
#define OUTPUT_SCALE        0.00390625f   /* = 1/256 */
#define OUTPUT_ZP           (-128)
#define OUTPUT_HAS_SOFTMAX  1    /* 1 = softmax baked in — do NOT re-apply */
#define OUTPUT_FLOAT_MIN    0.0f
#define OUTPUT_FLOAT_MAX    0.99609375f  /* (127 - (-128)) * (1/256) */

#ifdef __cplusplus
}
#endif

#endif /* MODEL_METADATA_H */
