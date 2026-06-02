/**
 * @file    model_metadata.h
 * @brief   Compile-time metadata for SqueezeNet1.1 TFLite models (FP32 & INT8).
 *
 * Generated from:
 *   Model  : squeezenet1_1_INT8.tflite / squeezenet1_1_FP32.tflite
 *   Source : SqueezeNet 1.1 (PyTorch) → ONNX → TFLite
 *   Task   : Image Classification — 1000 ImageNet classes
 *   Date   : 2026-04-07
 *
 * I/O details verified with:
 *   interpreter.get_input_details()[0]   → shape [1, 224, 224, 3]  int8   (INT8 model)
 *                                        → shape [1, 224, 224, 3]  float32 (FP32 model)
 *   interpreter.get_output_details()[0]  → shape [1, 1000]         int8   (INT8 model)
 *                                        → shape [1, 1000]         float32 (FP32 model)
 *
 * MERA compilation targets:
 *   CPU INT8 (TFLite) : TFLite_INT8_cpu_quantized_c/squeezenet1_1_INT8_CPU_external/
 *   NPU INT8 (TFLite) : TFLite_INT8_npu_quantized_c/squeezenet1_1_INT8_NPU_external/
 *
 * ──────────────────────────────────────────────────────────────────────────
 * QUICK-START (bare-metal / TFLite Micro — INT8 model)
 * ──────────────────────────────────────────────────────────────────────────
 *
 *  1. Capture  : raw uint8 BGR image (any resolution)
 *  2. Convert  : BGR → RGB
 *  3. Resize   : bilinear → [224 × 224]
 *  4. Normalise: per-channel ImageNet mean/std
 *                  f = pixel / 255.0f
 *                  n = (f - mean[c]) / std[c]      →  range ≈ [-2.1, 2.6]
 *  5. Quantize : q = clamp(round(n / INPUT_SCALE) + INPUT_ZP, -128, 127)
 *  6. Copy int8 tensor [1×224×224×3] → model input
 *  7. Run inference
 *  8. Dequantize output: logit = (out_q[i] - OUTPUT_ZP) * OUTPUT_SCALE
 *  9. Apply softmax over 1000 logits → probability distribution
 * 10. argmax / top-K → class index → labels.h class name
 *
 * ──────────────────────────────────────────────────────────────────────────
 *
 * Copyright 2026 Renesas Electronics Corporation
 * SPDX-License-Identifier: Apache-2.0
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
#define MODEL_NAME              "SqueezeNet1.1"

/** Model source. */
#define MODEL_SOURCE            "SqueezeNet 1.1 (PyTorch) → ONNX → TFLite"

/** Relative paths to TFLite flatbuffers (from project root). */
#define MODEL_FILE_INT8         "Models/squeezenet1_1_INT8.tflite"
#define MODEL_FILE_FP32         "Models/squeezenet1_1_FP32.tflite"
#define MODEL_FILE_ONNX         "Models/squeezenet1_1.onnx"

/** Task performed by the model. */
#define MODEL_TASK              "Image Classification"

/** Training dataset. */
#define MODEL_DATASET           "ImageNet ILSVRC-2012"

/** Model file sizes (bytes). */
#define CPU_RAM_TENSOR_ARENA_BYTES   984449   /* ~984 KB */
#define NPU_RAM_TENSOR_ARENA_BYTES   982144   /* ~982 KB */

/** Number of output classes. */
#define MODEL_NUM_CLASSES       1000


/* ══════════════════════════════════════════════════════════════════════
 * 2.  INPUT TENSOR
 *     Verified with: interpreter.get_input_details()[0]
 *     Name  : "serving_default_input:0"
 *     Index : 0
 *     Shape : [1, 224, 224, 3]
 *     DType : int8 (INT8 model) / float32 (FP32 model)
 * ══════════════════════════════════════════════════════════════════════ */

/** Expected input height in pixels. */
#define MODEL_INPUT_H           224

/** Expected input width in pixels. */
#define MODEL_INPUT_W           224

/** Number of colour channels (RGB). */
#ifndef MODEL_CHANNELS
#define MODEL_INPUT_C           3
#else
#define MODEL_INPUT_C           MODEL_CHANNELS
#endif

/** Flat element count of one input tensor (H × W × C). */
#ifndef MODEL_INPUT_SIZE
#define MODEL_INPUT_SIZE        (MODEL_INPUT_H * MODEL_INPUT_W * MODEL_INPUT_C)
#endif
/* = 224 × 224 × 3 = 150 528 */

/** Memory layout of the input tensor. */
#define MODEL_INPUT_LAYOUT      "NHWC"   /* batch=1, Height, Width, Channels */

/**
 * INT8 input quantization scale.
 * Source: interpreter.get_input_details()[0]['quantization_parameters']['scales'][0]
 *
 * Maps int8 value q → normalised float via:  n = (q - INPUT_ZP) * INPUT_SCALE
 * To quantize normalised float → int8:       q = clamp(round(n / INPUT_SCALE) + INPUT_ZP, -128, 127)
 *
 * "Normalised" here means AFTER ImageNet mean/std subtraction (see INPUT_MEAN / INPUT_STD).
 */
#define INPUT_SCALE             0.01865844801068306f

/**
 * INT8 input quantization zero-point.
 * Source: interpreter.get_input_details()[0]['quantization_parameters']['zero_points'][0]
 */
#define INPUT_ZP                (-14)

/**
 * ImageNet per-channel normalisation applied BEFORE quantisation.
 * Applied to the [0, 1] float image (after /255):
 *   normalised[c] = (pixel_f[c] - INPUT_MEAN[c]) / INPUT_STD[c]
 *
 * Normalised range: R ≈ [-2.12, 2.64], G ≈ [-2.04, 2.43], B ≈ [-1.80, 2.64]
 *
 * Full pipeline (uint8 pixel → int8 tensor value):
 *   f = pixel / 255.0f
 *   n = (f - INPUT_MEAN[c]) / INPUT_STD[c]
 *   q = clamp(round(n / INPUT_SCALE) + INPUT_ZP, -128, 127)
 */
#define INPUT_NORM_DIV          255.0f

/** ImageNet channel means [R, G, B]. */
#define INPUT_MEAN_R            0.485f
#define INPUT_MEAN_G            0.456f
#define INPUT_MEAN_B            0.406f

/** ImageNet channel standard deviations [R, G, B]. */
#define INPUT_STD_R             0.229f
#define INPUT_STD_G             0.224f
#define INPUT_STD_B             0.225f

/**
 * Static arrays for use in C/C++ code.
 * Usage: float n = (pixel/255.0f - INPUT_MEAN[c]) / INPUT_STD[c];
 */
static const float INPUT_MEAN[3] = { 0.485f, 0.456f, 0.406f };  /* R, G, B */
static const float INPUT_STD[3]  = { 0.229f, 0.224f, 0.225f };  /* R, G, B */

/** Pixel value range expected by the model BEFORE normalisation (raw camera output). */
#define INPUT_PIXEL_MIN         0
#define INPUT_PIXEL_MAX         255

/** Resize method used during pre-processing. */
#define INPUT_RESIZE_METHOD     "Bilinear (half-pixel centre convention)"


/* ══════════════════════════════════════════════════════════════════════
 * 3.  OUTPUT TENSOR
 *     Verified with: interpreter.get_output_details()[0]
 *     Name  : "PartitionedCall:0"
 *     Index : 0
 *     Shape : [1, 1000]
 *     DType : int8 (INT8 model) / float32 (FP32 model)
 * ══════════════════════════════════════════════════════════════════════ */

/** Flat size of one output tensor (= MODEL_NUM_CLASSES). */
#define MODEL_OUTPUT_SIZE       1000

/**
 * INT8 output dequantization scale.
 * Source: interpreter.get_output_details()[0]['quantization_parameters']['scales'][0]
 * Dequantize: logit = (out_q[i] - OUTPUT_ZP) * OUTPUT_SCALE
 *
 * ⚠ IMPORTANT: SqueezeNet's output tensor contains RAW LOGITS, NOT probabilities.
 *   You MUST apply softmax after dequantization before comparing confidences.
 *   Dequantized range ≈ [0.0, 46.7]  (unnormalised logit values).
 */
#define OUTPUT_SCALE            0.18399938941001892f

/**
 * INT8 output dequantization zero-point.
 * Source: interpreter.get_output_details()[0]['quantization_parameters']['zero_points'][0]
 */
#define OUTPUT_ZP               (-128)

/**
 * 0 = output is raw logits — softmax MUST be applied after dequantization.
 * (contrast with MobileNetV1/V2 where OUTPUT_HAS_SOFTMAX = 1)
 */
#define OUTPUT_HAS_SOFTMAX      0

/** Minimum dequantized logit value (q = -128 → (−128 − (−128)) × scale = 0.0). */
#define OUTPUT_LOGIT_MIN        0.0f

/** Maximum dequantized logit value (q = 127 → (127 − (−128)) × 0.184 ≈ 46.9). */
#define OUTPUT_LOGIT_MAX        46.919f   /* (127 - (-128)) * OUTPUT_SCALE */

/** Byte size of one INT8 output tensor buffer. */
#define MODEL_OUTPUT_BUF_BYTES  (MODEL_OUTPUT_SIZE * sizeof(int8_t))   /* 1 000 B */


/* ══════════════════════════════════════════════════════════════════════
 * 4.  MERA COMPILATION TARGETS
 * ══════════════════════════════════════════════════════════════════════ */

/**
 * CPU compiled C-code — TFLite INT8 (externally quantized).
 *   Source tree: TFLite_INT8_cpu_quantized_c/squeezenet1_1_INT8_CPU_external/
 */
#define MERA_CPU_INT8_TFLITE_DIR \
    "TFLite_INT8_cpu_quantized_c/squeezenet1_1_INT8_CPU_external"

/**
 * NPU compiled C-code — TFLite INT8 (ARM Ethos-U55, externally quantized).
 *   Source tree: TFLite_INT8_npu_quantized_c/squeezenet1_1_INT8_NPU_external/
 */
#define MERA_NPU_INT8_TFLITE_DIR \
    "TFLite_INT8_npu_quantized_c/squeezenet1_1_INT8_NPU_external"


#ifdef __cplusplus
}
#endif

#endif /* MODEL_METADATA_H */
