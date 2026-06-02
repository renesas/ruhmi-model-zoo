/**
 * @file    model_metadata.h
 * @brief   Compile-time metadata for MobileNetV1 INT8 TFLite model.
 *
 *
 * Generated from:
 *   Model  : mobilenet_v1_INT8.tflite
 *   Python : inference.py  (MobileNetV1, ImageNet 1000-class, INT8)
 *   Date   : 2026-04-02
 *
 * ──────────────────────────────────────────────────────────────────────
 * QUICK-START (bare-metal / TFLite Micro)
 * ──────────────────────────────────────────────────────────────────────
 *
 *  1. Feed  : raw uint8 RGB image  (any resolution)
 *  2. Resize: bilinear → [MODEL_INPUT_H × MODEL_INPUT_W]
 *  3. Pre-process each pixel:
 *       float f = pixel / 127.5f - 1.0f;                  // [-1, 1]
 *       int8  q = clamp(round(f / INPUT_SCALE) + INPUT_ZP, -128, 127);
 *  4. Copy q into the INT8 input tensor  [1×224×224×3]
 *  5. Run inference
 *  6. Read int8 output tensor  [1×1000]
 *  7. De-quantize each output element:
 *       float prob = (out_q[i] - OUTPUT_ZP) * OUTPUT_SCALE;
 *       // prob is already a probability (softmax baked in); range [0, ~1]
 *  8. argsort descending → top-K class indices → look up CLASS_NAMES[]
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
#define MODEL_NAME          "MobileNetV1"

/** Relative path to the INT8 TFLite flatbuffer (from project root). */
#define MODEL_FILE_INT8     "mobilenet_v1_INT8.tflite"

/** Relative path to the FP32 TFLite flatbuffer (for reference / validation). */
#define MODEL_FILE_FP32     "mobilenet_v1_FP32.tflite"

/** Task performed by the model. */
#define MODEL_TASK          "Image Classification"

/** Dataset the model was trained on. */
#define MODEL_DATASET       "ImageNet ILSVRC-2012"

/** Number of output classes. */
#define MODEL_NUM_CLASSES   1000


/* ══════════════════════════════════════════════════════════════════════
 * 2.  INPUT TENSOR
 *     Verified with: interpreter.get_input_details()[0]
 *     Name  : "serving_default_input_2:0"
 *     Index : 0
 *     Shape : [1, 224, 224, 3]
 *     DType : int8
 * ══════════════════════════════════════════════════════════════════════ */

/** Expected input height in pixels. */
#define MODEL_INPUT_H       224

/** Expected input width in pixels. */
#define MODEL_INPUT_W       224

/** Number of colour channels (RGB). */
#ifndef MODEL_CHANNELS          /* also defined in preprocess.h as MODEL_CHANNELS */
#define MODEL_INPUT_C       3
#else
#define MODEL_INPUT_C       MODEL_CHANNELS
#endif

/** Flat size of one input tensor (H × W × C). */
#ifndef MODEL_INPUT_SIZE        /* preprocess.h defines this too — avoid redefinition */
#define MODEL_INPUT_SIZE    (MODEL_INPUT_H * MODEL_INPUT_W * MODEL_INPUT_C)
#endif
/* = 224 × 224 × 3 = 150528 bytes */

/** Memory layout of the input tensor. */
#define MODEL_INPUT_LAYOUT  "NHWC"   /* batch=1, Height, Width, Channels */

/**
 * INT8 input quantization scale.
 * Source: interpreter.get_input_details()[0]['quantization'][0]
 * Maps int8 value q → float f via:  f = (q - INPUT_ZP) * INPUT_SCALE
 * Or to quantize:  q = clamp(round(f / INPUT_SCALE) + INPUT_ZP, -128, 127)
 */
#define INPUT_SCALE         0.007843137718737125f   /* ≈ 1/127.5 */

/**
 * INT8 input quantization zero-point.
 * Source: interpreter.get_input_details()[0]['quantization'][1]
 */
#define INPUT_ZP            (-1)

/**
 * Pixel normalization applied BEFORE quantization (MobileNetV1 standard):
 *   normalized = raw_pixel / 127.5 - 1.0    →  range [-1.0, 1.0]
 *
 * Full pipeline (uint8 pixel → int8 tensor value):
 *   f = pixel / 127.5f - 1.0f
 *   q = clamp( round(f / INPUT_SCALE) + INPUT_ZP , -128, 127 )
 *
 * Numerically equivalent shortcut:
 *   q = clamp( round(pixel / 127.5f / INPUT_SCALE
 *                    - 1.0f / INPUT_SCALE
 *                    + INPUT_ZP) , -128, 127 )
 *   ≈  clamp( round(pixel - 128) , -128, 127 )   [for INPUT_SCALE≈1/128]
 */
#define INPUT_NORM_SCALE    127.5f   /* divisor  */
#define INPUT_NORM_BIAS     (-1.0f)  /* subtracted after division */

/** Pixel value range expected by the model BEFORE normalisation. */
#define INPUT_PIXEL_MIN     0
#define INPUT_PIXEL_MAX     255

/** Resize method used during pre-processing. */
#define INPUT_RESIZE_METHOD "Bilinear (half-pixel centre convention)"


/* ══════════════════════════════════════════════════════════════════════
 * 3.  OUTPUT TENSOR
 *     Verified with: interpreter.get_output_details()[0]
 *     Name  : "StatefulPartitionedCall:0"
 *     Index : 94
 *     Shape : [1, 1000]
 *     DType : int8
 * ══════════════════════════════════════════════════════════════════════ */

/** Flat size of one output tensor (= MODEL_NUM_CLASSES). */
#define MODEL_OUTPUT_SIZE   1000

/**
 * INT8 output dequantization scale.
 * Source: interpreter.get_output_details()[0]['quantization'][0]
 * Dequantize: prob = (out_q - OUTPUT_ZP) * OUTPUT_SCALE
 * Dequantized range ≈ [0.0, 0.996] — already a valid probability distribution.
 */
#define OUTPUT_SCALE        0.00390625f   /* = 1/256 */

/**
 * INT8 output dequantization zero-point.
 * Source: interpreter.get_output_details()[0]['quantization'][1]
 */
#define OUTPUT_ZP           (-128)

/**
 * The model graph includes a Softmax layer at the output
 * (classifier_activation='softmax' at training time).
 * The output is ALREADY a probability distribution — DO NOT apply softmax
 * again after dequantization; doing so would push predictions toward
 * uniform and destroy accuracy.
 *
 * Post-processing (int8 tensor → top-K class):
 *   for (int i = 0; i < MODEL_OUTPUT_SIZE; i++)
 *       prob[i] = (out_q[i] - OUTPUT_ZP) * OUTPUT_SCALE;
 *   // prob[i] ∈ [0, ~1], sum(prob) ≈ 1.0
 *   // argsort descending → take first K indices
 */
#define OUTPUT_HAS_SOFTMAX  1   /* 1 = softmax already applied in graph */

/** Minimum valid dequantized output value (zero_point maps to 0.0). */
#define OUTPUT_FLOAT_MIN    0.0f

/** Maximum possible dequantized output value (127 - (-128)) * (1/256). */
#define OUTPUT_FLOAT_MAX    0.99609375f   /* (127 - (-128)) * (1/256) */


/* ══════════════════════════════════════════════════════════════════════
 * 4.  MEMORY FOOTPRINT  (approximate, read from tflite flatbuffer)
 * ══════════════════════════════════════════════════════════════════════ */

/** Input tensor buffer size in bytes. */
#define MODEL_INPUT_BUF_BYTES    MODEL_INPUT_SIZE            /* 150 528 B ≈ 147 KB */

/** Output tensor buffer size in bytes. */
#define MODEL_OUTPUT_BUF_BYTES   MODEL_OUTPUT_SIZE           /* 1 000 B */

/** Approximate working / activation RAM required at inference time (bytes).
 *  Measure precisely with TFLite Micro arena size tuning. */
#define MODEL_ARENA_SIZE_BYTES   (512 * 1024)                /* ~512 KB — tune for target */


/* ══════════════════════════════════════════════════════════════════════
 * 5.  COMPLETE PREPROCESSING PIPELINE REFERENCE
 *     (mirrors inference.py step-by-step)
 * ══════════════════════════════════════════════════════════════════════
 *
 * Python:
 *   img   = Image.open(path).convert("RGB")
 *   img   = img.resize((224, 224), Image.BILINEAR)
 *   arr   = np.array(img, dtype=np.float32)          # [H,W,3], [0,255]
 *   arr   = arr / 127.5 - 1.0                         # [H,W,3], [-1,1]
 *   arr_q = np.round(arr / 0.007843137718737125)
 *           + (-1)                                    # quantize
 *   arr_q = np.clip(arr_q, -128, 127).astype(np.int8) # → input tensor
 *
 *
 * ══════════════════════════════════════════════════════════════════════ */


/* ══════════════════════════════════════════════════════════════════════
 * 6.  COMPLETE POSTPROCESSING PIPELINE REFERENCE
 *     (mirrors inference.py step-by-step)
 * ══════════════════════════════════════════════════════════════════════
 *
 * Python:
 *   out_q  = interpreter.get_tensor(output_index)    # shape (1,1000) int8
 *   probs  = (out_q.astype(np.float32) - OUTPUT_ZP) * OUTPUT_SCALE
 *                                                     # already probabilities
 *   top5   = np.argsort(probs[0])[::-1][:5]
 *   for i in top5:
 *       print(labels[i], probs[0][i])
 *
 * Equivalent C:
 *
 *   float prob[MODEL_OUTPUT_SIZE];
 *   for (int i = 0; i < MODEL_OUTPUT_SIZE; i++)
 *       prob[i] = ((float)out_q[i] - (float)OUTPUT_ZP) * OUTPUT_SCALE;
 *   // DO NOT apply softmax — output is already a probability distribution
 *   // Run top-K selection on prob[]
 *   // Map index → CLASS_NAMES[index]   (see labels.h / labels.c)
 *
 * ══════════════════════════════════════════════════════════════════════ */



/* ══════════════════════════════════════════════════════════════════════
 * 8.  LABELS
 * ══════════════════════════════════════════════════════════════════════ */

/**
 * The 1000 ImageNet class name strings are provided by labels.c / labels.h.
 *
 *   #include "labels.h"
 *   const char **names = get_class_names();   // names[i] for class i
 *   int          n     = get_num_classes();   // == MODEL_NUM_CLASSES
 */


#ifdef __cplusplus
}
#endif

#endif /* MODEL_METADATA_H */
