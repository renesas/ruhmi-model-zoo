/**
 * @file    model_metadata.h
 * @brief   Compile-time metadata for ShuffleNetV2 x0.5 INT8 (224×224 input).
 *
 * Model  : shufflenet_v2_x0_5_INT8 (EdgeCortix / Renesas)
 * Task   : Image Classification (ImageNet 1000-class)
 * Date   : 2026-05-13
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
#define MODEL_NAME          "ShuffleNetV2 x0.5"
#define MODEL_TASK          "Image Classification"
#define MODEL_DATASET       "ImageNet ILSVRC-2012"
#define MODEL_DTYPE         "int8"
#define MODEL_NUM_CLASSES   (1000)

/* ══════════════════════════════════════════════════════════════════════
 * 2.  INPUT TENSOR  [1, 224, 224, 3]  int8
 * ══════════════════════════════════════════════════════════════════════ */
#define MODEL_INPUT_H       (224)
#define MODEL_INPUT_W       (224)
#define MODEL_INPUT_C       (3)
#define MODEL_INPUT_SIZE    (MODEL_INPUT_H * MODEL_INPUT_W * MODEL_INPUT_C)
/* = 150,528 bytes */

/** Input quantization parameters from TFLite model. */
#define INPUT_SCALE         (0.01865845f)
#define INPUT_ZP            (-14)

/**
 * ImageNet channel-wise normalization (applied BEFORE quantization).
 *   normalized = (pixel / 255.0 - mean) / std
 *
 * Python reference:
 *   arr = np.array(img) / 255.0
 *   arr = (arr - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
 */
#define INPUT_NORM_MEAN_R   (0.485f)
#define INPUT_NORM_MEAN_G   (0.456f)
#define INPUT_NORM_MEAN_B   (0.406f)
#define INPUT_NORM_STD_R    (0.229f)
#define INPUT_NORM_STD_G    (0.224f)
#define INPUT_NORM_STD_B    (0.225f)

/**
 * Preprocessing: resize shortest edge to 256, center-crop to 224×224.
 */
#define RESIZE_SHORT_EDGE   (256)

/* ══════════════════════════════════════════════════════════════════════
 * 3.  OUTPUT TENSOR  [1, 1000]  int8  (raw logits, no softmax)
 * ══════════════════════════════════════════════════════════════════════ */
#define MODEL_OUTPUT_SIZE   (1000)
#define OUTPUT_SCALE        (0.18189311f)
#define OUTPUT_ZP           (-57)
#define OUTPUT_HAS_SOFTMAX  (1)

/* ══════════════════════════════════════════════════════════════════════
 * 4.  POST-PROCESSING
 * ══════════════════════════════════════════════════════════════════════ */
#define TOP_K               (5)

/* ══════════════════════════════════════════════════════════════════════
 * 5.  TEST CONFIGURATION
 * ══════════════════════════════════════════════════════════════════════ */
#define NUM_TEST_IMAGES     (3U)

#ifdef __cplusplus
}
#endif

#endif /* MODEL_METADATA_H */
