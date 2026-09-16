/*
 * Copyright (c) 2020 - 2026 Renesas Electronics Corporation and/or its affiliates
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
/**********************************************************************************************************************
 * File Name    : model_metadata.h
 * Description  : Compile-time metadata for the MCUNet-in0 model deployed on the RA8P1 CPU target.
 *              : Source model : mcunet_in0_FP32.tflite (MIT HAN Lab mcunet-10fps_imagenet)
 *              : Input  : serving_default_input_0  shape=[1,48,48,3]  dtype=float32
 *              :          range ~ [-1.0, 1.0]  (pre-normalised by the host preprocessor)
 *              : Output : PartitionedCall_0_70154  shape=[1,1000]  dtype=float32
 *              :          raw logits (no softmax in graph)
 *              : MCUNet has no Softmax layer in the exported graph; apply softmax at
 *              : runtime for calibrated probabilities (top-K ordering is unaffected
 *              : since softmax is monotonic).
 **********************************************************************************************************************/

#ifndef MODEL_METADATA_H
#define MODEL_METADATA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ======================================================================
 * 1.  MODEL IDENTITY
 * ====================================================================== */
#define MODEL_NAME          ("MCUNet")
#define MODEL_FILE_INT8     ("mcunet_in0_INT8.tflite")
#define MODEL_FILE_FP32     ("mcunet_in0_FP32.tflite")
#define MODEL_TASK          ("Image Classification")
#define MODEL_DATASET       ("ImageNet ILSVRC-2012")
#define MODEL_NUM_CLASSES   (1000)

/* ======================================================================
 * 2.  INPUT TENSOR  [1, 48, 48, 3] float32
 * ====================================================================== */
#define MODEL_INPUT_H       (48)
#define MODEL_INPUT_W       (48)

#ifndef MODEL_CHANNELS
#define MODEL_INPUT_C       (3)
#else
#define MODEL_INPUT_C       (MODEL_CHANNELS)
#endif

#ifndef MODEL_INPUT_SIZE
#define MODEL_INPUT_SIZE    (MODEL_INPUT_H * MODEL_INPUT_W * MODEL_INPUT_C)
#endif
/* = 48 * 48 * 3 = 6 912 floats = 27 648 bytes */

#define MODEL_INPUT_LAYOUT  ("NHWC")
#define INPUT_SCALE         (0.007843137718737125f) 
#define INPUT_ZP            (-1)
/* Pixel normalization applied by the host preprocessor:
 *   normalized = raw_pixel / 127.5 - 1.0    ->  range [-1.0, 1.0] */
#define INPUT_NORM_SCALE    (127.5f)
#define INPUT_NORM_BIAS     (-1.0f)

#define INPUT_PIXEL_MIN     (0)
#define INPUT_PIXEL_MAX     (255)

#define INPUT_RESIZE_METHOD ("Bilinear (half-pixel centre convention)")

/* ======================================================================
 * 3.  OUTPUT TENSOR  [1, 1000] float32  (raw logits - no softmax in graph)
 * ====================================================================== */
#define MODEL_OUTPUT_SIZE   (1000)
#define OUTPUT_SCALE        (0.17176611721515656f)
#define OUTPUT_ZP           (-31)
#define OUTPUT_HAS_SOFTMAX  (1)

#ifdef __cplusplus
}
#endif

#endif /* MODEL_METADATA_H */
