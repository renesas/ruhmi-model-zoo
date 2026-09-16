/*
* Copyright (c) 2020 - 2026 Renesas Electronics Corporation and/or its affiliates
*
* SPDX-License-Identifier: BSD-3-Clause
*/
/**********************************************************************************************************************
 * File Name    : model_metadata.h
 * Description  : YAMNet model input/output dimension constants shared across
 *                preprocessing and postprocessing modules.
 *********************************************************************************************************************/

#ifndef MODEL_METADATA_H
#define MODEL_METADATA_H

#include <stdint.h>

#define MODEL_INPUT_FRAMES      (96)
#define MODEL_INPUT_BANDS       (64)
#define MODEL_INPUT_ELEMENTS    (MODEL_INPUT_FRAMES * MODEL_INPUT_BANDS)

#define MODEL_OUTPUT_CLASSES    (521)
#define TOP_K_COUNT             (5)

/* Audio frontend geometry. */
#define MODEL_SAMPLE_RATE_HZ      (16000U)
#define MODEL_STFT_WINDOW_SAMPLES (400U)    /* 25 ms */
#define MODEL_STFT_HOP_SAMPLES    (160U)    /* 10 ms */
#define MODEL_PATCH_HOP_FRAMES    (48U)
#define MODEL_LOG_OFFSET          (0.001f)

/* Model input quantization: q = round(x / scale + zp), clipped to int16. */
#define MODEL_INPUT_SCALE         (0.0002108143963f)
#define MODEL_INPUT_ZERO_POINT    (0)

/* Model output dequantization: y = (q - zp) * scale. */
#define MODEL_OUTPUT_SCALE        (3.051757812e-05f)
#define MODEL_OUTPUT_ZERO_POINT   (0)

#endif /* MODEL_METADATA_H */
