/*
 * Copyright (c) 2020 - 2026 Renesas Electronics Corporation and/or its affiliates
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * MISRA C 2012 Compliant MobileFaceNet Preprocessing Declarations
 */
/**********************************************************************************************************************
 * File Name    : preprocessing.h
 * Description  :  Convert a pre-aligned 112x112 RGB face into the int8 model input tensor.
 *********************************************************************************************************************/

#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include <stdint.h>

/**
 * @brief Convert a pre-aligned 112x112 RGB face into the int8 model input
 *        tensor.
 *
 * Two-step pipeline:
 *   1. Normalise each uint8 pixel to float in [-1, 1]:
 *        float_norm = (float)pixel / 127.5f - 1.0f
 *   2. Quantize to int8 using the model's input quantization parameters
 *      (INPUT_QUANT_SCALE, INPUT_QUANT_ZP from model_metadata.h):
 *        q = clamp(round(float_norm / INPUT_QUANT_SCALE) + INPUT_QUANT_ZP,
 *                  -128, 127)
 *
 * @param p_source_image_hwc  Source uint8 face in HWC interleaved RGB layout
 *                             (exactly MOBILEFACENET_INPUT_BYTES bytes).
 * @param p_input_tensor      Destination int8 tensor (MODEL_INPUT_SIZE
 *                             elements, HWC), quantized per INPUT_QUANT_SCALE
 *                             / INPUT_QUANT_ZP.
 */
void preprocess(const uint8_t *p_source_image_hwc,
                int8_t        *p_input_tensor);

#endif /* PREPROCESSING_H */
