/*
 * Copyright (c) 2020 - 2026 Renesas Electronics Corporation and/or its affiliates
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
/**********************************************************************************************************************
 * File Name    : preprocessing.h
 * Description  : MCUNet input preprocessing declarations - bilinear resize and [-1, 1] float normalization.
 **********************************************************************************************************************/

#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include <stdint.h>

/**
 * @brief Convert a source RGB image into the int8 model input tensor.
 *
 * Pipeline:
 * 1. Bilinear resize (separable triangle filter, half-pixel centres).
 * 2. Centre-crop to [MODEL_INPUT_H x MODEL_INPUT_W].
 * 3. Normalize to float:  f = pixel / INPUT_NORM_SCALE + INPUT_NORM_BIAS
 *                           = pixel / 127.5 - 1.0   (range [-1.0, 1.0]).
 * 4. Quantize to int8:    q = clamp(round(f / INPUT_SCALE) + INPUT_ZP, -128, 127).
 * @note Not reentrant / not thread-safe. Uses module-private static
 *       scratch buffers for the resize intermediate and coefficient
 *       tables. Intended for single-threaded bare-metal use.
 *
 * @param p_source_image_hwc  Source uint8 image in HWC layout.
 * @param _source_width        Source image width in pixels.
 * @param _source_height       Source image height in pixels.
 * @param p_input_tensor      Destination int8 tensor (HWC, MODEL_INPUT_SIZE
 *                             elements) quantized per INPUT_SCALE / INPUT_ZP.
 */
void preprocess(const uint8_t *p_source_image_hwc,
                int _source_width,
                int _source_height,
                int8_t *p_input_tensor);

#endif /* PREPROCESSING_H */
