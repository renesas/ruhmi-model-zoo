/**
 * @file    preprocessing.h
 * @brief   MobileNetV1 input preprocessing declarations - bilinear resize, normalization, and INT8 quantization.
 *
 * @author  Renesas Electronics
 * @date    2025
 * @version 1.0
 */

/*
 * Copyright (C) 2025 Renesas Electronics Corporation. All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include <stdint.h>

/**
 * @brief Convert a source RGB image into the model input tensor.
 *
 * Pipeline:
 * 1. Bilinear resize to [MODEL_INPUT_H x MODEL_INPUT_W].
 * 2. Normalize using INPUT_NORM_SCALE and INPUT_NORM_BIAS.
 * 3. Quantize using INPUT_SCALE and INPUT_ZP.
 *
 * @param _p_source_image_hwc       Source uint8 image in HWC layout.
 * @param _source_width             Source image width in pixels.
 * @param _source_height            Source image height in pixels.
 * @param _p_quantized_input_tensor Destination int8 tensor in HWC layout.
 */
void preprocess(const uint8_t *_p_source_image_hwc,
                int _source_width,
                int _source_height,
                int8_t *_p_quantized_input_tensor);

#endif /* PREPROCESSING_H */
