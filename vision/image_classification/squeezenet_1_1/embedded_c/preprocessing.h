/*
 * Copyright (C) 2025 Renesas Electronics Corporation.
 *
 * This software is supplied by Renesas Electronics Corporation and is only
 * intended for use with Renesas products. No other uses are authorized.
 */

/**
 * @file    preprocessing.h
 * @brief   SqueezeNet (NPU) input preprocessing declarations - bilinear resize, mean/std normalization, and INT8 quantization.
 *
 * @author  Renesas Electronics
 * @date    2025
 * @version 1.0
 */

#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include <stdint.h>
#include "../common/model_metadata.h"

/**
 * Bilinear resize a uint8 RGB source image to [MODEL_INPUT_W x MODEL_INPUT_H x MODEL_INPUT_C],
 * apply ImageNet mean/std normalization, then quantize to int8.
 *
 * Steps per pixel:
 *   1. Bilinear interpolation  (PIL/OpenCV half-pixel centre convention)
 *   2. Scale  : f = pixel / INPUT_NORM_DIV
 *   3. Normalize : f = (f - INPUT_MEAN[c]) / INPUT_STD[c]
 *   4. Quantize  : q = round(f / INPUT_SCALE) + INPUT_ZP
 *   5. Clamp     : q = clamp(q, -128, 127)
 *
 * @param p_source_image       Source image, HWC layout, uint8, [0, 255].
 * @param source_width       Source image width in pixels.
 * @param source_height      Source image height in pixels.
 * @param p_destination_tensor Output buffer [MODEL_INPUT_SIZE] int8.
 */
void preprocess(const uint8_t *p_source_image,
                int source_width,
                int source_height,
                int8_t *p_destination_tensor);

#endif /* PREPROCESSING_H */
