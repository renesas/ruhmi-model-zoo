/*
 * Copyright (c) 2020 - 2026 Renesas Electronics Corporation and/or its affiliates
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * @file    preprocessing.h
 * @brief   ShuffleNetV2 x0.5 INT8 preprocessing - resize shortest edge + center crop + ImageNet normalize + quantize.
 */

#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include <stdint.h>

/**
 * @brief  Preprocess a raw RGB image for ShuffleNetV2 INT8 inference.
 *
 * Pipeline (matches Python inference.py exactly):
 *   1. Resize shortest edge to 256 using bilinear interpolation.
 *   2. Center-crop to 224×224.
 *   3. Normalize: (pixel/255 - mean) / std   per channel (ImageNet stats).
 *   4. Quantize: int8 = clamp(round(normalized / INPUT_SCALE) + INPUT_ZP, -128, 127).
 *
 * @param[in]  source_image   Raw RGB uint8 image, HWC layout.
 * @param[in]  source_width   Source image width in pixels.
 * @param[in]  source_height  Source image height in pixels.
 * @param[out] quantized_output  Output int8 tensor [224×224×3], HWC.
 */
void preprocess(const uint8_t *p_source_image,
                int            source_width,
                int            source_height,
                int8_t        *p_quantized_output);

#endif /* PREPROCESSING_H */
