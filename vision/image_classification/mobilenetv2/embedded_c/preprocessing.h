/*
* Copyright (c) 2020 - 2025 Renesas Electronics Corporation and/or its affiliates
*
* SPDX-License-Identifier: BSD-3-Clause
*/
/**********************************************************************************************************************
 * File Name    : preprocessing.h
 * Description  : MobileNetV2 (NPU) input preprocessing declarations - bilinear resize, normalization, and INT8 quantization.
 **********************************************************************************************************************/

#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include <stdint.h>

/*******************************************************************************************************************//**
 * @brief Convert a source RGB image into the model input tensor.
 *
 * Pipeline:
 * 1. Bilinear resize to [MODEL_INPUT_H x MODEL_INPUT_W].
 * 2. Normalize using INPUT_NORM_SCALE and INPUT_NORM_BIAS.
 * 3. Quantize using INPUT_SCALE and INPUT_ZP.
 *
 * @param[in]  p_source_image       Source uint8 image in HWC layout.
 * @param[in]  source_width         Source image width in pixels.
 * @param[in]  source_height        Source image height in pixels.
 * @param[out] p_destination_tensor Destination int8 tensor in HWC layout.
 **********************************************************************************************************************/
void preprocess(const uint8_t *p_source_image,
				int source_width,
				int source_height,
				int8_t *p_destination_tensor);

#endif /* PREPROCESSING_H */
