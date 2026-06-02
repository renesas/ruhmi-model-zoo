/**********************************************************************************************************************
 * Copyright (c) 2020 - 2026 Renesas Electronics Corporation and/or its affiliates
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * File Name    : preprocessing.h
 * Description  : Public API for BlazeFace input preprocessing.
 *********************************************************************************************************************/

#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include <stdint.h>
#include "../common/model_metadata.h"

/**
 * @brief Resize and quantize an RGB888 image for BlazeFace inference.
 * @param[in] p_source_rgb Packed RGB888 input buffer of size
 *            source_width * source_height * 3 bytes.
 * @param[in] source_width Source image width in pixels.
 * @param[in] source_height Source image height in pixels.
 * @param[out] p_output_tensor Output INT8 tensor buffer of size MODEL_INPUT_SIZE.
 * @note The output tensor is written in NHWC order.
 * @return void
 */
void preprocess(const uint8_t *p_source_rgb,
                int32_t        source_width,
                int32_t        source_height,
                int8_t         p_output_tensor[MODEL_INPUT_SIZE]);

#endif /* PREPROCESSING_H */
