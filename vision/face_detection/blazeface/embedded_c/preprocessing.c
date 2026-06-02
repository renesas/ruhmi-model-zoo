/**********************************************************************************************************************
 * Copyright (c) 2020 - 2026 Renesas Electronics Corporation and/or its affiliates
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * File Name    : preprocessing.c
 * Description  : BlazeFace preprocessing implementation.
 *********************************************************************************************************************/

#include "preprocessing.h"
#include "../common/model_metadata.h"
#include <string.h>

#define U8_MAX_VALUE  (255U)

/**
 * @brief Resize RGB888 image to model input size using bilinear interpolation.
 * @param[in] p_source_rgb Source RGB888 image buffer.
 * @param[in] source_width Source image width.
 * @param[in] source_height Source image height.
 * @param[out] resized_output Resized RGB888 output buffer
 *             [MODEL_INPUT_H][MODEL_INPUT_W][MODEL_INPUT_C].
 * @note Coordinate mapping matches align_corners=false behavior.
 * @return void
 */
static void resize_rgb888_bilinear(const uint8_t *p_source_rgb,
                                   int32_t        source_width,
                                   int32_t        source_height,
                                   uint8_t        resized_output[MODEL_INPUT_H][MODEL_INPUT_W][MODEL_INPUT_C])
{
    float x_scale = (float)source_width / (float)MODEL_INPUT_W;
    float y_scale = (float)source_height / (float)MODEL_INPUT_H;

    for (int dst_y = 0; dst_y < MODEL_INPUT_H; dst_y++) {
        float src_y_float = ((float)dst_y + 0.5f) * y_scale - 0.5f;
        int   src_y0 = (int)src_y_float;
        float y_weight = src_y_float - (float)src_y0;
        if (src_y0 < 0) {
            src_y0 = 0;
            y_weight = 0.0f;
        }
        int   src_y1 = src_y0 + 1;
        if (src_y1 >= source_height) {
            src_y1 = source_height - 1;
        }

        for (int dst_x = 0; dst_x < MODEL_INPUT_W; dst_x++) {
            float src_x_float = ((float)dst_x + 0.5f) * x_scale - 0.5f;
            int   src_x0 = (int)src_x_float;
            float x_weight = src_x_float - (float)src_x0;
            if (src_x0 < 0) {
                src_x0 = 0;
                x_weight = 0.0f;
            }
            int   src_x1 = src_x0 + 1;
            if (src_x1 >= source_width) {
                src_x1 = source_width - 1;
            }

            const uint8_t *p_top_left_pixel = p_source_rgb + (src_y0 * source_width + src_x0) * MODEL_INPUT_C;
            const uint8_t *p_top_right_pixel = p_source_rgb + (src_y0 * source_width + src_x1) * MODEL_INPUT_C;
            const uint8_t *p_bottom_left_pixel = p_source_rgb + (src_y1 * source_width + src_x0) * MODEL_INPUT_C;
            const uint8_t *p_bottom_right_pixel = p_source_rgb + (src_y1 * source_width + src_x1) * MODEL_INPUT_C;

            for (int channel = 0; channel < MODEL_INPUT_C; channel++) {
                float interpolated_value = (1.0f - y_weight) * ((1.0f - x_weight) * p_top_left_pixel[channel] + x_weight * p_top_right_pixel[channel])
                                       +          y_weight  * ((1.0f - x_weight) * p_bottom_left_pixel[channel] + x_weight * p_bottom_right_pixel[channel]);
                int rounded_value = (int)(interpolated_value + 0.5f);
                resized_output[dst_y][dst_x][channel] = (uint8_t)((rounded_value < 0) ? 0 : ((rounded_value > (int32_t)U8_MAX_VALUE) ? (int32_t)U8_MAX_VALUE : rounded_value));
            }
        }
    }
}

/**
 * @brief Resize and quantize source image into model input tensor.
 * @param[in] p_source_rgb Packed RGB888 input buffer.
 * @param[in] source_width Source image width.
 * @param[in] source_height Source image height.
 * @param[out] p_output_tensor Quantized output tensor.
 * @return void
 */
void preprocess(const uint8_t *p_source_rgb,
                int32_t        source_width,
                int32_t        source_height,
                int8_t         p_output_tensor[MODEL_INPUT_SIZE])
{
    /* Temporary resized buffer — static to avoid stack pressure on MCU. */
    static uint8_t resized_rgb[MODEL_INPUT_H][MODEL_INPUT_W][MODEL_INPUT_C];
    const uint8_t *p_resized_linear = &resized_rgb[0][0][0];

    if (((int32_t)MODEL_INPUT_W == source_width) && ((int32_t)MODEL_INPUT_H == source_height)) {
        (void)memcpy(resized_rgb, p_source_rgb, (size_t)MODEL_INPUT_SIZE);
    } else {
        resize_rgb888_bilinear(p_source_rgb, source_width, source_height, resized_rgb);
    }

    /* Quantise: q = pixel + INPUT_ZP  (INPUT_ZP = -128, so q = pixel - 128) */
    for (int index = 0; index < MODEL_INPUT_SIZE; index++) {
        p_output_tensor[index] = (int8_t)((int32_t)p_resized_linear[index] + INPUT_ZP);
    }
}
