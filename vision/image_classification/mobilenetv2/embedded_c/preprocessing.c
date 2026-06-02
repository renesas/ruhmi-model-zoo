/*
* Copyright (c) 2020 - 2025 Renesas Electronics Corporation and/or its affiliates
*
* SPDX-License-Identifier: BSD-3-Clause
*/
/**********************************************************************************************************************
 * File Name    : preprocessing.c
 * Description  : MobileNetV2 (NPU) input preprocessing - bilinear resize, normalization, and INT8 quantization.
 **********************************************************************************************************************/

#include "preprocessing.h"
#include "../common/model_metadata.h"

/*******************************************************************************************************************//**
 * @brief Preprocess an RGB image into MobileNet INT8 input tensor format.
 * @details Performs bilinear resize to model dimensions, applies normalization,
 *          then quantizes to int8 using model metadata scale and zero-point.
 * @param[in]  p_source_image       Source RGB image in HWC uint8 layout.
 * @param[in]  source_width         Source image width in pixels.
 * @param[in]  source_height        Source image height in pixels.
 * @param[out] p_destination_tensor Destination INT8 tensor in model HWC layout.
 **********************************************************************************************************************/
void preprocess(const uint8_t *p_source_image,
                int source_width,
                int source_height,
                int8_t *p_destination_tensor)
{
    const float horizontal_scale = (float)source_width / MODEL_INPUT_W;
    const float vertical_scale = (float)source_height / MODEL_INPUT_H;
    int output_y;

    /* Resize the source image into the model input grid using bilinear sampling. */
    for (output_y = 0; output_y < MODEL_INPUT_H; output_y++) {
        float source_y_position;
        int source_y_index0;
        int source_y_index1;
        float vertical_weight;
        int output_x;

        /* Compute the source-space Y coordinate with half-pixel center alignment. */
        source_y_position = ((float)output_y + 0.5f) * vertical_scale - 0.5f;
        source_y_index0 = (int)source_y_position;
        source_y_index1 = source_y_index0 + 1;
        vertical_weight = source_y_position - (float)source_y_index0;

        /* Clamp vertical sample indices so interpolation stays inside the source image. */
        if (source_y_index0 < 0) {
            source_y_index0 = 0;
        }
        if (source_y_index1 >= source_height) {
            source_y_index1 = source_height - 1;
        }

        for (output_x = 0; output_x < MODEL_INPUT_W; output_x++) {
            float source_x_position;
            int source_x_index0;
            int source_x_index1;
            float horizontal_weight;
            int channel_index;

            /* Compute the corresponding X coordinate in the source image. */
            source_x_position = ((float)output_x + 0.5f) * horizontal_scale - 0.5f;
            source_x_index0 = (int)source_x_position;
            source_x_index1 = source_x_index0 + 1;
            horizontal_weight = source_x_position - (float)source_x_index0;

            /* Clamp horizontal sample indices so edge pixels are reused at the border. */
            if (source_x_index0 < 0) {
                source_x_index0 = 0;
            }
            if (source_x_index1 >= source_width) {
                source_x_index1 = source_width - 1;
            }

            for (channel_index = 0; channel_index < MODEL_INPUT_C; channel_index++) {
                float top_left_sample;
                float top_right_sample;
                float bottom_left_sample;
                float bottom_right_sample;
                float interpolated_top_row;
                float interpolated_bottom_row;
                float resized_pixel;
                float normalized_pixel;
                float quantized_value_float;
                int quantized_value;

                /* Read the 2x2 source neighborhood used for bilinear interpolation. */
                top_left_sample = (float)p_source_image[((source_y_index0 * source_width) + source_x_index0) * MODEL_INPUT_C + channel_index];
                top_right_sample = (float)p_source_image[((source_y_index0 * source_width) + source_x_index1) * MODEL_INPUT_C + channel_index];
                bottom_left_sample = (float)p_source_image[((source_y_index1 * source_width) + source_x_index0) * MODEL_INPUT_C + channel_index];
                bottom_right_sample = (float)p_source_image[((source_y_index1 * source_width) + source_x_index1) * MODEL_INPUT_C + channel_index];

                /* Interpolate first across X, then across Y, to obtain the resized pixel. */
                interpolated_top_row = (top_left_sample * (1.0f - horizontal_weight)) + (top_right_sample * horizontal_weight);
                interpolated_bottom_row = (bottom_left_sample * (1.0f - horizontal_weight)) + (bottom_right_sample * horizontal_weight);
                resized_pixel = (interpolated_top_row * (1.0f - vertical_weight)) + (interpolated_bottom_row * vertical_weight);

                /* Normalize the uint8 pixel into the floating-point range expected by the model. */
                normalized_pixel = (resized_pixel / INPUT_NORM_SCALE) + INPUT_NORM_BIAS;

                /* Convert the normalized value into the model's int8 quantized domain. */
                quantized_value_float = (normalized_pixel / INPUT_SCALE) + (float)INPUT_ZP;

                /* Apply symmetric rounding and saturate to the representable int8 range. */
                if (quantized_value_float >= 0.0f) {
                    quantized_value = (int)(quantized_value_float + 0.5f);
                } else {
                    quantized_value = (int)(quantized_value_float - 0.5f);
                }

                if (quantized_value < -128) {
                    quantized_value = -128;
                }
                if (quantized_value > 127) {
                    quantized_value = 127;
                }

                /* Store the final quantized HWC tensor element for this pixel and channel. */
                p_destination_tensor[((output_y * MODEL_INPUT_W) + output_x) * MODEL_INPUT_C + channel_index] = (int8_t)quantized_value;
            }
        }
    }
}
