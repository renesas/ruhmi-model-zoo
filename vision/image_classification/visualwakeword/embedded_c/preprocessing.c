/**
 * @file    preprocessing.c
 * @brief   Visual Wake Words (VWW, NPU) input preprocessing - bilinear resize and INT8 quantization.
 *
 * @author  Renesas Electronics
 * @date    2025
 * @version 1.0
 */

#include "preprocessing.h"

/**
 * @brief Preprocess an RGB image into VWW INT8 input tensor format.
 * @details Resizes input with bilinear interpolation, applies model-specific
 *          normalization, and quantizes to int8 tensor values.
 * @param[in] p_source_image Source RGB image in HWC uint8 layout.
 * @param[in] source_width Source image width in pixels.
 * @param[in] source_height Source image height in pixels.
 * @param[out] p_destination_tensor Destination INT8 tensor in model HWC layout.
 */
void preprocess(const uint8_t *p_source_image, int source_width, int source_height,
                int8_t *p_destination_tensor)
{
    /* Step 1: compute scaling factors from source image size to model input size. */
    const float horizontal_scale = (float)source_width / MODEL_INPUT_W;
    const float vertical_scale = (float)source_height / MODEL_INPUT_H;
    const int int8_min = -128;
    const int int8_max = 127;

    for (int destination_row = 0; destination_row < MODEL_INPUT_H; destination_row++)
    {
        /* Half-pixel centre convention (matches PIL/OpenCV BILINEAR) */
        float source_y_position = (destination_row + 0.5f) * vertical_scale - 0.5f;
        int   source_y_index0 = (int)source_y_position;
        int   source_y_index1 = source_y_index0 + 1;
        float vertical_weight = source_y_position - (float)source_y_index0;
        if (source_y_index0 < 0)
        {
            source_y_index0 = 0;
        }
        if (source_y_index1 >= source_height)
        {
            source_y_index1 = source_height - 1;
        }

        for (int destination_col = 0; destination_col < MODEL_INPUT_W; destination_col++)
        {
            float source_x_position = (destination_col + 0.5f) * horizontal_scale - 0.5f;
            int   source_x_index0 = (int)source_x_position;
            int   source_x_index1 = source_x_index0 + 1;
            float horizontal_weight = source_x_position - (float)source_x_index0;
            if (source_x_index0 < 0)
            {
                source_x_index0 = 0;
            }
            if (source_x_index1 >= source_width)
            {
                source_x_index1 = source_width - 1;
            }

            for (int channel_index = 0; channel_index < MODEL_INPUT_C; channel_index++)
            {
                /* Bilinear interpolation over uint8 pixels */
                float top_left_pixel = (float)p_source_image[(source_y_index0 * source_width + source_x_index0) * MODEL_INPUT_C + channel_index];
                float top_right_pixel = (float)p_source_image[(source_y_index0 * source_width + source_x_index1) * MODEL_INPUT_C + channel_index];
                float bottom_left_pixel = (float)p_source_image[(source_y_index1 * source_width + source_x_index0) * MODEL_INPUT_C + channel_index];
                float bottom_right_pixel = (float)p_source_image[(source_y_index1 * source_width + source_x_index1) * MODEL_INPUT_C + channel_index];

                float top_row_interpolated = top_left_pixel * (1.0f - horizontal_weight) + top_right_pixel * horizontal_weight;
                float bottom_row_interpolated = bottom_left_pixel * (1.0f - horizontal_weight) + bottom_right_pixel * horizontal_weight;
                float resized_pixel = top_row_interpolated * (1.0f - vertical_weight) + bottom_row_interpolated * vertical_weight;

                /* Normalize using model_metadata.h normalization parameters. */
                float normalized_pixel = (resized_pixel / INPUT_NORM_SCALE) + INPUT_NORM_BIAS;

                /* Quantize using INPUT_SCALE and INPUT_ZP from model_metadata.h */
                float quantized_value_float = normalized_pixel / INPUT_SCALE + (float)INPUT_ZP;

                /* Round away from zero, clamp to [-128, 127] */
                int quantized_value = (int)(quantized_value_float >= 0.0f ? quantized_value_float + 0.5f : quantized_value_float - 0.5f);
                if (quantized_value < int8_min)
                {
                    quantized_value = int8_min;
                }
                if (quantized_value > int8_max)
                {
                    quantized_value = int8_max;
                }

                /* Step 5: store quantized pixel in destination HWC tensor buffer. */
                p_destination_tensor[(destination_row * MODEL_INPUT_W + destination_col) * MODEL_INPUT_C + channel_index] = (int8_t)quantized_value;
            }
        }
    }
}
