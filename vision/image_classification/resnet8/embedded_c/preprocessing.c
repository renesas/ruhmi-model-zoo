/**
 ******************************************************************************
 * @file    preprocessing.c
 * @brief   Portable NN image preprocessing
 *
 ******************************************************************************
 */

#include "preprocessing.h"
#include "../common/model_metadata.h"

void preprocess(const uint8_t *source_image,
                uint16_t source_width, uint16_t source_height,
                float    *destination_image,
                uint16_t destination_width, uint16_t destination_height)
{
    int32_t destination_row_index;
    int32_t destination_column_index;
    int32_t channel_index;
    const float horizontal_scale = (float)source_width / (float)destination_width;
    const float vertical_scale = (float)source_height / (float)destination_height;
    float source_row_position;
    int32_t source_row_low;
    int32_t source_row_high;
    float row_weight;
    float source_column_position;
    int32_t source_column_low;
    int32_t source_column_high;
    float column_weight;
    float top_left_pixel;
    float top_right_pixel;
    float bottom_left_pixel;
    float bottom_right_pixel;
    float upper_row_blend;
    float lower_row_blend;

    /* AI reference preprocessing does not apply channel swapping. */

    /* Map each destination row to the corresponding source rows. */
    for (destination_row_index = 0; destination_row_index < (int32_t)destination_height; destination_row_index++)
    {
        source_row_position = ((float)destination_row_index + 0.5F) * vertical_scale - 0.5F;
        source_row_low = (int32_t)source_row_position;
        source_row_high = source_row_low + 1;
        row_weight = source_row_position - (float)source_row_low;

        /* Clamp the vertical sample window at the image border. */
        if (source_row_low < 0)
        {
            source_row_low = 0;
        }
        if (source_row_high >= (int32_t)source_height)
        {
            source_row_high = (int32_t)source_height - 1;
        }

        /* Map each destination column to the corresponding source columns. */
        for (destination_column_index = 0; destination_column_index < (int32_t)destination_width; destination_column_index++)
        {
            source_column_position = ((float)destination_column_index + 0.5F) * horizontal_scale - 0.5F;
            source_column_low = (int32_t)source_column_position;
            source_column_high = source_column_low + 1;
            column_weight = source_column_position - (float)source_column_low;

            /* Clamp the horizontal sample window at the image border. */
            if (source_column_low < 0)
            {
                source_column_low = 0;
            }
            if (source_column_high >= (int32_t)source_width)
            {
                source_column_high = (int32_t)source_width - 1;
            }

            /* Interpolate each RGB channel independently. */
            for (channel_index = 0; channel_index < MODEL_INPUT_C; channel_index++)
            {
                /* Read the four neighboring source pixels used by bilinear interpolation. */
                top_left_pixel = (float)source_image[((source_row_low * (int32_t)source_width) + source_column_low) * MODEL_INPUT_C + channel_index];
                top_right_pixel = (float)source_image[((source_row_low * (int32_t)source_width) + source_column_high) * MODEL_INPUT_C + channel_index];
                bottom_left_pixel = (float)source_image[((source_row_high * (int32_t)source_width) + source_column_low) * MODEL_INPUT_C + channel_index];
                bottom_right_pixel = (float)source_image[((source_row_high * (int32_t)source_width) + source_column_high) * MODEL_INPUT_C + channel_index];

                /* Blend horizontally first, then vertically, to produce the output sample. */
                upper_row_blend = (top_left_pixel * (1.0F - column_weight)) + (top_right_pixel * column_weight);
                lower_row_blend = (bottom_left_pixel * (1.0F - column_weight)) + (bottom_right_pixel * column_weight);

                destination_image[((destination_row_index * (int32_t)destination_width) + destination_column_index) * MODEL_INPUT_C + channel_index] =
                    (upper_row_blend * (1.0F - row_weight)) + (lower_row_blend * row_weight);
            }
        }
    }
}
