/**
 * @file    preprocessing.c
 * @brief   MobileNetV1 input preprocessing - bilinear resize, normalization, and INT8 quantization.
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

#include "../common/model_metadata.h"
#include "preprocessing.h"

/**
 * @brief Preprocess an RGB image into MobileNet INT8 input tensor format.
 * @details Performs bilinear resize to model dimensions, applies normalization,
 *          then quantizes to int8 using model metadata scale and zero-point.
 * @param[in] _p_source_image_hwc Source RGB image in HWC uint8 layout.
 * @param[in] _source_width Source image width in pixels.
 * @param[in] _source_height Source image height in pixels.
 * @param[out] _p_quantized_input_tensor Destination INT8 tensor in model HWC layout.
 */
void preprocess(const uint8_t *_p_source_image_hwc,
                int _source_width,
                int _source_height,
                int8_t *_p_quantized_input_tensor)
{
    const float _half_pixel = 0.5F;
    const float _one        = 1.0F;
    const float _zero       = 0.0F;
    const int   _q_min      = -128;
    const int   _q_max      = 127;
    const float _scale_x = (float)_source_width / (float)MODEL_INPUT_W;
    const float _scale_y = (float)_source_height / (float)MODEL_INPUT_H;
    int _destination_row;

    /* Iterate over each destination row and map it back to source space. */
    for (_destination_row = 0; _destination_row < MODEL_INPUT_H; _destination_row++)
    {
        /* Half-pixel center convention (matches PIL bilinear behavior). */
        float _source_y_float = ((float)_destination_row + _half_pixel) * _scale_y - _half_pixel;
        int   _source_y0 = (int)_source_y_float;
        int   _source_y1 = _source_y0 + 1;
        float _weight_y = _source_y_float - (float)_source_y0;
        int _destination_col;

        /* Clamp vertical sampling coordinates to valid source bounds. */
        if (0 > _source_y0)
        {
            _source_y0 = 0;
        }
        if (_source_y1 >= _source_height)
        {
            _source_y1 = _source_height - 1;
        }

        /* Iterate over each destination column and map it back to source space. */
        for (_destination_col = 0; _destination_col < MODEL_INPUT_W; _destination_col++)
        {
            float _source_x_float = ((float)_destination_col + _half_pixel) * _scale_x - _half_pixel;
            int   _source_x0 = (int)_source_x_float;
            int   _source_x1 = _source_x0 + 1;
            float _weight_x = _source_x_float - (float)_source_x0;
            int _channel_index;

            /* Clamp horizontal sampling coordinates to valid source bounds. */
            if (0 > _source_x0)
            {
                _source_x0 = 0;
            }
            if (_source_x1 >= _source_width)
            {
                _source_x1 = _source_width - 1;
            }

            /* Process each color channel independently (HWC layout). */
            for (_channel_index = 0; _channel_index < MODEL_INPUT_C; _channel_index++)
            {
                /* Fetch the four neighboring source pixels used for interpolation. */
                float _pixel_00 = (float)_p_source_image_hwc[(_source_y0 * _source_width + _source_x0) * MODEL_INPUT_C + _channel_index];
                float _pixel_10 = (float)_p_source_image_hwc[(_source_y0 * _source_width + _source_x1) * MODEL_INPUT_C + _channel_index];
                float _pixel_01 = (float)_p_source_image_hwc[(_source_y1 * _source_width + _source_x0) * MODEL_INPUT_C + _channel_index];
                float _pixel_11 = (float)_p_source_image_hwc[(_source_y1 * _source_width + _source_x1) * MODEL_INPUT_C + _channel_index];

                /* Bilinear interpolation along X then Y. */
                float _interp_top    = _pixel_00 * (_one - _weight_x) + _pixel_10 * _weight_x;
                float _interp_bottom = _pixel_01 * (_one - _weight_x) + _pixel_11 * _weight_x;
                float _interp_pixel  = _interp_top  * (_one - _weight_y) + _interp_bottom * _weight_y;

                /* Normalize uint8 pixel to model range [-1, 1]. */
                float _normalized = _interp_pixel / INPUT_NORM_SCALE + INPUT_NORM_BIAS;

                /* Convert float input to model quantized domain. */
                float _quantized_float = _normalized / INPUT_SCALE + (float)INPUT_ZP;

                /* Round to nearest integer (ties away from zero), then clamp to int8. */
                int _quantized_int;
                if (_zero <= _quantized_float)
                {
                    _quantized_int = (int)(_quantized_float + _half_pixel);
                }
                else
                {
                    _quantized_int = (int)(_quantized_float - _half_pixel);
                }

                if (_q_min > _quantized_int)
                {
                    _quantized_int = _q_min;
                }
                if (_q_max < _quantized_int)
                {
                    _quantized_int = _q_max;
                }

                /* Store quantized value into destination tensor (HWC layout). */
                _p_quantized_input_tensor[(_destination_row * MODEL_INPUT_W + _destination_col) * MODEL_INPUT_C + _channel_index] = (int8_t)_quantized_int;
            }
        }
    }
}
