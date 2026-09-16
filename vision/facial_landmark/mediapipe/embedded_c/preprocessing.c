/**********************************************************************************************************************
 * Copyright (c) 2020 - 2026 Renesas Electronics Corporation and/or its affiliates
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * DISCLAIMER
 * This software is supplied by Renesas Electronics Corporation and is only intended for use with Renesas products.
 * No other uses are authorized.
 */
/**********************************************************************************************************************
 * File Name    : preprocessing.c
 * Description  : MediaPipe face-landmark input preprocessing - direct 2-tap
 *                bilinear resize (half-pixel-centre, edge replicate) from any
 *                RGB888 source resolution to MODEL_INPUT_H x MODEL_INPUT_W
 *                (192x192), then float32 [0,1] normalisation and INT8
 *                quantisation matching the tflite input tensor's quant params.
 *
 *                Algorithm:
 *                  1. Source coordinate (half-pixel-centre):
 *                       sx = (dx + 0.5) * src_w / dst_w - 0.5
 *                       sy = (dy + 0.5) * src_h / dst_h - 0.5
 *                  2. 2-tap separable bilinear with edge replication.
 *                  3. Normalise: f = pixel * (1.0f / 255.0f)  ->  [0, 1]
 *                  4. Quantise : q = round(f / scale + zp)   -> int8
 **********************************************************************************************************************/

#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include "model_metadata.h"
#include "preprocessing.h"


/**
 * @brief Round a float to the nearest integer using ties-to-even (IEEE 754
 *        default), matching the offline reference rounding flow.
 *
 * We roll our own implementation instead of relying on nearbyintf() because
 * some Cortex-M libc builds do not track FE_TONEAREST at runtime.
 *
 * @param[in]  _input_value  Float value to round.
 * @return int32_t Rounded integer.
 */
static int32_t round_ties_to_even_f(float _input_value)
{
    float _floor_value = floorf(_input_value);
    float _fraction_value = _input_value - _floor_value;
    int32_t _base_value = (int32_t)_floor_value;

    if (0.5f > _fraction_value)
    {
        return _base_value;
    }
    if (0.5f < _fraction_value)
    {
        return _base_value + 1;
    }
    /* Exactly halfway - pick the even integer. */
    return (0 == (_base_value & 1)) ? _base_value : (_base_value + 1);
}


/**
 * @brief Saturate a rounded quantised value into the int8 range.
 *
 * @param[in]  _rounded_value  32-bit rounded quantised value.
 * @return int8_t Value clipped to [-128, 127].
 */
static int8_t clip_to_int8(int32_t _rounded_value)
{
    if ((int32_t)INT8_MIN > _rounded_value)
    {
        return INT8_MIN;
    }

    if ((int32_t)INT8_MAX < _rounded_value)
    {
        return INT8_MAX;
    }

    return (int8_t)_rounded_value;
}


/**
 * @brief Build the per-axis source-coordinate + fractional weight tables for
 *        one resize axis.
 *
 * Implements the half-pixel-centre mapping:
 *     sx = (dx + 0.5) * src_n / dst_n - 0.5
 * with edge replication at both endpoints.
 *
 * @param[in]  _source_axis_length   Source axis length (pixels).
 * @param[in]  _destination_axis_length Destination axis length (pixels).
 * @param[out] p_source_index  Integer lower-tap source index array (destination_axis_length entries).
 * @param[out] p_fraction      Weight of the upper tap array (destination_axis_length entries); the
 *                      lower-tap weight is implicitly (1 - p_fraction[k]).
 */
static void compute_bilinear_table(int32_t _source_axis_length,
                                   int32_t _destination_axis_length,
                                   int32_t *p_source_index,
                                   float   *p_fraction)
{
    float _axis_scale;
    int32_t _destination_index;

    if ((NULL == p_source_index) || (NULL == p_fraction))
    {
        return;
    }

    if ((1 >= _source_axis_length) || (0 >= _destination_axis_length))
    {
        return;
    }

    _axis_scale = (float)_source_axis_length / (float)_destination_axis_length;

    for (_destination_index = 0; _destination_index < _destination_axis_length; _destination_index++)
    {
        float _source_position = ((float)_destination_index + 0.5f) * _axis_scale - 0.5f;
        int32_t _source_index_low = (int32_t)floorf(_source_position);
        float _source_fraction = _source_position - (float)_source_index_low;

        /* Low-edge clamp: replicate src[0]. */
        if (0 > _source_index_low)
        {
            _source_index_low = 0;
            _source_fraction = 0.0f;
        }
        /* High-edge clamp: replicate src[src_n - 1]. */
        if (_source_index_low >= (_source_axis_length - 1))
        {
            _source_index_low = _source_axis_length - 2;
            _source_fraction = 1.0f;
        }

        p_source_index[_destination_index] = _source_index_low;
        p_fraction[_destination_index] = _source_fraction;
    }
}


/**
 * @brief Preprocess RGB888 image -> face-landmark INT8 input tensor.
 *
 * @param[in]  p_source_image_hwc Source RGB888 image (uint8, HWC layout).
 * @param[in]  _source_width       Source image width  (pixels).
 * @param[in]  _source_height      Source image height (pixels).
 * @param[out] p_input_tensor     Destination int8 tensor (HWC, MODEL_INPUT_SIZE
 *                                 elements) quantised to the tflite input
 *                                 tensor's numerical range.
 */
void preprocess(const uint8_t *p_source_image_hwc,
                int32_t        _source_width,
                int32_t        _source_height,
                int8_t        *p_input_tensor)
{
    /* Per-axis precomputed look-up tables. */
    static int32_t s_source_x_index_table[MAX_DST_DIM];
    static float   s_source_x_fraction_table[MAX_DST_DIM];
    static int32_t s_source_y_index_table[MAX_DST_DIM];
    static float   s_source_y_fraction_table[MAX_DST_DIM];

    if ((NULL == p_source_image_hwc) || (NULL == p_input_tensor))
    {
        return;
    }

    if ((1 >= _source_width) || (1 >= _source_height))
    {
        return;
    }

    compute_bilinear_table(_source_height,
                           (int32_t)MODEL_INPUT_H,
                           s_source_y_index_table,
                           s_source_y_fraction_table);
    compute_bilinear_table(_source_width,
                           (int32_t)MODEL_INPUT_W,
                           s_source_x_index_table,
                           s_source_x_fraction_table);
    const float _pixel_normalization_factor = INV_255_F;
    const float _inv_quantization_scale = INV_INPUT_QUANT_SCALE;
    const float _quantization_zero_point = (float)INPUT_QUANT_ZERO_POINT;

    int32_t _row_index;
    int32_t _column_index;
    int32_t _channel_index;

    for (_row_index = 0; (int32_t)MODEL_INPUT_H > _row_index; _row_index++)
    {
        const int32_t _source_y_index = s_source_y_index_table[_row_index];
        const float _fraction_y = s_source_y_fraction_table[_row_index];
        const float _weight_y0 = 1.0f - _fraction_y;
        const float _weight_y1 = _fraction_y;
        const int32_t _row0_offset = _source_y_index * _source_width * (int32_t)MODEL_INPUT_C;
        const int32_t _row1_offset = (_source_y_index + 1) * _source_width * (int32_t)MODEL_INPUT_C;

        for (_column_index = 0; (int32_t)MODEL_INPUT_W > _column_index; _column_index++)
        {
            const int32_t _source_x_index = s_source_x_index_table[_column_index];
            const float _fraction_x = s_source_x_fraction_table[_column_index];
            const float _weight_x0 = 1.0f - _fraction_x;
            const float _weight_x1 = _fraction_x;
            const int32_t _column0_offset = _source_x_index * (int32_t)MODEL_INPUT_C;
            const int32_t _column1_offset = (_source_x_index + 1) * (int32_t)MODEL_INPUT_C;

            const uint8_t *p_top_left     = &p_source_image_hwc[_row0_offset + _column0_offset];
            const uint8_t *p_top_right    = &p_source_image_hwc[_row0_offset + _column1_offset];
            const uint8_t *p_bottom_left  = &p_source_image_hwc[_row1_offset + _column0_offset];
            const uint8_t *p_bottom_right = &p_source_image_hwc[_row1_offset + _column1_offset];

            for (_channel_index = 0; (int32_t)MODEL_INPUT_C > _channel_index; _channel_index++)
            {
                /* 2-tap separable bilinear (values remain in [0, 255]). */
                const float _top_value = _weight_x0 * (float)p_top_left[_channel_index] +
                                         _weight_x1 * (float)p_top_right[_channel_index];
                const float _bottom_value = _weight_x0 * (float)p_bottom_left[_channel_index] +
                                            _weight_x1 * (float)p_bottom_right[_channel_index];
                const float _pixel_value = _weight_y0 * _top_value + _weight_y1 * _bottom_value;

                /* Normalise to [0, 1] using multiplication for ULP stability. */
                const float _normalized_value = _pixel_value * _pixel_normalization_factor;
                /* Quantise to INT8 using the tflite input tensor scale/zp. */
                const float _quantized_float_value = _normalized_value * _inv_quantization_scale + _quantization_zero_point;
                const int32_t _quantized_int_value = round_ties_to_even_f(_quantized_float_value);

                p_input_tensor[
                    (_row_index * (int32_t)MODEL_INPUT_W + _column_index) * (int32_t)MODEL_INPUT_C + _channel_index] =
                    clip_to_int8(_quantized_int_value);
            }
        }
    }
}
