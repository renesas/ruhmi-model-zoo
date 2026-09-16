/*
 * The Renesas Electronics Corporation
 * DISCLAIMER
 * This software is supplied by Renesas Electronics Corporation and is only intended for use with Renesas products. No
 * other uses are authorized. This software is owned by Renesas Electronics Corporation and is protected under all
 * applicable laws, including copyright laws.
 * THIS SOFTWARE IS PROVIDED 'AS IS' AND RENESAS MAKES NO WARRANTIES REGARDING
 * THIS SOFTWARE, WHETHER EXPRESS, IMPLIED OR STATUTORY, INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. ALL SUCH WARRANTIES ARE EXPRESSLY DISCLAIMED. TO THE MAXIMUM
 * EXTENT PERMITTED NOT PROHIBITED BY LAW, NEITHER RENESAS ELECTRONICS CORPORATION NOR ANY OF ITS AFFILIATED COMPANIES
 * SHALL BE LIABLE FOR ANY DIRECT, INDIRECT, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES FOR ANY REASON RELATED TO THIS
 * SOFTWARE, EVEN IF RENESAS OR ITS AFFILIATES HAVE BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
 * Renesas reserves the right, without notice, to make changes to this software and to discontinue the availability of
 * this software. By using this software, you agree to the additional terms and conditions found by accessing the
 * following link:
 * http://www.renesas.com/disclaimer
 */

/*
* Copyright (c) 2020 - 2026 Renesas Electronics Corporation and/or its affiliates
*
* SPDX-License-Identifier: BSD-3-Clause
*/
/**
 ******************************************************************************
 * @file    preprocessing.c
 * @brief   NanoDet input preprocessing - bilinear resize to model input dimensions.
 *
 * @author  Renesas Electronics
 * @date    2026
 * @version 1.0
 ******************************************************************************
 */

#include "preprocessing.h"
#include "model_metadata.h"

#include <math.h>
#include <stddef.h>

/* Normalization constants in BGR output order (sourced from model_metadata.h).
 * Index 0 = B channel, 1 = G channel, 2 = R channel. */
static const float _k_mean_bgr[3] = {MODEL_MEAN_B, MODEL_MEAN_G, MODEL_MEAN_R};
static const float _k_std_bgr[3]  = {MODEL_STD_B,  MODEL_STD_G,  MODEL_STD_R};

/* RGB888 source-to-BGR-output channel remap.
 * Input  channel 0=R, 1=G, 2=B.
 * Output channel 0=B, 1=G, 2=R  (BGR order expected by the model).
 * k_rgb_to_bgr[out_c] gives the source channel index to read. */
static const int32_t _k_rgb_to_bgr[3] = {2, 1, 0};

/**
 * @brief  Clamp a signed integer value to the int8 range.
 *
 * @param[in]  value  Integer value to clamp.
 * @return            Clamped int8 value.
 */
static int8_t quantize_input_to_int8(float _value)
{
    const float _quantized_float_value = (_value / MODEL_INPUT_SCALE) + (float) MODEL_INPUT_ZERO_POINT;
    const int32_t _quantized_integer_value =
        (int32_t) ((_quantized_float_value >= 0.0F) ? (_quantized_float_value + 0.5F) : (_quantized_float_value - 0.5F));

    if (_quantized_integer_value > 127)
    {
        return 127;
    }
    if (_quantized_integer_value < -128)
    {
        return -128;
    }

    return (int8_t) _quantized_integer_value;
}

/**
 * @brief  Bilinearly sample one source-image channel at a floating-point coordinate.
 *
 * @param[in]  p_source_image         Source RGB image buffer.
 * @param[in]  source_width_pixels    Source image width in pixels.
 * @param[in]  source_height_pixels   Source image height in pixels.
 * @param[in]  source_channel_count   Number of channels in the source image.
 * @param[in]  source_channel_index   Channel index to sample.
 * @param[in]  source_x_coordinate    Horizontal sample coordinate.
 * @param[in]  source_y_coordinate    Vertical sample coordinate.
 * @return                           Interpolated channel value.
 */
static float bilinear_sample_channel(const uint8_t *p_source_image,
                                     int32_t _source_width_pixels,
                                     int32_t _source_height_pixels,
                                     int32_t _source_channel_count,
                                     int32_t _source_channel_index,
                                     float _source_x_coordinate,
                                     float _source_y_coordinate)
{
    /* Use floorf to match OpenCV INTER_LINEAR floor convention,
     * including negative coordinate edge cases. */
    int32_t _source_x_index_floor = (int32_t) floorf(_source_x_coordinate);
    int32_t _source_y_index_floor = (int32_t) floorf(_source_y_coordinate);
    int32_t _source_x_index_ceil = _source_x_index_floor + 1;
    int32_t _source_y_index_ceil = _source_y_index_floor + 1;

    if (_source_x_index_floor < 0) { _source_x_index_floor = 0; }
    if (_source_y_index_floor < 0) { _source_y_index_floor = 0; }
    if (_source_x_index_ceil >= _source_width_pixels) { _source_x_index_ceil = _source_width_pixels - 1; }
    if (_source_y_index_ceil >= _source_height_pixels) { _source_y_index_ceil = _source_height_pixels - 1; }

    {
        const float _x_weight = _source_x_coordinate - (float) _source_x_index_floor;
        const float _y_weight = _source_y_coordinate - (float) _source_y_index_floor;

        const float _top_left_value =
            (float) p_source_image[(_source_y_index_floor * _source_width_pixels + _source_x_index_floor) * _source_channel_count + _source_channel_index];
        const float _top_right_value =
            (float) p_source_image[(_source_y_index_floor * _source_width_pixels + _source_x_index_ceil) * _source_channel_count + _source_channel_index];
        const float _bottom_left_value =
            (float) p_source_image[(_source_y_index_ceil * _source_width_pixels + _source_x_index_floor) * _source_channel_count + _source_channel_index];
        const float _bottom_right_value =
            (float) p_source_image[(_source_y_index_ceil * _source_width_pixels + _source_x_index_ceil) * _source_channel_count + _source_channel_index];

        const float _top_interpolated_value = _top_left_value * (1.0F - _x_weight) + _top_right_value * _x_weight;
        const float _bottom_interpolated_value = _bottom_left_value * (1.0F - _x_weight) + _bottom_right_value * _x_weight;
        return _top_interpolated_value * (1.0F - _y_weight) + _bottom_interpolated_value * _y_weight;
    }
}

/**
 * @brief  Resize, normalize, and quantize an RGB888 image for the NanoDet model input.
 *
 * @param[in]  p_source_image         Source image, HWC layout, RGB888.
 * @param[in]  _source_width           Source image width in pixels.
 * @param[in]  _source_height          Source image height in pixels.
 * @param[out] p_destination_image    Destination int8 tensor buffer.
 * @param[in]  _destination_width      Destination image width in pixels.
 * @param[in]  _destination_height     Destination image height in pixels.
 * @param[out] p_letterbox_params     Output scale metadata used by postprocessing.
 */
void preprocess(const uint8_t *p_source_image,
                uint16_t _source_width,
                uint16_t _source_height,
                int8_t *p_destination_image,
                uint16_t _destination_width,
                uint16_t _destination_height,
                letterbox_params_t *p_letterbox_params)
{
    const int32_t _source_width_pixels = (int32_t) _source_width;
    const int32_t _source_height_pixels = (int32_t) _source_height;
    const int32_t _destination_width_pixels = (int32_t) _destination_width;
    const int32_t _destination_height_pixels = (int32_t) _destination_height;
    int32_t y;

    if ((NULL == p_source_image) || (NULL == p_destination_image))
    {
        return;
    }

    if (NULL != p_letterbox_params)
    {
        p_letterbox_params->_scale_w = (float) _destination_width_pixels / (float) _source_width_pixels;
        p_letterbox_params->_scale_h = (float) _destination_height_pixels / (float) _source_height_pixels;
        p_letterbox_params->_pad_x = 0;
        p_letterbox_params->_pad_y = 0;
    }

    for (y = 0; y < _destination_height_pixels; y++)
    {
        int32_t x;
        const float _source_y_coordinate =
            ((float) y + 0.5F) * ((float) _source_height_pixels / (float) _destination_height_pixels) - 0.5F;

        for (x = 0; x < _destination_width_pixels; x++)
        {
            const float _source_x_coordinate =
                ((float) x + 0.5F) * ((float) _source_width_pixels / (float) _destination_width_pixels) - 0.5F;
            const int32_t _output_pixel_index = (y * _destination_width_pixels + x) * MODEL_INPUT_C;
            int32_t c;

            for (c = 0; c < (int32_t) MODEL_INPUT_C; c++)
            {
                /* Read from the RGB source using the remapped channel index so
                 * that the output tensor is in BGR order as the model expects. */
                const float _sampled_pixel_value = bilinear_sample_channel(p_source_image,
                                                                          _source_width_pixels,
                                                                          _source_height_pixels,
                                                                          (int32_t) MODEL_INPUT_C,
                                                                          _k_rgb_to_bgr[c],
                                                                          _source_x_coordinate,
                                                                          _source_y_coordinate);
                const float _normalized_pixel_value = (_sampled_pixel_value - _k_mean_bgr[c]) / _k_std_bgr[c];
                p_destination_image[_output_pixel_index + c] = quantize_input_to_int8(_normalized_pixel_value);
            }
        }
    }
}
