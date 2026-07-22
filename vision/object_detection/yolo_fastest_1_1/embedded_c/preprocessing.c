/*
* Copyright (c) 2020 - 2026 Renesas Electronics Corporation and/or its affiliates
*
* SPDX-License-Identifier: BSD-3-Clause
*/
/**
 ******************************************************************************
 * @file    preprocessing.c
 * @brief   YOLO-Fastest 1.1 INT8 letterbox resize + quantize to int8.
 *
 * Bilinear interpolation with half-pixel centre convention, matching the
 * Python cv2.resize() / PIL behaviour used during training.
 *
 * Quantization: int8_value = round(pixel_value / 255.0 / scale) + zero_point
 *   With scale≈1/255, zero_point=-128:  int8_value = pixel_value - 128
 *
 * All buffers are caller-provided — no heap allocation.
 ******************************************************************************
 */

#include "preprocessing.h"
#include "../common/model_metadata.h"
#include <stddef.h>
#include <string.h>

/* ── Bilinear sample helper ──────────────────────────────────────────── */

/**
 * @brief Sample one channel from a uint8 HWC image with bilinear interpolation.
 *
 * @param[in] p_source_image Source image in HWC uint8 layout.
 * @param[in] image_width    Source image width in pixels.
 * @param[in] image_height   Source image height in pixels.
 * @param[in] channels       Number of channels per pixel.
 * @param[in] channel_idx    Channel index to sample.
 * @param[in] float_x        Floating-point x coordinate in source space.
 * @param[in] float_y        Floating-point y coordinate in source space.
 *
 * @return Interpolated channel value in float domain.
 */
static float bilinear_sample_channel(const uint8_t *p_source_image,
                             int32_t image_width, int32_t image_height,
                             int32_t channels, int32_t channel_idx,
                             float float_x, float float_y)
{
    /* Integer neighbors around the floating-point sample location. */
    int32_t _left_x  = (int32_t)float_x;
    int32_t _top_y   = (int32_t)float_y;
    int32_t _right_x = _left_x + 1;
    int32_t _bot_y   = _top_y + 1;

    /* Clamp to image boundaries */
    if (_left_x  < 0)            { _left_x  = 0; }
    if (_top_y   < 0)            { _top_y   = 0; }
    if (_right_x >= image_width) { _right_x = image_width  - 1; }
    if (_bot_y   >= image_height){ _bot_y   = image_height - 1; }

    /* Fractional offsets used as interpolation weights. */
    float _weight_x = float_x - (float)_left_x;
    float _weight_y = float_y - (float)_top_y;

    /* Read the four surrounding pixels for the requested channel. */
    float _top_left_val     = (float)p_source_image[(_top_y * image_width + _left_x)  * channels + channel_idx];
    float _top_right_val    = (float)p_source_image[(_top_y * image_width + _right_x) * channels + channel_idx];
    float _bottom_left_val  = (float)p_source_image[(_bot_y * image_width + _left_x)  * channels + channel_idx];
    float _bottom_right_val = (float)p_source_image[(_bot_y * image_width + _right_x) * channels + channel_idx];

    /* Interpolate horizontally at the top row and bottom row. */
    float _interpolated_top = _top_left_val * (BILINEAR_BLEND_ONE - _weight_x) + _top_right_val * _weight_x;
    float _interpolated_bottom = _bottom_left_val * (BILINEAR_BLEND_ONE - _weight_x) + _bottom_right_val * _weight_x;

    /* Interpolate vertically between the top and bottom blended values. */
    return _interpolated_top * (BILINEAR_BLEND_ONE - _weight_y) + _interpolated_bottom * _weight_y;
}

/* ── Quantize helper ─────────────────────────────────────────────────── */

/**
 * @brief Quantize a float pixel value [0,255] to int8 using input quant params.
 *        int8 = clamp(round(value / scale) + zero_point, -128, 127)
 *        With scale≈1/255, zp=-128: int8 = round(value) - 128
 *
 * Input range [0,255] maps to [-128,127] which fits int8 exactly.
 *
 * @param[in] pixel_value Pixel intensity in float domain.
 *
 * @return Quantized int8 value.
 */
static inline int8_t quantize_to_int8(float pixel_value)
{
    /* Round to nearest integer in [0,255]-like range. */
    int32_t _rounded_int_value = (int32_t)(pixel_value + PIXEL_CENTER_OFFSET);
    /* Shift uint8-like range [0..255] to int8 range [-128..127]. */
    return (int8_t)(_rounded_int_value + MODEL_INPUT_ZERO_POINT);
}

/* ── Main preprocess function ────────────────────────────────────────── */

/**
 * @brief Preprocess source image to model input using letterbox and quantization.
 *
 * @param[in]  p_source_image      Source image, HWC layout, uint8, RGB order.
 * @param[in]  source_width        Source image width in pixels.
 * @param[in]  source_height       Source image height in pixels.
 * @param[out] p_destination_image Output buffer for quantized model input.
 * @param[in]  destination_width   Target width in pixels.
 * @param[in]  destination_height  Target height in pixels.
 * @param[out] p_params            Filled with scale and padding for postprocessing.
 */
void preprocess(const uint8_t   *p_source_image,
                uint16_t         source_width,
                uint16_t         source_height,
                int8_t          *p_destination_image,
                uint16_t         destination_width,
                uint16_t         destination_height,
                letterbox_params_t *p_params)
{
    /* Convert dimensions to signed integers for arithmetic operations. */
    int32_t _source_width_px = (int32_t)source_width;
    int32_t _source_height_px = (int32_t)source_height;
    int32_t _dest_width_px = (int32_t)destination_width;
    int32_t _dest_height_px = (int32_t)destination_height;

    /* Step 1: Compute uniform scale (preserve aspect ratio) */
    float _scale_factor_x = (float)_dest_width_px / (float)_source_width_px;
    float _scale_factor_y = (float)_dest_height_px / (float)_source_height_px;
    float _uniform_scale = (_scale_factor_x < _scale_factor_y) ? _scale_factor_x : _scale_factor_y;

    /* Compute resized image size in model space before padding (rounded to nearest pixel, matching Python int(round(...))). */
    int32_t _resized_width_px = (int32_t)((float)_source_width_px * _uniform_scale + PIXEL_CENTER_OFFSET);
    int32_t _resized_height_px = (int32_t)((float)_source_height_px * _uniform_scale + PIXEL_CENTER_OFFSET);

    /* Compute symmetric letterbox padding around resized content. */
    int32_t _padding_x = (_dest_width_px - _resized_width_px) / 2;
    int32_t _padding_y = (_dest_height_px - _resized_height_px) / 2;

    /* Return letterbox parameters for coordinate unscaling */
    if (p_params != NULL)
    {
        p_params->scale = _uniform_scale;
        p_params->pad_x = _padding_x;
        p_params->pad_y = _padding_y;
    }

    /* Step 2: Fill ONLY the pad regions with quantized pad value (gray 114 → int8 -14).
     * This avoids writing 307,200 bytes only to overwrite the interior again in Step 3.
     * Pad regions:
     *   - Top:    rows [0, padding_y)
     *   - Bottom: rows [padding_y + resized_height_px, dest_height_px)
     *   - Left:   cols [0, padding_x) on interior rows
     *   - Right:  cols [padding_x + resized_width_px, dest_width_px) on interior rows
     */
    {
        int8_t _pad_quantized = quantize_to_int8((float)INPUT_LETTERBOX_PAD);
        int32_t _row_stride = _dest_width_px * MODEL_INPUT_C;

        /* Top pad rows (full rows of pad) */
        if (_padding_y > 0)
        {
            (void)memset(p_destination_image, (int)_pad_quantized,
                         (size_t)(_padding_y * _row_stride));
        }

        /* Bottom pad rows (full rows of pad) */
        int32_t _bottom_pad_start_row = _padding_y + _resized_height_px;
        if (_bottom_pad_start_row < _dest_height_px)
        {
            int32_t _bottom_pad_bytes = (_dest_height_px - _bottom_pad_start_row) * _row_stride;
            (void)memset(&p_destination_image[_bottom_pad_start_row * _row_stride],
                         (int)_pad_quantized, (size_t)_bottom_pad_bytes);
        }

        /* Left and right pad on interior rows */
        if (_padding_x > 0 || (_padding_x + _resized_width_px) < _dest_width_px)
        {
            int32_t _left_pad_bytes = _padding_x * MODEL_INPUT_C;
            int32_t _right_pad_start = (_padding_x + _resized_width_px) * MODEL_INPUT_C;
            int32_t _right_pad_bytes = (_dest_width_px - _padding_x - _resized_width_px) * MODEL_INPUT_C;
            int32_t _interior_row;

            for (_interior_row = _padding_y; _interior_row < _bottom_pad_start_row; _interior_row++)
            {
                int32_t _row_offset = _interior_row * _row_stride;
                if (_left_pad_bytes > 0)
                {
                    (void)memset(&p_destination_image[_row_offset],
                                 (int)_pad_quantized, (size_t)_left_pad_bytes);
                }
                if (_right_pad_bytes > 0)
                {
                    (void)memset(&p_destination_image[_row_offset + _right_pad_start],
                                 (int)_pad_quantized, (size_t)_right_pad_bytes);
                }
            }
        }
    }

    /* Step 3: Bilinear resize into padded region, preserve RGB channel order,
     *         then quantize each channel to int8 (pixel - 128). */
    {
        float _inv_scale_x = (float)_source_width_px / (float)_resized_width_px;
        float _inv_scale_y = (float)_source_height_px / (float)_resized_height_px;
        int32_t _dest_row_y;
        int32_t _dest_col_x;

        for (_dest_row_y = 0; _dest_row_y < _resized_height_px; _dest_row_y++)
        {
            float _source_float_y = ((float)_dest_row_y + PIXEL_CENTER_OFFSET) * _inv_scale_y - PIXEL_CENTER_OFFSET;
            int32_t _dest_pixel_y = _dest_row_y + _padding_y;

            if ((_dest_pixel_y < 0) || (_dest_pixel_y >= _dest_height_px))
            {
                continue;
            }

            for (_dest_col_x = 0; _dest_col_x < _resized_width_px; _dest_col_x++)
            {
                float _source_float_x = ((float)_dest_col_x + PIXEL_CENTER_OFFSET) * _inv_scale_x - PIXEL_CENTER_OFFSET;
                int32_t _dest_pixel_x = _dest_col_x + _padding_x;

                if ((_dest_pixel_x < 0) || (_dest_pixel_x >= _dest_width_px))
                {
                    continue;
                }

                int32_t _dest_buffer_idx = (_dest_pixel_y * _dest_width_px + _dest_pixel_x) * MODEL_INPUT_C;

                /*
                 * Source pixels are RGB (test_data_N.h generated via PIL .convert("RGB")).
                 * Python inference: cv2.imread() loads BGR, cv2.COLOR_BGR2RGB converts to RGB before
                 * quantizing and feeding to model. So the model expects RGB input.
                 * Source test images are already RGB - no channel swap needed.
                 * Quantize: int8 = round(pixel_value) - 128
                 */
                float _red_channel = bilinear_sample_channel(p_source_image,
                                                            _source_width_px,
                                                            _source_height_px,
                                                            MODEL_INPUT_C,
                                                                                                                        CHANNEL_INDEX_RED,
                                                            _source_float_x,
                                                            _source_float_y);
                float _green_channel = bilinear_sample_channel(p_source_image,
                                                              _source_width_px,
                                                              _source_height_px,
                                                              MODEL_INPUT_C,
                                                                                                                            CHANNEL_INDEX_GREEN,
                                                              _source_float_x,
                                                              _source_float_y);
                float _blue_channel = bilinear_sample_channel(p_source_image,
                                                             _source_width_px,
                                                             _source_height_px,
                                                             MODEL_INPUT_C,
                                                                                                                         CHANNEL_INDEX_BLUE,
                                                             _source_float_x,
                                                             _source_float_y);

                /* Write quantized RGB channels into destination model input tensor. */
                p_destination_image[_dest_buffer_idx + CHANNEL_INDEX_RED] = quantize_to_int8(_red_channel);
                p_destination_image[_dest_buffer_idx + CHANNEL_INDEX_GREEN] = quantize_to_int8(_green_channel);
                p_destination_image[_dest_buffer_idx + CHANNEL_INDEX_BLUE] = quantize_to_int8(_blue_channel);
            }
        }
    }
}
