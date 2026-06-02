/*
 * Copyright (c) 2020 - 2026 Renesas Electronics Corporation and/or its affiliates
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * @file    preprocessing.c
 * @brief   ShuffleNetV2 x0.5 INT8 preprocessing for MCU.
 *
 * Matches the Python inference pipeline exactly:
 *   1. Resize shortest edge to 256 (bilinear, half-pixel center convention).
 *   2. Center-crop to 224x224.
 *   3. Normalize per-channel: (pixel/255.0 - mean) / std.
 *   4. Quantize: int8 = clamp(round(normalized / INPUT_SCALE) + INPUT_ZP, -128, 127).
 *
 * No heap allocation. All arithmetic is float32 for accuracy.
 */

#include "preprocessing.h"
#include "../common/model_metadata.h"
#include <stddef.h>

/* ── Per-channel normalization constants ──────────────────────────────── */
static const float s_mean[3] = { INPUT_NORM_MEAN_R, INPUT_NORM_MEAN_G, INPUT_NORM_MEAN_B };
static const float s_inv_std[3] = {
    1.0F / INPUT_NORM_STD_R,
    1.0F / INPUT_NORM_STD_G,
    1.0F / INPUT_NORM_STD_B
};

/* ── Bilinear sample helper ──────────────────────────────────────────── */
/**
 * @brief Bilinear-sample one channel value from the source image.
 *
 * @param[in] p_img  Source RGB image in HWC layout.
 * @param[in] width  Source image width.
 * @param[in] height Source image height.
 * @param[in] channel Channel index in RGB.
 * @param[in] fx     Floating-point X coordinate.
 * @param[in] fy     Floating-point Y coordinate.
 *
 * @return Interpolated pixel value.
 */
static inline float bilinear_sample(const uint8_t *p_img, int width, int height,
                                    int channel, float fx, float fy)
{
    int x0 = (int)fx;
    int y0 = (int)fy;
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    if (x0 < 0) { x0 = 0; }
    if (y0 < 0) { y0 = 0; }
    if (x1 >= width) { x1 = width - 1; }
    if (y1 >= height) { y1 = height - 1; }

    float wx = fx - (float)x0;
    float wy = fy - (float)y0;

    float p00 = (float)p_img[(y0 * width + x0) * 3 + channel];
    float p10 = (float)p_img[(y0 * width + x1) * 3 + channel];
    float p01 = (float)p_img[(y1 * width + x0) * 3 + channel];
    float p11 = (float)p_img[(y1 * width + x1) * 3 + channel];

    float top    = p00 * (1.0F - wx) + p10 * wx;
    float bottom = p01 * (1.0F - wx) + p11 * wx;
    return top * (1.0F - wy) + bottom * wy;
}

/* ── Quantize helper ─────────────────────────────────────────────────── */
/**
 * @brief Quantize a normalized float to int8 tensor value.
 *
 * @param[in] normalized_value Normalized scalar value.
 *
 * @return Saturated int8 quantized value.
 */
static inline int8_t quantize_to_int8(float normalized_value)
{
    float q = normalized_value / INPUT_SCALE + (float)INPUT_ZP;
    int32_t qi;
    if (q >= 0.0F)
    {
        qi = (int32_t)(q + 0.5F);
    }
    else
    {
        qi = (int32_t)(q - 0.5F);
    }
    if (qi < -128) { qi = -128; }
    if (qi >  127) { qi =  127; }
    return (int8_t)qi;
}

/* ── Main preprocessing function ─────────────────────────────────────── */
/**
 * @brief Preprocess source RGB image into model-ready int8 tensor.
 *
 * @param[in] p_source_image Source image in HWC uint8 RGB format.
 * @param[in] source_width Source image width.
 * @param[in] source_height Source image height.
 * @param[out] p_quantized_output Output model input tensor.
 */
void preprocess(const uint8_t *p_source_image,
                int            source_width,
                int            source_height,
                int8_t        *p_quantized_output)
{
    /*
     * Step 1: Compute resized dimensions (shortest edge → 256).
     * Python: scale = 256.0 / min(w, h)
     *         new_w, new_h = int(round(w * scale)), int(round(h * scale))
     */
    int min_dim = (source_width < source_height) ? source_width : source_height;
    float resize_scale = (float)RESIZE_SHORT_EDGE / (float)min_dim;
    int resized_w = (int)((float)source_width  * resize_scale + 0.5F);
    int resized_h = (int)((float)source_height * resize_scale + 0.5F);

    /*
     * Step 2: Center-crop offsets in resized space.
     * Python: left = (new_w - 224) // 2;  top = (new_h - 224) // 2
     */
    int crop_left = (resized_w - MODEL_INPUT_W) / 2;
    int crop_top  = (resized_h - MODEL_INPUT_H) / 2;

    /*
     * Step 3+4: For each output pixel, compute source coordinate via
     *           the resize mapping, apply bilinear interpolation,
     *           then normalize and quantize.
     *
     * Mapping from crop-space pixel (dx, dy) to source pixel:
     *   resized_pixel = (crop_left + dx, crop_top + dy)
     *   source_float  = (resized_pixel + 0.5) * (src_dim / resized_dim) - 0.5
     *
     * This is the half-pixel centre convention matching PIL BILINEAR.
     */
    float inv_scale_x = (float)source_width  / (float)resized_w;
    float inv_scale_y = (float)source_height / (float)resized_h;

    int dy;
    int dx;
    int channel;

    for (dy = 0; dy < MODEL_INPUT_H; dy++)
    {
        float src_fy = ((float)(crop_top + dy) + 0.5F) * inv_scale_y - 0.5F;

        for (dx = 0; dx < MODEL_INPUT_W; dx++)
        {
            float src_fx = ((float)(crop_left + dx) + 0.5F) * inv_scale_x - 0.5F;
            int out_idx = (dy * MODEL_INPUT_W + dx) * MODEL_INPUT_C;

            for (channel = 0; channel < MODEL_INPUT_C; channel++)
            {
                float pixel = bilinear_sample(p_source_image,
                                              source_width, source_height,
                                              channel, src_fx, src_fy);

                /* Normalize: (pixel/255.0 - mean) / std */
                float normalized = (pixel / 255.0F - s_mean[channel]) * s_inv_std[channel];

                /* Quantize to int8 */
                p_quantized_output[out_idx + channel] = quantize_to_int8(normalized);
            }
        }
    }
}
