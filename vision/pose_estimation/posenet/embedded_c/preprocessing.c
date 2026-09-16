
/*
 * Copyright (C) 2026 Renesas Electronics Corporation. All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

/***********************************************************************************************************************
 * File Name    : preprocessing.c
 * Description  : PoseNet input preprocessing implementation - bilinear resize,
 *                normalization to [-1, 1], and INT8 quantization.
 **********************************************************************************************************************/



#include <math.h>
#include <stdint.h>
#include "model_metadata.h"
#include "preprocessing.h"




/**
 * @brief Build the per-axis source-coordinate + fractional weight tables
 *        for one resize axis.
 *
 * Implements the half-pixel-centre mapping:
 *     sx = (dx + 0.5) * src_n / dst_n - 0.5
 * with edge replication at both endpoints.
 *
 * @param[in]  _src_n       Source axis length (pixels).
 * @param[in]  _dst_n       Destination axis length (pixels).
 * @param[out] p_index      Integer lower-tap source index (_dst_n entries).
 * @param[out] p_frac       Float weight of the upper tap (_dst_n entries).
 *                          Lower-tap weight is implicitly `1 - p_frac[_dst_idx]`.
 * @note Variable use:
 *       - _inv_scale : Source-to-destination scale factor for this axis.
 *       - _src_pos   : Floating source coordinate mapped from destination index.
 *       - _src_lo    : Lower source tap index for bilinear interpolation.
 *       - _frac      : Fractional weight of upper source tap.
 */
static void compute_bilinear_table(int    _src_n,
                                   int    _dst_n,
                                   int   *p_index,
                                   float *p_frac)
{
    const float _inv_scale = (float)_src_n / (float)_dst_n;
    int _dst_idx;

    for (_dst_idx = 0; _dst_idx < _dst_n; _dst_idx++)
    {
        float _src_pos = ((float)_dst_idx + 0.5f) * _inv_scale - 0.5f;
        int   _src_lo  = (int)floorf(_src_pos);
        float _frac    = _src_pos - (float)_src_lo;

        /* Low-edge clamp: replicate src[0]. */
        if (0 > _src_lo)
        {
            _src_lo = 0;
            _frac   = 0.0f;
        }
        /* High-edge clamp: replicate src[_src_n - 1]. */
        if ((_src_n - 1) <= _src_lo)
        {
            _src_lo = _src_n - 2;
            _frac   = 1.0f;
        }

        p_index[_dst_idx] = _src_lo;
        p_frac[_dst_idx]  = _frac;
    }
}


/**
 * @brief Preprocess RGB image -> PoseNet INT8 input tensor.
 *
 * @param[in]  p_source_image_hwc  Source RGB image (uint8, HWC layout).
 * @param[in]  _source_width       Source image width (pixels).
 * @param[in]  _source_height      Source image height (pixels).
 * @param[out] p_input_tensor      Destination int8 tensor (HWC, MODEL_INPUT_SIZE
 *                                 elements), quantized per INPUT_QUANT_SCALE / INPUT_QUANT_ZP.
 * @note Variable use:
 *       - s_ix/s_fx : X-axis bilinear lookup tables (index and fractional weight).
 *       - s_iy/s_fy : Y-axis bilinear lookup tables (index and fractional weight).
 *       - _row0/_row1 and _col0/_col1 : Source pointer offsets to the 2x2 bilinear neighborhood.
 *       - _top/_bottom/_pixel : Interpolated pixel values before normalization.
 *       - _norm/_q : Normalized float and final int8 quantized value.
 */
void preprocess(const uint8_t *p_source_image_hwc,
                int _source_width,
                int _source_height,
                int8_t *p_input_tensor)
{
    /* Per-axis precomputed look-up tables. */
    static int   s_ix[MAX_DST_DIM];
    static float s_fx[MAX_DST_DIM];
    static int   s_iy[MAX_DST_DIM];
    static float s_fy[MAX_DST_DIM];

    compute_bilinear_table(_source_height, MODEL_INPUT_H, s_iy, s_fy);
    compute_bilinear_table(_source_width,  MODEL_INPUT_W, s_ix, s_fx);

    int _dst_y;
    int _dst_x;
    int _channel_idx;

    for (_dst_y = 0; _dst_y < MODEL_INPUT_H; _dst_y++)
    {
        const int   _iy    = s_iy[_dst_y];
        const float _fy    = s_fy[_dst_y];
        const float _wy0   = 1.0f - _fy;
        const float _wy1   = _fy;
        const int   _row0  = _iy       * _source_width * MODEL_INPUT_C;
        const int   _row1  = (_iy + 1) * _source_width * MODEL_INPUT_C;

        for (_dst_x = 0; _dst_x < MODEL_INPUT_W; _dst_x++)
        {
            const int   _ix   = s_ix[_dst_x];
            const float _fx   = s_fx[_dst_x];
            const float _wx0  = 1.0f - _fx;
            const float _wx1  = _fx;
            const int   _col0 = _ix       * MODEL_INPUT_C;
            const int   _col1 = (_ix + 1) * MODEL_INPUT_C;

            const uint8_t *p_00 = &p_source_image_hwc[_row0 + _col0];
            const uint8_t *p_01 = &p_source_image_hwc[_row0 + _col1];
            const uint8_t *p_10 = &p_source_image_hwc[_row1 + _col0];
            const uint8_t *p_11 = &p_source_image_hwc[_row1 + _col1];

            for (_channel_idx = 0; _channel_idx < MODEL_INPUT_C; _channel_idx++)
            {
                /* 2-tap separable bilinear. */
                const float _top    = _wx0 * (float)p_00[_channel_idx] + _wx1 * (float)p_01[_channel_idx];
                const float _bottom = _wx0 * (float)p_10[_channel_idx] + _wx1 * (float)p_11[_channel_idx];
                const float _pixel  = _wy0 * _top + _wy1 * _bottom;

                /* Normalize to [-1, 1], then quantize to int8. */
                const float _norm = _pixel * INPUT_NORM_MUL - 1.0f;
                int _q = (int)lroundf(_norm / INPUT_QUANT_SCALE) + INPUT_QUANT_ZP;
                if (-128 > _q) { _q = -128; }
                if ( 127 < _q) { _q =  127; }
                p_input_tensor[
                    (_dst_y * MODEL_INPUT_W + _dst_x) * MODEL_INPUT_C + _channel_idx] = (int8_t)_q;
            }
        }
    }
}
