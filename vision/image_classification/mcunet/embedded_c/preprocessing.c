/*
 * Copyright (c) 2020 - 2026 Renesas Electronics Corporation and/or its affiliates
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
/**********************************************************************************************************************
 * File Name    : preprocessing.c
 * Description  : MCUNet-in0 input preprocessing - bilinear resize-then-centre-crop and [-1, 1] float normalisation.
 *              : Pipeline:
 *              :   - short edge resized to round(MODEL_INPUT_H * 256 / 224) (= 55 for 48-px input)
 *              :   - separable triangle-filter bilinear, half-pixel centres
 *              :   - centre-crop to MODEL_INPUT_W x MODEL_INPUT_H
 *              :   - f = pixel/127.5 - 1  (range [-1, 1] float32)
 **********************************************************************************************************************/

#include <math.h>
#include <stdint.h>
#include "model_metadata.h"
#include "preprocessing.h"

/* Resampling buffer caps. MAX_RESIZED_DIM bounds the resized intermediate
 * (long edge after the short-edge resize); MAX_FILTER_TAPS bounds the
 * triangle-filter window per output sample. */
#define MAX_RESIZED_DIM     (96)
#define MAX_FILTER_TAPS     (32)

/* Resized intermediate image, sized [MODEL_INPUT_H][MAX_RESIZED_DIM][C] */
static float s_resized[MAX_RESIZED_DIM * MAX_RESIZED_DIM * MODEL_INPUT_C];


/**
 * @brief Precompute resampling coefficients for one output axis.
 *
 * Triangle filter centred at (k + 0.5) / ratio, support = max(1, 1/ratio).
 * Weights are clipped to [0, src_n) and normalised to sum to 1.
 *
 * @param[in]  _src_n     Number of source samples along the axis.
 * @param[in]  _dst_n     Number of destination samples along the axis.
 * @param[in]  _ratio     Scale factor (_dst_n / _src_n).
 * @param[out] p_first    First contributing source index per destination sample [_dst_n].
 * @param[out] p_count    Number of contributing source samples per destination sample [_dst_n].
 * @param[out] p_weights  Normalised filter weights [_dst_n * MAX_FILTER_TAPS].
 */
static void compute_coeffs(int     _src_n,
                           int     _dst_n,
                           float   _ratio,
                           int    *p_first,        /* [_dst_n]                   */
                           int    *p_count,        /* [_dst_n]                   */
                           float  *p_weights)      /* [_dst_n * MAX_TAPS] */
{
    float _filterscale = (_ratio < 1.0f) ? (1.0f / _ratio) : 1.0f;
    float _support     = _filterscale;
    int   _dst_index;
    int   _tap_index;

    for (_dst_index = 0; _dst_index < _dst_n; _dst_index++)
    {
        float _center = ((float)_dst_index + 0.5f) / _ratio;
        int   _left   = (int)floorf(_center - _support + 0.5f);
        int   _right  = (int)floorf(_center + _support + 0.5f);
        int   _count;
        float _sum_w = 0.0f;

        if (_left  < 0)     _left  = 0;
        if (_right > _src_n) _right = _src_n;
        _count = _right - _left;
        if (_count > MAX_FILTER_TAPS) _count = MAX_FILTER_TAPS;

        p_first[_dst_index] = _left;
        p_count[_dst_index] = _count;

        for (_tap_index = 0; _tap_index < _count; _tap_index++)
        {
            float _weight = 1.0f - fabsf(((float)(_left + _tap_index) + 0.5f - _center) / _filterscale);
            if (_weight < 0.0f) _weight = 0.0f;
            p_weights[_dst_index * MAX_FILTER_TAPS + _tap_index] = _weight;
            _sum_w += _weight;
        }
        if (_sum_w > 0.0f)
        {
            float _inv = 1.0f / _sum_w;
            for (_tap_index = 0; _tap_index < _count; _tap_index++)
            {
                p_weights[_dst_index * MAX_FILTER_TAPS + _tap_index] *= _inv;
            }
        }
    }
}


/**
 * @brief Preprocess RGB image -> MCUNet INT8 input tensor.
 *
 * @param[in]  p_source_image_hwc  Source RGB image (uint8, HWC layout).
 * @param[in]  _source_width        Source image width (pixels).
 * @param[in]  _source_height       Source image height (pixels).
 * @param[out] p_input_tensor      Destination int8 tensor (HWC, quantized).
 */
void preprocess(const uint8_t *p_source_image_hwc,
                int _source_width,
                int _source_height,
                int8_t *p_input_tensor)
{
    /* Coefficient tables for the two separable passes. */
    static int   s_first_h[MAX_RESIZED_DIM];
    static int   s_count_h[MAX_RESIZED_DIM];
    static float s_w_h[MAX_RESIZED_DIM * MAX_FILTER_TAPS];
    static int   s_first_w[MAX_RESIZED_DIM];
    static int   s_count_w[MAX_RESIZED_DIM];
    static float s_w_w[MAX_RESIZED_DIM * MAX_FILTER_TAPS];

    const int   _target_short = (int)((MODEL_INPUT_H * 256 + 112) / 224);   /* round(48*256/224) = 55 */
    const int   _short_edge   = (_source_width < _source_height) ? _source_width : _source_height;
    const float _ratio        = (float)_target_short / (float)_short_edge;

    /* Resized dimensions (round half-up). */
    int _new_w = (int)floorf((float)_source_width  * _ratio + 0.5f);
    int _new_h = (int)floorf((float)_source_height * _ratio + 0.5f);

    /* Defensive clamps to scratch capacity. */
    if (_new_w > MAX_RESIZED_DIM) _new_w = MAX_RESIZED_DIM;
    if (_new_h > MAX_RESIZED_DIM) _new_h = MAX_RESIZED_DIM;
    if (_new_w < MODEL_INPUT_W)   _new_w = MODEL_INPUT_W;
    if (_new_h < MODEL_INPUT_H)   _new_h = MODEL_INPUT_H;

    /* Precompute resampling kernels for both axes. */
    compute_coeffs(_source_height, _new_h, _ratio, s_first_h, s_count_h, s_w_h);
    compute_coeffs(_source_width,  _new_w, _ratio, s_first_w, s_count_w, s_w_w);

    const int _crop_left = (_new_w - MODEL_INPUT_W) / 2;
    const int _crop_top  = (_new_h - MODEL_INPUT_H) / 2;
    int _dy;
    int _dx;
    int _ch;

    /* Resize directly into s_resized as [MODEL_INPUT_H][_new_w][C] using the
     * cropped vertical range (only rows that the centre crop will keep). */
    for (_dy = 0; _dy < MODEL_INPUT_H; _dy++)
    {
        const int _resize_y = _dy + _crop_top;
        const int _first_y = s_first_h[_resize_y];
        const int _count_y = s_count_h[_resize_y];
        const float *p_weights_y  = &s_w_h[_resize_y * MAX_FILTER_TAPS];

        int _sx;
        for (_sx = 0; _sx < _new_w; _sx++)
        {
            const int _first_x = s_first_w[_sx];
            const int _count_x = s_count_w[_sx];
            const float *p_weights_x   = &s_w_w[_sx * MAX_FILTER_TAPS];

            float _partial[MODEL_INPUT_C] = {0};
            int   _t_y;
            int   _t_x;

            for (_t_y = 0; _t_y < _count_y; _t_y++)
            {
                const int   _src_y = _first_y + _t_y;
                const float _wgt_y = p_weights_y[_t_y];
                float _row_h[MODEL_INPUT_C] = {0};

                for (_t_x = 0; _t_x < _count_x; _t_x++)
                {
                    const int   _src_x = _first_x + _t_x;
                    const float _wgt_x = p_weights_x[_t_x];
                    const uint8_t *p_src_pixel = &p_source_image_hwc[
                        (_src_y * _source_width + _src_x) * MODEL_INPUT_C];
                    for (_ch = 0; _ch < MODEL_INPUT_C; _ch++)
                    {
                        _row_h[_ch] += _wgt_x * (float)p_src_pixel[_ch];
                    }
                }
                for (_ch = 0; _ch < MODEL_INPUT_C; _ch++)
                {
                    _partial[_ch] += _wgt_y * _row_h[_ch];
                }
            }
            for (_ch = 0; _ch < MODEL_INPUT_C; _ch++)
            {
                s_resized[(_dy * _new_w + _sx) * MODEL_INPUT_C + _ch] = _partial[_ch];
            }
        }
    }

    /* Centre-crop + normalise + quantize into the int8 output tensor. */
    for (_dy = 0; _dy < MODEL_INPUT_H; _dy++)
    {
        for (_dx = 0; _dx < MODEL_INPUT_W; _dx++)
        {
            const int _src_x = _dx + _crop_left;
            for (_ch = 0; _ch < MODEL_INPUT_C; _ch++)
            {
                float _pix  = s_resized[(_dy * _new_w + _src_x) * MODEL_INPUT_C + _ch];
                float _norm = _pix / INPUT_NORM_SCALE + INPUT_NORM_BIAS;
                int32_t _quant_val  = (int32_t)lroundf(_norm / INPUT_SCALE) + INPUT_ZP;
                if (_quant_val < -128) { _quant_val = -128; }
                if (_quant_val >  127) { _quant_val =  127; }
                p_input_tensor[
                    (_dy * MODEL_INPUT_W + _dx) * MODEL_INPUT_C + _ch] = (int8_t)_quant_val;
            }
        }
    }
}

