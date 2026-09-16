/*
 * Copyright (c) 2020 - 2026 Renesas Electronics Corporation and/or its affiliates
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
/**********************************************************************************************************************
 * File Name    : preprocessing.c
 * Description  : TinyWav2letter preprocessing - PCM to float conversion,
 *                MFCC/delta/delta-delta extraction, normalization, and int8
 *                quantization for model input.
 *********************************************************************************************************************/

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "preprocessing.h"
#include "mfcc_librosa.h"   /* librosa-compatible MFCC + delta backend */

/* Static scratch buffers used during preprocessing. Sized to the maximum
 * supported PCM input length and maximum model input time dimension. */
static float s_audio_f32 [TW2L_PCM_MAX_SAMPLES];
static float s_mfcc_t13  [MODEL_INPUT_TIME * TW2L_MFCC_COEFFS];
static float s_delta_t13 [MODEL_INPUT_TIME * TW2L_MFCC_COEFFS];
static float s_delta2_t13[MODEL_INPUT_TIME * TW2L_MFCC_COEFFS];

/**
 * @brief Normalize a float array in place using global mean and standard deviation.
 *
 * Applies p_x[i] = (p_x[i] - mean) / std across _elem_count elements.
 *
 * @param[in,out] p_x Pointer to the float array to normalize in place.
 * @param[in] _elem_count Number of elements in p_x.
 */
static void normalize_global_inplace(float *p_x, uint32_t _elem_count)
{
    double _sum    = 0.0;
    double _sq_sum = 0.0;
    uint32_t i;

    /* Accumulate sum and sum-of-squares for mean and variance computation. */
    for (i = 0U; i < _elem_count; ++i) {
        const double _dbl_val = (double)p_x[i];
        _sum    += _dbl_val;
        _sq_sum += _dbl_val * _dbl_val;
    }

    const double _mean = _sum / (double)_elem_count;
    const double _var  = (_sq_sum / (double)_elem_count) - (_mean * _mean);
    const double _std  = sqrt(_var);

    /* Apply normalization: subtract mean and divide by standard deviation. */
    for (i = 0U; i < _elem_count; ++i) {
        p_x[i] = (float)(((double)p_x[i] - _mean) / _std);
    }
}

/**
 * @brief Quantize one float feature value to int8 using model quantization parameters.
 *
 * Uses scale and zero-point, rounds to nearest integer, and clamps to [-128, 127].
 *
 * @param[in] _value Float feature value to quantize.
 *
 * @return Quantized int8 representation of _value.
 */
static int8_t quantize_int8(float _value)
{
    const float _scaled  = _value / (float)MODEL_INPUT_SCALE
                         + (float)MODEL_INPUT_ZERO_POINT;
    /* Round half-away-from-zero by adding/subtracting 0.5 before truncation. */
    const float _rounded = (_scaled >= 0.0f) ? (_scaled + 0.5f) : (_scaled - 0.5f);
    long _q_val = (long)_rounded;

    /* Clamp to int8 range to prevent overflow. */
    if (_q_val >  127L) { _q_val =  127L; }
    if (_q_val < -128L) { _q_val = -128L; }
    return (int8_t)_q_val;
}

/**
 * @brief Convert int16 mono PCM samples to quantized model input features.
 *
 * Extracts MFCC, delta, and delta-delta features, normalizes each stream,
 * and pads to MODEL_INPUT_TIME frames.
 *
 * @param[in] p_pcm_in Pointer to int16 mono PCM samples at 16 kHz.
 * @param[in] _pcm_sample_count Number of samples in p_pcm_in.
 * @param[out] p_out_q Output buffer for quantized model input [MODEL_INPUT_SIZE].
 * @param[out] p_out_time_frames Pointer to store MFCC frame count before padding.
 *
 * @return PREPROCESS_OK on success.
 * @return PREPROCESS_ERR_INVALID_ARG if a required pointer argument is NULL.
 * @return PREPROCESS_ERR_TOO_SHORT if input is too short to extract features.
 * @return PREPROCESS_ERR_TOO_LONG if input exceeds TW2L_PCM_MAX_SAMPLES.
 * @return PREPROCESS_ERR_BACKEND if MFCC or delta computation fails.
 */
preprocess_status_t preprocess(const int16_t *p_pcm_in,
                               uint32_t _pcm_sample_count,
                               int8_t p_out_q[MODEL_INPUT_SIZE],
                               uint32_t *p_out_time_frames)
{
    uint32_t _frames = 0U;
    uint32_t t;
    uint32_t f;

    /* Validate all required pointer arguments. */
    if ((NULL == p_pcm_in) || (NULL == p_out_q) || (NULL == p_out_time_frames)) {
        return PREPROCESS_ERR_INVALID_ARG;
    }
    if (0U == _pcm_sample_count) {
        return PREPROCESS_ERR_TOO_SHORT;
    }
    if (_pcm_sample_count > TW2L_PCM_MAX_SAMPLES) {
        return PREPROCESS_ERR_TOO_LONG;
    }

    /* Convert int16 PCM samples to float32 in the range [-1, 1). */
    for (t = 0U; t < _pcm_sample_count; ++t) {
        s_audio_f32[t] = (float)p_pcm_in[t] / 32768.0f;
    }

    /* Extract MFCC features using the librosa-compatible backend. */
    if (0 != mfcc_librosa_extract(s_audio_f32,
                                  _pcm_sample_count,
                                  s_mfcc_t13,
                                  MODEL_INPUT_TIME,
                                  &_frames)) {
        return PREPROCESS_ERR_BACKEND;
    }
    if (_frames < 2U) {
        return PREPROCESS_ERR_TOO_SHORT;
    }
    if (_frames > MODEL_INPUT_TIME) {
        return PREPROCESS_ERR_TOO_LONG;
    }

    /* Compute first-order delta (velocity) features. */
    if (0 != mfcc_librosa_delta(s_mfcc_t13, _frames, 1U, s_delta_t13)) {
        return PREPROCESS_ERR_BACKEND;
    }
    /* Compute second-order delta (acceleration) features. */
    if (0 != mfcc_librosa_delta(s_mfcc_t13, _frames, 2U, s_delta2_t13)) {
        return PREPROCESS_ERR_BACKEND;
    }

    /* Apply per-stream global mean/std normalization. */
    const uint32_t _norm_elem_count = _frames * TW2L_MFCC_COEFFS;
    normalize_global_inplace(s_mfcc_t13,   _norm_elem_count);
    normalize_global_inplace(s_delta_t13,  _norm_elem_count);
    normalize_global_inplace(s_delta2_t13, _norm_elem_count);

    /* Interleave [mfcc | delta | delta2] and quantize to int8. */
    for (t = 0U; t < _frames; ++t) {
        const uint32_t _row_base = t * MODEL_INPUT_FEATURES;
        const uint32_t _src_base = t * TW2L_MFCC_COEFFS;
        for (f = 0U; f < TW2L_MFCC_COEFFS; ++f) {
            p_out_q[_row_base + f]                          = quantize_int8(s_mfcc_t13  [_src_base + f]);
            p_out_q[_row_base + TW2L_MFCC_COEFFS + f]       = quantize_int8(s_delta_t13 [_src_base + f]);
            p_out_q[_row_base + 2U * TW2L_MFCC_COEFFS + f]  = quantize_int8(s_delta2_t13[_src_base + f]);
        }
    }

    /* Right-pad to MODEL_INPUT_TIME by repeating the (current_len - 2) row. */
    if (_frames < MODEL_INPUT_TIME) {
        uint32_t _cur_len = _frames;
        while (_cur_len < MODEL_INPUT_TIME) {
            const uint32_t _src = (_cur_len - 2U) * MODEL_INPUT_FEATURES;
            const uint32_t _dst = _cur_len * MODEL_INPUT_FEATURES;
            memcpy(&p_out_q[_dst], &p_out_q[_src], MODEL_INPUT_FEATURES);
            _cur_len++;
        }
    }

    *p_out_time_frames = _frames;
    return PREPROCESS_OK;
}
