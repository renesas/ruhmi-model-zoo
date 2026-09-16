/*
 * Copyright (c) 2020 - 2026 Renesas Electronics Corporation and/or its affiliates.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * DISCLAIMER
 * This software is supplied by Renesas Electronics Corporation and is only
 * intended for use with Renesas products. No other uses are authorized.
 * This software is owned by Renesas Electronics Corporation and is protected
 * under all applicable laws, including copyright laws.
 *
 * THIS SOFTWARE IS PROVIDED "AS IS" AND RENESAS MAKES NO WARRANTIES REGARDING
 * THIS SOFTWARE, WHETHER EXPRESS, IMPLIED OR STATUTORY, INCLUDING BUT NOT
 * LIMITED TO WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE
 * AND NON-INFRINGEMENT. ALL SUCH WARRANTIES ARE EXPRESSLY DISCLAIMED.
 */

/**
 * @file    preprocessing.c
 * @brief   MicroNet audio preprocessing: int16 PCM -> int8 model patches.
 *
 * Implements the DSP pipeline described in preprocessing.h without any heap
 * allocation. Numerics are matched to the trained model parameters
 * (librosa mel spectrogram, Slaney norm,
 * Hann periodic window, center=False STFT, 2x2 mean pooling, and int8
 * quantization with the TFLite input scale / zero-point).
 */

#include "preprocessing.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/*  Static working buffers (no heap use).                             */
/* ------------------------------------------------------------------ */

/** Hann periodic window coefficients, cached after the first call. */
static float s_hann_window[PREP_N_FFT];
static bool  s_hann_ready = false;

/** Mel triangle centre frequencies in Hz (Slaney). */
static float s_mel_freqs[PREP_N_MELS + 2];
static bool  s_mel_freqs_ready = false;
static int32_t   s_mel_freqs_sr    = 0;

/** FFT scratch buffers (real / imaginary parts, interleaved by array). */
static float s_fft_real[PREP_N_FFT];
static float s_fft_imag[PREP_N_FFT];

/** Power spectrum for one STFT frame. */
static float s_power_spectrum[PREP_N_BINS];

/**
 * Full log-mel matrix for one clip, laid out as [mel_bin][frame].
 * For a 10 s / 16 kHz clip the effective size is 64 x 311; the extra columns
 * up to PREP_MAX_MEL_FRAMES are unused but kept static for MCU determinism.
 */
static float s_log_mel[PREP_N_MELS * PREP_MAX_MEL_FRAMES];


/* ------------------------------------------------------------------ */
/*  Radix-2 Cooley-Tukey FFT (in-place).                              */
/* ------------------------------------------------------------------ */

/**
 * @brief  Compute an in-place radix-2 Cooley-Tukey FFT.
 *
 * @param[in,out]  p_real       Real parts of the input/output (length _fft_length).
 * @param[in,out]  p_imag       Imaginary parts of the input/output (length _fft_length).
 * @param[in]      _fft_length  Transform length; must be a power of two.
 */
static void fft_radix2_inplace(float *p_real, float *p_imag, int32_t _fft_length)
{
    int32_t _bit_reversed_index = 0;

    /* Bit-reversal permutation. */
    for (int32_t _sample_index = 1; _sample_index < _fft_length; _sample_index++)
    {
        int32_t _bit_mask = _fft_length >> 1;
        while ((_bit_reversed_index & _bit_mask) != 0)
        {
            _bit_reversed_index ^= _bit_mask;
            _bit_mask >>= 1;
        }
        _bit_reversed_index ^= _bit_mask;

        if (_sample_index < _bit_reversed_index)
        {
            float _temp_real = p_real[_sample_index];
            float _temp_imag = p_imag[_sample_index];
            p_real[_sample_index] = p_real[_bit_reversed_index];
            p_imag[_sample_index] = p_imag[_bit_reversed_index];
            p_real[_bit_reversed_index] = _temp_real;
            p_imag[_bit_reversed_index] = _temp_imag;
        }
    }

    /* Butterflies. */
    for (int32_t _butterfly_span = 2; _butterfly_span <= _fft_length; _butterfly_span <<= 1)
    {
        float _twiddle_angle = (-2.0f * (float)M_PI) / (float)_butterfly_span;
        float _twiddle_step_real = cosf(_twiddle_angle);
        float _twiddle_step_imag = sinf(_twiddle_angle);

        for (int32_t _block_start = 0; _block_start < _fft_length; _block_start += _butterfly_span)
        {
            float _current_twiddle_real = 1.0f;
            float _current_twiddle_imag = 0.0f;
            int32_t _half_span = _butterfly_span / 2;

            for (int32_t _offset = 0; _offset < _half_span; _offset++)
            {
                float _upper_real = p_real[_block_start + _offset];
                float _upper_imag = p_imag[_block_start + _offset];
                float _lower_real = p_real[_block_start + _offset + _half_span];
                float _lower_imag = p_imag[_block_start + _offset + _half_span];

                float _rotated_lower_real =
                    _current_twiddle_real * _lower_real - _current_twiddle_imag * _lower_imag;
                float _rotated_lower_imag =
                    _current_twiddle_real * _lower_imag + _current_twiddle_imag * _lower_real;

                p_real[_block_start + _offset]              = _upper_real + _rotated_lower_real;
                p_imag[_block_start + _offset]              = _upper_imag + _rotated_lower_imag;
                p_real[_block_start + _offset + _half_span] = _upper_real - _rotated_lower_real;
                p_imag[_block_start + _offset + _half_span] = _upper_imag - _rotated_lower_imag;

                float _next_twiddle_real =
                    _current_twiddle_real * _twiddle_step_real - _current_twiddle_imag * _twiddle_step_imag;
                _current_twiddle_imag =
                    _current_twiddle_real * _twiddle_step_imag + _current_twiddle_imag * _twiddle_step_real;
                _current_twiddle_real = _next_twiddle_real;
            }
        }
    }
}


/* ------------------------------------------------------------------ */
/*  Table initializers.                                               */
/* ------------------------------------------------------------------ */

/**
 * @brief  Initialise the periodic Hann window table (idempotent).
 *
 *         Populated on the first call; subsequent calls return immediately.
 *         Matches librosa's 'hann' window with center=False.
 */
static void init_hann_window(void)
{
    if (s_hann_ready)
    {
        return;
    }

    /* Periodic Hann window (matches librosa's 'hann' window with center=False). */
    for (int32_t _n = 0; _n < PREP_N_FFT; _n++)
    {
        s_hann_window[_n] =
            0.5f - 0.5f * cosf((2.0f * (float)M_PI * (float)_n) / (float)PREP_N_FFT);
    }
    s_hann_ready = true;
}

/**
 * @brief  Convert a frequency in Hz to the Slaney mel scale.
 *
 * @param[in]  _frequency_hz  Frequency in Hz.
 * @return  Mel-scale value (librosa htk=False, norm='slaney').
 */
static float hz_to_mel_slaney(float _frequency_hz)
{
    if (MEL_MIN_LOG_HZ > _frequency_hz)
    {
        return (_frequency_hz - MEL_F_MIN) / MEL_F_SP;
    }
    return MEL_MIN_LOG_MEL + logf(_frequency_hz / MEL_MIN_LOG_HZ) / MEL_LOGSTEP;
}

/**
 * @brief  Convert a Slaney mel value to Hz.
 *
 * @param[in]  _mel  Mel-scale value (Slaney).
 * @return  Frequency in Hz.
 */
static float mel_to_hz_slaney(float _mel)
{
    if (MEL_MIN_LOG_MEL > _mel)
    {
        return MEL_F_MIN + _mel * MEL_F_SP;
    }
    return MEL_MIN_LOG_HZ * expf(MEL_LOGSTEP * (_mel - MEL_MIN_LOG_MEL));
}

/**
 * @brief  Initialise the mel filter centre-frequency table (idempotent).
 *
 * @param[in]  _sample_rate  Audio sample rate in Hz; table is rebuilt when the
 *                           rate differs from the last call.
 */
static void init_mel_freqs(int32_t _sample_rate)
{
    if (s_mel_freqs_ready && s_mel_freqs_sr == _sample_rate)
    {
        return;
    }

    const int32_t _n_points            = PREP_N_MELS + 2;
    const float   _max_frequency_hz    = PREP_MEL_FMAX;
    const float   _min_mel_value       = hz_to_mel_slaney(PREP_MEL_FMIN);
    const float   _max_mel_value       = hz_to_mel_slaney(_max_frequency_hz);

    for (int32_t _i = 0; _i < _n_points; _i++)
    {
        float _mel_step = _min_mel_value +
                          ((float)_i / (float)(_n_points - 1)) * (_max_mel_value - _min_mel_value);
        s_mel_freqs[_i] = mel_to_hz_slaney(_mel_step);
    }

    s_mel_freqs_sr    = _sample_rate;
    s_mel_freqs_ready = true;
}


/* ------------------------------------------------------------------ */
/*  Log-mel matrix.                                                   */
/* ------------------------------------------------------------------ */

/**
 * @brief  Fill s_log_mel with the (64 x n_frames) log-mel matrix for the clip.
 *
 * Uses center=False STFT semantics (matches librosa default when center=False):
 *   frame t covers samples [t * hop, t * hop + n_fft), no reflection padding.
 *
 * @param[in]  p_pcm         Mono int16 PCM samples.
 * @param[in]  _num_samples  Number of int16 samples in the clip.
 * @param[in]  _sample_rate  Audio sample rate in Hz.
 * @return  Number of mel frames written, or -1 on error.
 */
static int32_t compute_log_mel_matrix(const int16_t *p_pcm, int32_t _num_samples, int32_t _sample_rate)
{
    /* Number of STFT frames when center=False. */
    if (PREP_N_FFT > _num_samples)
    {
        return -1;   /* clip too short */
    }
    const int32_t _n_frames = 1 + ((_num_samples - PREP_N_FFT) / PREP_HOP);
    if (PREP_MAX_MEL_FRAMES < _n_frames)
    {
        return -1;   /* clip too long for static buffers */
    }

    init_hann_window();
    init_mel_freqs(_sample_rate);

    (void)memset(s_log_mel, 0, sizeof(s_log_mel));

    const float _log_factor = 10.0f;

    for (int32_t _t = 0; _t < _n_frames; _t++)
    {
        const int32_t _frame_start = _t * PREP_HOP;

        /* Windowed real signal for the STFT frame. */
        for (int32_t _n = 0; _n < PREP_N_FFT; _n++)
        {
            float _sample = (float)p_pcm[_frame_start + _n] / PCM_NORM_FACTOR;
            s_fft_real[_n] = _sample * s_hann_window[_n];
            s_fft_imag[_n] = 0.0f;
        }

        fft_radix2_inplace(s_fft_real, s_fft_imag, PREP_N_FFT);

        /* Power spectrum (|X|^2). */
        for (int32_t _k = 0; _k < PREP_N_BINS; _k++)
        {
            s_power_spectrum[_k] = s_fft_real[_k] * s_fft_real[_k]
                                 + s_fft_imag[_k] * s_fft_imag[_k];
        }

        /*
         * Mel projection.  Loop order: FFT bin (outer) -> mel bin (inner) so
         * that the bin -> Hz mapping is computed once per FFT bin.
         *
         * For each triangular filter _m with lower / mid / upper centre
         * frequencies (_lower_edge_hz, _center_hz, _upper_edge_hz):
         *     _weight = max(0, min((f - _lower_edge_hz)/(_center_hz - _lower_edge_hz),
         *                          (_upper_edge_hz - f)/(_upper_edge_hz - _center_hz)))
         *     slaney_scale = 2 / (_upper_edge_hz - _lower_edge_hz)
         *     _mel_energy[_m] += _weight * slaney_scale * _power[_k]
         */
        float _mel_energy[PREP_N_MELS];
        (void)memset(_mel_energy, 0, sizeof(_mel_energy));

        for (int32_t _k = 0; _k < PREP_N_BINS; _k++)
        {
            float _bin_hz = (float)_k * (float)_sample_rate / (float)PREP_N_FFT;
            float _power  = s_power_spectrum[_k];

            for (int32_t _m = 0; _m < PREP_N_MELS; _m++)
            {
                float _lower_edge_hz = s_mel_freqs[_m];
                float _center_hz     = s_mel_freqs[_m + 1];
                float _upper_edge_hz = s_mel_freqs[_m + 2];

                float _lower_slope = (_bin_hz - _lower_edge_hz) / (_center_hz - _lower_edge_hz);
                float _upper_slope = (_upper_edge_hz - _bin_hz) / (_upper_edge_hz - _center_hz);
                float _weight = (_lower_slope < _upper_slope) ? _lower_slope : _upper_slope;

                if (0.0f < _weight)
                {
                    _mel_energy[_m] += _weight * (2.0f / (_upper_edge_hz - _lower_edge_hz)) * _power;
                }
            }
        }

        /* Store as log-dB, shifted by -TRAINING_MEAN so the resulting matrix */
        for (int32_t _m = 0; _m < PREP_N_MELS; _m++)
        {
            float _mel_val = _mel_energy[_m];
            if (PREP_EPSILON > _mel_val)
            {
                _mel_val = PREP_EPSILON;
            }
            float _db_value = _log_factor * log10f(_mel_val);
            s_log_mel[_m * PREP_MAX_MEL_FRAMES + _t] = _db_value - PREP_TRAINING_MEAN;
        }
    }

    return _n_frames;
}


/* ------------------------------------------------------------------ */
/*  Windowing + 2x2 mean pool + int8 quantization.                    */
/* ------------------------------------------------------------------ */

/**
 * @brief  Quantize a float value to int8 using TFLite affine quantization.
 *
 * @param[in]  _value       Float value to quantize.
 * @param[in]  _scale       Quantization scale factor.
 * @param[in]  _zero_point  Quantization zero-point offset.
 * @return  Clamped int8 result in [-128, 127].
 */
static int8_t quantize_to_int8(float _value, float _scale, int32_t _zero_point)
{
    float _quantized_value = roundf(_value / _scale) + (float)_zero_point;
    if (-128.0f > _quantized_value) { _quantized_value = -128.0f; }
    if (127.0f < _quantized_value)  { _quantized_value =  127.0f; }
    return (int8_t)_quantized_value;
}


/**
 * @brief  Extract one (32 x 32) int8 patch for the window starting at
 *         mel-frame index start_frame.
 *
 * @param[in]  _start_frame  Starting mel-frame column index in s_log_mel.
 * @param[out] p_patch_int8  Output buffer of at least INPUT_DIM int8 entries.
 */
static void extract_patch(int32_t _start_frame, int8_t *p_patch_int8)
{
    for (int32_t _i = 0; _i < PATCH_MODEL_DIM; _i++)
    {
        for (int32_t _j = 0; _j < PATCH_MODEL_DIM; _j++)
        {
            /* 2x2 mean-pool over rows (2*_i, 2*_i+1) and cols (2*_j, 2*_j+1). */
            float _sum = 0.0f;
            for (int32_t _di = 0; _di < 2; _di++)
            {
                int32_t _row = (2 * _i) + _di;
                for (int32_t _dj = 0; _dj < 2; _dj++)
                {
                    int32_t _col = _start_frame + (2 * _j) + _dj;
                    _sum += s_log_mel[_row * PREP_MAX_MEL_FRAMES + _col];
                }
            }
            float _pooled = _sum * 0.25f;
            p_patch_int8[_i * PATCH_MODEL_DIM + _j] =
                quantize_to_int8(_pooled, MODEL_INPUT_SCALE, MODEL_INPUT_ZP);
        }
    }
}


/* ------------------------------------------------------------------ */
/*  Public API.                                                       */
/* ------------------------------------------------------------------ */

/**
 * @brief  Convert one clip of int16 PCM audio into a batch of int8 model patches.
 *
 * @param[in]  p_pcm              Mono int16 PCM samples.
 * @param[in]  _num_samples       Number of int16 samples in the clip.
 * @param[in]  _sample_rate       Audio sample rate in Hz (typically 16000).
 * @param[out] p_patches_int8     Caller-provided buffer for int8 patches.
 * @param[in]  _patches_capacity  Capacity of p_patches_int8 in int8 elements.
 * @param[out] p_num_windows      Number of windows actually produced.
 * @return  0 on success, -1 on any error.
 */
int32_t preprocess(const int16_t *p_pcm,
                               int32_t            _num_samples,
                               int32_t            _sample_rate,
                               int8_t        *p_patches_int8,
                               int32_t            _patches_capacity,
                               int32_t           *p_num_windows)
{
    if ((NULL == p_pcm) || (NULL == p_patches_int8) || (NULL == p_num_windows))
    {
        return -1;
    }
    *p_num_windows = 0;

    if ((0 >= _num_samples) || (PREP_MAX_PCM_SAMPLES < _num_samples))
    {
        return -1;
    }

    int32_t _n_frames = compute_log_mel_matrix(p_pcm, _num_samples, _sample_rate);
    if (PREP_PATCH_RAW_FRAMES > _n_frames)
    {
        return -1;
    }

    int32_t       _start_frame = 0;
    int32_t       _win_index   = 0;
    const int32_t _stride      = PREP_OUTER_STRIDE;

    while ((_start_frame + PREP_PATCH_RAW_FRAMES) <= _n_frames)
    {
        if (PREP_MAX_WINDOWS <= _win_index)
        {
            /* Should never happen for the supported clip length. */
            return -1;
        }
        int32_t _offset = _win_index * INPUT_DIM;
        if (_patches_capacity < (_offset + INPUT_DIM))
        {
            return -1;
        }
        extract_patch(_start_frame, &p_patches_int8[_offset]);

        _win_index   += 1;
        _start_frame += _stride;
    }

    if (0 == _win_index)
    {
        return -1;
    }

    *p_num_windows = _win_index;
    return 0;
}
