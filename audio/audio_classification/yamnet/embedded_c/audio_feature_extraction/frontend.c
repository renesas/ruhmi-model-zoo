/*
* Copyright (c) 2020 - 2026 Renesas Electronics Corporation and/or its affiliates
*
* SPDX-License-Identifier: BSD-3-Clause
*/
/**
 * @file frontend.c
 * @brief YAMNet audio front-end implementation: raw int16 waveform to log-mel feature patches.
 */

#include <math.h>
#include <string.h>

#include "frontend.h"

/* Scratch buffers. Kept static so they land in .bss (~5 KB total). */
static float s_frame_real[YAMNET_FFT_SIZE];
static float s_frame_imag[YAMNET_FFT_SIZE];
static float s_magnitude[YAMNET_NUM_SPECTROGRAM_BINS];

/**
 * @brief Run an in-place radix-2 decimation-in-time complex FFT of length 512.
 * @param[in,out] p_re Real-part array of length `YAMNET_FFT_SIZE`.
 * @param[in,out] p_im Imaginary-part array of length `YAMNET_FFT_SIZE`.
 */
static void fft_512_dit(float *p_real_buf, float *p_imag_buf)
{
    uint32_t _i;
    uint32_t _m;
    uint32_t _base;
    uint32_t _k;

    /* Bit-reversal permutation. */
    uint32_t _bit_rev_idx = 0U;
    for (_i = 1U; _i < YAMNET_FFT_SIZE; _i++)
    {
        uint32_t _bit = YAMNET_FFT_HALF_SIZE;
        while (_bit_rev_idx & _bit)
        {
            _bit_rev_idx ^= _bit;
            _bit >>= YAMNET_BIT_SHIFT_STEP;
        }
        _bit_rev_idx ^= _bit;

        if (_i < _bit_rev_idx)
        {
            float _tr = p_real_buf[_i]; p_real_buf[_i] = p_real_buf[_bit_rev_idx]; p_real_buf[_bit_rev_idx] = _tr;
            float _ti = p_imag_buf[_i]; p_imag_buf[_i] = p_imag_buf[_bit_rev_idx]; p_imag_buf[_bit_rev_idx] = _ti;
        }
    }

    /* Cooley-Tukey butterflies. At stage `m` (m = 2, 4, ..., N), the twiddle
     * for butterfly index k is W_m^k = exp(-j 2 pi k / m). We express this as
     * an index into the length-N/2 LUT: W_m^k = W_N^(k * N / m). */
    for (_m = 2U; _m <= YAMNET_FFT_SIZE; _m <<= 1)
    {
        const uint32_t _half = _m >> 1;
        const uint32_t _step = YAMNET_FFT_SIZE / _m;

        for (_base = 0U; _base < YAMNET_FFT_SIZE; _base += _m)
        {
            for (_k = 0U; _k < _half; _k++)
            {
                const float _wr = g_fft_cos[_k * _step];
                const float _wi = g_fft_sin[_k * _step];

                const uint32_t _idx_a = _base + _k;
                const uint32_t _idx_b = _idx_a + _half;

                const float _ar = p_real_buf[_idx_a];
                const float _ai = p_imag_buf[_idx_a];
                const float _br = p_real_buf[_idx_b];
                const float _bi = p_imag_buf[_idx_b];

                const float _vr = _br * _wr - _bi * _wi;
                const float _vi = _br * _wi + _bi * _wr;

                p_real_buf[_idx_a] = _ar + _vr;
                p_imag_buf[_idx_a] = _ai + _vi;
                p_real_buf[_idx_b] = _ar - _vr;
                p_imag_buf[_idx_b] = _ai - _vi;
            }
        }
    }
}

/**
 * @brief Calculate the number of complete STFT frames available from an input waveform.
 * @param[in] num_samples Number of input PCM samples.
 * @return Number of complete STFT frames that can be produced with `pad_end=False`.
 */
size_t yamnet_frontend_num_stft_frames(size_t num_samples)
{
    if (num_samples < YAMNET_WINDOW_SAMPLES)
    {
        return 0U;
    }
    return 1U + (num_samples - YAMNET_WINDOW_SAMPLES) / YAMNET_HOP_SAMPLES;
}

/**
 * @brief Calculate the number of complete 96-row patches available from STFT rows.
 * @param[in] num_stft_frames Number of STFT frame rows available.
 * @return Number of complete 96-row patches that can be produced with `pad_end=False`.
 */
size_t yamnet_frontend_num_patches(size_t num_stft_frames)
{
    if (num_stft_frames < YAMNET_PATCH_FRAMES)
    {
        return 0U;
    }
    return 1U + (num_stft_frames - YAMNET_PATCH_FRAMES) / YAMNET_PATCH_HOP_FRAMES;
}

/**
 * @brief Compute one log-mel spectrogram row for a single STFT frame.
 * @param[in] p_wav Pointer to int16 PCM samples at `YAMNET_SAMPLE_RATE`.
 * @param[in] frame_start Sample index of the start of the STFT frame.
 * @param[out] p_out_row Output buffer holding `YAMNET_NUM_MEL_BINS` floats.
 */
static void compute_one_log_mel_row(const int16_t *p_wav,
                                    size_t         frame_start,
                                    float         *p_out_row)
{
    uint32_t _i;
    uint32_t _k;
    uint32_t _m;

    /* Step 1+2: framing and periodic Hann windowing.
     * Convert int16 PCM to float in [-1, 1) exactly as Python does. */
    for (_i = 0U; _i < YAMNET_WINDOW_SAMPLES; _i++)
    {
        const float _pcm_norm_sample = (float)p_wav[frame_start + _i] / YAMNET_PCM_INT16_NORM_SCALE;
        s_frame_real[_i] = _pcm_norm_sample * g_yamnet_hann[_i];
        s_frame_imag[_i] = 0.0f;
    }
    /* Step 3a: right-zero-pad to FFT_SIZE. Matches tf.signal.stft behaviour
     * when fft_length > frame_length. */
    for (_i = YAMNET_WINDOW_SAMPLES; _i < YAMNET_FFT_SIZE; _i++)
    {
        s_frame_real[_i] = 0.0f;
        s_frame_imag[_i] = 0.0f;
    }

    /* Step 3b: complex FFT of length YAMNET_FFT_SIZE. */
    fft_512_dit(s_frame_real, s_frame_imag);

    /* Step 4: magnitude of the first N/2 + 1 bins. */
    for (_k = 0U; _k < YAMNET_NUM_SPECTROGRAM_BINS; _k++)
    {
        const float _re = s_frame_real[_k];
        const float _im = s_frame_imag[_k];
        s_magnitude[_k] = sqrtf(_re * _re + _im * _im);
    }

    /* Step 5 + 6: mel filterbank projection followed by log. */
    for (_m = 0U; _m < YAMNET_NUM_MEL_BINS; _m++)
    {
        float _acc = 0.0f;
        for (_k = 0U; _k < YAMNET_NUM_SPECTROGRAM_BINS; _k++)
        {
            _acc += s_magnitude[_k] * g_mel_matrix[_k][_m];
        }
        p_out_row[_m] = logf(_acc + YAMNET_MEL_LOG_OFFSET);
    }
}

/**
 * @brief Compute the log-mel row stream for an input waveform.
 * @param[in] p_wav Pointer to int16 PCM samples at `YAMNET_SAMPLE_RATE`.
 * @param[in] num_samples Length of `p_wav` in samples.
 * @param[out] p_out_rows Caller-owned output buffer holding rows of `YAMNET_NUM_MEL_BINS` floats.
 * @param[in] max_rows Capacity of `p_out_rows` expressed in rows.
 * @return Number of rows written, or `0` for invalid input or a waveform that is too short.
 */
size_t yamnet_frontend_compute_log_mel(const int16_t *p_wav,
                                       size_t         num_samples,
                                       float         *p_out_rows,
                                       size_t         max_rows)
{
    if ((NULL == p_wav) || (NULL == p_out_rows) || (0U == max_rows))
    {
        return 0U;
    }

    size_t       _f;
    const size_t _total_frames  = yamnet_frontend_num_stft_frames(num_samples);
    const size_t _rows_to_write = (_total_frames < max_rows) ? _total_frames : max_rows;

    for (_f = 0U; _f < _rows_to_write; _f++)
    {
        const size_t _frame_start = _f * YAMNET_HOP_SAMPLES;
        compute_one_log_mel_row(p_wav, _frame_start, &p_out_rows[_f * YAMNET_NUM_MEL_BINS]);
    }
    return _rows_to_write;
}

/**
 * @brief Copy one log-mel patch from a precomputed row buffer.
 * @param[in] p_log_mel_rows Pointer to the row buffer produced by `yamnet_frontend_compute_log_mel`.
 * @param[in] num_rows Number of valid rows in `p_log_mel_rows`.
 * @param[in] patch_idx Zero-based patch index to extract.
 * @param[out] p_out_patch Caller-owned output buffer of at least `YAMNET_PATCH_ELEMENTS` floats.
 * @retval 1 Patch copied successfully.
 * @retval 0 `patch_idx` is out of range or an input pointer is invalid.
 */
int yamnet_frontend_extract_patch(const float *p_log_mel_rows,
                                  size_t       num_rows,
                                  size_t       patch_idx,
                                  float       *p_out_patch)
{
    if ((NULL == p_log_mel_rows) || (NULL == p_out_patch))
    {
        return 0;
    }

    const size_t _total_patches = yamnet_frontend_num_patches(num_rows);
    if (patch_idx >= _total_patches)
    {
        return 0;
    }

    const size_t  _row_start = patch_idx * YAMNET_PATCH_HOP_FRAMES;
    const float  *p_src      = &p_log_mel_rows[_row_start * YAMNET_NUM_MEL_BINS];
    memcpy(p_out_patch, p_src, YAMNET_PATCH_ELEMENTS * sizeof(float));
    return 1;
}
