/*
 * Copyright (c) 2020 - 2026 Renesas Electronics Corporation and/or its affiliates
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
/**********************************************************************************************************************
 * File Name    : mfcc_librosa.c
 * Description  : Librosa-compatible MFCC and delta feature extraction implementation.
 *********************************************************************************************************************/

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "mfcc_librosa.h"

/* ---------------------------------------------------------------------
 * Fixed parameters
 * ------------------------------------------------------------------ */
#define TW2L_SR             (16000)
#define TW2L_N_FFT          (512)
#define TW2L_HOP            (160)
#define TW2L_N_BINS         ((TW2L_N_FFT / 2) + 1)   /* 257 */
#define TW2L_N_MELS         (128)
#define TW2L_N_MFCC         (13)
#define TW2L_MAX_FRAMES     (296)
#define TW2L_FMIN           (0.0f)
#define TW2L_FMAX           ((float)TW2L_SR / 2.0f)  /* 8000 */
#define TW2L_POWER_TO_DB_AMIN   (1.0e-10f)
#define TW2L_POWER_TO_DB_TOPDB  (80.0f)
#define TW2L_DELTA_WIDTH    (9)

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---------------------------------------------------------------------
 * Cached tables (initialised on first extract call)
 * ------------------------------------------------------------------ */
static int   s_is_tables_ready = 0;
static float s_hann      [TW2L_N_FFT];
static float s_mel_fb    [TW2L_N_MELS][TW2L_N_BINS];
static float s_dct_ortho [TW2L_N_MFCC][TW2L_N_MELS];
static float s_logmel_t_m[TW2L_MAX_FRAMES][TW2L_N_MELS];

/* FFT twiddle tables for a 512-point complex radix-2 FFT. */
static float s_fft_cos[TW2L_N_FFT];
static float s_fft_sin[TW2L_N_FFT];
static int   s_bit_rev[TW2L_N_FFT];

/* ---------------------------------------------------------------------
 * Slaney (auditory-toolbox) mel scale used by librosa when htk=False.
 * ------------------------------------------------------------------ */
static float hz_to_mel_slaney(float hz_value)
{
    const float f_min = 0.0f;
    const float f_sp  = 200.0f / 3.0f;
    const float min_log_hz  = 1000.0f;
    const float min_log_mel = (min_log_hz - f_min) / f_sp;         /* 15 */
    const float logstep     = logf(6.4f) / 27.0f;

    if (hz_value >= min_log_hz) {
        return min_log_mel + logf(hz_value / min_log_hz) / logstep;
    }
    return (hz_value - f_min) / f_sp;
}

static float mel_to_hz_slaney(float mel_value)
{
    const float f_min = 0.0f;
    const float f_sp  = 200.0f / 3.0f;
    const float min_log_hz  = 1000.0f;
    const float min_log_mel = (min_log_hz - f_min) / f_sp;
    const float logstep     = logf(6.4f) / 27.0f;

    if (mel_value >= min_log_mel) {
        return min_log_hz * expf(logstep * (mel_value - min_log_mel));
    }
    return f_min + f_sp * mel_value;
}

/* ---------------------------------------------------------------------
 * Table builders
 * ------------------------------------------------------------------ */
static void build_hann_periodic(void)
{
    /* Librosa uses scipy.signal.get_window('hann', N) with fftbins=True,
     * i.e. the periodic Hann window: w[n] = 0.5 * (1 - cos(2*pi*n / N)). */
    for (int n = 0; n < TW2L_N_FFT; ++n) {
        s_hann[n] = 0.5f * (1.0f - cosf((float)(2.0 * M_PI) * (float)n / (float)TW2L_N_FFT));
    }
}

static void build_mel_filterbank_slaney(void)
{
    const float mel_min = hz_to_mel_slaney(TW2L_FMIN);
    const float mel_max = hz_to_mel_slaney(TW2L_FMAX);

    float mel_pts[TW2L_N_MELS + 2];
    float hz_pts [TW2L_N_MELS + 2];
    float bin_hz [TW2L_N_BINS];

    for (int i = 0; i < (TW2L_N_MELS + 2); ++i) {
        const float t = (float)i / (float)(TW2L_N_MELS + 1);
        mel_pts[i] = mel_min + t * (mel_max - mel_min);
        hz_pts [i] = mel_to_hz_slaney(mel_pts[i]);
    }

    /* FFT bin centre frequencies for a 512-point FFT at 16 kHz. */
    for (int k = 0; k < TW2L_N_BINS; ++k) {
        bin_hz[k] = (float)k * (float)TW2L_SR / (float)TW2L_N_FFT;
    }

    for (int m = 0; m < TW2L_N_MELS; ++m) {
        const float f_lo = hz_pts[m];
        const float f_ct = hz_pts[m + 1];
        const float f_hi = hz_pts[m + 2];

        /* Slaney normalisation: 2 / (f_hi - f_lo). */
        const float norm = 2.0f / (f_hi - f_lo);

        for (int k = 0; k < TW2L_N_BINS; ++k) {
            const float f = bin_hz[k];
            float w = 0.0f;
            if (f >= f_lo && f <= f_hi) {
                if (f <= f_ct) {
                    w = (f - f_lo) / (f_ct - f_lo);
                } else {
                    w = (f_hi - f) / (f_hi - f_ct);
                }
                if (w < 0.0f) { w = 0.0f; }
                w *= norm;
            }
            s_mel_fb[m][k] = w;
        }
    }
}

static void build_dct_ortho(void)
{
    /* DCT-II with norm='ortho', keeping the first n_mfcc rows.
     * scipy convention:
     *   D[0][n]  = sqrt(1/N)
     *   D[k][n]  = sqrt(2/N) * cos(pi * k * (2n + 1) / (2N))
     */
    const float scale0 = sqrtf(1.0f / (float)TW2L_N_MELS);
    const float scalek = sqrtf(2.0f / (float)TW2L_N_MELS);
    for (int k = 0; k < TW2L_N_MFCC; ++k) {
        const float s = (k == 0) ? scale0 : scalek;
        for (int n = 0; n < TW2L_N_MELS; ++n) {
            s_dct_ortho[k][n] = s * cosf((float)M_PI * (float)k *
                                         ((2.0f * (float)n) + 1.0f) /
                                         (2.0f * (float)TW2L_N_MELS));
        }
    }
}

static void build_fft_tables(void)
{
    for (int n = 0; n < TW2L_N_FFT; ++n) {
        const float angle = -2.0f * (float)M_PI * (float)n / (float)TW2L_N_FFT;
        s_fft_cos[n] = cosf(angle);
        s_fft_sin[n] = sinf(angle);
    }

    /* Bit-reversal permutation table for radix-2 FFT of size 512 (9 bits). */
    const int log2n = 9;
    for (int i = 0; i < TW2L_N_FFT; ++i) {
        int r = 0;
        int x = i;
        for (int b = 0; b < log2n; ++b) {
            r = (r << 1) | (x & 1);
            x >>= 1;
        }
        s_bit_rev[i] = r;
    }
}

static void ensure_tables(void)
{
    if (0 != s_is_tables_ready) {
        return;
    }
    build_hann_periodic();
    build_mel_filterbank_slaney();
    build_dct_ortho();
    build_fft_tables();
    s_is_tables_ready = 1;
}

/* ---------------------------------------------------------------------
 * In-place radix-2 Cooley-Tukey FFT, 512 complex points.
 * Inputs/outputs interleaved as [re, im, re, im, ...].
 * ------------------------------------------------------------------ */
static void fft_512(float *p_fft_data)
{
    /* Bit-reverse permutation. */
    for (int i = 0; i < TW2L_N_FFT; ++i) {
        const int j = s_bit_rev[i];
        if (j > i) {
            const float temp_real = p_fft_data[2 * i];
            const float temp_imag = p_fft_data[2 * i + 1];
            p_fft_data[2 * i]     = p_fft_data[2 * j];
            p_fft_data[2 * i + 1] = p_fft_data[2 * j + 1];
            p_fft_data[2 * j]     = temp_real;
            p_fft_data[2 * j + 1] = temp_imag;
        }
    }

    /* Butterfly stages. */
    for (int size = 2; size <= TW2L_N_FFT; size <<= 1) {
        const int half = size >> 1;
        const int step = TW2L_N_FFT / size;
        for (int i = 0; i < TW2L_N_FFT; i += size) {
            for (int j = 0; j < half; ++j) {
                const int  ti  = (j * step);
                const float weight_real = s_fft_cos[ti];
                const float weight_imag = s_fft_sin[ti];
                const int   a  = 2 * (i + j);
                const int   b  = 2 * (i + j + half);
                const float mix_real = p_fft_data[b] * weight_real - p_fft_data[b + 1] * weight_imag;
                const float mix_imag = p_fft_data[b] * weight_imag + p_fft_data[b + 1] * weight_real;
                p_fft_data[b]        = p_fft_data[a]     - mix_real;
                p_fft_data[b + 1]    = p_fft_data[a + 1] - mix_imag;
                p_fft_data[a]        = p_fft_data[a]     + mix_real;
                p_fft_data[a + 1]    = p_fft_data[a + 1] + mix_imag;
            }
        }
    }
}

/* ---------------------------------------------------------------------
 * Constant-zero padding used by librosa STFT when center=True and
 * pad_mode='constant' (the default path for feature.mfcc()).
 * ------------------------------------------------------------------ */
static void read_padded_constant(const float *p_source, int32_t source_len,
                                 int32_t source_index, float *p_out_sample)
{
    if ((source_index >= 0) && (source_index < source_len)) {
        *p_out_sample = p_source[source_index];
        return;
    }
    *p_out_sample = 0.0f;
}

/**
 * @brief Extract MFCC[13] features in row-major layout [T, 13].
 *
 * Implements librosa.feature.mfcc with n_mfcc=13, n_fft=512, hop_length=160,
 * n_mels=128, sr=16000, center=True, htk=False, and power_to_db with top_db=80.
 *
 * @param[in]  p_audio         Input PCM audio samples as float32.
 * @param[in]  audio_samples   Number of samples in the audio buffer.
 * @param[out] p_out_mfcc_t13  Output buffer [max_frames * 13] for MFCC features.
 * @param[in]  max_frames      Maximum number of frames the output buffer can hold.
 * @param[out] p_out_frames    Actual number of frames written on success.
 *
 * @return  0 on success.
 * @return -1 if a required pointer argument is NULL.
 * @return -2 if audio_samples is zero.
 * @return -3 if the computed frame count exceeds max_frames or TW2L_MAX_FRAMES.
 */
int mfcc_librosa_extract(const float *p_audio,
                         uint32_t audio_samples,
                         float *p_out_mfcc_t13,
                         uint32_t max_frames,
                         uint32_t *p_out_frames)
{
    if ((NULL == p_audio) || (NULL == p_out_mfcc_t13) || (NULL == p_out_frames)) {
        return -1;
    }
    if (0U == audio_samples) {
        return -2;
    }

    ensure_tables();

    /* librosa center=True with a 512-point window gives:
     *   n_frames = 1 + floor(audio_samples / hop_length)
     * The first frame is centred on sample 0 (i.e. left half is
     * reflect-padded); the last frame is centred on the largest
     * multiple of hop that fits within audio_samples. */
    const uint32_t n_frames = 1U + (audio_samples / (uint32_t)TW2L_HOP);
    if (n_frames > max_frames) {
        return -3;
    }
    if (n_frames > (uint32_t)TW2L_MAX_FRAMES) {
        return -3;
    }

    const int32_t pad = TW2L_N_FFT / 2;

    float frame_buf[TW2L_N_FFT * 2];   /* interleaved complex I/O for FFT */
    float mel_buf  [TW2L_N_MELS];
    float global_max_db_value = -1.0e30f;

    for (uint32_t t = 0U; t < n_frames; ++t) {
        const int32_t centre = (int32_t)t * TW2L_HOP;
        const int32_t start  = centre - pad;

        /* Fill windowed frame with reflection at the edges. */
        for (int n = 0; n < TW2L_N_FFT; ++n) {
            float sample;
            read_padded_constant(p_audio, (int32_t)audio_samples, start + n, &sample);
            frame_buf[2 * n]     = sample * s_hann[n];
            frame_buf[2 * n + 1] = 0.0f;
        }

        fft_512(frame_buf);

        /* Mel filterbank on the power spectrogram. */
        for (int m = 0; m < TW2L_N_MELS; ++m) {
            float acc = 0.0f;
            for (int k = 0; k < TW2L_N_BINS; ++k) {
                const float real_part = frame_buf[2 * k];
                const float imag_part = frame_buf[2 * k + 1];
                const float power_val = (real_part * real_part) + (imag_part * imag_part);
                acc += s_mel_fb[m][k] * power_val;
            }
            mel_buf[m] = acc;
        }

        /* power_to_db(ref=1.0, amin=1e-10) pre-step; top_db clipping is
         * applied in a second pass against the GLOBAL max over the full
         * spectrogram, matching librosa semantics. */
        for (int m = 0; m < TW2L_N_MELS; ++m) {
            float power_val = mel_buf[m];
            if (power_val < TW2L_POWER_TO_DB_AMIN) { power_val = TW2L_POWER_TO_DB_AMIN; }
            const float decibel_val = 10.0f * log10f(power_val);
            s_logmel_t_m[t][m] = decibel_val;
            if (decibel_val > global_max_db_value) { global_max_db_value = decibel_val; }
        }
    }

    const float floor_db = global_max_db_value - TW2L_POWER_TO_DB_TOPDB;

    for (uint32_t t = 0U; t < n_frames; ++t) {
        /* DCT-II orthonormal, first 13 coefficients. */
        float *p_out_row = &p_out_mfcc_t13[t * (uint32_t)TW2L_N_MFCC];
        for (int k = 0; k < TW2L_N_MFCC; ++k) {
            float acc = 0.0f;
            for (int n = 0; n < TW2L_N_MELS; ++n) {
                float decibel_val = s_logmel_t_m[t][n];
                if (decibel_val < floor_db) { decibel_val = floor_db; }
                acc += s_dct_ortho[k][n] * decibel_val;
            }
            p_out_row[k] = acc;
        }
    }

    *p_out_frames = n_frames;
    return 0;
}

/**
 * @brief Compute first or second temporal derivative of MFCC features.
 *
 * Uses a Savitzky-Golay filter (width=9) matching librosa.feature.delta with
 * mode='interp'. Edge frames are filled by fitting the nearest 9 frames and
 * repeating the boundary derivative value across the first/last half-window.
 *
 * Interior weight vectors for width=9:
 *   order=1: c[k] = k / 60              for k in -4..4
 *   order=2: c[k] = SG(deriv=2, poly=2) for k in -4..4
 */
static const float k_sg_order1_w9[TW2L_DELTA_WIDTH] = {
    -4.0f / 60.0f, -3.0f / 60.0f, -2.0f / 60.0f, -1.0f / 60.0f,
     0.0f / 60.0f,
     1.0f / 60.0f,  2.0f / 60.0f,  3.0f / 60.0f,  4.0f / 60.0f
};

static const float k_sg_order2_w9[TW2L_DELTA_WIDTH] = {
    28.0f / 462.0f,  7.0f / 462.0f, -8.0f / 462.0f, -17.0f / 462.0f,
   -20.0f / 462.0f,
   -17.0f / 462.0f, -8.0f / 462.0f,  7.0f / 462.0f,  28.0f / 462.0f
};

/**
 * @brief Compute first/second temporal derivative for [T, 13] MFCC features.
 *
 * @param[in]  p_mfcc_t13  Input MFCC array [frames * 13] in row-major order.
 * @param[in]  frames    Number of time frames in the input array.
 * @param[in]  order     Derivative order: 1 (delta) or 2 (delta-delta).
 * @param[out] p_out_t13 Output buffer [frames * 13] for the computed delta.
 *
 * @return  0 on success.
 * @return -1 if a required pointer argument is NULL.
 * @return -2 if order is not 1 or 2.
 * @return -3 if frames is less than TW2L_DELTA_WIDTH.
 */
int mfcc_librosa_delta(const float *p_mfcc_t13,
                       uint32_t frames,
                       uint32_t order,
                       float *p_out_t13)
{
    if ((NULL == p_mfcc_t13) || (NULL == p_out_t13)) {
        return -1;
    }
    if ((1U != order) && (2U != order)) {
        return -2;
    }
    if (frames < (uint32_t)TW2L_DELTA_WIDTH) {
        return -3;
    }

    const float *p_weights = (1U == order) ? k_sg_order1_w9 : k_sg_order2_w9;
    const int    half    = TW2L_DELTA_WIDTH / 2;

    /* Interior samples: apply the fixed SG weight vector across time
     * independently for each of the 13 MFCC coefficients. */
    for (uint32_t t = (uint32_t)half; t + (uint32_t)half < frames; ++t) {
        float *p_out_row = &p_out_t13[t * (uint32_t)TW2L_N_MFCC];
        for (int c = 0; c < TW2L_N_MFCC; ++c) {
            float acc = 0.0f;
            for (int k = 0; k < TW2L_DELTA_WIDTH; ++k) {
                const int32_t time_index = (int32_t)t + (k - half);
                acc += p_weights[k] * p_mfcc_t13[(uint32_t)time_index * (uint32_t)TW2L_N_MFCC + (uint32_t)c];
            }
            p_out_row[c] = acc;
        }
    }

    /* Edge samples: librosa's mode='interp' with polyorder==deriv
     * (which feature.delta sets by default) evaluates to the same SG
     * dot kernel on the first/last `width` samples and repeats that
     * value across the first/last half-window frames. */
    float coeff_column[TW2L_DELTA_WIDTH];
    for (int c = 0; c < TW2L_N_MFCC; ++c) {
        /* Leading edge: fit over frames [0 .. width-1]. */
        for (int i = 0; i < TW2L_DELTA_WIDTH; ++i) {
            coeff_column[i] = p_mfcc_t13[(uint32_t)i * (uint32_t)TW2L_N_MFCC + (uint32_t)c];
        }
        float lead = 0.0f;
        for (int i = 0; i < TW2L_DELTA_WIDTH; ++i) {
            lead += p_weights[i] * coeff_column[i];
        }
        for (int t = 0; t < half; ++t) {
            p_out_t13[(uint32_t)t * (uint32_t)TW2L_N_MFCC + (uint32_t)c] = lead;
        }

        /* Trailing edge: fit over frames [frames-width .. frames-1]. */
        for (int i = 0; i < TW2L_DELTA_WIDTH; ++i) {
            const uint32_t source_frame_idx = frames - (uint32_t)TW2L_DELTA_WIDTH + (uint32_t)i;
            coeff_column[i] = p_mfcc_t13[source_frame_idx * (uint32_t)TW2L_N_MFCC + (uint32_t)c];
        }
        float tail = 0.0f;
        for (int i = 0; i < TW2L_DELTA_WIDTH; ++i) {
            tail += p_weights[i] * coeff_column[i];
        }
        for (int t = 0; t < half; ++t) {
            const uint32_t output_frame_idx = frames - (uint32_t)half + (uint32_t)t;
            p_out_t13[output_frame_idx * (uint32_t)TW2L_N_MFCC + (uint32_t)c] = tail;
        }
    }

    return 0;
}
