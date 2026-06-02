/**
 * @file    preprocessing.c
 * @brief   KWS MFCC preprocessing implementation.
 *
 * Copyright 2026 Renesas Electronics Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#include "preprocessing.h"
#include "arm_math.h"
#include <math.h>
#include <string.h>

/* ========== Internal Constants ========== */
#define FFT_HALF_SIZE   (MFCC_FFT_SIZE / 2 + 1)   /* 257 bins */
#define MEL_LOW_FREQ    20.0f
#define MEL_HIGH_FREQ   4000.0f
#define LOG_OFFSET      1e-6f    /* Same as TF: log(mel + 1e-6) */

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

/* ========== Static Buffers ========== */
static float g_frame_buffer[MFCC_FFT_SIZE];
static float g_fft_output[MFCC_FFT_SIZE];
static float g_magnitude_spectrum[FFT_HALF_SIZE];
static float g_mel_energies[MFCC_NUM_MEL_BINS];

/* Precomputed tables */
static float g_hann_window[MFCC_FRAME_LEN];
static int g_hann_initialized = 0;

static float g_mel_filterbank[MFCC_NUM_MEL_BINS][FFT_HALF_SIZE];
static int g_mel_initialized = 0;

static float g_dct_matrix[MFCC_NUM_COEFFS][MFCC_NUM_MEL_BINS];
static int g_dct_initialized = 0;

static arm_rfft_fast_instance_f32 g_fft_instance;
static int g_fft_initialized = 0;

/**
 * @brief Safely fetch and normalize a PCM sample for MFCC frame construction.
 *
 * @details This helper mirrors the Python behavior where audio shorter than
 *          DESIRED_SAMPLES is effectively zero-padded. If the requested index
 *          is out of range, 0.0f is returned before normalization.
 *
 * @param[in] pcm_input                  Pointer to raw PCM sample array.
 * @param[in] sample_index               Index of the requested sample.
 * @param[in] valid_sample_count         Number of valid samples in pcm_input.
 * @param[in] normalization_denominator  Max-abs normalization denominator.
 *
 * @return Normalized sample value in approximately [-1.0, 1.0].
 */
static float get_normalized_audio_sample(const int16_t *p_pcm_input,
                                         int32_t sample_index,
                                         int32_t valid_sample_count,
                                         float normalization_denominator)
{
    float sample_value = 0.0f;

    if ((sample_index >= 0) && (sample_index < valid_sample_count))
    {
        sample_value = (float)p_pcm_input[sample_index];
    }

    return sample_value / normalization_denominator;
}

/* ========== Helper Functions ========== */

/**
 * @brief Convert linear frequency in Hz to Mel scale (HTK formula).
 *
 * @param frequency_hz Frequency value in Hz.
 * @return Equivalent value on Mel scale.
 */
static float hz_to_mel(float frequency_hz)
{
    return 2595.0f * log10f(1.0f + frequency_hz / 700.0f);
}

/**
 * @brief Convert Mel-scale value back to linear frequency in Hz.
 *
 * @param mel_value Frequency value in Mel scale.
 * @return Equivalent value in Hz.
 */
static float mel_to_hz(float mel_value)
{
    return 700.0f * (powf(10.0f, mel_value / 2595.0f) - 1.0f);
}

/**
 * @brief Initialize periodic Hann window coefficients.
 *
 * @details Matches TensorFlow periodic Hann behavior used by tf.signal.stft.
 *          This function runs only once and caches coefficients.
 */
static void init_hann_window(void)
{
    int32_t sample_index;

    if (0 != g_hann_initialized)
    {
        return;
    }

    for (sample_index = 0; sample_index < MFCC_FRAME_LEN; sample_index++)
    {
        g_hann_window[sample_index] =
            0.5f * (1.0f - cosf(2.0f * (float)M_PI * (float)sample_index / (float)MFCC_FRAME_LEN));
    }

    g_hann_initialized = 1;
}

/**
 * @brief Initialize Mel filterbank weights for MFCC extraction.
 *
 * @details Builds triangular Mel filters equivalent to
 *          tf.signal.linear_to_mel_weight_matrix with:
 *          - 40 mel bins
 *          - lower edge 20 Hz
 *          - upper edge 4000 Hz
 *          - FFT_HALF_SIZE spectrogram bins
 *
 *          This function runs only once and caches weights.
 */
static void init_mel_filterbank(void)
{
    int32_t edge_point_index;
    int32_t mel_bin_index;
    int32_t fft_bin_index;
    float mel_low;
    float mel_high;
    float mel_points[MFCC_NUM_MEL_BINS + 2];
    float hz_points[MFCC_NUM_MEL_BINS + 2];
    int32_t fft_bin_points[MFCC_NUM_MEL_BINS + 2];
    int32_t left_bin;
    int32_t center_bin;
    int32_t right_bin;

    if (0 != g_mel_initialized)
    {
        return;
    }

    mel_low = hz_to_mel(MEL_LOW_FREQ);
    mel_high = hz_to_mel(MEL_HIGH_FREQ);

    for (edge_point_index = 0; edge_point_index < (MFCC_NUM_MEL_BINS + 2); edge_point_index++)
    {
        mel_points[edge_point_index] = mel_low +
            (mel_high - mel_low) * (float)edge_point_index / (float)(MFCC_NUM_MEL_BINS + 1);
        hz_points[edge_point_index] = mel_to_hz(mel_points[edge_point_index]);
        fft_bin_points[edge_point_index] =
            (int32_t)floorf((float)(MFCC_FFT_SIZE + 1) * hz_points[edge_point_index] / (float)MFCC_SAMPLE_RATE);
    }

    (void)memset(g_mel_filterbank, 0, sizeof(g_mel_filterbank));

    for (mel_bin_index = 0; mel_bin_index < MFCC_NUM_MEL_BINS; mel_bin_index++)
    {
        left_bin = fft_bin_points[mel_bin_index];
        center_bin = fft_bin_points[mel_bin_index + 1];
        right_bin = fft_bin_points[mel_bin_index + 2];

        for (fft_bin_index = left_bin;
             (fft_bin_index < center_bin) && (fft_bin_index < FFT_HALF_SIZE);
             fft_bin_index++)
        {
            if (center_bin != left_bin)
            {
                g_mel_filterbank[mel_bin_index][fft_bin_index] =
                    (float)(fft_bin_index - left_bin) / (float)(center_bin - left_bin);
            }
        }

        for (fft_bin_index = center_bin;
             (fft_bin_index <= right_bin) && (fft_bin_index < FFT_HALF_SIZE);
             fft_bin_index++)
        {
            if (right_bin != center_bin)
            {
                g_mel_filterbank[mel_bin_index][fft_bin_index] =
                    (float)(right_bin - fft_bin_index) / (float)(right_bin - center_bin);
            }
        }
    }

    g_mel_initialized = 1;
}

/**
 * @brief Initialize DCT-II matrix for MFCC coefficient projection.
 *
 * @details Builds orthonormal DCT-II matrix equivalent to
 *          tf.signal.mfccs_from_log_mel_spectrograms and applies first-row
 *          orthonormal scaling (1/sqrt(2)).
 *
 *          This function runs only once and caches coefficients.
 */
static void init_dct_matrix(void)
{
    int32_t coefficient_index;
    int32_t mel_bin_index;
    float normalization_factor;
    float first_row_scale;

    if (0 != g_dct_initialized)
    {
        return;
    }

    normalization_factor = sqrtf(2.0f / (float)MFCC_NUM_MEL_BINS);

    for (coefficient_index = 0; coefficient_index < MFCC_NUM_COEFFS; coefficient_index++)
    {
        for (mel_bin_index = 0; mel_bin_index < MFCC_NUM_MEL_BINS; mel_bin_index++)
        {
            g_dct_matrix[coefficient_index][mel_bin_index] =
                normalization_factor *
                cosf((float)M_PI * (float)coefficient_index *
                     ((float)mel_bin_index + 0.5f) / (float)MFCC_NUM_MEL_BINS);
        }
    }

    first_row_scale = 1.0f / sqrtf(2.0f);
    for (mel_bin_index = 0; mel_bin_index < MFCC_NUM_MEL_BINS; mel_bin_index++)
    {
        g_dct_matrix[0][mel_bin_index] *= first_row_scale;
    }

    g_dct_initialized = 1;
}

/**
 * @brief Initialize CMSIS-DSP real FFT instance.
 *
 * @details Initializes arm_rfft_fast_instance_f32 for 512-point FFT,
 *          once per boot/session.
 */
static void init_fft(void)
{
    if (0 != g_fft_initialized)
    {
        return;
    }

    (void)arm_rfft_fast_init_f32(&g_fft_instance, MFCC_FFT_SIZE);
    g_fft_initialized = 1;
}

/* ========== Main MFCC Extraction ========== */

/**
 * @brief Convert PCM audio to quantized MFCC features.
 */
void preprocess(const int16_t *p_pcm_input, int32_t sample_count, int8_t *p_mfcc_output)
{
    int32_t valid_sample_count;
    float normalization_denominator;
    int32_t sample_index;
    int32_t frame_index;

    /* ------------------------------------------------------------------
     * Block 1: One-time initialization of lookup tables and FFT context.
     * ------------------------------------------------------------------ */
    init_hann_window();
    init_mel_filterbank();
    init_dct_matrix();
    init_fft();

    /* ------------------------------------------------------------------
     * Block 2: Sanitize and bound input sample count.
     * ------------------------------------------------------------------ */
    valid_sample_count =
        (sample_count < MFCC_AUDIO_LENGTH) ? sample_count : MFCC_AUDIO_LENGTH;
    if (valid_sample_count < 0)
    {
        valid_sample_count = 0;
    }

    /* ------------------------------------------------------------------
     * Block 3: Compute max(abs(audio)) for TF-equivalent normalization.
     * ------------------------------------------------------------------ */
    normalization_denominator = 0.0f;
    for (sample_index = 0; sample_index < valid_sample_count; sample_index++)
    {
        float absolute_sample_value = (float)p_pcm_input[sample_index];
        if (absolute_sample_value < 0.0f)
        {
            absolute_sample_value = -absolute_sample_value;
        }
        if (absolute_sample_value > normalization_denominator)
        {
            normalization_denominator = absolute_sample_value;
        }
    }
    if (normalization_denominator < 1.0f)
    {
        normalization_denominator = 1.0f;
    }

    /* ------------------------------------------------------------------
     * Block 4: Process each analysis frame independently.
     * ------------------------------------------------------------------ */
    for (frame_index = 0; frame_index < MFCC_NUM_FRAMES; frame_index++)
    {
        int32_t frame_start_index = frame_index * MFCC_FRAME_STEP;
        int32_t window_sample_index;
        int32_t spectrum_bin_index;
        int32_t mel_bin_index;
        int32_t coefficient_index;
        float mfcc_float[MFCC_NUM_COEFFS];

        /* Step 4.1: Build windowed time-domain frame (with zero-padding). */
        (void)memset(g_frame_buffer, 0, sizeof(g_frame_buffer));

        for (window_sample_index = 0; window_sample_index < MFCC_FRAME_LEN; window_sample_index++)
        {
            float normalized_sample = get_normalized_audio_sample(p_pcm_input,
                                                                   frame_start_index + window_sample_index,
                                                                   valid_sample_count,
                                                                   normalization_denominator);
            g_frame_buffer[window_sample_index] = normalized_sample * g_hann_window[window_sample_index];
        }

        /* Step 4.2: Run real FFT (TF STFT-equivalent frequency transform). */
        arm_rfft_fast_f32(&g_fft_instance, g_frame_buffer, g_fft_output, 0);

        /* Step 4.3: Convert complex FFT bins to magnitude spectrum. */
        g_magnitude_spectrum[0] = fabsf(g_fft_output[0]);
        g_magnitude_spectrum[FFT_HALF_SIZE - 1] = fabsf(g_fft_output[1]);

        for (spectrum_bin_index = 1; spectrum_bin_index < (FFT_HALF_SIZE - 1); spectrum_bin_index++)
        {
            float real_part = g_fft_output[2 * spectrum_bin_index];
            float imag_part = g_fft_output[2 * spectrum_bin_index + 1];
            g_magnitude_spectrum[spectrum_bin_index] =
                sqrtf(real_part * real_part + imag_part * imag_part);
        }

        /* Step 4.4: Apply mel filterbank projection and log compression. */
        for (mel_bin_index = 0; mel_bin_index < MFCC_NUM_MEL_BINS; mel_bin_index++)
        {
            float mel_energy = 0.0f;
            for (spectrum_bin_index = 0; spectrum_bin_index < FFT_HALF_SIZE; spectrum_bin_index++)
            {
                mel_energy += g_mel_filterbank[mel_bin_index][spectrum_bin_index] *
                              g_magnitude_spectrum[spectrum_bin_index];
            }
            g_mel_energies[mel_bin_index] = logf(mel_energy + LOG_OFFSET);
        }

        /* Step 4.5: Apply DCT-II projection and keep first 10 coefficients. */
        for (coefficient_index = 0; coefficient_index < MFCC_NUM_COEFFS; coefficient_index++)
        {
            int32_t dct_column_index;
            float dct_sum = 0.0f;
            for (dct_column_index = 0; dct_column_index < MFCC_NUM_MEL_BINS; dct_column_index++)
            {
                dct_sum += g_dct_matrix[coefficient_index][dct_column_index] *
                           g_mel_energies[dct_column_index];
            }
            mfcc_float[coefficient_index] = dct_sum;
        }

        /* Step 4.6: Quantize MFCC floats to INT8 model input domain. */
        for (coefficient_index = 0; coefficient_index < MFCC_NUM_COEFFS; coefficient_index++)
        {
            float quantized_value =
                roundf(mfcc_float[coefficient_index] / MFCC_QUANT_SCALE) +
                (float)MFCC_QUANT_ZERO_POINT;

            if (quantized_value < -128.0f)
            {
                quantized_value = -128.0f;
            }
            if (quantized_value > 127.0f)
            {
                quantized_value = 127.0f;
            }

            p_mfcc_output[frame_index * MFCC_NUM_COEFFS + coefficient_index] =
                (int8_t)quantized_value;
        }
    }
}
