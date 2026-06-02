/**
 * @file    preprocessing.c
 * @brief   Log-mel spectrogram feature extraction from raw int16 PCM audio.
 *
 * Implements the complete DSP pipeline for audio feature extraction with static buffer
 * allocation (no heap use) suitable for bare-metal embedded systems:
 *   - STFT computation (Hann window, FFT-based)
 *   - Mel-scale filterbank application (Slaney normalization)
 *   - Log-dB conversion
 *   - Central region cropping
 *   - Sliding-window vectorization
 *
 * All intermediate buffers are statically allocated to avoid heap fragmentation
 * on resource-constrained embedded platforms like RA8P1.
 *
 * @author  Renesas Electronics
 * @date    2025
 * @version 2.0  Revised for MISRA C 2025 (ISO/IEC 9899:2018) compliance:
 *               - Replaced magic literal 32768.0f with named constant PCM_NORM_FACTOR
 *               - Corrected controlling boolean expression (Rule 14.4)
 *               - Corrected composite-expression casts before size computation
 *                 (Rules 10.7, 10.8)
 */

#include "preprocessing.h"
#include "../common/model_metadata.h"

#include <stdbool.h>
#include <string.h>
#include <math.h>

/* Hann window coefficients */
static float g_hann[PREP_N_FFT];
static bool  g_hann_ready = false;

/* FFT working buffers */
static float g_fft_re[PREP_N_FFT];
static float g_fft_im[PREP_N_FFT];

/* Power spectrum for one frame */
static float g_pwr[LOCAL_PREP_N_BINS];

/*
 * Mel centre-frequency table [PREP_N_MELS + 2] in Hz.
 * Replaces the full filterbank weight matrix (was PREP_N_MELS × LOCAL_PREP_N_BINS floats ~256 KB).
 * Triangular weights are derived on-the-fly during each frame's mel projection.
 */
static float g_mel_freqs[PREP_N_MELS + 2];
static bool  g_mel_freqs_ready = false;
static int   g_mel_freqs_sr    = 0;

/*
 * Cropped spectrogram [PREP_N_MELS × PREP_CROP_COLS].
 * Written directly from the STFT loop — g_mel_spec is no longer needed (~180 KB removed).
 */
static float g_cropped[PREP_N_MELS * PREP_CROP_COLS];

/**
 * @brief   Compute Cooley-Tukey radix-2 FFT in-place on complex signal.
 *
 * Performs fast Fourier transform on complex input using the Cooley-Tukey
 * radix-2 DIT algorithm. Real and imaginary parts are stored in separate
 * arrays of size n (must be a power of 2).
 *
 * @param[in,out] p_real_part   Array of real coefficients (length n, modified in-place)
 * @param[in,out] p_imag_part   Array of imaginary coefficients (length n, modified in-place)
 * @param[in]     fft_size    FFT size, must be power of 2 (e.g., 1024)
 *
 * @return        None
 */
static void compute_radix2_fft_inplace(float *p_real_part, float *p_imag_part, int fft_size)
{
    int bit_reverse_index = 0;
    int element_index;

    /* Bit-reversal permutation */
    for (element_index = 1; element_index < fft_size; element_index++)
    {
        int bit_mask = fft_size >> 1;
        while ((bit_reverse_index & bit_mask) != 0)
        {
            bit_reverse_index ^= bit_mask;
            bit_mask >>= 1;
        }
        bit_reverse_index ^= bit_mask;
        if (element_index < bit_reverse_index)
        {
            float temp_real;
            float temp_imag;

            temp_real = p_real_part[element_index];
            p_real_part[element_index] = p_real_part[bit_reverse_index];
            p_real_part[bit_reverse_index] = temp_real;

            temp_imag = p_imag_part[element_index];
            p_imag_part[element_index] = p_imag_part[bit_reverse_index];
            p_imag_part[bit_reverse_index] = temp_imag;
        }
    }

    /* Butterflies */
    {
        int stage_length;
        for (stage_length = 2; stage_length <= fft_size; stage_length <<= 1)
        {
            float angle_step = (-2.0f * (float)M_PI) / (float)stage_length;
            float twiddle_real_step = cosf(angle_step);
            float twiddle_imag_step = sinf(angle_step);
            for (element_index = 0; element_index < fft_size; element_index += stage_length)
            {
                float twiddle_real = 1.0f;
                float twiddle_imag = 0.0f;
                int butterfly_index;
                for (butterfly_index = 0; butterfly_index < stage_length / 2; butterfly_index++)
                {
                    float upper_real = p_real_part[element_index + butterfly_index];
                    float upper_imag = p_imag_part[element_index + butterfly_index];
                    float lower_real = p_real_part[element_index + butterfly_index + stage_length / 2];
                    float lower_imag = p_imag_part[element_index + butterfly_index + stage_length / 2];
                    float twiddle_prod_real = twiddle_real * lower_real - twiddle_imag * lower_imag;
                    float twiddle_prod_imag = twiddle_real * lower_imag + twiddle_imag * lower_real;
                    p_real_part[element_index + butterfly_index]           = upper_real + twiddle_prod_real;
                    p_imag_part[element_index + butterfly_index]           = upper_imag + twiddle_prod_imag;
                    p_real_part[element_index + butterfly_index + stage_length / 2] = upper_real - twiddle_prod_real;
                    p_imag_part[element_index + butterfly_index + stage_length / 2] = upper_imag - twiddle_prod_imag;
                    {
                        float twiddle_real_next = twiddle_real * twiddle_real_step - twiddle_imag * twiddle_imag_step;
                        twiddle_imag           = twiddle_real * twiddle_imag_step + twiddle_imag * twiddle_real_step;
                        twiddle_real           = twiddle_real_next;
                    }
                }
            }
        }
    }
}

/**
 * @brief   Initialise Hann window coefficients and cache them for reuse.
 *
 * Generates periodic Hann window coefficients matching librosa's default
 * behaviour. Results are cached so computation occurs only on the first call.
 *
 * @return   None
 */
static void init_hann_window(void)
{
    int window_index;

    if (g_hann_ready)
    {
        return;
    }

    for (window_index = 0; window_index < PREP_N_FFT; window_index++)
    {
        g_hann[window_index] = 0.5f - 0.5f * cosf((2.0f * (float)M_PI * (float)window_index) / (float)PREP_N_FFT);
    }

    g_hann_ready = true;
}

/**
 * @brief   Convert frequency in Hz to mel scale using the Slaney definition.
 *
 * @param[in] frequency_hz   Frequency value in Hz
 *
 * @return    Frequency converted to mel scale
 */
static float convert_hz_to_mel_slaney(float frequency_hz)
{
    if (frequency_hz < MEL_MIN_LOG_HZ)
    {
        return (frequency_hz - MEL_F_MIN) / MEL_F_SP;
    }

    return MEL_MIN_LOG_MEL + logf(frequency_hz / MEL_MIN_LOG_HZ) / MEL_LOGSTEP;
}

/**
 * @brief   Convert mel scale to frequency in Hz using the Slaney definition.
 *
 * @param[in] mel_value   Frequency value in mel scale
 *
 * @return    Frequency converted to Hz
 */
static float convert_mel_to_hz_slaney(float mel_value)
{
    if (mel_value < MEL_MIN_LOG_MEL)
    {
        return MEL_F_MIN + mel_value * MEL_F_SP;
    }

    return MEL_MIN_LOG_HZ * expf(MEL_LOGSTEP * (mel_value - MEL_MIN_LOG_MEL));
}

/**
 * @brief   Build and cache the mel centre-frequency table (PREP_N_MELS + 2 values).
 *
 * Stores only the triangular-filter centre frequencies in Hz — the actual
 * per-bin weights are derived on-the-fly during mel projection, eliminating
 * the PREP_N_MELS × LOCAL_PREP_N_BINS weight matrix (~256 KB).
 * Rebuilt only when the sample rate changes.
 *
 * @param[in] sample_rate   Audio sample rate in Hz
 *
 * @return    None
 */
static void init_mel_filterbank(int sample_rate)
{
    int point_index;
    int num_points = PREP_N_MELS + 2;
    float frequency_max;
    float mel_minimum;
    float mel_maximum;

    /* Guard first — avoids any computation on cache hits */
    if ((g_mel_freqs_ready) && (g_mel_freqs_sr == sample_rate))
    {
        return;
    }

    frequency_max = (float) sample_rate / 2.0f;
    mel_minimum   = convert_hz_to_mel_slaney(0.0f);
    mel_maximum   = convert_hz_to_mel_slaney(frequency_max);

    for (point_index = 0; point_index < num_points; point_index++)
    {
        float mel_interp = mel_minimum + ((float) point_index / (float) (num_points - 1)) * (mel_maximum - mel_minimum);

        g_mel_freqs[point_index] = convert_mel_to_hz_slaney(mel_interp);
    }

    g_mel_freqs_sr    = sample_rate;
    g_mel_freqs_ready = true;
}

/**
 * @brief   Extract sliding-window log-mel spectrogram feature vectors from
 *          raw int16 PCM audio.
 *
 * This is the preprocessing interface function called from application code.
 * Implements the complete preprocessing pipeline without heap allocation:
 *   1. STFT computation using Hann window and Cooley-Tukey radix-2 FFT
 *   2. Mel-scale filterbank application with power spectrum
 *   3. Log-dB conversion for dynamic range compression
 *   4. Central region cropping [50, 250)
 *   5. Sliding-window vectorization (5-frame concatenation)
 *
 * @param[in]  p_pcm              Mono int16_t PCM samples (signed 16-bit, -32768 to 32767)
 * @param[in]  num_samples      Total number of PCM samples provided
 * @param[in]  sample_rate      Audio sample rate in Hz (typically 16000)
 * @param[out] p_output_vectors   Pre-allocated float buffer for output vectors
 *                              Expected size: [NUM_VECTORS_PER_CLIP][INPUT_DIM]
 * @param[out] p_output_count     Pointer to store the actual number of vectors generated
 *
 * @return     0 on successful completion, -1 on error (audio too short or too long)
 *
 * @note       All intermediate buffers are statically allocated and sized using
 *             preprocessing dimension macros and local derived constants.
 */
int preprocess(const int16_t *p_pcm,
               int            num_samples,
               int            sample_rate,
               float         *p_output_vectors,
               int           *p_output_count)
{
    int zero_padding;
    int num_stft_frames;
    int num_frequency_bins;
    int frame_index;
    int mel_bin_index;
    int frequency_bin_index;
    int num_output_vectors;
    int feature_dimension;

    *p_output_count = 0;

    /* Build lookup tables on first call */
    init_hann_window();
    init_mel_filterbank(sample_rate);

    /* Compute number of STFT frames */
    zero_padding = PREP_N_FFT / 2;
    num_stft_frames = 1 + (num_samples / PREP_HOP);
    num_frequency_bins = LOCAL_PREP_N_BINS;

    if (num_stft_frames > LOCAL_PREP_MAX_STFT_FRAMES)
    {
        return -1;  /* audio too long — exceeds supported STFT frame count */
    }

    if (num_stft_frames < PREP_CROP_END)
    {
        return -1;  /* audio too short */
    }

    /* ── STFT + on-the-fly Mel filterbank → g_cropped ── */
    /*
     * Only frames inside the crop window [PREP_CROP_START, PREP_CROP_END) are
     * processed. Results are written directly into g_cropped, so g_mel_spec is
     * no longer needed.
     */
    (void)memset(g_cropped, 0, sizeof(g_cropped));

    for (frame_index = PREP_CROP_START; frame_index < PREP_CROP_END; frame_index++)
    {
        int start_sample_index = frame_index * PREP_HOP - zero_padding;
        int sample_index;
        int crop_col = frame_index - PREP_CROP_START;

        /* Window and zero-pad */
        for (sample_index = 0; sample_index < PREP_N_FFT; sample_index++)
        {
            int current_sample_idx = start_sample_index + sample_index;
            float sample_value = 0.0f;

            if ((current_sample_idx >= 0) && (current_sample_idx < num_samples))
            {
                sample_value = (float)p_pcm[current_sample_idx] / PCM_NORM_FACTOR;
            }

            g_fft_re[sample_index] = sample_value * g_hann[sample_index];
            g_fft_im[sample_index] = 0.0f;
        }

        /* FFT */
        compute_radix2_fft_inplace(g_fft_re, g_fft_im, PREP_N_FFT);

        /* Power spectrum */
        for (frequency_bin_index = 0; frequency_bin_index < num_frequency_bins; frequency_bin_index++)
        {
            g_pwr[frequency_bin_index] = (g_fft_re[frequency_bin_index] * g_fft_re[frequency_bin_index]) +
                                         (g_fft_im[frequency_bin_index] * g_fft_im[frequency_bin_index]);
        }

        /*
         * On-the-fly mel filterbank application.
         *
         * Loop order: frequency bin (outer) → mel bin (inner).
         * This allows freq_f to be computed once per FFT bin and reused across
         * all 128 mel bins, eliminating 128× redundant multiplications per frame.
         *
         * mel_energy[] accumulates the weighted power for each mel bin.
         * For bin m, FFT bin f:
         *   lower = (freq_f - mel_freqs[m])   / (mel_freqs[m+1] - mel_freqs[m])
         *   upper = (mel_freqs[m+2] - freq_f) / (mel_freqs[m+2] - mel_freqs[m+1])
         *   w     = max(0, min(lower, upper)) * 2 / (mel_freqs[m+2] - mel_freqs[m])
         */
        {
            float mel_energy[PREP_N_MELS];
            (void)memset(mel_energy, 0, sizeof(mel_energy));

            for (frequency_bin_index = 0; frequency_bin_index < num_frequency_bins; frequency_bin_index++)
            {
                /* Compute FFT bin frequency once — shared by all mel bins */
                float freq = (float) frequency_bin_index * (float) sample_rate / (float) PREP_N_FFT;
                float pwr  = g_pwr[frequency_bin_index];

                for (mel_bin_index = 0; mel_bin_index < PREP_N_MELS; mel_bin_index++)
                {
                    float f_lo       = g_mel_freqs[mel_bin_index];
                    float f_mid      = g_mel_freqs[mel_bin_index + 1];
                    float f_hi       = g_mel_freqs[mel_bin_index + 2];
                    float lower_weight = (freq - f_lo)  / (f_mid - f_lo);
                    float upper_weight = (f_hi - freq)  / (f_hi  - f_mid);
                    float weight_value = (lower_weight < upper_weight) ? lower_weight : upper_weight;

                    if (weight_value > 0.0f)
                    {
                        mel_energy[mel_bin_index] += weight_value * (2.0f / (f_hi - f_lo)) * pwr;
                    }
                }
            }

            for (mel_bin_index = 0; mel_bin_index < PREP_N_MELS; mel_bin_index++)
            {
                g_cropped[mel_bin_index * PREP_CROP_COLS + crop_col] = mel_energy[mel_bin_index];
            }
        }
    }

    /* ── Log-dB conversion (applied directly to g_cropped) ── */
    {
        float factor = 20.0f / PREP_POWER;   /* = 10.0 */
        int total_elements = PREP_N_MELS * PREP_CROP_COLS;
        int element_index;

        for (element_index = 0; element_index < total_elements; element_index++)
        {
            float mel_value = g_cropped[element_index];

            if (mel_value < PREP_EPSILON)
            {
                mel_value = PREP_EPSILON;
            }

            g_cropped[element_index] = factor * log10f(mel_value);
        }
    }

    /* ── Sliding window → float32 vectors ── */
    num_output_vectors = NUM_VECTORS_PER_CLIP;
    feature_dimension = INPUT_DIM;

    for (frame_index = 0; frame_index < num_output_vectors; frame_index++)
    {
        float *p_output_vector_ptr = &p_output_vectors[frame_index * feature_dimension];
        int frame_offset;

        for (frame_offset = 0; frame_offset < PREP_FRAMES; frame_offset++)
        {
            for (mel_bin_index = 0; mel_bin_index < PREP_N_MELS; mel_bin_index++)
            {
                p_output_vector_ptr[frame_offset * PREP_N_MELS + mel_bin_index] =
                    g_cropped[mel_bin_index * PREP_CROP_COLS + (frame_index + frame_offset)];
            }
        }
    }

    *p_output_count = num_output_vectors;
    return 0;
}
