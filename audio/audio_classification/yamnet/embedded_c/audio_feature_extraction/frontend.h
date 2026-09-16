/*
* Copyright (c) 2020 - 2026 Renesas Electronics Corporation and/or its affiliates
*
* SPDX-License-Identifier: BSD-3-Clause
*/
/**
 * @file frontend.h
 * @brief YAMNet audio front-end API: raw int16 waveform to log-mel feature patches.
 */

#ifndef YAMNET_FRONTEND_H
#define YAMNET_FRONTEND_H

#include <stdint.h>
#include <stddef.h>
#include "../../common/model_metadata.h"

/* YAMNet frontend: raw int16 waveform -> log-mel patches (float32).
 *
 * Mirrors the Python reference in yamnet_src/features.py with:
 *   sample_rate        = 16000
 *   stft window        = 25 ms (400 samples)
 *   stft hop           = 10 ms (160 samples)
 *   fft length         = 512
 *   mel bands          = 64
 *   mel range          = 125 .. 7500 Hz
 *   log offset         = 0.001
 *   patch window       = 96 STFT frames (0.96 s)
 *   patch hop          = 48 STFT frames (0.48 s)
 *
 * Steps performed (matching Python element-for-element up to float rounding):
 *   1. Framing        : slice waveform into 400-sample frames, hop 160.
 *   2. Windowing      : elementwise multiply by periodic Hann of length 400.
 *   3. Zero-pad + FFT : center-pad frame to 512 samples, run a length-512
 *                       radix-2 real FFT.
 *   4. Magnitude      : |FFT bin| for bins 0 .. 256 (257 bins total).
 *   5. Mel filterbank : mel[m] = sum_k mag[k] * mel_matrix[k][m].
 *   6. Log            : log_mel = log(mel + 0.001).
 *   7. Patching       : slice the log_mel row stream into 96-row patches,
 *                       hop 48 rows.
 */

#define YAMNET_SAMPLE_RATE          MODEL_SAMPLE_RATE_HZ
#define YAMNET_HOP_SAMPLES          MODEL_STFT_HOP_SAMPLES
#define YAMNET_PATCH_FRAMES         MODEL_INPUT_FRAMES
#define YAMNET_PATCH_HOP_FRAMES     MODEL_PATCH_HOP_FRAMES
#define YAMNET_MEL_LOG_OFFSET       MODEL_LOG_OFFSET
#define YAMNET_PCM_INT16_NORM_SCALE (32768.0f)

/* YAMNET_WINDOW_SAMPLES, YAMNET_FFT_SIZE, YAMNET_FFT_HALF_SIZE,
 * YAMNET_NUM_MEL_BINS, YAMNET_NUM_SPECTROGRAM_BINS come from the auto-generated
 * headers below. Each LUT header exposes only extern declarations; the
 * corresponding storage lives in hann_window.c, fft_twiddles.c and
 * mel_filterbank.c so no storage is declared in any header file. */
#include "hann_window.h"
#include "fft_twiddles.h"
#include "mel_filterbank.h"

/* Elements per patch (96 frames * 64 mel bins = 6144). */
#define YAMNET_PATCH_ELEMENTS       (YAMNET_PATCH_FRAMES * YAMNET_NUM_MEL_BINS)

/**
 * @brief Calculate the number of complete STFT frames available from an input waveform.
 * @param[in] num_samples Number of input PCM samples.
 * @return Number of complete STFT frames that can be produced with `pad_end=False`.
 */
size_t yamnet_frontend_num_stft_frames(size_t num_samples);

/**
 * @brief Calculate the number of complete 96-row patches available from STFT rows.
 * @param[in] num_stft_frames Number of STFT frame rows available.
 * @return Number of complete 96-row patches that can be produced with `pad_end=False`.
 */
size_t yamnet_frontend_num_patches(size_t num_stft_frames);

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
                                       size_t         max_rows);

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
                                  float       *p_out_patch);

#endif /* YAMNET_FRONTEND_H */
