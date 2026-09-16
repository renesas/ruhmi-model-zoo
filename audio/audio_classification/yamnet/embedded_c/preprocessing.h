/*
* Copyright (c) 2020 - 2026 Renesas Electronics Corporation and/or its affiliates
*
* SPDX-License-Identifier: BSD-3-Clause
*/
/**
 * @file preprocessing.h
 * @brief YAMNet pre-processing APIs for feature extraction and model input quantization.
 */

#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include <stdint.h>
#include <stddef.h>

/* Full YAMNet preprocessing pipeline: raw waveform -> quantized patch batch.
 *
 * The main inference driver calls this single entry point; every step
 * required to turn a PCM clip into model-ready int16 inputs happens inside.
 * The steps mirror the Python reference (YamnetTFLite.score_patches, int16
 * path) element-for-element up to float rounding:
 *
 *   1. Framing + periodic Hann window (25 ms window, 10 ms hop)
 *   2. Length-512 FFT + magnitude spectrogram
 *   3. Mel filterbank (64 bands, 125 .. 7500 Hz) + log(mel + 0.001)
 *   4. Slice into 96-row patches with hop 48
 *   5. Model-boundary quantize: q = round(x / scale + zp), clipped to int16
 *
 * Parameters:
 *   p_waveform          : int16 PCM samples at YAMNET_SAMPLE_RATE.
 *   num_samples         : length of p_waveform.
 *   p_log_mel_scratch   : caller-owned scratch for the log-mel row stream.
 *                         Size >= max_rows * YAMNET_NUM_MEL_BINS floats.
 *   max_rows            : capacity of p_log_mel_scratch in rows.
 *   p_quantized_patches : caller-owned output buffer laid out as
 *                         p_quantized_patches[patch_idx * YAMNET_PATCH_ELEMENTS + i].
 *                         Size >= _max_patches * YAMNET_PATCH_ELEMENTS int16.
 *   _max_patches         : capacity of p_quantized_patches in patches.
 *
 * Returns: number of quantized patches actually produced (0 if the clip is
 * too short to form a full patch or if buffer capacities are too small).
 */

/**
 * @brief Run the full YAMNet pre-processing pipeline on an input waveform.
 * @param[in] p_waveform Pointer to int16 PCM input samples.
 * @param[in] _num_samples Length of `p_waveform` in samples.
 * @param[out] p_log_mel_scratch Caller-owned scratch buffer for log-mel rows.
 * @param[in] _max_rows Capacity of `p_log_mel_scratch` expressed in rows.
 * @param[out] p_quantized_patches Caller-owned output buffer for quantized int16 patches.
 * @param[in] _max_patches Capacity of `p_quantized_patches` expressed in patches.
 * @return Number of quantized patches produced, or `0` on invalid input or a short clip.
 */
size_t preprocess(const int16_t *p_waveform,
                  size_t         _num_samples,
                  float         *p_log_mel_scratch,
                  size_t         _max_rows,
                  int16_t       *p_quantized_patches,
                  size_t         _max_patches);

#endif /* PREPROCESSING_H */
