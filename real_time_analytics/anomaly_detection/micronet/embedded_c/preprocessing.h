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
 * @file    preprocessing.h
 * @brief   MicroNet audio preprocessing (int16 PCM -> int8 model patches).
 *
 * DSP pipeline (static buffers only):
 *
 *   int16 PCM  ->  x / 32768.0f       (float32 mono, [-1, 1])
 *              ->  STFT (n_fft=1024, hop=512, Hann periodic, center=False)
 *              ->  Mel power spectrogram (64 bins, Slaney norm, fmin=0, fmax=8000)
 *              ->  10 * log10(max(spec, tiny)) - PREP_TRAINING_MEAN
 *              ->  Sliding windows (64 mel frames, stride 20)
 *              ->  2x2 mean-pool -> 32x32 float patch
 *              ->  Quantize -> int8 patch for the TFLite model
 */

#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include <stdint.h>

#include "model_metadata.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief  Convert one clip of int16 PCM audio into a batch of int8 model patches.
 *
 * Produces at most PREP_MAX_WINDOWS patches, each PATCH_MODEL_DIM x PATCH_MODEL_DIM
 * int8 samples, laid out contiguously as [window][row][col].
 *
 * @param[in]  p_pcm              Mono int16 PCM samples.
 * @param[in]  _num_samples       Number of int16 samples in the clip.
 * @param[in]  _sample_rate       Audio sample rate in Hz (typically 16000).
 * @param[out] p_patches_int8     Caller-provided buffer of at least
 *                                PREP_MAX_WINDOWS * INPUT_DIM int8 entries.
 * @param[in]  _patches_capacity  Capacity of p_patches_int8 in int8 elements.
 * @param[out] p_num_windows      Number of windows actually produced.
 *
 * @return  0 on success, -1 on any error (null pointer, buffer too small,
 *          clip too short, or clip exceeds PREP_MAX_PCM_SAMPLES).
 */
int32_t preprocess(const int16_t *p_pcm,
                               int32_t            _num_samples,
                               int32_t            _sample_rate,
                               int8_t        *p_patches_int8,
                               int32_t            _patches_capacity,
                               int32_t           *p_num_windows);

#ifdef __cplusplus
}
#endif

#endif /* PREPROCESSING_H */
