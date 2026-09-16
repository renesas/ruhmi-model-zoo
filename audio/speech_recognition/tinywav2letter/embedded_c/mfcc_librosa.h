/*
 * Copyright (c) 2020 - 2026 Renesas Electronics Corporation and/or its affiliates
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
/**********************************************************************************************************************
 * File Name    : mfcc_librosa.h
 * Description  : Librosa-compatible MFCC and delta backend declarations.
 *********************************************************************************************************************/

#ifndef MFCC_LIBROSA_H_
#define MFCC_LIBROSA_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Extract MFCC[13] features in row-major layout [T, 13].
 *
 * @param[in]  p_audio          Input PCM audio samples as float32.
 * @param[in]  audio_samples    Number of input samples.
 * @param[out] p_out_mfcc_t13   Output MFCC buffer [max_frames * 13].
 * @param[in]  max_frames       Maximum number of output frames.
 * @param[out] p_out_frames     Actual number of frames written.
 *
 * @return 0 on success, negative value on error.
 */
int mfcc_librosa_extract(const float *p_audio,
                         uint32_t audio_samples,
                         float *p_out_mfcc_t13,
                         uint32_t max_frames,
                         uint32_t *p_out_frames);

/**
 * @brief Compute first/second temporal derivative for [T, 13] MFCC features.
 *
 * @param[in]  p_mfcc_t13   Input MFCC buffer [frames * 13].
 * @param[in]  frames       Number of input frames.
 * @param[in]  order        Derivative order: 1 (delta) or 2 (delta-delta).
 * @param[out] p_out_t13    Output buffer [frames * 13].
 *
 * @return 0 on success, negative value on error.
 */
int mfcc_librosa_delta(const float *p_mfcc_t13,
                       uint32_t frames,
                       uint32_t order,
                       float *p_out_t13);

#ifdef __cplusplus
}
#endif

#endif /* MFCC_LIBROSA_H_ */
