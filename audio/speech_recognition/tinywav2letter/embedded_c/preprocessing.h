/*
 * Copyright (c) 2020 - 2026 Renesas Electronics Corporation and/or its affiliates
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
/**
 * @file preprocessing.h
 * @brief TinyWav2letter preprocessing API.
 *
 * Converts int16 mono PCM (16 kHz) to the quantized model input tensor
 * [MODEL_INPUT_TIME, MODEL_INPUT_FEATURES].
 */

#ifndef PREPROCESSING_H_
#define PREPROCESSING_H_

#include <stdint.h>

#include "../common/model_metadata.h"   /* MODEL_INPUT_SIZE, MODEL_INPUT_TIME, ... */

#ifdef __cplusplus
extern "C" {
#endif

/* MFCC front-end constants. */
#define TW2L_MFCC_COEFFS          (13U)
#define TW2L_SAMPLE_RATE_HZ       (16000U)
#define TW2L_N_FFT                (512U)
#define TW2L_HOP_LENGTH           (160U)

/* Upper bound for input PCM samples accepted by preprocess(). */
#define TW2L_PCM_MAX_SAMPLES      (48000U)

typedef enum {
    PREPROCESS_OK              =  0,
    PREPROCESS_ERR_INVALID_ARG = -1,
    PREPROCESS_ERR_TOO_SHORT   = -2,
    PREPROCESS_ERR_TOO_LONG    = -3,
    PREPROCESS_ERR_BACKEND     = -4
} preprocess_status_t;

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
                               uint32_t *p_out_time_frames);

#ifdef __cplusplus
}
#endif

#endif /* PREPROCESSING_H_ */
