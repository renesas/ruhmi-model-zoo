/**
 * @file    preprocessing.h
 * @brief   KWS MFCC preprocessing API.
 *
 * Copyright 2026 Renesas Electronics Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include <stdint.h>
#include "../common/model_metadata.h"

/* ========== Configuration (Synced with model_metadata.h) ========== */
#define MFCC_SAMPLE_RATE       SAMPLE_RATE
#define MFCC_AUDIO_LENGTH      DESIRED_SAMPLES          /* 1 second (clip_duration_ms=1000) */
#define MFCC_FRAME_LEN         WINDOW_SIZE_SAMPLES      /* 30ms window (window_size_ms=30) */
#define MFCC_FRAME_STEP        WINDOW_STRIDE_SAMPLES    /* 20ms hop  (window_stride_ms=20) */
#define MFCC_NUM_FRAMES        NUM_MFCC_FRAMES          /* (16000 - 480) / 320 + 1 = 49   */
#define MFCC_FFT_SIZE          512                      /* Next power of 2 >= 480          */
#define MFCC_NUM_MEL_BINS      NUM_MEL_BINS
#define MFCC_NUM_COEFFS        DCT_COEFFICIENT_COUNT    /* dct_coefficient_count = 10      */
#define MFCC_OUTPUT_SIZE       (MFCC_NUM_FRAMES * MFCC_NUM_COEFFS) /* 49 * 10 = 490 */

/* Quantization parameters (from model_metadata.h) */
#define MFCC_QUANT_SCALE       INPUT_SCALE
#define MFCC_QUANT_ZERO_POINT  INPUT_ZP

/* ========== API ========== */

/**
 * @brief Convert PCM audio to quantized MFCC features.
 */
void preprocess(const int16_t *p_pcm_input, int32_t sample_count, int8_t *p_mfcc_output);

#endif /* PREPROCESSING_H */
