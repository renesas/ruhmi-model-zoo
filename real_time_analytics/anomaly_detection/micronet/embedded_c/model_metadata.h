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
 * @file    model_metadata.h
 * @brief   Compile-time metadata for MicroNet Anomaly Detection (INT8 CPU).
 *
 * Generated from:
 *   Model  : ad_medium_int8.tflite  (ARM ML-zoo MicroNet Medium anomaly detection)
 *   Task   : Anomaly Detection (DCASE 2020 Task 2 slider)
 *   Target : RA8P1 CPU (RUHMI-compiled C-codegen, int8 I/O)
 *
 * The pre/post processing values below match the trained model parameters.
 * Do NOT edit unless the reference model is retrained or the ML-zoo pinned
 * commit changes.
 *
 * Pipeline summary (one clip):
 *   1. Load int16 PCM audio (16 kHz mono).
 *   2. Log-mel spectrogram (64 mel bins, n_fft=1024, hop=512, Slaney norm,
 *      Hann periodic window, center=False, power=2.0).
 *   3. Log-dB shift : 10 * log10(max(spec, tiny)) - TRAINING_MEAN.
 *   4. Sliding window over the log-mel matrix (width 64 frames, stride 20).
 *   5. 2x2 mean-pool each 64x64 window to a 32x32 patch.
 *   6. Quantize the patch with the TFLite input scale/zero-point (int8).
 *   7. Run compute_sub_0000() to obtain 8 int8 output logits per window.
 *   8. Dequantize logits and take -logit[target_output_index] as the score.
 *   9. Clip-level score = mean of window scores. Verdict = score > threshold.
 */

#ifndef MODEL_METADATA_H
#define MODEL_METADATA_H

#ifdef __cplusplus
extern "C" {
#endif


/* ======================================================================
 * 1.  MODEL IDENTITY
 * ====================================================================== */

#define MODEL_NAME              "MicroNet Anomaly Detection (INT8)"
#define MODEL_TASK              "Anomaly Detection"
#define MODEL_DTYPE             "int8"


/* ======================================================================
 * 2.  INPUT / OUTPUT TENSOR SHAPES
 *     Input  : [1, 32, 32, 1] int8
 *     Output : [1, 8]         int8
 * ====================================================================== */

/** Spatial size of the model input after 2x2 mean pooling. */
#define PATCH_MODEL_DIM         (32)

/** Flat input tensor length (elements). */
#define INPUT_DIM               (PATCH_MODEL_DIM * PATCH_MODEL_DIM)  /* 1024 */

/** Number of output logits per window. */
#define OUTPUT_DIM              (8)


/* ======================================================================
 * 3.  DSP / PREPROCESSING PARAMETERS
 * ====================================================================== */

/** WAV resample target, Hz. */
#define PREP_SAMPLE_RATE        (16000)

/** FFT size for the STFT. */
#define PREP_N_FFT              (1024)

/** STFT hop length in samples. */
#define PREP_HOP                (512)

/** Number of mel bins produced by the filterbank. */
#define PREP_N_MELS             (64)

/** Mel filter lower / upper frequency bounds, Hz. */
#define PREP_MEL_FMIN           (0.0f)
#define PREP_MEL_FMAX           (8000.0f)

/** Training-time mean shift subtracted from the log-mel matrix. */
#define PREP_TRAINING_MEAN      (-30.0f)

/** Small positive value used as a floor before log10 (numpy float32 tiny). */
#define PREP_EPSILON            (1.175494351e-38f)

/** Sliding-window width in mel frames. */
#define PREP_PATCH_RAW_FRAMES   (64)

/** Sliding-window stride in mel frames. */
#define PREP_OUTER_STRIDE       (20)

/** Number of real-valued FFT bins retained (N_FFT/2 + 1). */
#define PREP_N_BINS             ((PREP_N_FFT / 2) + 1)   /* 513 */

/** PCM normalization factor for signed 16-bit samples. */
#define PCM_NORM_FACTOR         (32768.0f)

#ifndef M_PI
#  define M_PI (3.14159265358979323846f)
#endif

/* -------- Slaney mel scale constants (librosa htk=False, norm='slaney') -------- */
#define MEL_F_MIN               (0.0f)
#define MEL_F_SP                (200.0f / 3.0f)
#define MEL_MIN_LOG_HZ          (1000.0f)
#define MEL_MIN_LOG_MEL         ((MEL_MIN_LOG_HZ - MEL_F_MIN) / MEL_F_SP)
#define MEL_LOGSTEP             (0.06875177742094912f)   /* log(6.4)/27 */


/* ======================================================================
 * 4.  RUNTIME SIZE CEILINGS (buffers sized against these)
 * ====================================================================== */

/** Maximum supported clip length in PCM samples (10 s @ 16 kHz). */
#define PREP_MAX_PCM_SAMPLES        (160000)

/**
 * Maximum number of STFT frames buffered.
 * For 10 s / 16 kHz / hop=512, center=False:
 *   T = 1 + floor((160000 - 1024) / 512) = 311
 * A small margin is added to tolerate slightly longer clips.
 */
#define PREP_MAX_MEL_FRAMES         (320)

/**
 * Maximum number of sliding windows per clip.
 * For T=311, patch=64, stride=20: W = 1 + floor((311-64)/20) = 13.
 */
#define PREP_MAX_WINDOWS            (16)


/* ======================================================================
 * 5.  QUANTIZATION PARAMETERS  (read from ad_medium_int8.tflite)
 * ====================================================================== */

/** TFLite input tensor scale / zero-point (per-tensor). */
#define MODEL_INPUT_SCALE           (0.19243663549423218f)
#define MODEL_INPUT_ZP              (11)

/** TFLite output tensor scale / zero-point (per-tensor). */
#define MODEL_OUTPUT_SCALE          (0.04889114573597908f)
#define MODEL_OUTPUT_ZP             (-30)


/* ======================================================================
 * 6.  POST-PROCESSING PARAMETERS
 * ====================================================================== */

/**
 * Anomaly threshold on the mean -logit score (matches DEFAULT_THRESHOLD).
 * Higher score => more anomalous.
 * Verdict: ANOMALY if mean_score > ANOMALY_THRESHOLD, else normal.
 */
#define ANOMALY_THRESHOLD           (-7.2f)


/* ======================================================================
 * 7.  TEST CONFIGURATION
 * ====================================================================== */

/** Number of test audio clips baked into test_input/. */
#define NUM_TEST_CLIPS              (10U)


#ifdef __cplusplus
}
#endif

#endif /* MODEL_METADATA_H */
