/*
 * This file is developed by EdgeCortix Inc. to be used with certain Renesas Electronics Hardware only.
 *
 * Copyright (C) 2025 EdgeCortix Inc. Licensed to Renesas Electronics Corporation with the
 * right to sublicense under the Apache License, Version 2.0.
 *
 * This file also includes source code originally developed by the Renesas Electronics Corporation.
 * The Renesas disclaimer below applies to any Renesas-originated portions for usage of the code.
 *
 * The Renesas Electronics Corporation
 * DISCLAIMER
 * This software is supplied by Renesas Electronics Corporation and is only intended for use with Renesas products. No
 * other uses are authorized. This software is owned by Renesas Electronics Corporation and is protected under all
 * applicable laws, including copyright laws.
 * THIS SOFTWARE IS PROVIDED 'AS IS' AND RENESAS MAKES NO WARRANTIES REGARDING
 * THIS SOFTWARE, WHETHER EXPRESS, IMPLIED OR STATUTORY, INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. ALL SUCH WARRANTIES ARE EXPRESSLY DISCLAIMED. TO THE MAXIMUM
 * EXTENT PERMITTED NOT PROHIBITED BY LAW, NEITHER RENESAS ELECTRONICS CORPORATION NOR ANY OF ITS AFFILIATED COMPANIES
 * SHALL BE LIABLE FOR ANY DIRECT, INDIRECT, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES FOR ANY REASON RELATED TO THIS
 * SOFTWARE, EVEN IF RENESAS OR ITS AFFILIATES HAVE BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
 * Renesas reserves the right, without notice, to make changes to this software and to discontinue the availability of
 * this software. By using this software, you agree to the additional terms and conditions found by accessing the
 * following link:
 * http://www.renesas.com/disclaimer
 */

/**
 * @file    model_metadata.h
 * @brief   Compile-time metadata for the Arm ML-zoo RNNoise INT8 model
 *          deployed on the RA8P1 CPU target.
 *
 * Source model : rnnoise_INT8.tflite (Arm ML-zoo, derived from
 *                xiph/rnnoise commit 7f449bf, INT8-quantised, 113 472 B)
 * Task         : Single-channel speech denoising at 48 kHz / 10 ms frames
 * Architecture : 3-GRU stack (24 / 48 / 96 hidden units) + 2 dense heads
 *                producing per-frame band-gain mask + VAD probability.
 *
 * The model is invoked once per 10 ms audio frame and threads three hidden
 * states across frames. Per-frame data flow:
 *
 *   480 int16 PCM samples
 *      |
 *      |  preprocessing.c
 *      v
 *   42 features  +  prev VAD/NOISE/DENOISE GRU states  (4 inputs total)
 *      |
 *      |  compute_sub_0000 (...)            <- MERA CMSIS-NN code-gen
 *      v
 *   22 band gains  +  VAD prob  +  next VAD/NOISE/DENOISE states  (5 outputs)
 *      |
 *      |  postprocessing.c
 *      v
 *   480 int16 PCM samples (denoised)
 *
 * Boundary tensors on the MERA-generated C entry point
 * (alphabetical input order — DO NOT REORDER, MERA fixes positional binding):
 *
 *   Inputs:
 *     denoise_gru_prev_state_int8 [96]   scale = 0.0078431359   zp =   -1
 *     main_input_int8             [42]   scale = 0.2215006053   zp =  +14
 *     noise_gru_prev_state_int8   [48]   scale = 0.0479424894   zp = -128
 *     vad_gru_prev_state_int8     [24]   scale = 0.0078431359   zp =   -1
 *
 *   Outputs (in declared order in compute_sub_0000()):
 *     Identity_3_int8 [24]   VAD GRU next state         scale = 0.0078431377  zp =  -1
 *     Identity_2_int8 [48]   NOISE GRU next state       scale = 0.0479424857  zp = -128
 *     Identity_int8   [96]   DENOISE GRU next state     scale = 0.0078431377  zp =  -1
 *     Identity_1_int8 [22]   22 band gains in [0, 1]    scale = 0.0039062500  zp = -128
 *     Identity_4_int8 [1]    VAD probability in [0, 1]  scale = 0.0039062500  zp = -128
 */

#ifndef MODEL_METADATA_H
#define MODEL_METADATA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ======================================================================
 * 1.  MODEL IDENTITY
 * ====================================================================== */
#define MODEL_NAME                  "RNNoise-Arm-INT8"
#define MODEL_FILE_INT8             "rnnoise_INT8.tflite"
#define MODEL_TASK                  "Speech Denoising (single channel)"
#define MODEL_DATASET               "Edinburgh Noisy Speech DB (validation only)"
#define MODEL_QUANTIZATION          "Symmetric per-tensor INT8 (Arm ML-zoo)"

/* ======================================================================
 * 2.  AUDIO & FRAME GEOMETRY
 * ====================================================================== */
#define MODEL_SAMPLE_RATE_HZ        48000     /* RNNoise is hard-coded 48 kHz */
#define MODEL_FRAME_SIZE            480       /* samples = 10 ms              */
#define MODEL_FRAME_DURATION_MS     10
#define MODEL_WINDOW_SIZE           960       /* 2 * FRAME_SIZE for overlap   */
#define MODEL_FREQ_SIZE             481       /* FRAME_SIZE + 1 (FFT bins)    */

#define MODEL_NB_BANDS              22        /* Bark-scale critical bands    */
#define MODEL_NB_DELTA_CEPS         6         /* delta-cepstra used per frame */
#define MODEL_CEPS_MEM              8         /* cepstral memory ring length  */

/* 42 = 22 BFCC + 6 dBFCC + 6 ddBFCC + 6 pitch-correlation BFCC
 *      + 1 pitch period + 1 pitch gain */
#define MODEL_NB_FEATURES           42

/* Pitch analyser working set */
#define MODEL_PITCH_MIN_PERIOD      60
#define MODEL_PITCH_MAX_PERIOD      768
#define MODEL_PITCH_FRAME_SIZE      960
#define MODEL_PITCH_BUF_SIZE        (MODEL_PITCH_MAX_PERIOD + MODEL_PITCH_FRAME_SIZE)
                                          /* = 1 728 float32 samples         */

/* ======================================================================
 * 3.  MODEL INPUT TENSORS  (4 inputs — alphabetical order matters)
 * ====================================================================== */
/* 3.1  denoise_gru_prev_state_int8  shape=[1, 96]  int8 */
#define MODEL_INPUT_DENOISE_GRU_SIZE      96
#define MODEL_INPUT_DENOISE_GRU_SCALE     0.0078431359f
#define MODEL_INPUT_DENOISE_GRU_ZP        (-1)

/* 3.2  main_input_int8              shape=[1, 1, 42]  int8 */
#define MODEL_INPUT_FEATURES_SIZE         MODEL_NB_FEATURES   /* 42 */
#define MODEL_INPUT_FEATURES_SCALE        0.2215006053f
#define MODEL_INPUT_FEATURES_ZP           (+14)

/* 3.3  noise_gru_prev_state_int8    shape=[1, 48]  int8 */
#define MODEL_INPUT_NOISE_GRU_SIZE        48
#define MODEL_INPUT_NOISE_GRU_SCALE       0.0479424894f
#define MODEL_INPUT_NOISE_GRU_ZP          (-128)

/* 3.4  vad_gru_prev_state_int8      shape=[1, 24]  int8 */
#define MODEL_INPUT_VAD_GRU_SIZE          24
#define MODEL_INPUT_VAD_GRU_SCALE         0.0078431359f
#define MODEL_INPUT_VAD_GRU_ZP            (-1)

/* ======================================================================
 * 4.  MODEL OUTPUT TENSORS  (5 outputs, in declared C order)
 * ====================================================================== */
/* 4.1  Identity_3_int8  shape=[1, 1, 24]  int8  - VAD GRU next state */
#define MODEL_OUTPUT_VAD_GRU_SIZE         MODEL_INPUT_VAD_GRU_SIZE      /* 24 */
#define MODEL_OUTPUT_VAD_GRU_SCALE        0.0078431377f
#define MODEL_OUTPUT_VAD_GRU_ZP           (-1)

/* 4.2  Identity_2_int8  shape=[1, 1, 48]  int8  - NOISE GRU next state */
#define MODEL_OUTPUT_NOISE_GRU_SIZE       MODEL_INPUT_NOISE_GRU_SIZE    /* 48 */
#define MODEL_OUTPUT_NOISE_GRU_SCALE      0.0479424857f
#define MODEL_OUTPUT_NOISE_GRU_ZP         (-128)

/* 4.3  Identity_int8    shape=[1, 1, 96]  int8  - DENOISE GRU next state */
#define MODEL_OUTPUT_DENOISE_GRU_SIZE     MODEL_INPUT_DENOISE_GRU_SIZE  /* 96 */
#define MODEL_OUTPUT_DENOISE_GRU_SCALE    0.0078431377f
#define MODEL_OUTPUT_DENOISE_GRU_ZP       (-1)

/* 4.4  Identity_1_int8  shape=[1, 1, 22]  int8  - 22 band gains in [0, 1] */
#define MODEL_OUTPUT_GAINS_SIZE           MODEL_NB_BANDS                /* 22 */
#define MODEL_OUTPUT_GAINS_SCALE          0.0039062500f
#define MODEL_OUTPUT_GAINS_ZP             (-128)

/* 4.5  Identity_4_int8  shape=[1, 1, 1]  int8  - VAD probability in [0, 1] */
#define MODEL_OUTPUT_VAD_PROB_SIZE        1
#define MODEL_OUTPUT_VAD_PROB_SCALE       0.0039062500f
#define MODEL_OUTPUT_VAD_PROB_ZP          (-128)

/* ======================================================================
 * 5.  STATEFUL EXECUTION
 * ======================================================================
 * The 3 GRU states are read at the start of every frame and written back
 * at the end. They must be **zero-initialised** at boot and never reset
 * between frames in a continuous denoising session.
 *
 * Storage requirement (3 buffers of int8):
 *     24 + 48 + 96 = 168 bytes  (per channel)
 *
 * preprocessing.c also owns persistent DSP state across frames:
 *     analysis_mem    : float32[FRAME_SIZE]      = 1 920 B
 *     synthesis_mem   : float32[FRAME_SIZE]      = 1 920 B
 *     pitch_buf       : float32[PITCH_BUF_SIZE]  = 6 912 B
 *     pitch_enh_buf   : float32[PITCH_BUF_SIZE]  = 6 912 B
 *     cepstral_mem    : float32[CEPS_MEM*BANDS]  =   704 B
 *     lastg           : float32[NB_BANDS]        =    88 B
 *                                            sum = ~18 KB  per stream
 * ====================================================================== */
#define MODEL_GRU_STATE_TOTAL_BYTES \
    (MODEL_INPUT_VAD_GRU_SIZE + MODEL_INPUT_NOISE_GRU_SIZE + MODEL_INPUT_DENOISE_GRU_SIZE)
/* = 168 bytes */

/* ======================================================================
 * 6.  MEMORY FOOTPRINT  (MERA-generated scratch / model size)
 * ======================================================================
 * The MERA-generated function compute_sub_0000() needs a per-call scratch
 * buffer. The size depends on whether the compiler defines ARM_MATH_MVEI
 * (RA8P1's Helium SIMD lane). Both values come from
 *     embedded_c/2_6_0_pkg_4815/INT8/src_mcu/rnnoise_INT8_CPU/
 *              deploy/build/MCU/compilation/src/compute_sub_0000.h
 *
 * The firmware allocates max(MVEI, DSP) = 2 209 B unconditionally so
 * either code path links cleanly.
 * ====================================================================== */
#define MODEL_MERA_BUFFER_BYTES_MVEI      2209U
#define MODEL_MERA_BUFFER_BYTES_DSP       1061U
#define MODEL_MERA_BUFFER_BYTES           MODEL_MERA_BUFFER_BYTES_MVEI

/* Total static flash budget of the model graph itself (CMSIS-NN code-gen).
 * Reported by `ls -la` of the 6 generated files on the dev container; the
 * exported .text / .rodata footprint of the firmware will be slightly
 * smaller after the linker dead-strips unused kernel variants. */
#define MODEL_C_SOURCE_BYTES              612002U    /* sum of 6 .c/.h files */

/* ======================================================================
 * 7.  ON-MCU EVALUATION CONFIGURATION
 * ======================================================================
 * The firmware does not run PESQ (too heavy for the MCU). accuracy_eval.c
 * reports per-frame SNR(denoised, clean) instead. The pass /
 * fail threshold is set conservatively to flag a *broken* deployment, not
 * to enforce parity with the host's PESQ score.
 * ====================================================================== */
#define MODEL_EVAL_METRIC                 "SNR_dB(denoised, clean)"
#define MODEL_EVAL_MIN_PASS_SNR_DB        2.0f      /* below this = broken    */

/* ======================================================================
 * 8.  PIPELINE REFERENCE  (matches the host-side denoise loop)
 * ======================================================================
 * Per-frame (10 ms):
 *
 *   1.  int16 PCM [480]
 *           -> preprocessing.c: high-pass biquad
 *                              -> windowed FFT
 *                              -> 22-band energy + cepstrum + pitch search
 *                              -> 42 features (host-side: features = float32)
 *
 *   2.  Quantise features:       int8 = clip(features / 0.2215 + 14)
 *
 *   3.  compute_sub_0000(
 *           main_storage[MODEL_MERA_BUFFER_BYTES],
 *           denoise_gru_prev[96], main_input[42],
 *           noise_gru_prev[48], vad_gru_prev[24],
 *           vad_next[24], noise_next[48], denoise_next[96],
 *           gains[22], vad_prob[1]
 *       );
 *
 *   4.  Dequantise gains:        g[k] = (gains[k] + 128) * 0.00390625
 *       Dequantise GRU states:   feed back into the next frame as-is
 *
 *   5.  postprocessing.c: interpolate 22 band gains -> 481 FFT-bin gains
 *                         -> apply to magnitude spectrum
 *                         -> pitch-aware comb filtering
 *                         -> IFFT + overlap-add -> int16 [480]
 *
 * Silent frames bypass step (3) and emit 480 zero samples (matches host).
 * ====================================================================== */

#ifdef __cplusplus
}
#endif

#endif /* MODEL_METADATA_H */
