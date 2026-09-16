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
 * @brief   Compile-time metadata for TinyWav2letter INT8 model
 *          deployed on RA8P1 CPU (MERA generated C kernel).
 *
 * Source model:
 *   tiny_wav2letter_pruned_int8.tflite
 *
 * Task:
 *   Streaming speech-to-text (character-level CTC logits)
 *
 * Tensor interface on compute_sub_0000():
 *   Input:
 *     input_1_int8            [1,296,39]   int8   (flattened 11544)
 *   Output:
 *     Identity_int8_70036     [1,1,148,29] int8   (flattened 4292)
 *
 * Pipeline summary:
 *   1) Host/offline feature extractor builds MFCC+delta+delta2 (39 dims).
 *   2) Features are arranged as [296,39] and quantized to int8.
 *   3) compute_sub_0000() produces 148 CTC frames x 29 classes.
 *   4) Firmware can run greedy CTC decode:
 *        - argmax per frame
 *        - collapse repeats
 *        - drop blank index (28)
 */

#ifndef MODEL_METADATA_H
#define MODEL_METADATA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ======================================================================
 * 1. MODEL IDENTITY
 * ====================================================================== */
#define MODEL_NAME                       "TinyWav2letter-INT8"
#define MODEL_FILE_INT8                  "tiny_wav2letter_pruned_int8.tflite"
#define MODEL_TASK                       "Speech Recognition (CTC char-level)"
#define MODEL_DATASET                    "Fluent Speech Commands"
#define MODEL_QUANTIZATION               "Per-tensor INT8 (TFLite/MERA)"

/* ======================================================================
 * 2. INPUT / OUTPUT GEOMETRY
 * ====================================================================== */
#define MODEL_INPUT_BATCH                (1U)
#define MODEL_INPUT_TIME                 (296U)
#define MODEL_INPUT_FEATURES             (39U)
#define MODEL_INPUT_SIZE                 (MODEL_INPUT_BATCH * MODEL_INPUT_TIME * MODEL_INPUT_FEATURES) /* 11544 */

#define MODEL_OUTPUT_BATCH               (1U)
#define MODEL_OUTPUT_AXIS1               (1U)
#define MODEL_OUTPUT_TIME                (148U)
#define MODEL_OUTPUT_CLASSES             (29U)
#define MODEL_OUTPUT_SIZE                (MODEL_OUTPUT_BATCH * MODEL_OUTPUT_AXIS1 * MODEL_OUTPUT_TIME * MODEL_OUTPUT_CLASSES) /* 4292 */

/* ======================================================================
 * 3. CTC DECODE METADATA
 * ====================================================================== */
#define MODEL_CTC_BLANK_INDEX            (28U)
#define MODEL_ALPHABET_SIZE              (29U)

/* ======================================================================
 * 4. QUANTIZATION PARAMETERS
 * ======================================================================
 */
#define MODEL_INPUT_SCALE                (0.17129258811473846f)
#define MODEL_INPUT_ZERO_POINT           (4)

#define MODEL_OUTPUT_SCALE               (0.4950784146785736f)
#define MODEL_OUTPUT_ZERO_POINT          (5)

/* ======================================================================
 * 5. MERA SCRATCH BUFFER
 * ====================================================================== */
#if defined(ARM_MATH_MVEI)
#define MODEL_MERA_BUFFER_BYTES          (259105U)
#else
#define MODEL_MERA_BUFFER_BYTES          (264417U)
#endif


#ifdef __cplusplus
}
#endif

#endif /* MODEL_METADATA_H */