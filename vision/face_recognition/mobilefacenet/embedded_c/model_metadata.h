/*
 * Copyright (c) 2020 - 2026 Renesas Electronics Corporation and/or its affiliates
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * MISRA C 2012 Compliant MobileFaceNet Metadata Definitions
 */
/**********************************************************************************************************************
 * File Name    : model_metadata.h
 * Description  : Compile-time metadata for the MobileFaceNet model deployed on
 *                the RA8P1 CPU target (MobileFaceNet face-recognition demo).
 *
 *                Source model : mobilefacenet_FP32.tflite
 *
 * Boundary tensors exposed to host code (full-INT8 deployment):
 *   Input  : serving_default_input0:0      shape=[1,112,112,3] dtype=int8
 *            Quantized from float range [-1.0, 1.0]:
 *              q = clamp(round(float / INPUT_QUANT_SCALE) + INPUT_QUANT_ZP, -128, 127)
 *   Output : PartitionedCall:0             shape=[1,128]       dtype=int8
 *            Raw quantized 128-D face embedding (no L2 norm in graph).
 *            Dequantize before use:
 *              float = (q - OUTPUT_QUANT_ZP) * OUTPUT_QUANT_SCALE
 *
 * Face verification protocol:
 *   1. Quantize 112x112 uint8 face -> int8 input tensor.
 *   2. Run model -> int8 embedding output.
 *   3. Dequantize int8 output -> float[128].
 *   4. L2-normalise both embeddings.
 *   5. Cosine similarity = dot product of the unit vectors.
 *   6. The pair is declared a match iff sim >= MATCH_THRESHOLD.
 *
 * The match threshold (0.28) was tuned on the LFW 6000-pair verification
 * protocol using the MERA-quantised model and gives 96.83% accuracy.
 **********************************************************************************************************************/

#ifndef MODEL_METADATA_H
#define MODEL_METADATA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ======================================================================
 * 1.  MODEL IDENTITY
 * ====================================================================== */
#define MODEL_NAME          "MobileFaceNet"
#define MODEL_FILE_INT8     "mobilefacenet_INT8.tflite"
#define MODEL_TASK          "Face Recognition (1:1 verification)"
#define MODEL_DATASET       "MS-Celeb-1M (training) / LFW (evaluation)"

/* ======================================================================
 * 2.  INPUT TENSOR  [1, 112, 112, 3] int8
 * ====================================================================== */
#define MODEL_INPUT_H       (112)
#define MODEL_INPUT_W       (112)
#define MODEL_INPUT_C       (3)
#define MODEL_INPUT_SIZE    (MODEL_INPUT_H * MODEL_INPUT_W * MODEL_INPUT_C)
                                                         /* = 37 632 elements */
#define MODEL_INPUT_LAYOUT  "NHWC"

/* Input quantization parameters (from mobilefacenet_INT8.tflite tensor 0).
 * The preprocessor normalises uint8 pixels to float in [-1, 1] and then
 * quantizes to int8:
 *
 *   float_norm = (float)raw_pixel / 127.5f - 1.0f          ->  [-1.0, 1.0]
 *   q = clamp(round(float_norm / INPUT_QUANT_SCALE) + INPUT_QUANT_ZP, -128, 127)
 */

#define INPUT_QUANT_SCALE   (0.007843137718737125f)   /* ≈ 2/255  */
#define INPUT_QUANT_ZP      (-1)

#define INPUT_PIXEL_MIN     (0)
#define INPUT_PIXEL_MAX     (255)

/* The host receives pre-aligned 112x112 faces (MTCNN alignment is done off
 * device and the crops are baked into test_faces.h), so no resize, crop or
 * rotation is performed in the preprocessor - only the per-pixel float
 * normalisation above. */
#define INPUT_RESIZE_METHOD "None (input is pre-aligned to 112x112)"

/* ======================================================================
 * 3.  OUTPUT TENSOR  [1, 128] float32  (raw face embedding)
 * ====================================================================== */
#define MODEL_OUTPUT_SIZE   (128)

/* Output quantization parameters (from mobilefacenet_INT8.tflite tensor 1).
 * The postprocessor dequantizes int8 output to float before L2 normalisation:
 *
 *   float_emb[i] = ((float)q_emb[i] - OUTPUT_QUANT_ZP) * OUTPUT_QUANT_SCALE
 */
#define OUTPUT_QUANT_SCALE  (0.054182425141334534f)
#define OUTPUT_QUANT_ZP     (5)

/* 0 = the embedding is NOT unit length out of the model.  The
 * postprocessor dequantizes then applies L2 normalisation before cosine
 * similarity comparison. */
#define OUTPUT_HAS_L2_NORM  (0)

/* Cosine-similarity decision threshold (tuned on LFW 6000-pair set). */
#define MATCH_THRESHOLD     (0.280f)

/* ======================================================================
 * 4.  MEMORY FOOTPRINT
 * ====================================================================== */
#define MODEL_INPUT_BUF_BYTES    (MODEL_INPUT_SIZE  * (uint32_t)sizeof(int8_t))
                                                /* 37 632 * 1 = 37 632 B */
#define MODEL_OUTPUT_BUF_BYTES   (MODEL_OUTPUT_SIZE * (uint32_t)sizeof(int8_t))
                                                /* 128 * 1 = 128 B */

/* Float buffer for dequantized embedding (used by postprocessor). */
#define MODEL_OUTPUT_FLOAT_BUF_BYTES  (MODEL_OUTPUT_SIZE * (uint32_t)sizeof(float))
                                                /* 128 * 4 = 512 B */

/* ======================================================================
 * 5.  PIPELINE REFERENCE
 * ======================================================================
 *   Input  : pre-aligned 112x112 RGB uint8
 *            -> (x/127.5 - 1.0) quantized to int8  -> preprocess()
 *   Model  : compute_sub_0000(arena, int8 in[37632], int8 out[128])
 *   Output : int8[128] -> dequantize to float[128] -> L2 normalise
 *            -> cosine sim vs reference embedding
 *            -> compare against MATCH_THRESHOLD    -> postprocess()
 * ====================================================================== */

#ifdef __cplusplus
}
#endif

#endif /* MODEL_METADATA_H */
