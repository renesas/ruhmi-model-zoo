/*
 * Copyright (c) 2020 - 2026 Renesas Electronics Corporation and/or its affiliates
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
/**********************************************************************************************************************
 * File Name    : postprocessing.h
 * Description  : MCUNet-in0 output postprocessing declarations - softmax and top-K classification on float32 logits.
 **********************************************************************************************************************/

#ifndef POSTPROCESSING_H
#define POSTPROCESSING_H

#include <stdint.h>

/* Number of top predictions returned by postprocessing. */
#define TOP_K       (5)

/**
 * @brief Compute top-k class predictions from the int8 MCUNet output tensor.
 *
 * Pipeline: 
 *   1. Dequantize each int8 element to float probability:
 *        prob[i] = ((float)out_q[i] - OUTPUT_ZP) * OUTPUT_SCALE
 *   2. Select top-K classes by probability (no softmax; already in graph).
 *
 * @note Not reentrant / not thread-safe. Uses a module-private static
 *       probability buffer of MODEL_OUTPUT_SIZE floats. Intended for
 *       single-threaded bare-metal use.
 *
 * @param p_output_q      Raw int8 output tensor (length _class_count).
 * @param _class_count    Number of classes (must equal MODEL_OUTPUT_SIZE).
 * @param _top_k_count    Number of top predictions requested.
 * @param p_top_indices   Output: class indices ordered high->low.
 * @param p_top_scores    Output: dequantized probabilities for those indices.
 */
void postprocess(const int8_t *p_output_q,
                 int _class_count,
                 int _top_k_count,
                 int *p_top_indices,
                 float *p_top_scores);

#endif /* POSTPROCESSING_H */
