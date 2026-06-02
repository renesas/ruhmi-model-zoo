/*
 * Copyright (c) 2020 - 2026 Renesas Electronics Corporation and/or its affiliates
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * @file    postprocessing.h
 * @brief   ShuffleNetV2 x0.5 INT8 postprocessing - dequantize + top-K.
 */

#ifndef POSTPROCESSING_H
#define POSTPROCESSING_H

#include <stdint.h>

/**
 * @brief  Dequantize INT8 logits and find top-K classes.
 *
 * @param[in]  quantized_output  INT8 output tensor [1000].
 * @param[in]  class_count       Number of classes (1000).
 * @param[in]  output_scale      Output quantization scale.
 * @param[in]  output_zp         Output quantization zero point.
 * @param[in]  top_k_count       Number of top results to return.
 * @param[out] top_indices       Array of top-K class indices (descending score).
 * @param[out] top_scores        Array of top-K dequantized logit scores.
 */
void postprocess(const int8_t *p_quantized_output,
                 int           class_count,
                 float         output_scale,
                 int           output_zp,
                 int           top_k_count,
                 int          *p_top_indices,
                 float        *p_top_scores);

#endif /* POSTPROCESSING_H */
