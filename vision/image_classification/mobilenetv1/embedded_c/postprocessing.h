/**
 * @file    postprocessing.h
 * @brief   MobileNetV1 output postprocessing declarations - INT8 dequantization and top-k classification.
 *
 * @author  Renesas Electronics
 * @date    2025
 * @version 1.0
 */

/*
 * Copyright (C) 2025 Renesas Electronics Corporation. All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef POSTPROCESSING_H
#define POSTPROCESSING_H

#include <stdint.h>

/* Number of top predictions returned by postprocessing. */
#define TOP_K       (5)

/**
 * @brief Dequantize INT8 model output into floating-point class scores.
 *
 * Dequantization uses OUTPUT_SCALE and OUTPUT_ZP from model metadata.
 *
 * @param _p_quantized_output    Raw int8 output tensor.
 * @param _class_count           Number of classes in output tensor.
 * @param _p_dequantized_scores  Destination float score buffer.
 */
void dequantize_output_to_scores(const int8_t *_p_quantized_output,
                                 int _class_count,
                                 float *_p_dequantized_scores);

/**
 * @brief Select top-k predictions directly from quantized model output.
 *
 * @param _p_quantized_output  Raw int8 output tensor.
 * @param _class_count         Number of classes in output tensor.
 * @param _top_k_count         Number of highest scores to keep.
 * @param _p_top_indices       Output class indices sorted by score descending.
 * @param _p_top_scores        Output class scores sorted by score descending.
 */
void postprocess(const int8_t *_p_quantized_output,
                 int _class_count,
                 int _top_k_count,
                 int *_p_top_indices,
                 float *_p_top_scores);

#endif /* POSTPROCESSING_H */
