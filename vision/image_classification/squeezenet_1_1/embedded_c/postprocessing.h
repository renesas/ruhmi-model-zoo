/*
 * Copyright (C) 2025 Renesas Electronics Corporation.
 *
 * This software is supplied by Renesas Electronics Corporation and is only
 * intended for use with Renesas products. No other uses are authorized.
 */

/**
 * @file    postprocessing.h
 * @brief   SqueezeNet (NPU) output postprocessing declarations - INT8 dequantization, softmax, and top-k classification.
 *
 * @author  Renesas Electronics
 * @date    2025
 * @version 1.0
 */

#ifndef POSTPROCESSING_H
#define POSTPROCESSING_H

#include <stdint.h>
#include "../common/model_metadata.h"

/**
 * @brief Convert int8 output tensor values to floating-point class p_scores.
 *
 * @param p_out_q        Raw int8 output tensor.
 * @param class_count  Number of classes in output tensor.
 * @param p_scores       Destination float score buffer.
 */
void dequantize_output_to_scores(const int8_t *p_out_q,
                                 int class_count,
                                 float *p_scores);

/**
 * @brief Select highest scoring classes and return them in descending order.
 *
 * @param p_scores       Input class p_scores.
 * @param score_count  Number of score elements.
 * @param top_count    Number of highest results to keep.
 * @param p_top_indices  Output class indices.
 * @param p_top_scores   Output class p_scores.
 */
void top_k(const float *p_scores,
           int score_count,
           int top_count,
           int *p_top_indices,
           float *p_top_scores);

/**
 * @brief Main postprocessing entry point for application flow.
 * @details Dequantizes output, conditionally applies softmax, then computes top-k.
 * @param[in] p_out_q         Raw int8 output tensor.
 * @param[in] class_count   Number of classes in output tensor.
 * @param[in] top_count     Number of top classes requested.
 * @param[out] p_scores       Working/output float p_scores buffer [class_count].
 * @param[out] p_top_indices  Output class indices sorted by descending score.
 * @param[out] p_top_scores   Output class p_scores sorted by descending score.
 */
void postprocess(const int8_t *p_out_q,
                 int class_count,
                 int top_count,
                 float *p_scores,
                 int *p_top_indices,
                 float *p_top_scores);

#endif /* POSTPROCESSING_H */
