/*
* Copyright (c) 2020 - 2025 Renesas Electronics Corporation and/or its affiliates
*
* SPDX-License-Identifier: BSD-3-Clause
*/
/**********************************************************************************************************************
 * File Name    : postprocessing.h
 * Description  : MobileNetV2 (NPU) output postprocessing declarations - INT8 dequantization and top-k classification.
 **********************************************************************************************************************/

#ifndef POSTPROCESSING_H
#define POSTPROCESSING_H

#include <stdint.h>

#define TOP_K (5)   /* Number of top predictions returned by postprocessing. */

/*******************************************************************************************************************//**
 * @brief Dequantize output and extract top-k predictions.
 *
 * @param[in]  p_quantized_output   Raw int8 output tensor.
 * @param[in]  class_count          Number of classes in output tensor.
 * @param[in]  top_count            Number of highest scores to keep.
 * @param[out] p_dequantized_scores Working/output float score buffer.
 * @param[out] p_top_indices        Output class indices sorted by score descending.
 * @param[out] p_top_scores         Output class scores sorted by score descending.
 **********************************************************************************************************************/
void postprocess(const int8_t *p_quantized_output,
                 int class_count,
                 int top_count,
                 float *p_dequantized_scores,
                 int *p_top_indices,
                 float *p_top_scores);

#endif /* POSTPROCESSING_H */
