/*
* Copyright (c) 2020 - 2025 Renesas Electronics Corporation and/or its affiliates
*
* SPDX-License-Identifier: BSD-3-Clause
*/
/**********************************************************************************************************************
 * File Name    : postprocessing.c
 * Description  : MobileNetV2 (NPU) output postprocessing - INT8 dequantization and top-k classification.
 **********************************************************************************************************************/

#include "postprocessing.h"

/* Use project-level quantization metadata directly inside postprocessing. */
#include "../common/model_metadata.h"

/*******************************************************************************************************************//**
 * @brief Dequantize INT8 output tensor into floating-point class scores.
 * @param[in]  p_quantized_output   Input INT8 model output buffer.
 * @param[in]  class_count          Number of class scores in the output buffer.
 * @param[out] p_dequantized_scores Output float scores after dequantization.
 **********************************************************************************************************************/
static void dequantize_output_to_scores(const int8_t *p_quantized_output,
                                        int class_count,
                                        float *p_dequantized_scores)
{
    int class_index;

    /* Convert each quantized output value into its floating-point score. */
    for (class_index = 0; class_index < class_count; class_index++) {
        p_dequantized_scores[class_index] = ((float)p_quantized_output[class_index] - (float)OUTPUT_ZP) * OUTPUT_SCALE;
    }
}

/*******************************************************************************************************************//**
 * @brief Select top_count highest scores from a score array.
 * @param[in]  p_scores      Input float score array.
 * @param[in]  score_count   Number of entries in p_scores.
 * @param[in]  top_count     Number of top entries to keep.
 * @param[out] p_top_indices Output indices of selected top entries.
 * @param[out] p_top_scores  Output values of selected top entries.
 **********************************************************************************************************************/
static void top_k(const float *p_scores,
                  int score_count,
                  int top_count,
                  int *p_top_indices,
                  float *p_top_scores)
{
    int score_index;
    int minimum_score_position;

    /* Seed the working top-k buffers with the first k class p_scores. */
    for (score_index = 0; score_index < top_count; score_index++) {
        p_top_indices[score_index] = score_index;
        p_top_scores[score_index] = p_scores[score_index];
    }

    /* Track the smallest score inside the current top-k window. */
    minimum_score_position = 0;
    for (score_index = 1; score_index < top_count; score_index++) {
        if (p_top_scores[score_index] < p_top_scores[minimum_score_position]) {
            minimum_score_position = score_index;
        }
    }

    /* Scan the remaining p_scores and replace the current minimum when a better score is found. */
    for (score_index = top_count; score_index < score_count; score_index++) {
        if (p_scores[score_index] > p_top_scores[minimum_score_position]) {
            int candidate_position;

            p_top_indices[minimum_score_position] = score_index;
            p_top_scores[minimum_score_position] = p_scores[score_index];

            for (candidate_position = 0; candidate_position < top_count; candidate_position++) {
                if (p_top_scores[candidate_position] < p_top_scores[minimum_score_position]) {
                    minimum_score_position = candidate_position;
                }
            }
        }
    }

    /* Order the selected classes from the highest score to the lowest score. */
    for (score_index = 0; score_index < (top_count - 1); score_index++) {
        int sorted_position;

        for (sorted_position = score_index + 1; sorted_position < top_count; sorted_position++) {
            if (p_top_scores[sorted_position] > p_top_scores[score_index]) {
                float temporary_score;
                int temporary_index;

                temporary_score = p_top_scores[score_index];
                p_top_scores[score_index] = p_top_scores[sorted_position];
                p_top_scores[sorted_position] = temporary_score;

                temporary_index = p_top_indices[score_index];
                p_top_indices[score_index] = p_top_indices[sorted_position];
                p_top_indices[sorted_position] = temporary_index;
            }
        }
    }
}

/*******************************************************************************************************************//**
 * @brief Run full postprocessing for MobileNet NPU output.
 * @details Dequantizes all output classes, then extracts ordered top-k results.
 * @param[in]  p_quantized_output   Input INT8 model output buffer.
 * @param[in]  class_count          Number of classes in output buffer.
 * @param[in]  top_count            Number of top predictions requested.
 * @param[out] p_dequantized_scores Workspace/output float scores for all classes.
 * @param[out] p_top_indices        Output indices for top_count predictions.
 * @param[out] p_top_scores         Output dequantized scores for top_count predictions.
 **********************************************************************************************************************/
void postprocess(const int8_t *p_quantized_output,
                 int class_count,
                 int top_count,
                 float *p_dequantized_scores,
                 int *p_top_indices,
                 float *p_top_scores)
{
    /* Expand the quantized network output into float p_scores for all classes. */
    dequantize_output_to_scores(p_quantized_output,
                                class_count,
                                p_dequantized_scores);

    /* Select and sort the highest scoring class indices requested by the caller. */
    top_k(p_dequantized_scores, class_count, top_count, p_top_indices, p_top_scores);
}
