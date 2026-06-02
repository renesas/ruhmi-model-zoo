/*
 * Copyright (C) 2025 Renesas Electronics Corporation.
 *
 * This software is supplied by Renesas Electronics Corporation and is only
 * intended for use with Renesas products. No other uses are authorized.
 */

/**
 * @file    postprocessing.c
 * @brief   SqueezeNet (NPU) output postprocessing - INT8 dequantization, softmax, and top-k classification.
 *
 * @author  Renesas Electronics
 * @date    2025
 * @version 1.0
 */

#include "postprocessing.h"

#include <math.h>

/**
 * @brief Apply softmax normalization in place.
 * @details Subtracts the maximum score for numerical stability before
 *          exponentiation, then normalizes so all probabilities sum to 1.
 * @param[in,out] p_scores Input logits and output probabilities.
 * @param[in] class_count Number of class elements in p_scores.
 */
static void softmax_inplace(float *p_scores, int class_count)
{
    int class_index;
    float max_value = p_scores[0];
    float sum = 0.0f;

    /* Pass 1: find maximum logit for numerical stability shift. */
    for (class_index = 1; class_index < class_count; class_index++)
    {
        if (max_value < p_scores[class_index])
        {
            max_value = p_scores[class_index];
        }
    }

    /* Pass 2: exponentiate shifted logits and accumulate denominator. */
    for (class_index = 0; class_index < class_count; class_index++)
    {
        p_scores[class_index] = expf(p_scores[class_index] - max_value);
        sum += p_scores[class_index];
    }

    /* Pass 3: divide by denominator to obtain probabilities. */
    if (0.0f < sum)
    {
        for (class_index = 0; class_index < class_count; class_index++)
        {
            p_scores[class_index] /= sum;
        }
    }
}

/**
 * @brief Dequantize INT8 output tensor into floating-point class p_scores.
 * @param[in] p_out_q Input INT8 model output buffer.
 * @param[in] class_count Number of class p_scores in the output buffer.
 * @param[out] p_scores Output float p_scores after dequantization.
 */
void dequantize_output_to_scores(const int8_t *p_out_q,
                                 int class_count,
                                 float *p_scores)
{
    /* Convert model output from int8 domain to float logits/p_scores. */
    for (int class_index = 0; class_index < class_count; class_index++) {
        p_scores[class_index] = ((float)p_out_q[class_index] - (float)OUTPUT_ZP) * OUTPUT_SCALE;
    }
}

/**
 * @brief Select top_count highest class p_scores from the score buffer.
 * @param[in] p_scores Input float class score buffer.
 * @param[in] score_count Number of score elements.
 * @param[in] top_count Number of top entries requested.
 * @param[out] p_top_indices Output class indices sorted by descending score.
 * @param[out] p_top_scores Output class p_scores sorted by descending score.
 */
void top_k(const float *p_scores,
           int score_count,
           int top_count,
           int *p_top_indices,
           float *p_top_scores)
{
    /* Seed top-k buffers with first top_count entries. */
    for (int rank_index = 0; rank_index < top_count; rank_index++) {
        p_top_indices[rank_index] = rank_index;
        p_top_scores[rank_index] = p_scores[rank_index];
    }

    /* Track position of the smallest element currently inside top-k set. */
    int minimum_score_position = 0;
    for (int rank_index = 1; rank_index < top_count; rank_index++) {
        if (p_top_scores[minimum_score_position] > p_top_scores[rank_index])
        {
            minimum_score_position = rank_index;
        }
    }

    /* Scan remaining p_scores and replace current minimum when a better score appears. */
    for (int score_index = top_count; score_index < score_count; score_index++) {
        if (p_top_scores[minimum_score_position] < p_scores[score_index]) {
            p_top_indices[minimum_score_position] = score_index;
            p_top_scores[minimum_score_position] = p_scores[score_index];

            /* Recompute minimum position after replacement. */
            for (int scan_index = 0; scan_index < top_count; scan_index++) {
                if (p_top_scores[minimum_score_position] > p_top_scores[scan_index])
                {
                    minimum_score_position = scan_index;
                }
            }
        }
    }

    /* Sort top-k entries by descending score for deterministic output ordering. */
    for (int rank_index = 0; rank_index < top_count - 1; rank_index++) {
        for (int compare_index = rank_index + 1; compare_index < top_count; compare_index++) {
            if (p_top_scores[rank_index] < p_top_scores[compare_index]) {
                float score_swap = p_top_scores[rank_index];
                p_top_scores[rank_index] = p_top_scores[compare_index];
                p_top_scores[compare_index] = score_swap;
                int index_swap = p_top_indices[rank_index];
                p_top_indices[rank_index] = p_top_indices[compare_index];
                p_top_indices[compare_index] = index_swap;
            }
        }
    }
}

/**
 * @brief Main postprocessing entry point for SqueezeNet application flow.
 * @details Dequantizes INT8 model output, optionally applies softmax when
 *          output is logits, then computes ordered top-k predictions.
 * @param[in] p_out_q Raw INT8 output tensor.
 * @param[in] class_count Number of classes in output tensor.
 * @param[in] top_count Number of top classes requested.
 * @param[out] p_scores Working/output float score buffer [class_count].
 * @param[out] p_top_indices Output class indices sorted by descending score.
 * @param[out] p_top_scores Output class p_scores sorted by descending score.
 */
void postprocess(const int8_t *p_out_q,
                 int class_count,
                 int top_count,
                 float *p_scores,
                 int *p_top_indices,
                 float *p_top_scores)
{
    /* Step 1: dequantize raw INT8 output into floating-point p_scores. */
    dequantize_output_to_scores(p_out_q, class_count, p_scores);
#if (0 == OUTPUT_HAS_SOFTMAX)
    /* Step 2: apply softmax only when model output is logits. */
    softmax_inplace(p_scores, class_count);
#endif

    /* Step 3: extract and sort top-k results for application output. */
    top_k(p_scores, class_count, top_count, p_top_indices, p_top_scores);
}
