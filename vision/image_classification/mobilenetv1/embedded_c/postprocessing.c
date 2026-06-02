/**
 * @file    postprocessing.c
 * @brief   MobileNetV1 output postprocessing - INT8 dequantization and top-k classification.
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

#include "postprocessing.h"
#include "../common/model_metadata.h"

/**
 * @brief Dequantize INT8 output tensor into floating-point class scores.
 * @param[in] _p_quantized_output Input INT8 model output buffer.
 * @param[in] _class_count Number of class scores in the output buffer.
 * @param[out] _p_dequantized_scores Output float scores after dequantization.
 */
void dequantize_output_to_scores(const int8_t *_p_quantized_output,
                                 int _class_count,
                                 float *_p_dequantized_scores)
{
    int _class_index;

    /* Convert each quantized output element to floating-point score. */
    for (_class_index = 0; _class_index < _class_count; _class_index++)
    {
        _p_dequantized_scores[_class_index] = ((float)_p_quantized_output[_class_index] - (float)OUTPUT_ZP) * OUTPUT_SCALE;
    }
}

/**
 * @brief Compute top-k class predictions from quantized model output.
 * @details Dequantizes candidate scores and returns class indices and scores
 *          ordered from highest to lowest confidence.
 * @param[in] _p_quantized_output Input INT8 model output buffer.
 * @param[in] _class_count Number of classes in the output buffer.
 * @param[in] _top_k_count Number of top predictions requested.
 * @param[out] _p_top_indices Output class indices for top-k predictions.
 * @param[out] _p_top_scores Output dequantized scores for top-k predictions.
 */
void postprocess(const int8_t *_p_quantized_output,
                 int _class_count,
                 int _top_k_count,
                 int *_p_top_indices,
                 float *_p_top_scores)
{
    int _top_index;
    int _min_score_position;

    /* Seed top-k buffers with the first k elements. */
    for (_top_index = 0; _top_index < _top_k_count; _top_index++)
    {
        _p_top_indices[_top_index] = _top_index;
        _p_top_scores[_top_index] = ((float)_p_quantized_output[_top_index] - (float)OUTPUT_ZP) * OUTPUT_SCALE;
    }

    /* Track the position of the current minimum score within the top-k set. */
    _min_score_position = 0;
    for (_top_index = 1; _top_index < _top_k_count; _top_index++)
    {
        if (_p_top_scores[_top_index] < _p_top_scores[_min_score_position])
        {
            _min_score_position = _top_index;
        }
    }

    /* Scan the remaining classes and keep only the k best scores seen so far. */
    for (_top_index = _top_k_count; _top_index < _class_count; _top_index++)
    {
        float _candidate_score = ((float)_p_quantized_output[_top_index] - (float)OUTPUT_ZP) * OUTPUT_SCALE;

        if (_candidate_score > _p_top_scores[_min_score_position])
        {
            int _search_index;

            /* Replace the current minimum element in top-k with the new candidate. */
            _p_top_indices[_min_score_position] = _top_index;
            _p_top_scores[_min_score_position] = _candidate_score;

            /* Recompute minimum position after replacement. */
            for (_search_index = 0; _search_index < _top_k_count; _search_index++)
            {
                if (_p_top_scores[_search_index] < _p_top_scores[_min_score_position])
                {
                    _min_score_position = _search_index;
                }
            }
        }
    }

    /* Sort top-k entries in descending score order for final output. */
    for (_top_index = 0; _top_index < (_top_k_count - 1); _top_index++)
    {
        int _compare_index;

        for (_compare_index = _top_index + 1; _compare_index < _top_k_count; _compare_index++)
        {
            if (_p_top_scores[_compare_index] > _p_top_scores[_top_index])
            {
                float _score_swap;
                int _index_swap;

                /* Swap score pair to keep index/score mapping aligned. */
                _score_swap = _p_top_scores[_top_index];
                _p_top_scores[_top_index] = _p_top_scores[_compare_index];
                _p_top_scores[_compare_index] = _score_swap;

                _index_swap = _p_top_indices[_top_index];
                _p_top_indices[_top_index] = _p_top_indices[_compare_index];
                _p_top_indices[_compare_index] = _index_swap;
            }
        }
    }
}
