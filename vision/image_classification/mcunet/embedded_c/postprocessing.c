/*
 * Copyright (c) 2020 - 2026 Renesas Electronics Corporation and/or its affiliates
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
/**********************************************************************************************************************
 * File Name    : postprocessing.c
 * Description  : MCUNet-in0 output postprocessing - softmax and top-K classification on int8 logits.
 *              : The model output encodes raw logits (the exported graph has no Softmax layer).
 *              : This module applies softmax for calibrated probabilities and returns the top-K
 *              : classes ordered high->low.
 **********************************************************************************************************************/
#include <stdint.h>
#include <math.h>
#include "postprocessing.h"
#include "model_metadata.h"

/**
 * @brief Compute top-K predictions from MCUNet int8 output.
 *
 * @param[in]  p_output_q       Raw int8 model output (MODEL_OUTPUT_SIZE elements).
 * @param[in]  _class_count      Number of classes.
 * @param[in]  _top_k_count      Number of top predictions requested.
 * @param[out] _p_top_indices    Top-K class indices (descending probability).
 * @param[out] _p_top_scores     Top-K softmax probabilities (descending).
 */
void postprocess(const int8_t *p_output_q,
                 int _class_count,
                 int _top_k_count,
                 int *p_top_indices,
                 float *p_top_scores)
{
    /* Stack buffer for the full 1000-class probability vector
     * (~4 KB float32; fits comfortably in MCU SRAM). */
    static float s_probs[MODEL_OUTPUT_SIZE];

    int   _class_index;
    int   _top_index;
    int   _min_score_position;
    float _max_logit;
    float _exp_sum;

    /* ── 1. Dequantize int8 -> float logits ────────────────────────── */
    for (_class_index = 0; _class_index < _class_count; _class_index++)
    {
        s_probs[_class_index] =
            ((float)p_output_q[_class_index] - (float)OUTPUT_ZP) * OUTPUT_SCALE;
    }

    /* ── 2. Numerically-stable softmax across all classes ─────────── */
    _max_logit = s_probs[0];
    for (_class_index = 1; _class_index < _class_count; _class_index++)
    {
        if (s_probs[_class_index] > _max_logit)
        {
            _max_logit = s_probs[_class_index];
        }
    }

    _exp_sum = 0.0f;
    for (_class_index = 0; _class_index < _class_count; _class_index++)
    {
        s_probs[_class_index] = expf(s_probs[_class_index] - _max_logit);
        _exp_sum += s_probs[_class_index];
    }

    if (_exp_sum > 0.0f)
    {
        float _inv_sum = 1.0f / _exp_sum;
        for (_class_index = 0; _class_index < _class_count; _class_index++)
        {
            s_probs[_class_index] *= _inv_sum;
        }
    }
    /* _probs[] now holds softmax probabilities in [0, 1] summing to ~1. */

    /* ── 3. Partial selection of top-K probabilities ──────────────── */

    /* Seed top-K buffers with the first K elements. */
    for (_top_index = 0; _top_index < _top_k_count; _top_index++)
    {
        p_top_indices[_top_index] = _top_index;
        p_top_scores[_top_index]  = s_probs[_top_index];
    }

    /* Track position of current minimum within the top-K window. */
    _min_score_position = 0;
    for (_top_index = 1; _top_index < _top_k_count; _top_index++)
    {
        if (p_top_scores[_top_index] < p_top_scores[_min_score_position])
        {
            _min_score_position = _top_index;
        }
    }

    /* Scan remaining classes and keep only the K best probabilities. */
    for (_top_index = _top_k_count; _top_index < _class_count; _top_index++)
    {
        float _candidate = s_probs[_top_index];

        if (_candidate > p_top_scores[_min_score_position])
        {
            int _search_index;

            p_top_indices[_min_score_position] = _top_index;
            p_top_scores[_min_score_position]  = _candidate;

            /* Recompute minimum position after replacement. */
            for (_search_index = 0; _search_index < _top_k_count; _search_index++)
            {
                if (p_top_scores[_search_index] < p_top_scores[_min_score_position])
                {
                    _min_score_position = _search_index;
                }
            }
        }
    }

    /* ── 4. Sort top-K entries in descending probability order ────── */
    for (_top_index = 0; _top_index < (_top_k_count - 1); _top_index++)
    {
        int _compare_index;

        for (_compare_index = _top_index + 1; _compare_index < _top_k_count; _compare_index++)
        {
            if (p_top_scores[_compare_index] > p_top_scores[_top_index])
            {
                float _score_swap;
                int   _index_swap;

                _score_swap = p_top_scores[_top_index];
                p_top_scores[_top_index]        = p_top_scores[_compare_index];
                p_top_scores[_compare_index]    = _score_swap;

                _index_swap = p_top_indices[_top_index];
                p_top_indices[_top_index]       = p_top_indices[_compare_index];
                p_top_indices[_compare_index]   = _index_swap;
            }
        }
    }
}
