/*
 * Copyright (c) 2020 - 2026 Renesas Electronics Corporation and/or its affiliates
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * @file    postprocessing.c
 * @brief   ShuffleNetV2 x0.5 INT8 postprocessing - dequantize + softmax + top-K selection.
 *
 *   1. Dequantize : logit = (q - zp) * scale
 *   2. Softmax    : P(i) = exp(logit_i - max) / sum(exp(logit_j - max))  [numerically stable]
 *   3. Top-K      : maintain a min-heap of K elements over all probabilities.
 *   4. Sort       : top-K descending by probability.
 */

#include "postprocessing.h"
#include <math.h>

/* Maximum number of output classes — sized for ImageNet 1000. */
#define POSTPROCESS_MAX_CLASSES (1000)

/**
 * @brief Dequantize model logits and select top-K classes.
 *
 * @param[in] p_quantized_output Model output tensor.
 * @param[in] class_count Number of output classes.
 * @param[in] output_scale Output tensor quantization scale.
 * @param[in] output_zp Output tensor quantization zero-point.
 * @param[in] top_k_count Number of candidates to keep.
 * @param[out] p_top_indices Output class index list.
 * @param[out] p_top_scores Output score list.
 */
void postprocess(const int8_t *p_quantized_output,
                 int           class_count,
                 float         output_scale,
                 int           output_zp,
                 int           top_k_count,
                 int          *p_top_indices,
                 float        *p_top_scores)
{
    int   i;
    int   min_pos;
    float max_logit;
    float sum_exp;

    /* Reused across calls; safe for single-threaded embedded execution. */
    static float s_probs[POSTPROCESS_MAX_CLASSES];

    /* ------------------------------------------------------------------ */
    /* Step 1 – Dequantize all logits                                       */
    /* ------------------------------------------------------------------ */
    for (i = 0; i < class_count; i++)
    {
        s_probs[i] = ((float)p_quantized_output[i] - (float)output_zp) * output_scale;
    }

    /* ------------------------------------------------------------------ */
    /* Step 2 – Numerically stable Softmax                                 */
    /*   Subtract max before exp() to prevent floating-point overflow.     */
    /* ------------------------------------------------------------------ */
    max_logit = s_probs[0];
    for (i = 1; i < class_count; i++)
    {
        if (s_probs[i] > max_logit)
        {
            max_logit = s_probs[i];
        }
    }

    sum_exp = 0.0f;
    for (i = 0; i < class_count; i++)
    {
        s_probs[i] = expf(s_probs[i] - max_logit);
        sum_exp += s_probs[i];
    }

    for (i = 0; i < class_count; i++)
    {
        s_probs[i] /= sum_exp;
    }

    /* ------------------------------------------------------------------ */
    /* Step 3 – Top-K selection (min-heap over softmax probabilities)       */
    /* ------------------------------------------------------------------ */

    /* Seed top-K with first K elements. */
    for (i = 0; i < top_k_count; i++)
    {
        p_top_indices[i] = i;
        p_top_scores[i]  = s_probs[i];
    }

    /* Find initial minimum position. */
    min_pos = 0;
    for (i = 1; i < top_k_count; i++)
    {
        if (p_top_scores[i] < p_top_scores[min_pos])
        {
            min_pos = i;
        }
    }

    /* Scan remaining classes, replace minimum when a better candidate is found. */
    for (i = top_k_count; i < class_count; i++)
    {
        if (s_probs[i] > p_top_scores[min_pos])
        {
            int j;
            p_top_indices[min_pos] = i;
            p_top_scores[min_pos]  = s_probs[i];

            /* Recompute minimum position. */
            min_pos = 0;
            for (j = 1; j < top_k_count; j++)
            {
                if (p_top_scores[j] < p_top_scores[min_pos])
                {
                    min_pos = j;
                }
            }
        }
    }

    /* Sort top-K descending by score (selection sort). */
    for (i = 0; i < top_k_count - 1; i++)
    {
        int j;
        for (j = i + 1; j < top_k_count; j++)
        {
            if (p_top_scores[j] > p_top_scores[i])
            {
                float tmp_s = p_top_scores[i];
                int tmp_i = p_top_indices[i];
                p_top_scores[i] = p_top_scores[j];
                p_top_indices[i] = p_top_indices[j];
                p_top_scores[j] = tmp_s;
                p_top_indices[j] = tmp_i;
            }
        }
    }
}
