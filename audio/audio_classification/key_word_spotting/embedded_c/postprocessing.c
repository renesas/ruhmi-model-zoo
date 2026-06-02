/**
 * @file    postprocessing.c
 * @brief   KWS top-1 postprocessing (Python-equivalent flow).
 *
 * Copyright 2026 Renesas Electronics Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#include "postprocessing.h"
#include "../common/model_metadata.h"

#include <stdbool.h>
#include <math.h>

/**
 * @brief Dequantize output, apply optional softmax, and return top-1 index.
 */
int32_t postprocess(const int8_t *p_output, int32_t num_classes)
{
    int32_t class_index;
    int32_t highest_score_class_index;
    float dequantized_scores[MODEL_NUM_CLASSES];
    float probabilities[MODEL_NUM_CLASSES];
    float score_sum;
    float max_score;
    float exp_sum;
    bool scores_already_probabilities;

    /* Step 1: Dequantize INT8 scores. */
    score_sum = 0.0f;
    scores_already_probabilities = true;

    for (class_index = 0; class_index < num_classes; class_index++)
    {
        dequantized_scores[class_index] =
            ((float)((int32_t)p_output[class_index] - OUTPUT_ZP)) * OUTPUT_SCALE;
        score_sum += dequantized_scores[class_index];

        if (dequantized_scores[class_index] < 0.0f)
        {
            scores_already_probabilities = false;
        }
    }

    if ((score_sum < 0.99f) || (score_sum > 1.01f))
    {
        scores_already_probabilities = false;
    }

    /* Step 2: Use scores directly if already probabilities; else softmax. */
    if (scores_already_probabilities)
    {
        for (class_index = 0; class_index < num_classes; class_index++)
        {
            probabilities[class_index] = dequantized_scores[class_index];
        }
    }
    else
    {
        max_score = dequantized_scores[0];
        for (class_index = 1; class_index < num_classes; class_index++)
        {
            if (dequantized_scores[class_index] > max_score)
            {
                max_score = dequantized_scores[class_index];
            }
        }

        exp_sum = 0.0f;
        for (class_index = 0; class_index < num_classes; class_index++)
        {
            probabilities[class_index] = expf(dequantized_scores[class_index] - max_score);
            exp_sum += probabilities[class_index];
        }

        for (class_index = 0; class_index < num_classes; class_index++)
        {
            probabilities[class_index] /= exp_sum;
        }
    }

    /* Step 3: Argmax. */
    highest_score_class_index = 0;

    for (class_index = 1; class_index < num_classes; class_index++)
    {
        if (probabilities[class_index] > probabilities[highest_score_class_index])
        {
            highest_score_class_index = class_index;
        }
    }

    return highest_score_class_index;
}

