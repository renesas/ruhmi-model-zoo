/**
 ******************************************************************************
 * @file    postprocessing.c
 * @brief   Portable NN postprocessing for image classification
 ******************************************************************************
 */

#include "postprocessing.h"

#include <stddef.h>

#define NN_POSTPROCESS_NEG_INF         (-3.4e38F)
#define NN_POSTPROCESS_INVALID_INDEX    (UINT32_MAX)

void postprocess(const float             *class_scores,
                 uint32_t                 class_count,
                 const char * const      *class_names,
                 NN_Postprocess_Result_t *top1_result,
                 uint32_t                 top_k_count,
                 NN_Postprocess_Result_t *top_k_results)
{
    uint32_t class_index;
    uint32_t rank_index;

    /* Exit early when the input score buffer is not usable. */
    if ((class_scores == NULL) || (class_count == 0U))
    {
        return;
    }

    /* Scan once to identify the single best class. */
    if (top1_result != NULL)
    {
        uint32_t best_class_index = 0U;
        float    best_score = class_scores[0];

        for (class_index = 1U; class_index < class_count; class_index++)
        {
            if (class_scores[class_index] > best_score)
            {
                best_score = class_scores[class_index];
                best_class_index = class_index;
            }
        }

        top1_result->top1_index      = best_class_index;
        top1_result->top1_score      = best_score;
        top1_result->top1_class_name = (class_names != NULL) ? class_names[best_class_index] : NULL;
    }

    /* Build the ordered top-k table when the caller requests it. */
    if ((top_k_results != NULL) && (top_k_count != 0U))
    {
        const uint32_t result_count = (top_k_count < class_count) ? top_k_count : class_count;
        float current_score;
        uint32_t insert_position;

        /* Seed all result slots with an invalid entry and the lowest score sentinel. */
        for (rank_index = 0U; rank_index < result_count; rank_index++)
        {
            top_k_results[rank_index].top1_index = NN_POSTPROCESS_INVALID_INDEX;
            top_k_results[rank_index].top1_score = NN_POSTPROCESS_NEG_INF;
            top_k_results[rank_index].top1_class_name = NULL;
        }

        /* Insert each class into the ranked list if its score is high enough. */
        for (class_index = 0U; class_index < class_count; class_index++)
        {
            current_score = class_scores[class_index];
            insert_position = result_count;

            /* Find the first rank where the current score should be inserted. */
            for (rank_index = 0U; rank_index < result_count; rank_index++)
            {
                if (current_score > top_k_results[rank_index].top1_score)
                {
                    insert_position = rank_index;
                    break;
                }
            }

            /* Shift lower-ranked entries down and place the new candidate. */
            if (insert_position < result_count)
            {
                for (rank_index = result_count - 1U; rank_index > insert_position; rank_index--)
                {
                    top_k_results[rank_index] = top_k_results[rank_index - 1U];
                }

                top_k_results[insert_position].top1_index = class_index;
                top_k_results[insert_position].top1_score = current_score;
                top_k_results[insert_position].top1_class_name = (class_names != NULL) ? class_names[class_index] : NULL;
            }
        }
    }
}
