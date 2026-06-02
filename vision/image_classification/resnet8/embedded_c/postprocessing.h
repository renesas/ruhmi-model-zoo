/**
 ******************************************************************************
 * @file    postprocessing.h
 * @brief   Portable NN postprocessing for image classification
 */

#ifndef POSTPROCESSING_H
#define POSTPROCESSING_H

#include <stdint.h>

typedef struct
{
    uint32_t     top1_index;       /**< Index of the class with highest score */
    float        top1_score;       /**< Score (probability) of the top-1 class */
    const char  *top1_class_name;  /**< Pointer to class label string, or NULL */
} NN_Postprocess_Result_t;

/**
 * @brief Find the top-1 and top-k classes from a score vector.
 * @param class_scores Pointer to the model output scores.
 * @param class_count Number of valid entries in @p class_scores.
 * @param class_names Optional class-label table aligned with @p class_scores.
 * @param top1_result Optional destination for the best-scoring class.
 * @param top_k_count Number of ranked results to populate in @p top_k_results.
 * @param top_k_results Optional destination array for ranked class results.
 */
void postprocess(const float             *class_scores,
                 uint32_t                 class_count,
                 const char * const      *class_names,
                 NN_Postprocess_Result_t *top1_result,
                 uint32_t                 top_k_count,
                 NN_Postprocess_Result_t *top_k_results);

#endif /* POSTPROCESSING_H */
