/**
 * @file    postprocessing.c
 * @brief   Visual Wake Words (VWW, NPU) output postprocessing - dequantization, softmax, and top-1 person/not-person detection.
 *
 * @author  Renesas Electronics
 * @date    2025
 * @version 1.0
 */

#include "postprocessing.h"

#include <math.h>

/**
 * @brief Dequantize INT8 output tensor into floating-point class p_scores.
 * @param[in] p_out_q Input INT8 model output buffer.
 * @param[in] class_count Number of class p_scores in the output buffer.
 * @param[out] p_scores Output float p_scores after dequantization.
 */
void dequantize_output_to_scores(const int8_t *p_out_q,
                                 int           class_count,
                                 float        *p_scores)
{
    /* Convert int8 logits to float values using output quantization metadata. */
    for (int class_index = 0; class_index < class_count; class_index++)
    {
        p_scores[class_index] = ((float)p_out_q[class_index] - (float)OUTPUT_ZP) * OUTPUT_SCALE;
    }
}

/**
 * @brief Apply softmax normalization in place.
 * @details Uses max-subtraction for numerical stability before exponentiation.
 * @param[in,out] p_scores Input logits and output probabilities.
 * @param[in] class_count Number of class elements in p_scores.
 */
void softmax_inplace(float *p_scores, int class_count)
{
    /* Subtract max logit first for numerical stability before exp(). */
    float max_val = p_scores[0];
    int   class_index;

    for (class_index = 1; class_index < class_count; class_index++)
    {
        if (p_scores[class_index] > max_val)
        {
            max_val = p_scores[class_index];
        }
    }

    /* Exponentiate and accumulate denominator. */
    float sum = 0.0f;
    for (class_index = 0; class_index < class_count; class_index++)
    {
        p_scores[class_index] = expf(p_scores[class_index] - max_val);
        sum                += p_scores[class_index];
    }

    /* Normalize probabilities so the distribution sums to 1.0. */
    for (class_index = 0; class_index < class_count; class_index++)
    {
        p_scores[class_index] /= sum;
    }
}

/**
 * @brief Return the index of the maximum score.
 * @param[in] p_scores Input float score/probability array.
 * @param[in] class_count Number of class elements in p_scores.
 * @return Index of the highest score element.
 */
int argmax(const float *p_scores, int class_count)
{
    /* Return index of class with maximum score. */
    int best_class_index = 0;
    for (int class_index = 1; class_index < class_count; class_index++)
    {
        if (p_scores[class_index] > p_scores[best_class_index])
        {
            best_class_index = class_index;
        }
    }
    return best_class_index;
}

/**
 * @brief Run full VWW postprocessing and return top-1 prediction.
 * @param[in] p_out_q Input INT8 model output buffer.
 * @param[in] class_count Number of classes in p_out_q.
 * @param[out] p_scores Output float probability p_scores.
 * @param[out] p_predicted_class Output top-1 class index.
 */
void postprocess(const int8_t *p_out_q,
                 int class_count,
                 float *p_scores,
                 int *p_predicted_class)
{
    int top1_class_index;

    /* Step 1: convert quantized model output to floating-point logits. */
    dequantize_output_to_scores(p_out_q, class_count, p_scores);

    /* Step 2: normalize logits into probabilities. */
    softmax_inplace(p_scores, class_count);

    /* Step 3: find top-1 class index from final score buffer. */
    top1_class_index = argmax(p_scores, class_count);

    /* Step 4: return selected class to caller. */
    *p_predicted_class = top1_class_index;
}
