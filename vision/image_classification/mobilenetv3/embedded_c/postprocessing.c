/***********************************************************************************************************************
 * Copyright [2020-2024] Renesas Electronics Corporation and/or its affiliates. All Rights Reserved.
 *
 * This software and documentation are supplied by Renesas Electronics Corporation and/or its affiliates and may only
 * be used with products of Renesas Electronics Corp. and its affiliates ("Renesas"). No other uses are authorized.
 * Renesas products are sold pursuant to Renesas terms and conditions of sale. Purchasers are solely responsible for
 * the selection and use of Renesas products and Renesas assumes no liability. No license, express or implied, to any
 * intellectual property right is granted by Renesas. This software is protected under all applicable laws, including
 * copyright laws. Renesas reserves the right to change or discontinue this software and/or this documentation.
 * THE SOFTWARE AND DOCUMENTATION IS DELIVERED TO YOU "AS IS," AND RENESAS MAKES NO REPRESENTATIONS OR WARRANTIES,
 * AND TO THE FULLEST EXTENT PERMISSIBLE UNDER APPLICABLE LAW, DISCLAIMS ALL WARRANTIES, WHETHER EXPLICITLY OR
 * IMPLICITLY, INCLUDING WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT, WITH
 * RESPECT TO THE SOFTWARE OR DOCUMENTATION. RENESAS SHALL HAVE NO LIABILITY ARISING OUT OF ANY SECURITY VULNERABILITY
 * OR BREACH. TO THE MAXIMUM EXTENT PERMITTED BY LAW, IN NO EVENT WILL RENESAS BE LIABLE TO YOU IN CONNECTION WITH THE
 * SOFTWARE OR DOCUMENTATION (OR ANY PERSON OR ENTITY CLAIMING RIGHTS DERIVED FROM YOU) FOR ANY LOSS, DAMAGE, OR
 * CLAIM, INCLUDING, BUT NOT LIMITED TO, LOST PROFITS, LOST DATA, OR INDIRECT, SPECIAL, CONSEQUENTIAL OR PUNITIVE
 * DAMAGES, REGARDLESS OF WHETHER SUCH CLAIM IS BASED ON CONTRACT, TORT, STRICT LIABILITY, OR OTHERWISE.
 **********************************************************************************************************************/

#include "postprocessing.h"

/***********************************************************************************************************************
 * Function Name: dequantize_output_to_scores
 * Description  : Converts each element of the quantized int8 output tensor to a floating-point score.
 *                dequantized_scores[i] = (quantized_output[i] - output_zero_point) * output_scale
 * Arguments    : p_quantized_output   - Pointer to int8 array of quantized model output values.
 *                _class_count         - Number of output classes.
 *                _output_scale        - Quantization scale factor.
 *                _output_zero_point   - Quantization zero-point offset.
 *                p_dequantized_scores - Pointer to float array to receive dequantized scores.
 * Return Value : None
 **********************************************************************************************************************/
void dequantize_output_to_scores(const int8_t *p_quantized_output,
                                 int _class_count,
                                 float _output_scale,
                                 int _output_zero_point,
                                 float *p_dequantized_scores)
{
    int _class_index;

    /* Convert each quantized output element to floating-point score. */
    for (_class_index = 0; _class_index < _class_count; _class_index++)
    {
        p_dequantized_scores[_class_index] = ((float)p_quantized_output[_class_index] - (float)_output_zero_point) * _output_scale;
    }
}

/***********************************************************************************************************************
 * Function Name: postprocess
 * Description  : Dequantizes int8 logits and selects the top-k classes.
 *                Output arrays contain results sorted in descending score order.
 * Arguments    : p_quantized_output   - Pointer to int8 array of quantized model output values.
 *                _class_count         - Total number of output classes.
 *                _output_scale        - Quantization scale factor.
 *                _output_zero_point   - Quantization zero-point offset.
 *                _top_k_count         - Number of top results to return.
 *                p_top_indices        - Pointer to int array to receive top-k class indices.
 *                p_top_scores         - Pointer to float array to receive top-k scores.
 * Return Value : None
 **********************************************************************************************************************/
void postprocess(const int8_t *p_quantized_output,
                 int _class_count,
                 float _output_scale,
                 int _output_zero_point,
                 int _top_k_count,
                 int *p_top_indices,
                 float *p_top_scores)
{
    int _top_index;
    int _min_score_position;

    /* Seed top-k buffers with the first k elements. */
    for (_top_index = 0; _top_index < _top_k_count; _top_index++)
    {
        p_top_indices[_top_index] = _top_index;
        p_top_scores[_top_index] = ((float)p_quantized_output[_top_index] - (float)_output_zero_point) * _output_scale;
    }

    /* Track the position of the current minimum score within the top-k set. */
    _min_score_position = 0;
    for (_top_index = 1; _top_index < _top_k_count; _top_index++)
    {
        if (p_top_scores[_top_index] < p_top_scores[_min_score_position])
        {
            _min_score_position = _top_index;
        }
    }

    /* Scan the remaining classes and keep only the k best scores seen so far. */
    for (_top_index = _top_k_count; _top_index < _class_count; _top_index++)
    {
        float _candidate_score = ((float)p_quantized_output[_top_index] - (float)_output_zero_point) * _output_scale;

        if (_candidate_score > p_top_scores[_min_score_position])
        {
            int _search_index;

            /* Replace the current minimum element in top-k with the new candidate. */
            p_top_indices[_min_score_position] = _top_index;
            p_top_scores[_min_score_position] = _candidate_score;

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

    /* Sort top-k entries in descending score order for final output. */
    for (_top_index = 0; _top_index < (_top_k_count - 1); _top_index++)
    {
        int _compare_index;

        for (_compare_index = _top_index + 1; _compare_index < _top_k_count; _compare_index++)
        {
            if (p_top_scores[_compare_index] > p_top_scores[_top_index])
            {
                float _score_swap;
                int _index_swap;

                /* Swap score pair to keep index/score mapping aligned. */
                _score_swap = p_top_scores[_top_index];
                p_top_scores[_top_index] = p_top_scores[_compare_index];
                p_top_scores[_compare_index] = _score_swap;

                _index_swap = p_top_indices[_top_index];
                p_top_indices[_top_index] = p_top_indices[_compare_index];
                p_top_indices[_compare_index] = _index_swap;
            }
        }
    }
}

