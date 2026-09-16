/*
* Copyright (c) 2020 - 2026 Renesas Electronics Corporation and/or its affiliates
*
* SPDX-License-Identifier: BSD-3-Clause
*/
/**
 * @file postprocessing.c
 * @brief YAMNet post-processing implementation for dequantization, peak aggregation, and top-K selection.
 */

#include <float.h>

#include "postprocessing.h"
#include "model_metadata.h"

/* Scratch buffers used by postprocess() to hold the per-patch dequantized
 * scores and the running class-wise peak. Kept file-static (not on the stack)
 * because MODEL_OUTPUT_CLASSES is 521 (~2 KB each). postprocess() is not
 * re-entrant. */
static float s_dequant_scratch[MODEL_OUTPUT_CLASSES];
static float s_peak_scores[MODEL_OUTPUT_CLASSES];

/**
 * @brief Dequantize an int16 model output buffer into float values.
 * @param[in] p_src_q_output Pointer to the quantized int16 model output.
 * @param[out] p_dst_output Pointer to the destination float buffer.
 * @param[in] _elem_count Number of elements to process.
 */
void yamnet_dequantize_output_int16(const int16_t *p_src_q_output,
                                    float         *p_dst_output,
                                    size_t         _elem_count)
{
    size_t _idx;

    for (_idx = 0; _idx < _elem_count; _idx++)
    {
        p_dst_output[_idx] = ((float)p_src_q_output[_idx] - (float)MODEL_OUTPUT_ZERO_POINT) * MODEL_OUTPUT_SCALE;
    }
}

/**
 * @brief Initialize the peak-score array to `-FLT_MAX` for each class.
 * @param[out] p_peak_scores Float array of length `class_count`.
 * @param[in] class_count Number of model output classes.
 */
static void yamnet_reset_peak_scores(float *p_peak_scores, size_t _class_count)
{
    size_t _idx;
    for (_idx = 0; _idx < _class_count; _idx++)
    {
        p_peak_scores[_idx] = -FLT_MAX;
    }
}

/**
 * @brief Update the running per-class peak scores using one patch of scores.
 * @param[in] p_patch_scores Float scores for the current patch.
 * @param[in,out] p_peak_scores Running peak scores to update.
 * @param[in] class_count Number of model output classes.
 */
static void yamnet_update_peak_scores(const float *p_patch_scores, float *p_peak_scores, size_t _class_count)
{
    size_t _idx;
    for (_idx = 0; _idx < _class_count; _idx++)
    {
        if (p_patch_scores[_idx] > p_peak_scores[_idx])
        {
            p_peak_scores[_idx] = p_patch_scores[_idx];
        }
    }
}

/**
 * @brief Extract the top-K class indices and scores from aggregated peak scores.
 * @param[in] p_peak_scores Aggregated peak-score array.
 * @param[in] class_count Number of elements in `p_peak_scores`.
 * @param[in] top_k Number of top results to return.
 * @param[out] p_top_indices Caller-owned output array of size at least `top_k`.
 * @param[out] p_top_scores Caller-owned output array of size at least `top_k`.
 */
static void yamnet_topk_from_peak_scores(const float *p_peak_scores, size_t _class_count, int _top_k,
                                         int *p_top_indices, float *p_top_scores)
{
    int _i;
    size_t _cls;

    for (_i = 0; _i < _top_k; _i++)
    {
        p_top_indices[_i] = -1;
        p_top_scores[_i]  = -FLT_MAX;
    }

    for (_cls = 0; _cls < _class_count; _cls++)
    {
        const float _score = p_peak_scores[_cls];
        int _insert_pos    = -1;

        for (_i = 0; _i < _top_k; _i++)
        {
            if (_score > p_top_scores[_i])
            {
                _insert_pos = _i;
                break;
            }
        }

        if (_insert_pos >= 0)
        {
            int _j;
            for (_j = _top_k - 1; _j > _insert_pos; _j--)
            {
                p_top_scores[_j]  = p_top_scores[_j - 1];
                p_top_indices[_j] = p_top_indices[_j - 1];
            }
            p_top_scores[_insert_pos]  = _score;
            p_top_indices[_insert_pos] = (int)_cls;
        }
    }
}

/**
 * @brief Run the full YAMNet post-processing pipeline on model outputs.
 * @param[in] p_quantized_outputs Pointer to concatenated int16 model outputs.
 * @param[in] num_patches Number of patches in the prediction.
 * @param[in] class_count Number of model output classes.
 * @param[in] top_k Number of top results to return.
 * @param[out] p_top_indices Caller-owned buffer for top-K class indices.
 * @param[out] p_top_scores Caller-owned buffer for top-K scores.
 */
void postprocess(const int16_t *p_quantized_outputs,
                 size_t         _num_patches,
                 size_t         _class_count,
                 int            _top_k,
                 int           *p_top_indices,
                 float         *p_top_scores)
{
    size_t _patch_idx;

    if ((NULL == p_quantized_outputs) || (NULL == p_top_indices) || (NULL == p_top_scores))
    {
        return;
    }
    if (_class_count > (size_t)MODEL_OUTPUT_CLASSES)
    {
        /* Static scratch is sized for MODEL_OUTPUT_CLASSES; reject anything
         * larger rather than overflowing. */
        return;
    }

    /* Step 1-2: dequantize each patch's output and fold into the class-wise
     * peak. */
    yamnet_reset_peak_scores(s_peak_scores, _class_count);

    for (_patch_idx = 0U; _patch_idx < _num_patches; _patch_idx++)
    {
        yamnet_dequantize_output_int16(&p_quantized_outputs[_patch_idx * _class_count],
                                       s_dequant_scratch,
                                       _class_count);
        yamnet_update_peak_scores(s_dequant_scratch, s_peak_scores, _class_count);
    }

    /* Step 3: top-K over the aggregated peak scores. */
    yamnet_topk_from_peak_scores(s_peak_scores,
                                 _class_count,
                                 _top_k,
                                 p_top_indices,
                                 p_top_scores);
}
