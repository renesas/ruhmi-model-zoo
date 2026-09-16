/*
* Copyright (c) 2020 - 2026 Renesas Electronics Corporation and/or its affiliates
*
* SPDX-License-Identifier: BSD-3-Clause
*/
/**********************************************************************************************************************
 * File Name    : postprocessing.h
 * Description  : YAMNet post-processing pipeline - dequantize, peak aggregate,
 *                and top-K selection.
 *********************************************************************************************************************/

#ifndef POSTPROCESSING_H
#define POSTPROCESSING_H

#include <stdint.h>
#include <stddef.h>

/* Full YAMNet postprocessing pipeline: quantized model outputs -> top-K.
 *
 * The main inference driver calls this single entry point after all patch
 * invocations have completed. The steps mirror the Python reference
 * (YamnetTFLite.score_patches, int16 path):
 *
 *   1. Per-patch dequantize:      y = (q - zp) * scale
 *   2. Class-wise max across all patches (peak aggregation)
 *   3. Top-K by descending peak score
 *
/* Parameters:
 *   p_quantized_outputs : concatenated int16 model outputs, one row per patch,
 *                         laid out as
 *                         p_quantized_outputs[patch_idx * class_count + class_idx].
 *   num_patches         : number of patches contributing to this prediction.
 *   class_count         : model output classes.
 *   top_k               : number of predictions to return.
 *   p_top_indices       : caller-owned output, size >= top_k int.
 *   p_top_scores        : caller-owned output, size >= top_k float.
 */

/**
 * @brief Dequantize an int16 model output buffer into float values.
 * @param[in] p_src_q_output Pointer to the quantized int16 model output.
 * @param[out] p_dst_output Pointer to the destination float buffer.
 * @param[in] _elem_count Number of elements to process.
 */
void yamnet_dequantize_output_int16(const int16_t *p_src_q_output,
                                    float         *p_dst_output,
                                    size_t         _elem_count);

/**
 * @brief Run the full YAMNet post-processing pipeline on model outputs.
 * @param[in] p_quantized_outputs Pointer to concatenated int16 model outputs.
 * @param[in] _num_patches Number of patches in the prediction.
 * @param[in] _class_count Number of model output classes.
 * @param[in] _top_k Number of top results to return.
 * @param[out] p_top_indices Caller-owned buffer for top-K class indices.
 * @param[out] p_top_scores Caller-owned buffer for top-K scores.
 */
void postprocess(const int16_t *p_quantized_outputs,
                 size_t         _num_patches,
                 size_t         _class_count,
                 int            _top_k,
                 int           *p_top_indices,
                 float         *p_top_scores);

#endif /* POSTPROCESSING_H */
