/*
 * Copyright (c) 2020 - 2026 Renesas Electronics Corporation and/or its affiliates.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * DISCLAIMER
 * This software is supplied by Renesas Electronics Corporation and is only
 * intended for use with Renesas products. No other uses are authorized.
 * This software is owned by Renesas Electronics Corporation and is protected
 * under all applicable laws, including copyright laws.
 *
 * THIS SOFTWARE IS PROVIDED "AS IS" AND RENESAS MAKES NO WARRANTIES REGARDING
 * THIS SOFTWARE, WHETHER EXPRESS, IMPLIED OR STATUTORY, INCLUDING BUT NOT
 * LIMITED TO WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE
 * AND NON-INFRINGEMENT. ALL SUCH WARRANTIES ARE EXPRESSLY DISCLAIMED.
 */

/**
 * @file    postprocessing.c
 * @brief   MicroNet clip-level anomaly scoring from int8 model outputs.
 */

#include "postprocessing.h"

#include <stddef.h>

/**
 * @brief  Map a DCASE slider machine ID to its output logit index.
 *
 * @param[in]  machine_id  Slider machine ID (0, 2, 4, or 6).
 * @return  Output index [0, 3], or -1 if the machine ID is unsupported.
 */
static int32_t model_machine_id_to_output_index(int32_t _machine_id)
{
    switch (_machine_id)
    {
        case 0: return 0;
        case 2: return 1;
        case 4: return 2;
        case 6: return 3;
        default: return -1;
    }
}

/**
 * @brief  Compute the clip-level anomaly score from per-window int8 logits.
 *
 * @param[in]  p_outputs_int8    Concatenated per-window int8 outputs of length
 *                               _num_windows * OUTPUT_DIM.
 * @param[in]  _num_windows      Number of windows scored for this clip.
 * @param[in]  _machine_id       DCASE slider machine ID (0, 2, 4, or 6).
 * @param[out] p_clip_score      Mean anomaly score for the clip.
 * @return  0 on success, -1 on error (null pointer, no windows, or unsupported
 *          machine ID).
 */
int32_t postprocess(const int8_t *p_outputs_int8,
                           int32_t           _num_windows,
                           int32_t           _machine_id,
                           float        *p_clip_score)
{
    if ((NULL == p_outputs_int8) || (NULL == p_clip_score) || (0 >= _num_windows))
    {
        return -1;
    }

    int32_t _target_index = model_machine_id_to_output_index(_machine_id);
    if ((0 > _target_index) || (OUTPUT_DIM <= _target_index))
    {
        return -1;
    }

    float _score_sum = 0.0f;

    for (int32_t _w = 0; _w < _num_windows; _w++)
    {
        int32_t _output_index    = (_w * OUTPUT_DIM) + _target_index;
        int32_t _quantized_logit = (int32_t)p_outputs_int8[_output_index];

        /* Dequantize: logit = scale * (q - zero_point). */
        float _logit = MODEL_OUTPUT_SCALE *
                       ((float)_quantized_logit - (float)MODEL_OUTPUT_ZP);

        /* Higher score => more anomalous. */
        _score_sum += -_logit;
    }

    *p_clip_score = _score_sum / (float)_num_windows;
    return 0;
}
