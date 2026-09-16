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
 * @file    postprocessing.h
 * @brief   MicroNet clip-level anomaly scoring from int8 model outputs.
 *
 * For each sliding window the model produces 8 int8 logits. The clip-level
 * anomaly score is the mean of per-window scores, where each window score is
 * the negative dequantized logit at the machine-ID target index (higher score
 * means more anomalous).
 */

#ifndef POSTPROCESSING_H
#define POSTPROCESSING_H

#include <stdint.h>

#include "model_metadata.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief  Compute the clip-level anomaly score from per-window int8 logits.
 *
 * @param[in]  p_outputs_int8    Concatenated per-window int8 outputs of length
 *                               _num_windows * OUTPUT_DIM.
 * @param[in]  _num_windows      Number of windows scored for this clip.
 * @param[in]  _machine_id       DCASE slider machine ID (0, 2, 4, or 6).
 * @param[out] p_clip_score      Mean anomaly score for the clip.
 *
 * @return  0 on success, -1 on error (null pointer, no windows, or unsupported
 *          machine ID).
 */
int32_t postprocess(const int8_t *p_outputs_int8,
                           int32_t           _num_windows,
                           int32_t           _machine_id,
                           float        *p_clip_score);

#ifdef __cplusplus
}
#endif

#endif /* POSTPROCESSING_H */
