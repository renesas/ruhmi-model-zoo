/**********************************************************************************************************************
 * Copyright (c) 2020 - 2026 Renesas Electronics Corporation and/or its affiliates
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * DISCLAIMER
 * This software is supplied by Renesas Electronics Corporation and is only intended for use with Renesas products.
 * No other uses are authorized.
 */
/**********************************************************************************************************************
 * File Name    : postprocessing.c
 * Description  : MediaPipe face-landmark output postprocessing - dequantises
 *                the two INT8 output tensors, evaluates the face-presence
 *                sigmoid, and rescales the 468 (x, y, z) landmarks from the
 *                192x192 model-input pixel frame back to the source-image
 *                pixel frame.
 **********************************************************************************************************************/

#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "postprocessing.h"
#include "model_metadata.h"


/**
 * @brief Sigmoid: 1 / (1 + exp(-x)). Used to gate the face-presence logit.
 *
 * @param[in]  _input_value  Pre-sigmoid float value.
 * @return float Sigmoid value in [0, 1].
 */
static float sigmoid_f(float _input_value)
{
    return 1.0f / (1.0f + expf(-_input_value));
}


/**
 * @brief Decode a single face from the face_landmark model output heads.
 *
 * @param[in]  p_raw_landmarks  Pointer to landmark tensor [1,1,1,1404] int8.
 * @param[in]  p_raw_face_flag  Pointer to face-flag tensor [1,1,1,1] int8.
 * @param[in]  _source_width     Original image width  (pixels) for rescaling.
 * @param[in]  _source_height    Original image height (pixels) for rescaling.
 * @param[out] p_result         Output aggregate face result.
 */
void postprocess(const int8_t   *p_raw_landmarks,
                 const int8_t   *p_raw_face_flag,
                 int32_t         _source_width,
                 int32_t         _source_height,
                 face_result_t  *p_result)
{
    if ((NULL == p_raw_landmarks) || (NULL == p_raw_face_flag) || (NULL == p_result))
    {
        return;
    }

    if ((0 >= _source_width) || (0 >= _source_height))
    {
        return;
    }

    /* --- Face presence gating ------------------------------------------- */
    const float _face_flag_scale = FACE_FLAG_QUANT_SCALE;
    const float _face_flag_zero_point = (float)FACE_FLAG_QUANT_ZERO_POINT;
    const float _face_flag_logit = _face_flag_scale * ((float)p_raw_face_flag[0] - _face_flag_zero_point);
    const float _face_flag_probability = sigmoid_f(_face_flag_logit);

    p_result->face_probability = _face_flag_probability;
    p_result->face_detected    = (FACE_DETECTION_THRESHOLD <= _face_flag_probability);

    /* --- Landmark dequantisation + rescale ------------------------------ */
    /* Pre-compute the rescale factors (model-input pixel -> source pixel). */
    const float _resize_scale_x = (float)_source_width  / (float)MODEL_INPUT_W;
    const float _resize_scale_y = (float)_source_height / (float)MODEL_INPUT_H;

    const float _landmark_scale  = LANDMARK_QUANT_SCALE;
    const float _landmark_zero_point = (float)LANDMARK_QUANT_ZERO_POINT;

    int32_t _landmark_index;
    for (_landmark_index = 0; (int32_t)MODEL_NUM_LANDMARKS > _landmark_index; _landmark_index++)
    {
        const int32_t _landmark_base_offset = _landmark_index * (int32_t)MODEL_LANDMARK_DIMS;

        const float _input_x_coordinate = _landmark_scale * ((float)p_raw_landmarks[_landmark_base_offset + 0] - _landmark_zero_point);
        const float _input_y_coordinate = _landmark_scale * ((float)p_raw_landmarks[_landmark_base_offset + 1] - _landmark_zero_point);
        const float _input_z_coordinate = _landmark_scale * ((float)p_raw_landmarks[_landmark_base_offset + 2] - _landmark_zero_point);

        face_landmark_t *p_landmark = &p_result->landmarks[_landmark_index];
        p_landmark->part_id = _landmark_index;
        p_landmark->x       = _input_x_coordinate * _resize_scale_x;
        p_landmark->y       = _input_y_coordinate * _resize_scale_y;
        p_landmark->z       = _input_z_coordinate * _resize_scale_x;
    }
}
