/**********************************************************************************************************************
 * Copyright (c) 2020 - 2026 Renesas Electronics Corporation and/or its affiliates
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * DISCLAIMER
 * This software is supplied by Renesas Electronics Corporation and is only intended for use with Renesas products.
 * No other uses are authorized.
 *********************************************************************************************************************/
/**********************************************************************************************************************
 * File Name    : postprocessing.h
 * Description  : MediaPipe face-landmark output postprocessing declarations -
 *                dequantise the two INT8 output tensors, evaluate the face
 *                presence sigmoid, and project the 468 (x, y, z) landmarks
 *                from the 192x192 model-input coordinate frame back to the
 *                original source-image pixel frame.
 *
 *                Decoder model (single face, no rotation, no letterbox):
 *                  face_flag_f = FACE_FLAG_QUANT_SCALE * (face_flag_i8 - FACE_FLAG_QUANT_ZERO_POINT)
 *                  face_prob   = 1 / (1 + exp(-face_flag_f))
 *                  for k in 0..467:
 *                      x_f = LANDMARK_QUANT_SCALE * (raw[k*3 + 0] - LANDMARK_QUANT_ZERO_POINT)
 *                      y_f = LANDMARK_QUANT_SCALE * (raw[k*3 + 1] - LANDMARK_QUANT_ZERO_POINT)
 *                      z_f = LANDMARK_QUANT_SCALE * (raw[k*3 + 2] - LANDMARK_QUANT_ZERO_POINT)
 *                      x_px = x_f * (src_w / MODEL_INPUT_W)
 *                      y_px = y_f * (src_h / MODEL_INPUT_H)
 *                      z_px = z_f * (src_w / MODEL_INPUT_W)  (same scale as x)
 **********************************************************************************************************************/

#ifndef POSTPROCESSING_H
#define POSTPROCESSING_H

#include <stdint.h>
#include <stdbool.h>

/**
 * @brief One decoded face landmark in source-image coordinates.
 *
 * @var face_landmark_t::part_id  Landmark index in [0, MODEL_NUM_LANDMARKS - 1].
 * @var face_landmark_t::x        Column in source-image pixels.
 * @var face_landmark_t::y        Row    in source-image pixels.
 * @var face_landmark_t::z        Depth (same axis-scale as x).
 */
typedef struct
{
    int32_t part_id;
    float   x;
    float   y;
    float   z;
} face_landmark_t;

/**
 * @brief Aggregate result for one decoded face.
 *
 * @var face_result_t::face_probability  Sigmoid probability of the face-flag head.
 * @var face_result_t::face_detected     face_probability >= FACE_DETECTION_THRESHOLD.
 * @var face_result_t::landmarks         MODEL_NUM_LANDMARKS decoded landmarks.
 */
typedef struct
{
    float           face_probability;
    bool            face_detected;
    face_landmark_t landmarks[468];   /* MODEL_NUM_LANDMARKS - kept explicit to avoid header cycle */
} face_result_t;

/**
 * @brief Decode a single face from the tflite face_landmark model heads.
 *
 * @param[in]  p_raw_landmarks  MODEL_LANDMARK_SIZE  int8 elements (flattened
 *                               (x0,y0,z0,x1,y1,z1,...) in 192x192 pixels).
 * @param[in]  p_raw_face_flag  1 int8 element (pre-sigmoid face-presence logit).
 * @param[in]  _source_width     Original image width  (pixels) for rescaling.
 * @param[in]  _source_height    Original image height (pixels) for rescaling.
 * @param[out] p_result         Output face result, filled with landmark
 *                               coordinates in source-image pixel space and
 *                               face_probability/face_detected flags.
 */
void postprocess(const int8_t   *p_raw_landmarks,
                 const int8_t   *p_raw_face_flag,
                 int32_t         _source_width,
                 int32_t         _source_height,
                 face_result_t  *p_result);

#endif /* POSTPROCESSING_H */
