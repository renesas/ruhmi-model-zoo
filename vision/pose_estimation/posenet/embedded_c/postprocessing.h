/**
 * @file    postprocessing.h
 * @brief   PoseNet output postprocessing declarations - multi-person
 *          PersonLab decoder from the int8 heatmap, offset, and
 *          displacement_fwd / displacement_bwd heads.
 *
 * Algorithm:
 *   1. Find local-max heatmap peaks whose score >= POSE_ROOT_SCORE_THRESHOLD
 *      and sort them by score (descending). These are root-keypoint
 *      candidates.
 *   2. For each candidate (highest score first, up to MAX_POSE_DETECTIONS
 *      accepted poses):
 *        a. Refine the root position with the offset head.
 *        b. NMS: skip if the refined root is within POSE_NMS_RADIUS_PX
 *           (model-input pixels) of an already-accepted pose's same-part
 *           keypoint.
 *        c. Walk POSE_CHAIN backward (using disp_bwd) and forward
 *           (using disp_fwd) to fill all remaining keypoints.
 *        d. Compute instance score as the mean of keypoint scores whose
 *           position is not within POSE_NMS_RADIUS_PX of the corresponding
 *           keypoint of any previously accepted pose. Keep if >= POSE_MIN_SCORE.
 *   3. Rescale all accepted poses from model-input pixels to source-image
 *      pixels using src_w / MODEL_INPUT_W, src_h / MODEL_INPUT_H.
 *
 * All int8 tensors are dequantized on-the-fly using the QUANT_SCALE /
 * QUANT_ZP constants defined in model_metadata.h; no intermediate float
 * buffers are allocated.
 *
 * @author  Renesas Electronics
 * @date    2026
 */

/*
 * Copyright (C) 2026 Renesas Electronics Corporation. All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef POSTPROCESSING_H
#define POSTPROCESSING_H

#include <stdint.h>
#include "model_metadata.h"

/* ------------------------------------------------------------------------- */
/* Candidate root peaks (from heatmap local-max NMS)                          */
/* ------------------------------------------------------------------------- */
#define MAX_PEAK_CANDIDATES  128U

/**
 * @brief One decoded keypoint in source-image coordinates.
 *
 * @var pose_keypoint_t::part_id   Keypoint index 0..16 (see g_pose_labels).
 * @var pose_keypoint_t::x         Column in source-image pixels.
 * @var pose_keypoint_t::y         Row    in source-image pixels.
 * @var pose_keypoint_t::score     Sigmoid confidence in [0, 1]. Zero means the
 *                                 chain walk never reached this keypoint.
 */
typedef struct
{
    int32_t part_id;
    float   x;
    float   y;
    float   score;
} pose_keypoint_t;

typedef struct
{
    int16_t gy;
    int16_t gx;
    int16_t kp;
    int16_t _pad;
    float   score;
} peak_candidate_t;

/**
 * @brief Decode up to ``_max_poses`` persons from PoseNet's four int8 heads.
 *
 * @param[in]  p_heatmap_q      [1,17,17,17] int8, dequant -> sigmoid [0,1].
 * @param[in]  p_offset_q       [1,17,17,34] int8, channels 0..16 = dy per kp,
 *                              17..33 = dx per kp (input-pixel units).
 * @param[in]  p_disp_fwd_q     [1,17,17,32] int8, channels 0..15 = dy per edge,
 *                              16..31 = dx per edge (input-pixel units).
 * @param[in]  p_disp_bwd_q     Same layout as disp_fwd but for backward walk.
 * @param[in]  _source_width    Original image width (pixels) for rescaling.
 * @param[in]  _source_height   Original image height (pixels) for rescaling.
 * @param[out] p_keypoints_out  Output pose keypoint sets;
 *                              ``p_keypoints_out[i][k]`` is the k-th keypoint
 *                              of the i-th detected pose, in source-image
 *                              pixel coordinates.
 * @param[out] p_pose_scores    Instance score per accepted pose (length
 *                              ``_max_poses``). Indices [0..return) valid.
 * @param[in]  _max_poses       Capacity of the output arrays
 *                              (typically ::MAX_POSE_DETECTIONS).
 *
 * @return Number of poses actually decoded (0..``_max_poses``).
 */
int postprocess(const int8_t   *p_heatmap_q,
                const int8_t   *p_offset_q,
                const int8_t   *p_disp_fwd_q,
                const int8_t   *p_disp_bwd_q,
                int              _source_width,
                int              _source_height,
                pose_keypoint_t (*p_keypoints_out)[MODEL_NUM_KEYPOINTS],
                float           *p_pose_scores,
                int              _max_poses);

#endif /* POSTPROCESSING_H */
