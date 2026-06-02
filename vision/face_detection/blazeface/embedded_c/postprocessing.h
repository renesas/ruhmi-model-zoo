/**********************************************************************************************************************
 * Copyright (c) 2020 - 2026 Renesas Electronics Corporation and/or its affiliates
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * File Name    : postprocessing.h
 * Description  : Public API for BlazeFace postprocessing.
 *********************************************************************************************************************/

#ifndef POSTPROCESSING_H
#define POSTPROCESSING_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Detection result ────────────────────────────────────────────────── */

#include "../common/model_metadata.h"

/** Maximum number of detections returned after NMS (sourced from model_metadata.h). */
#define BF_MAX_DETECTIONS  POSTPROC_MAX_DETS

/**
 * One face detection in normalised coordinates [0, 1].
 */
typedef struct
{
    float ymin;                     /**< Bounding box y minimum          */
    float xmin;                     /**< Bounding box x minimum          */
    float ymax;                     /**< Bounding box y maximum          */
    float xmax;                     /**< Bounding box x maximum          */
    float kp[6][2];                 /**< Keypoints [k][0]=x [k][1]=y     */
    float score;                    /**< Face confidence after sigmoid   */
} bf_detection_t;

/**
 * Collection of detections returned by postprocess().
 */
typedef struct
{
    bf_detection_t dets[BF_MAX_DETECTIONS];
    int32_t        n;
} bf_detections_t;

/* Backward-compatible aliases used by existing call sites. */
typedef bf_detection_t  BfDetection;
typedef bf_detections_t BfDetections;

/* ── Main post-processing entry point ───────────────────────────────── */

/**
 * @brief Execute full INT8 postprocessing pipeline.
 * @param[in] p_scores_stride8_quantized INT8 score tensor for stride-8 anchors.
 * @param[in] p_scores_stride16_quantized INT8 score tensor for stride-16 anchors.
 * @param[in] p_boxes_stride8_quantized INT8 box tensor for stride-8 anchors.
 * @param[in] p_boxes_stride16_quantized INT8 box tensor for stride-16 anchors.
 * @param[in] p_anchors Anchor array [896][4] in [cx, cy, w, h] format.
 * @param[in] score_threshold Minimum confidence threshold.
 * @param[in] nms_iou_threshold IoU threshold used by weighted NMS.
 * @param[out] p_output_detections Output detections structure.
 * @return void
 */
void postprocess(const int8_t  *p_scores_stride8_quantized,
                 const int8_t  *p_scores_stride16_quantized,
                 const int8_t  *p_boxes_stride8_quantized,
                 const int8_t  *p_boxes_stride16_quantized,
                 const float    p_anchors[896][4],
                 float          score_threshold,
                 float          nms_iou_threshold,
                 BfDetections  *p_output_detections);

#ifdef __cplusplus
}
#endif

#endif /* POSTPROCESSING_H */
