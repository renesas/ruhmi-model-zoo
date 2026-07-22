/*
* Copyright (c) 2020 - 2026 Renesas Electronics Corporation and/or its affiliates
*
* SPDX-License-Identifier: BSD-3-Clause
*/
/**
 ******************************************************************************
 * @file    postprocessing.h
 * @brief   YOLO-Fastest 1.1 INT8 anchor-based decoding + per-class NMS for MCU.
 *
 * Decodes two raw anchor-based YOLO heads:
 *   Head 0: [1, 10, 10, 255]  (stride=32, large objects)
 *   Head 1: [1, 20, 20, 255]  (stride=16, small objects)
 *
 * Each cell: 3 anchors × 85 channels = 255
 *   85 = 4 (tx,ty,tw,th) + 1 (objectness) + 80 (class logits)
 *
 * Detection layout:
 *   [x1, y1, x2, y2]  — absolute pixel coords in the ORIGINAL image
 *   score              — sigmoid(obj) × sigmoid(class) confidence
 *   cls_id             — COCO class index [0, 79]
 ******************************************************************************
 */

#ifndef POSTPROCESSING_H
#define POSTPROCESSING_H

#include <stdint.h>
#include "preprocessing.h"          /* LetterboxParams */
#include "../common/model_metadata.h"

#define SIGMOID_ONE_F                    (1.0F)
#define BOX_HALF_FACTOR                  (0.5F)

#define YOLO_BOX_TX_INDEX                (0U)
#define YOLO_BOX_TY_INDEX                (1U)
#define YOLO_BOX_TW_INDEX                (2U)
#define YOLO_BOX_TH_INDEX                (3U)
#define YOLO_OBJECTNESS_INDEX            (4U)
#define YOLO_CLASS_LOGITS_START_INDEX    (5U)

#define IOU_EPSILON_F                    (1e-6F)
#define SUPPRESSED_SCORE_F               (-1.0F)

#define MAX_RAW_DETS                     (MODEL_TOTAL_ANCHORS)

/* ── Max detections buffer (post-NMS) ────────────────────────────────── */
#define MAX_DETECTIONS          POSTPROC_MAX_DETS

/* ── Detection result structure ──────────────────────────────────────── */
typedef struct
{
    float    x1;        /**< Top-left x (original image pixels)              */
    float    y1;        /**< Top-left y (original image pixels)              */
    float    x2;        /**< Bottom-right x (original image pixels)          */
    float    y2;        /**< Bottom-right y (original image pixels)          */
    float    score;     /**< sigmoid(obj) × sigmoid(class) confidence        */
    uint32_t cls_id;    /**< COCO class index [0, 79]                        */
} Detection_t;

/**
 * @brief  Decode raw YOLO-Fastest INT8 dual-head output → filtered Detection_t array.
 *
 * Pipeline:
 *   1. Dequantize int8 output to float32 per head
 *   2. For each grid cell, for each anchor:
 *      a. sigmoid(objectness) → early exit if below threshold
 *      b. sigmoid(tx,ty) + grid offset → centre, exp(tw,th) × anchor → size
 *      c. sigmoid(class logits) → argmax class
 *      d. score = obj × class_conf
 *   3. Undo letterbox padding → original image coordinates
 *   4. Per-class greedy NMS
 *
 * @param[in]  p_raw_head0   Flat int8 array [10 × 10 × 255] (stride-32 head).
 * @param[in]  p_raw_head1   Flat int8 array [20 × 20 × 255] (stride-16 head).
 * @param[in]  p_params      Letterbox parameters from preprocess().
 * @param[in]  orig_w        Original image width  (pixels).
 * @param[in]  orig_h        Original image height (pixels).
 * @param[in]  score_thresh  Minimum detection score to keep.
 * @param[in]  nms_thresh    IoU threshold for per-class NMS.
 * @param[out] p_out_dets    Output array; must hold at least MAX_DETECTIONS entries.
 * @return                   Number of detections written to p_out_dets.
 */
int32_t postprocess(const int8_t           *p_raw_head0,
                    const int8_t           *p_raw_head1,
                    const letterbox_params_t *p_params,
                    uint32_t               orig_w,
                    uint32_t               orig_h,
                    float                  score_thresh,
                    float                  nms_thresh,
                    Detection_t           *p_out_dets);

#endif /* POSTPROCESSING_H */
