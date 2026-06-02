/*
* Copyright (c) 2020 - 2026 Renesas Electronics Corporation and/or its affiliates
*
* SPDX-License-Identifier: BSD-3-Clause
*/
/**
 ******************************************************************************
 * @file    postprocessing.h
 * @brief   YOLOX-Tiny INT8 anchor-free decoding + per-class NMS for MCU.
 *
 * Decodes the raw (1029 × 85) INT8 output into a list of bounding boxes.
 * Output is dequantized: float = (int8 - zero_point) * scale
 *
 * Detection layout:
 *   [x1, y1, x2, y2]  — absolute pixel coords in the ORIGINAL image
 *   score              — objectness × class confidence (dequantized)
 *   cls_id             — COCO class index [0, 79]
 ******************************************************************************
 */

#ifndef POSTPROCESSING_H
#define POSTPROCESSING_H

#include <stdint.h>
#include "preprocessing.h"          /* LetterboxParams */
#include "../common/model_metadata.h"

/* ── Model output constants (sourced from model_metadata.h) ──────────── */
#define NUM_ANCHORS             MODEL_NUM_ANCHORS
#define NUM_CLASSES             MODEL_NUM_CLASSES
#define OUTPUT_CHANNELS         MODEL_OUTPUT_CHANNELS

/* ── FPN stride / grid sizes (sourced from model_metadata.h) ─────────── */
#define STRIDE_S8               MODEL_STRIDE_S8
#define STRIDE_S16              MODEL_STRIDE_S16
#define STRIDE_S32              MODEL_STRIDE_S32
#define GRID_S8                 MODEL_GRID_S8
#define GRID_S16                MODEL_GRID_S16
#define GRID_S32                MODEL_GRID_S32

/* ── Post-processing thresholds (sourced from model_metadata.h) ──────── */
#define SCORE_THRESH            POSTPROC_SCORE_THRESH
#define NMS_IOU_THRESH          POSTPROC_NMS_THRESH

/* ── Max detections buffer (post-NMS) ────────────────────────────────── */
#define MAX_DETECTIONS          POSTPROC_MAX_DETS

/* ── Detection result structure ──────────────────────────────────────── */
typedef struct
{
    float    x1;        /**< Top-left x (original image pixels)              */
    float    y1;        /**< Top-left y (original image pixels)              */
    float    x2;        /**< Bottom-right x (original image pixels)          */
    float    y2;        /**< Bottom-right y (original image pixels)          */
    float    score;     /**< objectness × class confidence                   */
    uint32_t cls_id;    /**< COCO class index [0, 79]                        */
} Detection_t;

/**
 * @brief  Decode raw YOLOX INT8 output → filtered Detection_t array.
 *
 * Pipeline:
 *   1. Dequantize int8 output to float32
 *   2. Build grid offsets + strides (once, cached)
 *   3. Score-filter: score = obj × cls[argmax] > thresh
 *   4. Anchor-free box decode: cx=(off+grid)*stride, w=exp(log)*stride
 *   5. Undo letterbox padding → original image coordinates
 *   6. Per-class greedy NMS
 *
 * @param[in]  raw_output    Flat int8 array [NUM_ANCHORS × OUTPUT_CHANNELS].
 * @param[in]  params        Letterbox parameters from preprocess().
 * @param[in]  orig_w        Original image width  (pixels).
 * @param[in]  orig_h        Original image height (pixels).
 * @param[in]  score_thresh  Minimum detection score to keep.
 * @param[in]  nms_thresh    IoU threshold for per-class NMS.
 * @param[out] out_dets      Output array; must hold at least MAX_DETECTIONS entries.
 * @return                   Number of detections written to out_dets.
 */
int32_t postprocess(const int8_t           *raw_output,
                    const letterbox_params_t *p_params,
                    uint32_t               orig_w,
                    uint32_t               orig_h,
                    float                  score_thresh,
                    float                  nms_thresh,
                    Detection_t           *p_out_dets);

#endif /* POSTPROCESSING_H */
