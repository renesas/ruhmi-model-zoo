/*
 * The Renesas Electronics Corporation
 * DISCLAIMER
 * This software is supplied by Renesas Electronics Corporation and is only intended for use with Renesas products. No
 * other uses are authorized. This software is owned by Renesas Electronics Corporation and is protected under all
 * applicable laws, including copyright laws.
 * THIS SOFTWARE IS PROVIDED 'AS IS' AND RENESAS MAKES NO WARRANTIES REGARDING
 * THIS SOFTWARE, WHETHER EXPRESS, IMPLIED OR STATUTORY, INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. ALL SUCH WARRANTIES ARE EXPRESSLY DISCLAIMED. TO THE MAXIMUM
 * EXTENT PERMITTED NOT PROHIBITED BY LAW, NEITHER RENESAS ELECTRONICS CORPORATION NOR ANY OF ITS AFFILIATED COMPANIES
 * SHALL BE LIABLE FOR ANY DIRECT, INDIRECT, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES FOR ANY REASON RELATED TO THIS
 * SOFTWARE, EVEN IF RENESAS OR ITS AFFILIATES HAVE BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
 * Renesas reserves the right, without notice, to make changes to this software and to discontinue the availability of
 * this software. By using this software, you agree to the additional terms and conditions found by accessing the
 * following link:
 * http://www.renesas.com/disclaimer
 */

/*
* Copyright (c) 2020 - 2026 Renesas Electronics Corporation and/or its affiliates
*
* SPDX-License-Identifier: BSD-3-Clause
*/
/**
 ******************************************************************************
 * @file    postprocessing.h
 * @brief   NanoDet output postprocessing
 *
 * @author  Renesas Electronics
 * @date    2026
 * @version 1.0
 ******************************************************************************
 */

#ifndef POSTPROCESSING_H
#define POSTPROCESSING_H

#include <stdint.h>
#include "preprocessing.h"          /* LetterboxParams */
#include "model_metadata.h"

#define REG_MAX_PLUS_ONE         ((MODEL_REG_MAX) + 1)
#define NANODET_BBOX_DIMS        (4)

/* ── Detection result structure ──────────────────────────────────────── */
typedef struct
{
    float    _x1;        /**< Top-left x (original image pixels)              */
    float    _y1;        /**< Top-left y (original image pixels)              */
    float    _x2;        /**< Bottom-right x (original image pixels)          */
    float    _y2;        /**< Bottom-right y (original image pixels)          */
    float    _score;     /**< objectness × class confidence                   */
    uint32_t _cls_id;    /**< COCO class index [0, 79]                        */
} Detection_t;

/**
 * @brief  Dequantize and decode raw NanoDet int8 output into a filtered Detection_t array.
 *
 * @param[in]  p_raw_output            Flat int8 array [NUM_ANCHORS × OUTPUT_CHANNELS].
 * @param[in]  p_letterbox_params      Warp-resize parameters from preprocess().
 * @param[in]  _original_width_pixels   Original image width in pixels.
 * @param[in]  _original_height_pixels  Original image height in pixels.
 * @param[in]  _score_threshold         Minimum detection score to keep.
 * @param[in]  _nms_iou_threshold       IoU threshold for per-class NMS.
 * @param[out] p_output_detections     Output array; must hold at least MAX_DETECTIONS entries.
 * @return                              Number of detections written to p_output_detections.
 */
int32_t postprocess(const int8_t            *p_raw_output,
                    const letterbox_params_t *p_letterbox_params,
                    uint32_t                _original_width_pixels,
                    uint32_t                _original_height_pixels,
                    float                   _score_threshold,
                    float                   _nms_iou_threshold,
                    Detection_t            *p_output_detections);

#endif /* POSTPROCESSING_H */
