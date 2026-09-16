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
 * @file    preprocessing.h
 * @brief   NanoDet input preprocessing - bilinear resize to model input dimensions.
 *
 * @author  Renesas Electronics
 * @date    2026
 * @version 1.0
 ******************************************************************************
 */

#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include <stdint.h>

/* ── Letterbox parameters (needed by postprocessing to undo padding) ──── */
typedef struct
{
    float    _scale_w;   /**< Warp-resize width scale: dst_w / src_w          */
    float    _scale_h;   /**< Warp-resize height scale: dst_h / src_h         */
    int32_t  _pad_x;     /**< Kept for compatibility; always 0 in warp mode   */
    int32_t  _pad_y;     /**< Kept for compatibility; always 0 in warp mode   */
} letterbox_params_t;

/* Backward-compatible alias retained for existing call sites. */
typedef letterbox_params_t letterbox_params_alias_t;

/**
 * @brief  Warp-resize, normalize, and quantize an RGB888 image for NanoDet int8 input.
 *
 * @param[in]  p_source_image         Source image, HWC layout, uint8, RGB888, [0, 255]
 * @param[in]  _source_width           Source image width in pixels
 * @param[in]  _source_height          Source image height in pixels
 * @param[out] p_destination_image    Output buffer, int8 HWC in model input domain.
 * @param[in]  _destination_width      Target width
 * @param[in]  _destination_height     Target height
 * @param[out] p_letterbox_params     Filled with _scale_w/_scale_h for coord restore.
 */
void preprocess(const uint8_t      *p_source_image,
                uint16_t            _source_width,
                uint16_t            _source_height,
                int8_t             *p_destination_image,
                uint16_t            _destination_width,
                uint16_t            _destination_height,
                letterbox_params_t *p_letterbox_params);

#endif /* PREPROCESSING_H */
