/*
* Copyright (c) 2020 - 2026 Renesas Electronics Corporation and/or its affiliates
*
* SPDX-License-Identifier: BSD-3-Clause
*/
/**
 ******************************************************************************
 * @file    preprocessing.h
 * @brief   YOLO-Fastest 1.1 INT8 letterbox preprocessing - MCU portable, no heap.
 *
 * YOLO-Fastest preprocessing pipeline:
 *   1. Compute uniform scale = min(target_w/src_w, target_h/src_h)
 *   2. Resize to (new_w, new_h) with bilinear interpolation
 *   3. Pad with gray (114) to fill 320×320
 *   4. RGB channel order (model trained on RGB input)
 *   5. Quantize to int8: int8_val = pixel_value + (-128) = pixel_value - 128
 *      (input scale=0.00392157 ≈ 1/255, zero_point=-128)
 *
 * Output layout: HWC  [320 × 320 × 3]  int8
 ******************************************************************************
 */

#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include <stdint.h>

#define BILINEAR_BLEND_ONE               (1.0F)
#define PIXEL_CENTER_OFFSET              (0.5F)

#define CHANNEL_INDEX_RED                (0)
#define CHANNEL_INDEX_GREEN              (1)
#define CHANNEL_INDEX_BLUE               (2)

/* ── Letterbox parameters (needed by postprocessing to undo padding) ──── */
typedef struct
{
    float    scale;     /**< Uniform resize scale applied                    */
    int32_t  pad_x;     /**< Horizontal padding (pixels in model space)      */
    int32_t  pad_y;     /**< Vertical   padding (pixels in model space)      */
} letterbox_params_t;

/* Backward-compatible alias retained for existing call sites. */
typedef letterbox_params_t LetterboxParams;

/**
 * @brief  Letterbox-resize an RGB888 image for YOLO-Fastest INT8 input.
 *
 * @param[in]  p_source_image     Source image, HWC layout, uint8, RGB order, [0, 255]
 * @param[in]  source_width       Source image width in pixels
 * @param[in]  source_height      Source image height in pixels
 * @param[out] p_destination_image Output buffer, int8 HWC RGB (channel order matches model input).
 *                                 Must hold at least (dest_w × dest_h × 3) bytes.
 * @param[in]  destination_width  Target width  (320 for YOLO-Fastest)
 * @param[in]  destination_height Target height (320 for YOLO-Fastest)
 * @param[out] p_params           Filled with scale and padding for coordinate
 *                                unscaling in postprocessing.
 */
void preprocess(const uint8_t   *p_source_image,
                uint16_t         source_width,
                uint16_t         source_height,
                int8_t          *p_destination_image,
                uint16_t         destination_width,
                uint16_t         destination_height,
                letterbox_params_t *p_params);

#endif /* PREPROCESSING_H */
