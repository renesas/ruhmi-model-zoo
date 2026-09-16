/**
 * @file    preprocessing.h
 * @brief   PoseNet input preprocessing declarations - 2-tap bilinear resize
 *          to MODEL_INPUT_H x MODEL_INPUT_W, [-1, 1] normalisation, and
 *          int8 quantization.
 *
 * @author  Renesas Electronics
 * @date    2026
 */

/*
 * Copyright (C) 2026 Renesas Electronics Corporation. All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include <stdint.h>

/**
 * @brief Convert a source RGB image into the PoseNet INT8 input tensor.
 *
 * Pipeline:
 *
 *   1. 2-tap separable bilinear resize from (src_w, src_h) ->
 *      (MODEL_INPUT_W, MODEL_INPUT_H) with half-pixel-centre coordinates:
 *
 *          sx = (dx + 0.5) * src_w / dst_w - 0.5
 *          sy = (dy + 0.5) * src_h / dst_h - 0.5
 *
 *      Edges are replicated (clamp to [0, src - 1]).  No anti-aliasing pass
 *      is applied for downscaling.
 *
 *   2. Normalise to [-1, 1]:
 *          f = pixel * (2.0f / 255.0f) - 1.0f
 *
 *   3. Quantize to int8 (INPUT_QUANT_SCALE / INPUT_QUANT_ZP):
 *          q = clamp(round(f / INPUT_QUANT_SCALE) + INPUT_QUANT_ZP, -128, 127)
 *
 * @note  No aspect-preserving short-edge step is performed; the image is
 *        squashed to MODEL_INPUT_W x MODEL_INPUT_H.
 *
 * @param p_source_image_hwc  Source uint8 image in HWC RGB layout.
 * @param _source_width       Source image width in pixels.
 * @param _source_height      Source image height in pixels.
 * @param p_input_tensor      Destination int8 tensor (HWC, MODEL_INPUT_SIZE
 *                            elements) quantized per INPUT_QUANT_SCALE / INPUT_QUANT_ZP.
 */
void preprocess(const uint8_t *p_source_image_hwc,
                int _source_width,
                int _source_height,
                int8_t *p_input_tensor);

#endif /* PREPROCESSING_H */
