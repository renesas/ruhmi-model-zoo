/*
 * Copyright (c) 2020 - 2026 Renesas Electronics Corporation and/or its affiliates
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * DISCLAIMER
 * This software is supplied by Renesas Electronics Corporation and is only intended for use with Renesas products.
 * No other uses are authorized.
 */
/**********************************************************************************************************************
 * File Name    : preprocessing.h
 * Description  : MediaPipe face-landmark input preprocessing declarations -
 *                2-tap bilinear resize (half-pixel-centre, edge replicate) to
 *                192x192, per-pixel normalisation to [0, 1] and INT8
 *                quantisation using the tflite input tensor's quantisation
 *                parameters (scale = 2/255, zero_point = -1).
 *
 *                Source layout is RGB888 (RA8P1 camera produces RGB888).
 **********************************************************************************************************************/

#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include "model_metadata.h"

/* Largest output-axis extent (MODEL_INPUT_H == MODEL_INPUT_W for face landmark). */
#define MAX_DST_DIM              ((MODEL_INPUT_H > MODEL_INPUT_W) ? MODEL_INPUT_H : MODEL_INPUT_W)

/* Compile-time normalisation constant. Multiplication (not division) keeps
 * the float32 rounding order aligned with the reference preprocessing flow. */
#define INV_255_F                (1.0f / 255.0f)
#define INV_INPUT_QUANT_SCALE    (1.0f / INPUT_QUANT_SCALE)

#include <stdint.h>

/**
 * @brief Convert a source RGB888 image into the face-landmark INT8 input tensor.
 *
 * Pipeline (bit-exact mirror of the offline reference preprocessing flow):
 *
 *   1. 2-tap separable bilinear resize from (src_w, src_h) -> (192, 192) with
 *      half-pixel-centre coordinates:
 *
 *          sx = (dx + 0.5) * src_w / dst_w - 0.5
 *          sy = (dy + 0.5) * src_h / dst_h - 0.5
 *
 *      Edges are replicated (clamp to [0, src - 1]).  No anti-aliasing box
 *      filter is applied for downscaling.
 *
 *   2. Normalise to [0, 1] float32:
 *          f = pixel * (1.0f / 255.0f)
 *
 *   3. Quantise to INT8 using the tflite input tensor parameters:
 *          q = round( f / INPUT_QUANT_SCALE + INPUT_QUANT_ZERO_POINT )
 *          q clipped to [-128, 127]
 *
 * @note  No aspect-preserving letterbox is performed; the ROI (assumed to
 *        already be the cropped face) is stretched to MODEL_INPUT_W x MODEL_INPUT_H.
 *
 * @param[in]  p_source_image_hwc Source uint8 image in HWC RGB888 layout.
 * @param[in]  _source_width       Source image width  in pixels.
 * @param[in]  _source_height      Source image height in pixels.
 * @param[out] p_input_tensor     Destination int8 tensor (HWC, MODEL_INPUT_SIZE
 *                                 int8 elements).
 */
void preprocess(const uint8_t *p_source_image_hwc,
                int32_t        _source_width,
                int32_t        _source_height,
                int8_t        *p_input_tensor);

#endif /* PREPROCESSING_H */
