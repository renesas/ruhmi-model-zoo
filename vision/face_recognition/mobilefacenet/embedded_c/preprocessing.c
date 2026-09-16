/*
 * Copyright (c) 2020 - 2026 Renesas Electronics Corporation and/or its affiliates
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * MISRA C 2012 Compliant MobileFaceNet Preprocessing Implementation
 */
/**********************************************************************************************************************
 * File Name    : preprocessing.c
 * Description  : MobileFaceNet input preprocessing: per-pixel float
 *                normalisation of a pre-aligned 112x112 RGB face into the
 *                [-1, 1] model input tensor expected by the MERA-deployed
 *                MobileFaceNet graph.
 *
 *                The host receives an MTCNN-aligned 112x112 RGB crop
 *                (alignment is done off-device; the test faces are baked into
 *                test_faces.h). The only work left to do here is the
 *                per-pixel float normalisation.
 **********************************************************************************************************************/
#include <math.h>
#include <stdint.h>
#include "model_metadata.h"
#include "preprocessing.h"

/**
 * @brief Convert a pre-aligned 112x112 RGB face into the model input tensor.
 *
 * Applies per-pixel float normalisation:
 *     out[i] = ((float)in[i] / 255.0f - 0.5f) / 0.5f  ->  range [-1, 1]
 *
 * @param[in]  p_source_image_hwc  Source uint8 face in HWC interleaved RGB
 *                                  layout (MOBILEFACENET_INPUT_BYTES bytes).
 * @param[out] p_input_tensor      Destination float32 tensor
 *                                  (MODEL_INPUT_SIZE floats, HWC),
 *                                  values in range [-1, 1].
 */
void preprocess(const uint8_t *p_source_image_hwc,
                int8_t        *p_input_tensor)
{
    uint32_t _pixel_idx;

    for (_pixel_idx = 0U; _pixel_idx < (uint32_t)MODEL_INPUT_SIZE; _pixel_idx++)
    {
        /* Step 1: normalise uint8 pixel to float in [-1.0, 1.0] */
        float _norm = (float)p_source_image_hwc[_pixel_idx] / 127.5f - 1.0f;

        /* Step 2: quantize to int8 */
        int32_t _quant_val = (int32_t)lroundf(_norm / INPUT_QUANT_SCALE)
                    + INPUT_QUANT_ZP;
        if (_quant_val < -128) { _quant_val = -128; }
        if (_quant_val >  127) { _quant_val =  127; }
        p_input_tensor[_pixel_idx] = (int8_t)_quant_val;
    }
}
