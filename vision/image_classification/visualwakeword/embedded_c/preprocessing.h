/**
 * @file    preprocessing.h
 * @brief   Visual Wake Words (VWW, NPU) input preprocessing declarations - bilinear resize and INT8 quantization.
 *
 * @author  Renesas Electronics
 * @date    2025
 * @version 1.0
 */

#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include <stdint.h>
#include "../common/model_metadata.h"

/*
 * VWW normalization : pixel / INPUT_NORM_SCALE → [0.0, 1.0]
 * Quantize          : q = round(f / INPUT_SCALE) + INPUT_ZP → int8
 * (INPUT_NORM_SCALE, INPUT_SCALE, INPUT_ZP defined in model_metadata.h)
 */

/**
 * @brief Bilinear resize and INT8-quantize a source RGB888 image for the
 *        VWW model input tensor [MODEL_INPUT_H x MODEL_INPUT_W x MODEL_INPUT_C].
 *
 * @param p_source_image       Source image, HWC layout, uint8 [0, 255].
 * @param source_width       Source image width in pixels.
 * @param source_height      Source image height in pixels.
 * @param p_destination_tensor Output INT8 buffer [MODEL_INPUT_SIZE].
 */
void preprocess(const uint8_t *p_source_image,
                int source_width,
                int source_height,
                int8_t *p_destination_tensor);

#endif /* PREPROCESSING_H */
