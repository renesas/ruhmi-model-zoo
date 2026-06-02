/**
 ******************************************************************************
 * @file    preprocessing.h
 * @brief   Portable NN image preprocessing — board independent.
 */

#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include <stdint.h>

/**
 * @brief Resize an RGB image with bilinear interpolation for NN input.
 * @param source_image Source image buffer in interleaved RGB byte format.
 * @param source_width Source image width in pixels.
 * @param source_height Source image height in pixels.
 * @param destination_image Destination buffer for resized floating-point RGB pixels.
 * @param destination_width Destination image width in pixels.
 * @param destination_height Destination image height in pixels.
 */
void preprocess(const uint8_t *source_image,
                uint16_t source_width, uint16_t source_height,
                float    *destination_image,
                uint16_t destination_width, uint16_t destination_height);

#endif /* PREPROCESSING_H */