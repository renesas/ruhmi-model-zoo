/**
 * @file    postprocessing.h
 * @brief   KWS top-1 postprocessing API.
 *
 * Copyright 2026 Renesas Electronics Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef POSTPROCESSING_H
#define POSTPROCESSING_H

#include <stdint.h>

/**
 * @brief Dequantize/softmax output scores and return top-1 class index.
 */
int32_t postprocess(const int8_t *p_output, int32_t num_classes);

#endif /* POSTPROCESSING_H */
