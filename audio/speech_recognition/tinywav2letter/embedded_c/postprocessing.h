/*
 * Copyright (c) 2020 - 2026 Renesas Electronics Corporation and/or its affiliates
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
/**********************************************************************************************************************
 * File Name    : postprocessing.h
 * Description  : TinyWav2letter postprocessing API.
 *********************************************************************************************************************/

#ifndef POSTPROCESSING_H_
#define POSTPROCESSING_H_

#include <stdint.h>

#include "../common/model_metadata.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    POSTPROCESS_OK              =  0,
    POSTPROCESS_ERR_INVALID_ARG = -1,
    POSTPROCESS_ERR_CAPACITY    = -2
} postprocess_status_t;

/**
 * @brief Perform greedy CTC decoding on int8 model output logits.
 *
 * Produces a null-terminated transcript string. Repeated tokens and the
 * blank token (MODEL_CTC_BLANK_INDEX) are collapsed per CTC rules.
 *
 * @param[in] p_out_q Model output tensor [MODEL_OUTPUT_SIZE] in int8 format.
 * @param[out] p_transcript_out Output buffer for null-terminated transcript.
 * @param[in] _transcript_capacity Size of p_transcript_out in bytes (must be > 0).
 * @param[out] p_out_length Pointer to store written character count (excluding NUL).
 *
 * @return POSTPROCESS_OK on success.
 * @return POSTPROCESS_ERR_INVALID_ARG if a required pointer argument is NULL.
 * @return POSTPROCESS_ERR_CAPACITY if output buffer capacity is zero or too small.
 */
postprocess_status_t postprocess(const int8_t p_out_q[MODEL_OUTPUT_SIZE],
                                 char *p_transcript_out,
                                 uint32_t _transcript_capacity,
                                 uint32_t *p_out_length);

#ifdef __cplusplus
}
#endif

#endif /* POSTPROCESSING_H_ */
