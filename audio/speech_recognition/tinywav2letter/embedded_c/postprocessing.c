/*
 * Copyright (c) 2020 - 2026 Renesas Electronics Corporation and/or its affiliates
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
/**********************************************************************************************************************
 * File Name    : postprocessing.c
 * Description  : Greedy CTC decode for TinyWav2letter output logits.
 *********************************************************************************************************************/

#include <stddef.h>
#include <stdint.h>

#include "postprocessing.h"

/* a-z + apostrophe + space + blank(@) */
static const char s_alphabet[MODEL_ALPHABET_SIZE + 1U] = "abcdefghijklmnopqrstuvwxyz' @";

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
                                 uint32_t *p_out_length)
{
    uint32_t _written    = 0U;
    int32_t  _prev_index = -1;

    /* Validate all required pointer arguments. */
    if ((NULL == p_out_q) || (NULL == p_transcript_out) || (NULL == p_out_length)) {
        return POSTPROCESS_ERR_INVALID_ARG;
    }
    if (0U == _transcript_capacity) {
        return POSTPROCESS_ERR_CAPACITY;
    }

    /* Iterate over each output time step and perform argmax over class logits. */
    for (uint32_t t = 0U; t < MODEL_OUTPUT_TIME; ++t) {
        const int8_t *p_row  = &p_out_q[t * MODEL_OUTPUT_CLASSES];
        int8_t        _best_val = p_row[0];
        int32_t       _best_idx = 0;

        /* Find the class with the highest logit value (argmax). */
        for (uint32_t c = 1U; c < MODEL_OUTPUT_CLASSES; ++c) {
            if (p_row[c] > _best_val) {
                _best_val = p_row[c];
                _best_idx = (int32_t)c;
            }
        }

        /* CTC rule: skip repeated tokens to merge duplicates. */
        if (_best_idx == _prev_index) {
            continue;
        }
        _prev_index = _best_idx;

        /* CTC rule: drop the blank token. */
        if (MODEL_CTC_BLANK_INDEX == (uint32_t)_best_idx) {
            continue;
        }

        /* Guard against writing beyond the available transcript buffer capacity. */
        if ((_written + 1U) >= _transcript_capacity) {
            p_transcript_out[_transcript_capacity - 1U] = '\0';
            *p_out_length = _written;
            return POSTPROCESS_ERR_CAPACITY;
        }
        p_transcript_out[_written++] = s_alphabet[_best_idx];
    }

    p_transcript_out[_written] = '\0';
    *p_out_length = _written;
    return POSTPROCESS_OK;
}
