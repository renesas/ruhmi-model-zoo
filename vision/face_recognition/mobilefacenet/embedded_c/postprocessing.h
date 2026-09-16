/*
 * Copyright (c) 2020 - 2026 Renesas Electronics Corporation and/or its affiliates
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * MISRA C 2012 Compliant MobileFaceNet Postprocessing Declarations
 */
/**********************************************************************************************************************
 * File Name    : postprocessing.h
 * Description  : MobileFaceNet output postprocessing declarations.
 *
 *                Two entry points are exposed:
 *                  1. postprocess()         per-face L2 normalisation
 *                  2. postprocess_verify()  per-pair cosine similarity and
 *                                            threshold-based match decision
 **********************************************************************************************************************/

#ifndef POSTPROCESSING_H
#define POSTPROCESSING_H

#include <stdint.h>
#include <stdbool.h>

/**
 * @brief Per-face postprocess: dequantize int8 embedding and L2-normalise.
 *
 * Converts the raw int8 model output into a unit-norm float embedding
 * ready for cosine comparison:
 *
 *   Step 1 - Dequantize:
 *     f_emb[i] = ((float)q_emb[i] - OUTPUT_QUANT_ZP) * OUTPUT_QUANT_SCALE
 *
 *   Step 2 - L2 normalise:
 *     norm = sqrt(sum(f_emb[i]^2))
 *     f_emb[i] /= norm
 *
 * If the dequantized embedding has zero norm the output buffer is zeroed
 * (degenerate case).
 *
 * @param[in]  p_raw_q       Raw int8 output tensor from the model
 *                            (MODEL_OUTPUT_SIZE elements).
 * @param[out] p_embedding   Destination float32 buffer (MODEL_OUTPUT_SIZE
 *                            floats). Receives the dequantized, L2-normalised
 *                            embedding on return.
 * @param[in]  _dim           Embedding dimensionality (must equal
 *                            MODEL_OUTPUT_SIZE = 128).
 */
void postprocess(const int8_t *p_raw_q,
                 float        *p_embedding,
                 int32_t       _dim);

/**
 * @brief Per-pair postprocess: cosine similarity + match decision.
 *
 * Takes two already-L2-normalised float embeddings (produced by
 * postprocess()), computes cosine similarity (dot product) and decides
 * whether the pair represents the same identity.
 *
 * @param[in]  p_a         First embedding (length _dim, unit norm).
 * @param[in]  p_b         Second embedding (length _dim, unit norm).
 * @param[in]  _dim         Embedding dimensionality.
 * @param[in]  _threshold   Match threshold (typically MATCH_THRESHOLD).
 * @param[out] p_out_sim   If non-NULL, receives the cosine similarity
 *                          in [-1, 1].
 * @return     true if the pair matches (similarity >= threshold).
 */
bool postprocess_verify(const float *p_a,
                        const float *p_b,
                        int32_t      _dim,
                        float        _threshold,
                        float       *p_out_sim);

#endif /* POSTPROCESSING_H */
