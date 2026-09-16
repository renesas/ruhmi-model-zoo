/*
 * Copyright (c) 2020 - 2026 Renesas Electronics Corporation and/or its affiliates
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * MISRA C 2012 Compliant MobileFaceNet Postprocessing Implementation
 */
/**********************************************************************************************************************
 * File Name    : postprocessing.c
 * Description  : MobileFaceNet output postprocessing implementation.
 *
 *                Public API:
 *                  1. postprocess()         per-face L2 normalisation
 *                  2. postprocess_verify()  per-pair cosine similarity and
 *                                            threshold-based match decision
 *
 *                Internal helpers are kept static to keep the public surface
 *                minimal and to isolate verification math inside this module.
 **********************************************************************************************************************/

#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "model_metadata.h"
#include "postprocessing.h"


/* ============================================================================
 *  Internal helpers
 * ============================================================================ */

/**
 * @brief L2-normalise an embedding vector in place.
 *
 * Computes the L2 nordm then scales each element by 1/norm (reciprocal
 * multiply); the inner loop is a single MUL per element
 * (cheaper than per-element DIV on Cortex-M); the resulting <=0.5 ULP
 * perturbation is well below the cosine-similarity decision margin at the
 * 0.28 threshold.
 *
 * @param[in,out] p_embedding  Float32 vector to normalise in place.
 * @param[in]     _dim          Number of elements in the vector.
 */

/* Dequantize a raw int8 model output into a float embedding buffer.
 * Formula:  f[i] = ((float)q[i] - OUTPUT_QUANT_ZP) * OUTPUT_QUANT_SCALE */
static void dequantize(const int8_t *p_raw_q, float *p_out, int32_t _dim)
{
    int32_t _elem_idx;
    for (_elem_idx = 0; _elem_idx < _dim; _elem_idx++)
    {
        p_out[_elem_idx] = ((float)p_raw_q[_elem_idx] - (float)OUTPUT_QUANT_ZP)
                     * OUTPUT_QUANT_SCALE;
    }
}


/* L2-normalise an embedding in place.  Mirrors the Python reference
 * (np.linalg.norm + element-wise divide).  Uses a 1/norm reciprocal so the
 * inner loop is a single MUL per element (cheaper than per-element DIV on
 * Cortex-M). */
static void l2_normalize_inplace(float *p_embedding, int32_t _dim)
{
    int32_t   _elem_idx;
    float _sum_sq = 0.0f;
    float _norm;

    for (_elem_idx = 0; _elem_idx < _dim; _elem_idx++)
    {
        _sum_sq += p_embedding[_elem_idx] * p_embedding[_elem_idx];
    }

    /* sqrtf is correctly-rounded per IEEE-754, so this matches
     * np.linalg.norm to within numpy's internal accumulator width. */
    _norm = sqrtf(_sum_sq);

    if (_norm > 0.0f)
    {
        const float _inv = 1.0f / _norm;
        for (_elem_idx = 0; _elem_idx < _dim; _elem_idx++)
        {
            p_embedding[_elem_idx] *= _inv;
        }
    }
    else
    {
        /* Degenerate zero-norm case: zero the output buffer. */
        for (_elem_idx = 0; _elem_idx < _dim; _elem_idx++)
        {
            p_embedding[_elem_idx] = 0.0f;
        }
    }
}


/**
 * @brief Cosine similarity between two already-unit-norm embeddings.
 *
 * Since both inputs are unit-norm, cosine similarity reduces to a plain
 * dot product: sum(a[i] * b[i]).
 *
 * @param[in] p_a   First unit-norm embedding (length _dim).
 * @param[in] p_b   Second unit-norm embedding (length _dim).
 * @param[in] _dim   Embedding dimensionality.
 * @return           Cosine similarity in the range [-1, 1].
 */
static float cosine_similarity(const float *p_a,
                               const float *p_b,
                               int32_t      _dim)
{
    int32_t   _elem_idx;
    float _dot = 0.0f;

    for (_elem_idx = 0; _elem_idx < _dim; _elem_idx++)
    {
        _dot += p_a[_elem_idx] * p_b[_elem_idx];
    }
    return _dot;
}


/* ============================================================================
 *  Public API
 * ============================================================================ */

/**
 * @brief Per-face postprocess: L2-normalise the raw embedding in place.
 *
 * Wraps l2_normalize_inplace() with input validation. Returns immediately
 * without modifying any memory if the pointer is NULL or _dim is
 * non-positive.
 *
 * @param[in,out] p_embedding  int8 vector of length _dim. Overwritten
 *                              with its unit-norm version on return.
 * @param[in]     _dim          Embedding dimensionality (must equal
 *                              MOBILEFACENET_EMBEDDING_DIM = 128).
 */

void postprocess(const int8_t *p_raw_q,
                 float        *p_embedding,
                 int32_t       _dim)
{
    if ((p_raw_q == NULL) || (p_embedding == NULL) || (_dim <= 0))
    {
        return;
    }

    /* Step 1: dequantize int8 -> float */
    dequantize(p_raw_q, p_embedding, _dim);

    /* Step 2: L2 normalise in place */
    l2_normalize_inplace(p_embedding, _dim);
}


/**
 * @brief Per-pair postprocess: cosine similarity and match decision.
 *
 * Takes two already-L2-normalised embeddings, computes their cosine
 * similarity (dot product, since both are unit norm), and decides whether
 * the pair represents the same identity by comparing against _threshold.
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
                        float       *p_out_sim)
{
    if ((NULL == p_a) || (NULL == p_b) || (_dim <= 0))
    {
        if (NULL != p_out_sim)
        {
            *p_out_sim = 0.0f;
        }
        return false;
    }

    const float _sim = cosine_similarity(p_a, p_b, _dim);

    if (NULL != p_out_sim)
    {
        *p_out_sim = _sim;
    }
    return (_sim >= _threshold);
}
