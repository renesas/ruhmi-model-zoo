/**
 * @file    postprocessing.h
 * @brief   MSE-based anomaly scoring functions for autoencoder reconstruction error.
 *
 * This header defines the postprocessing interface for computing anomaly scores
 * based on mean squared error (MSE) between original input and autoencoder
 * reconstruction across all sliding-window vectors of an audio clip.
 *
 * @author  Renesas Electronics
 * @date    2025
 * @version 2.0  Revised for MISRA C 2025 (ISO/IEC 9899:2018) compliance
 */

#ifndef POSTPROCESSING_H
#define POSTPROCESSING_H

#include <stdint.h>

/**
 * @brief   Compute the clip-level anomaly score from autoencoder input and
 *          reconstruction vectors.
 *
 * Performs the complete postprocessing pipeline in a single call:
 *   1. For each sliding-window vector, compute MSE between the
 *      preprocessed input and the model's reconstruction output.
 *   2. Aggregate all per-vector MSE values into a single anomaly score
 *      (arithmetic mean).
 *
 * @param[in]  p_input_vectors    Flat array of preprocessed float32 input vectors
 *                              [num_vectors × vector_dim], row-major.
 * @param[in]  p_output_vectors   Flat array of model reconstruction float32 outputs
 *                              [num_vectors × vector_dim], row-major.
 * @param[in]  num_vectors      Number of sliding-window vectors in the clip.
 * @param[in]  vector_dim       Dimension of each vector (typically INPUT_DIM = 640).
 *
 * @return     Anomaly score (float32) — mean MSE across all vectors.
 *             Returns 0.0f if num_vectors is 0 or pointers are NULL.
 */
float postprocess(const float *p_input_vectors,
                  const float *p_output_vectors,
                  int          num_vectors,
                  int          vector_dim);

#endif /* POSTPROCESSING_H */
