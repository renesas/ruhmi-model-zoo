/**
 * @file    postprocessing.c
 * @brief   MSE-based anomaly scoring for AD01 autoencoder.
 *
 * Computes per-vector mean squared error between the preprocessed input
 * and the model reconstruction, then aggregates into a single anomaly score.
 *
 * @author  Renesas Electronics
 * @date    2025
 * @version 2.0  Revised for MISRA C 2025 (ISO/IEC 9899:2018) compliance:
 *               - Controlling expression uses correct essential type (Rule 14.4)
 *               - NULL pointer checks ordered before dereference (Rule 18.1)
 *               - All parameter types use explicit signedness conventions
 */

#include "postprocessing.h"
#include <stddef.h>

/**
 * @brief   Compute MSE between a single input/reconstruction vector pair
 *          (internal helper).
 *
 * @param[in] p_input_vec   Pointer to original float32 vector
 * @param[in] p_output_vec  Pointer to reconstructed float32 vector
 * @param[in] dim         Number of elements in each vector
 *
 * @return    Mean squared error for this vector pair
 */
static float compute_vector_mse(const float *p_input_vec,
                                const float *p_output_vec,
                                int dim)
{
    float sum = 0.0f;
    int i;

    for (i = 0; i < dim; i++)
    {
        float diff = p_input_vec[i] - p_output_vec[i];
        sum += diff * diff;
    }

    return sum / (float)dim;
}

/**
 * @brief   Compute the clip-level anomaly score from autoencoder input and
 *          reconstruction vectors.
 *
 * For each vector pair, computes MSE using compute_vector_mse() and returns
 * the arithmetic mean across all vectors as the final anomaly score.
 *
 * @param[in] p_input_vectors   Flat array of input vectors [num_vectors x vector_dim]
 * @param[in] p_output_vectors  Flat array of reconstructed vectors [num_vectors x vector_dim]
 * @param[in] num_vectors     Number of vectors in the clip
 * @param[in] vector_dim      Number of elements per vector
 *
 * @return    Mean MSE across all vectors, or 0.0f for invalid inputs
 */
float postprocess(const float *p_input_vectors,
                  const float *p_output_vectors,
                  int          num_vectors,
                  int          vector_dim)
{
    float mse_sum = 0.0f;
    int vec_idx;

    if ((num_vectors <= 0) || (NULL == p_input_vectors ) || (NULL == p_output_vectors ))
    {
        return 0.0f;
    }

    for (vec_idx = 0; vec_idx < num_vectors; vec_idx++)
    {
        const float *p_in  = &p_input_vectors[vec_idx * vector_dim];
        const float *p_out = &p_output_vectors[vec_idx * vector_dim];

        mse_sum += compute_vector_mse(p_in, p_out, vector_dim);
    }

    return mse_sum / (float)num_vectors;
}
