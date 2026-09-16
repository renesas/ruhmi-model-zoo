/* Copyright (c) 2018 Gregor Richards
 * Copyright (c) 2017 Mozilla */
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/**
 * @file rnnoise.h
 * @brief Public C API for the RNNoise recurrent-neural-network speech denoiser.
 * @details
 *   Vendored from xiph/rnnoise \@1cbdbcf (Gregor Richards / Mozilla). Declares
 *   the opaque DenoiseState handle, the RNNModel descriptor, and the six
 *   public entry points:
 *     - rnnoise_get_size() / rnnoise_get_frame_size()
 *     - rnnoise_init() / rnnoise_create() / rnnoise_destroy()
 *     - rnnoise_process_frame()
 *     - rnnoise_model_from_file() / rnnoise_model_free()
 *
 *   On the RA8P1 port heap-allocation functions (rnnoise_create,
 *   rnnoise_model_from_file) are not called; DenoiseState is allocated
 *   statically in ruhmi_perf_eval.c and initialised with rnnoise_init().
 *
 * @note Upstream BSD-3-Clause license retained above.
 *       Renesas RA8P1 port: Copyright (c) 2020 - 2026 Renesas Electronics
 *       Corporation and/or its affiliates.
 */

#ifndef RNNOISE_H
#define RNNOISE_H 1

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef RNNOISE_EXPORT
# if defined(WIN32)
#  if defined(RNNOISE_BUILD) && defined(DLL_EXPORT)
#   define RNNOISE_EXPORT __declspec(dllexport)
#  else
#   define RNNOISE_EXPORT
#  endif
# elif defined(__GNUC__) && defined(RNNOISE_BUILD)
#  define RNNOISE_EXPORT __attribute__ ((visibility ("default")))
# else
#  define RNNOISE_EXPORT
# endif
#endif

typedef struct DenoiseState DenoiseState;
typedef struct RNNModel RNNModel;

/**
 * @brief Return the size of DenoiseState in bytes.
 * @return Size of DenoiseState in bytes.
 */
RNNOISE_EXPORT int rnnoise_get_size();

/**
 * @brief Return the number of samples processed per rnnoise_process_frame() call.
 * @return Frame size in samples (480 for the 48 kHz, 10 ms configuration).
 */
RNNOISE_EXPORT int rnnoise_get_frame_size();

/**
 * @brief Initialise a pre-allocated DenoiseState.
 * @details If model is NULL the built-in default model (rnnoise_model_orig) is used.
 *          See rnnoise_create() and rnnoise_model_from_file().
 * @param[in,out] st    Pre-allocated DenoiseState to initialise.
 * @param[in]     model NN model descriptor, or NULL for the default model.
 * @return 0 on success.
 */
RNNOISE_EXPORT int rnnoise_init(DenoiseState *p_st, RNNModel *p_model);

/**
 * @brief Allocate and initialise a DenoiseState on the heap.
 * @details The returned pointer MUST be freed with rnnoise_destroy().
 *          If model is NULL the built-in default model is used.
 * @param[in] model NN model descriptor, or NULL for the default model.
 * @return Pointer to a newly allocated DenoiseState, or NULL on failure.
 */
RNNOISE_EXPORT DenoiseState *rnnoise_create(RNNModel *p_model);

/**
 * @brief Free a DenoiseState produced by rnnoise_create().
 * @details The optional custom model must be freed separately with rnnoise_model_free().
 * @param[in] st DenoiseState to destroy.
 */
RNNOISE_EXPORT void rnnoise_destroy(DenoiseState *p_st);

/**
 * @brief Denoise one frame of floating-point PCM samples.
 * @details Both \p in and \p out must point to at least rnnoise_get_frame_size() elements.
 * @param[in,out] st  Long-lived denoiser state (GRU hidden states, FFT memory).
 * @param[out]    out Denoised output frame (float, FRAME_SIZE samples).
 * @param[in]     in  Noisy input frame   (float, FRAME_SIZE samples).
 * @return Voice-activity probability for this frame in the range [0.0, 1.0].
 */
RNNOISE_EXPORT float rnnoise_process_frame(DenoiseState *p_st, float *p_out, const float *p_in);

/**
 * @brief Load a custom RNN model from an open file handle.
 * @details The returned pointer must be freed with rnnoise_model_free().
 * @param[in] f Open FILE* positioned at the start of the model binary.
 * @return Pointer to a newly allocated RNNModel, or NULL on failure.
 */
RNNOISE_EXPORT RNNModel *rnnoise_model_from_file(FILE *p_f);

/**
 * @brief Free a custom RNNModel created by rnnoise_model_from_file().
 * @details Must be called after all DenoiseState objects referencing this model
 *          have been destroyed.
 * @param[in] model RNNModel to free.
 */
RNNOISE_EXPORT void rnnoise_model_free(RNNModel *p_model);

#ifdef __cplusplus
}
#endif

#endif
