/*
 * Copyright (c) 2020 - 2026 Renesas Electronics Corporation and/or its affiliates
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
/**
 * @file denoise_internal.h
 * @brief Private DSP helpers hoisted from denoise.c for use by preprocessing and postprocessing.
 * @details
 *   Exposes three helpers that live inside denoise.c (vendored from
 *   xiph/rnnoise \@1cbdbcf, with their \c static qualifier removed):
 *
 *     - biquad()                 HP-filter biquad applied to the PCM frame
 *     - compute_frame_features() FFT feature-extraction front-end (42 floats)
 *     - frame_synthesis()        IFFT + overlap-add resynthesis back-end
 *
 *   Also contains the complete definition of struct DenoiseState, hoisted
 *   from denoise.c so that preprocessing.c, postprocessing.c, and
 *   ruhmi_perf_eval.c can access the internal fields and take sizeof().
 *
 *   Nothing outside io_processing/ should include this header — the public
 *   denoiser API is rnnoise.h.
 *
 * @note SPDX-License-Identifier: BSD-3-Clause
 *       Renesas RA8P1 port: Copyright (c) 2020 - 2026 Renesas Electronics
 *       Corporation and/or its affiliates.
 */

#ifndef DENOISE_INTERNAL_H_
#define DENOISE_INTERNAL_H_

#include "kiss_fft.h"                /* kiss_fft_cpx, kiss_fft_state          */
#include "rnnoise.h"                 /* DenoiseState (forward decl)           */
#include "mera_rnn_shim.h"           /* RNNState                              */
#include "../model_metadata.h"
                                     /* MODEL_FRAME_SIZE, MODEL_NB_BANDS,
                                      * MODEL_CEPS_MEM, MODEL_PITCH_BUF_SIZE  */

#ifdef __cplusplus
extern "C" {
#endif

/* ---- struct DenoiseState — hoisted from denoise.c@1cbdbcf:97-108 ----
 *
 * hoisted out of denoise.c so the wrapper layer
 * (preprocessing.c, postprocessing.c) can reach `st->mem_hp_x` and
 * `st->lastg` directly, and so ruhmi_perf_eval.c can take
 * `sizeof(struct DenoiseState)` at compile time to size its BSS
 * reservation. Layout is bit-identical to upstream — the MODEL_*
 * macro names are the project-portable spellings of upstream's
 * file-local FRAME_SIZE / NB_BANDS / CEPS_MEM / PITCH_BUF_SIZE, which
 * carry the same numeric values (see common/model_metadata.h §2). */
struct DenoiseState {
    float analysis_mem  [MODEL_FRAME_SIZE];                  /* 1 920 B */
    float cepstral_mem  [MODEL_CEPS_MEM][MODEL_NB_BANDS];    /*   704 B */
    int   memid;                                              /*     4 B */
    float synthesis_mem [MODEL_FRAME_SIZE];                  /* 1 920 B */
    float pitch_buf     [MODEL_PITCH_BUF_SIZE];              /* 6 912 B */
    float pitch_enh_buf [MODEL_PITCH_BUF_SIZE];              /* 6 912 B */
    float last_gain;                                          /*     4 B */
    int   last_period;                                        /*     4 B */
    float mem_hp_x      [2];                                  /*     8 B */
    float lastg         [MODEL_NB_BANDS];                    /*    88 B */
    RNNState rnn;                                             /*   ~184 B*/
};                                                            /* ~18.7 KB total */

/**
 * @brief Apply a 2nd-order IIR (biquad) high-pass filter to a PCM frame.
 * @details Used by preprocessing.c to pre-emphasise the audio before feature
 *          extraction. Upstream signature: denoise.c\@1cbdbcf:411.
 * @param[out]    y   Output float PCM frame, length N.
 * @param[in,out] mem 2-tap filter memory, carried across calls (DenoiseState::mem_hp_x[2]).
 * @param[in]     x   Input float PCM frame, length N.
 * @param[in]     b   Numerator (feed-forward) biquad coefficients, length 2.
 * @param[in]     a   Denominator (feedback) biquad coefficients, length 2.
 * @param[in]     N   Number of samples to process (FRAME_SIZE = 480).
 */
void biquad(float *p_y, float p_mem[2], const float *p_x,
            const float *p_b, const float *p_a, int N);

/**
 * @brief Compute the 42-element feature vector and per-frame DSP scratch.
 * @details FFT-based feature-extraction front-end. Upstream signature:
 *          denoise.c\@1cbdbcf:310. Called once per frame by preprocessing.c.
 * @param[in,out] st       Long-lived denoiser state (FFT memory, pitch buffers,
 *                         cepstral memory, last gains).
 * @param[out]    X        Analysis FFT of the current frame [FREQ_SIZE].
 * @param[out]    P        Pitch-prediction FFT              [WINDOW_SIZE].
 * @param[out]    Ex       Band energy of X                  [NB_BANDS].
 * @param[out]    Ep       Band energy of P                  [NB_BANDS].
 * @param[out]    Exp      Band cross-correlation X\u00b7P        [NB_BANDS].
 * @param[out]    features 42-element float feature vector for the NN [NB_FEATURES].
 * @param[in]     in       HP-filtered float PCM frame        [FRAME_SIZE].
 * @return 1 if the frame is silent (training mode only; always 0 in production
 *         with TRAINING=0), 0 otherwise.
 */
int compute_frame_features(DenoiseState *p_st,
                           kiss_fft_cpx *p_X, kiss_fft_cpx *p_P,
                           float *p_Ex, float *p_Ep, float *p_Exp,
                           float *p_features, const float *p_in);

/**
 * @brief IFFT and overlap-add resynthesis of the post-gain spectrum.
 * @details Inverse transforms the post-gain frequency-domain frame, windows
 *          it, and overlap-adds with the previous frame's tail stored in
 *          DenoiseState::synthesis_mem. Upstream signature:
 *          denoise.c\@1cbdbcf:401.
 * @param[in,out] st  Long-lived denoiser state (carries synthesis_mem across calls).
 * @param[out]    out Output float PCM frame [FRAME_SIZE].
 * @param[in]     y   Post-gain frequency-domain frame [FREQ_SIZE] (conjugate-symmetric
 *                    half; kiss_fft IFFT consumes WINDOW_SIZE bins internally).
 */
void frame_synthesis(DenoiseState *p_st, float *p_out, const kiss_fft_cpx *p_y);

#ifdef __cplusplus
}
#endif

#endif /* DENOISE_INTERNAL_H_ */
