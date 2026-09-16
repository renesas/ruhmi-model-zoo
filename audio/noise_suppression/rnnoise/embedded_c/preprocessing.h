/*
 * Copyright (c) 2020 - 2026 Renesas Electronics Corporation and/or its affiliates
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
/*
 * preprocessing.h
 *
 * Phase G.5 — preprocessing entry point for the RNNoise port.
 * One call: PCM in -> 42 features + per-frame DSP scratch.
 *
 * Matches the wrapper-layer contract documented in VENDOR.md:
 *
 *     int16_t pcm_in[480]
 *           |
 *           v  preprocess_frame()
 *           |
 *           |--> float features[42]              -> compute_sub_0000() in ruhmi_perf_eval.c
 *           '--> preprocess_scratch_t scratch    -> postprocess_frame()
 *
 * The opaque `preprocess_scratch_t` carries the per-frame FFT bins, the
 * band energies, the band cross-correlation, and the silence flag that
 * postprocess_frame() needs in order to apply the pitch filter and
 * resynthesise the time-domain signal. Caller allocates it once on the
 * stack and re-uses across frames.
 *
 * The long-lived `DenoiseState` (FFT memory, pitch buffers, cepstral
 * memory, HP-filter taps, last-frame gain memory, AND the embedded
 * RNN state) is **not owned by this module**. It lives in
 * `ruhmi_perf_eval.c` alongside the MERA scratch buffer so that all
 * long-lived inference storage is allocated in one place; this module
 * stays purely transformational. The caller passes a `DenoiseState *`
 * per call.
 */

#ifndef PREPROCESSING_H_
#define PREPROCESSING_H_

#include <stdint.h>
#include <math.h>    /* roundf() */

#include "model_metadata.h"             /* MODEL_FRAME_SIZE, MODEL_NB_BANDS, ... */
#include "rnnoise_dsp/rnnoise.h"        /* DenoiseState (opaque forward decl)    */
#include "rnnoise_dsp/kiss_fft.h"       /* kiss_fft_cpx                          */

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------------
 * Per-frame DSP scratch handed from preprocess to postprocess.
 *
 * Sizes are mirrored from the constants in `rnnoise_dsp/denoise.c`:
 *   FREQ_SIZE   = MODEL_FREQ_SIZE   = 481
 *   WINDOW_SIZE = MODEL_WINDOW_SIZE = 960
 *   NB_BANDS    = MODEL_NB_BANDS    = 22
 *
 * Memory: ~5 504 + 7 680 + 264 + 4 = 13 452 B per scratch instance.
 * Comfortably stack-allocatable on the RA8P1 main stack (>= 32 KB).
 * ------------------------------------------------------------------ */
typedef struct {
    kiss_fft_cpx X [MODEL_FREQ_SIZE];                  /* analysis FFT of current frame        */
    kiss_fft_cpx P [MODEL_WINDOW_SIZE];                /* pitch-prediction FFT                 */
    float        Ex[MODEL_NB_BANDS];                   /* band energy of X                     */
    float        Ep[MODEL_NB_BANDS];                   /* band energy of P                     */
    float        Exp[MODEL_NB_BANDS];                  /* band cross-correlation X . P         */
    int          silence;                              /* 1 = bypass NN + emit zeros                         */
    int8_t       features_int8[MODEL_INPUT_FEATURES_SIZE]; /* quantised features -> compute_sub_0000()      */
    int8_t       gains_int8[MODEL_OUTPUT_GAINS_SIZE];      /* raw gains     compute_sub_0000() -> postprocess_frame() */
    int8_t       vad_prob_int8;                            /* raw VAD prob  compute_sub_0000() -> postprocess_frame() */
} preprocess_scratch_t;

/* ---------------------------------------------------------------------
 * preprocess_frame() — one-shot front-end DSP for a single 10 ms frame.
 *
 *   st       : long-lived denoiser state owned by the caller           [in,out]
 *   pcm_in   : 480 int16 PCM samples @ 48 kHz (one frame)              [in]
 *   features : 42 float features that feed the NN                      [out]
 *   scratch  : per-frame DSP intermediates needed by postprocess       [out]
 *
 * Steps performed internally (matches upstream rnnoise_process_frame()
 * pre-NN body, denoise.c@1cbdbcf:478-494):
 *
 *   1. int16 -> float       (PCM in -> float buffer)
 *   2. HP biquad            (biquad() from denoise_internal.h)
 *   3. Windowed FFT + band energies + pitch + cepstral feature build
 *                           (compute_frame_features() from denoise_internal.h)
 *   4. Quantise float features[42] -> int8 (scratch->features_int8[]).
 *      Uses MODEL_INPUT_FEATURES_{SCALE,ZP} from model_metadata.h so the
 *      result is ready to pass directly to compute_sub_0000
 *
 * On exit `scratch->silence` is non-zero only in TRAINING builds (always 0
 * in our INT8 deployment). The downstream caller may still choose to
 * skip the NN on energy-based silence detection, but the upstream
 * contract is preserved: silent -> postprocess emits 480 zero samples.
 * ------------------------------------------------------------------ */
void preprocess_frame(DenoiseState *p_st,
                      const int16_t p_pcm_in[MODEL_FRAME_SIZE],
                      float p_features[MODEL_NB_FEATURES],
                      preprocess_scratch_t *p_scratch);

/* ---------------------------------------------------------------------
 * quantise_f32_to_int8() — symmetric per-tensor INT8 quantisation
 * matching TFLite's reference formula:
 *   q = clip( round(x / scale) + zero_point, -128, 127 )
 *
 * Defined in preprocessing.c because quantisation is the final step
 * of the preprocessing stage: it converts the float features produced
 * by the DSP front-end into the int8 tensor format expected by the
 * MERA-generated inference graph.
 * ------------------------------------------------------------------ */
int8_t quantise_f32_to_int8(float x, float scale, int32_t zero_point);

#ifdef __cplusplus
}
#endif

#endif /* PREPROCESSING_H_ */
