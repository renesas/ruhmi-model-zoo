/*
 * Copyright (c) 2020 - 2026 Renesas Electronics Corporation and/or its affiliates
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
/*
 * postprocessing.h
 *
 * Post-NN DSP entry point for a single 10 ms RNNoise frame.
 * Applies gain smoothing, spectral interpolation, and frame synthesis
 * to produce 16-bit PCM output.
 */

#ifndef POSTPROCESSING_H_
#define POSTPROCESSING_H_

#include <stdint.h>

#include "model_metadata.h"             /* MODEL_FRAME_SIZE, MODEL_NB_BANDS, ... */
#include "preprocessing.h"              /* preprocess_scratch_t                  */
#include "rnnoise_dsp/rnnoise.h"        /* DenoiseState (opaque forward decl)    */

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------------
 * postprocess_frame() — one-shot back-end DSP for a single 10 ms frame.
 *
 *   st           : long-lived denoiser state owned by the caller            [in,out]
 *   scratch      : per-frame DSP scratch from preprocess_frame();            [in,out]
 *                  .gains_int8 and .vad_prob_int8 carry the raw INT8 model
 *                  outputs written by (). .X is multiplied by
 *                  interpolated gains in place.
 *   pcm_out      : 480 int16 PCM samples @ 48 kHz (denoised)                [out]
 *   p_vad_prob_out: float VAD probability in [0.0, 1.0] dequantised here    [out]
 *
 * Step 1 (first action inside the function): dequantise scratch->gains_int8
 * and scratch->vad_prob_int8 to float. Subsequent steps match upstream
 * rnnoise_process_frame() post-NN body (pitch filter, gain smoothing,
 * interp_band_gain, spectrum multiply, IFFT/OLA).
 *
 * Silent frames (`scratch->silence != 0`) bypass the gain-application
 * path: the input spectrum is fed straight into frame_synthesis().
 * ------------------------------------------------------------------ */
void postprocess_frame(DenoiseState *p_st,
                       preprocess_scratch_t *p_scratch,
                       int16_t p_pcm_out[MODEL_FRAME_SIZE],
                       float *p_vad_prob_out);

/* ---------------------------------------------------------------------
 * dequantise_int8_to_f32() — symmetric per-tensor INT8 dequantisation
 * matching TFLite's reference formula:
 *   x = scale * (q - zero_point)
 *
 * Defined in postprocessing.c because dequantisation is the first step
 * of the postprocessing stage: it converts the int8 model output tensors
 * (band gains, VAD probability) into the float values consumed by the
 * DSP back-end.
 * ------------------------------------------------------------------ */
float dequantise_int8_to_f32(int8_t q, float scale, int32_t zero_point);

#ifdef __cplusplus
}
#endif

#endif /* POSTPROCESSING_H_ */
