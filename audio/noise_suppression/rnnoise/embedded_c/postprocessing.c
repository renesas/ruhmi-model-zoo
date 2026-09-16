/*
 * Copyright (c) 2020 - 2026 Renesas Electronics Corporation and/or its affiliates
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
/*
 * postprocessing.c
 *
 * thin wrapper that exposes the post-NN half of upstream
 * rnnoise's `rnnoise_process_frame()` as a single function call:
 *
 *     postprocess_frame(st, gains, scratch, pcm_out);
 *
 * Internally calls (in order, matching denoise.c@1cbdbcf:498-516):
 *     pitch_filter()       -- pitch-aware spectral filtering
 *     gain smoothing       -- max(g[i], 0.6 * lastg[i])  (in place)
 *     interp_band_gain()   -- 22 band gains -> 481 FFT-bin gains
 *     spectrum multiply    -- X *= gf (per FFT bin)
 *     frame_synthesis()    -- IFFT + windowed overlap-add
 *
 *
 * No state is owned here. The caller (ruhmi_perf_eval.c)
 * passes the long-lived `DenoiseState *`.
 */

#include <stdint.h>
#include <math.h>     /* fmaxf */

#include "postprocessing.h"
#include "rnnoise_dsp/denoise_internal.h"  /* frame_synthesis */

/* dequantise_int8_to_f32() — symmetric per-tensor INT8 dequantisation
 * matching TFLite's reference formula:
 *   x = scale * (q - zero_point) */
float dequantise_int8_to_f32(int8_t q, float scale, int32_t zero_point)
{
    return scale * ((float)q - (float)zero_point);
}

/* Externs from rnnoise_dsp/denoise.c — already non-static in upstream,
 * Declared here rather than via a header to
 * keep the include surface tight. */
extern void pitch_filter(kiss_fft_cpx *X, const kiss_fft_cpx *P,
                         const float *Ex, const float *Ep,
                         const float *Exp, const float *g);
extern void interp_band_gain(float *g, const float *bandE);

/* float_to_int16_saturate() — clamps a float sample to [-32768, +32767]
 * and rounds to nearest. Clips on the way out since IFFT + overlap-add
 * can theoretically exceed int16 range on very loud frames. */
static inline int16_t float_to_int16_saturate(float x)
{
    float r = (0.0f <= x) ? (x + 0.5f) : (x - 0.5f);
    if (32767.0f  < r) return  32767;
    if (-32768.0f > r) return -32768;
    return (int16_t)r;
}

void postprocess_frame(DenoiseState *p_st,
                       preprocess_scratch_t *p_scratch,
                       int16_t p_pcm_out[MODEL_FRAME_SIZE],
                       float *p_vad_prob_out)
{
    /* Per-frame frequency-domain gain mask, initialised to all-ones so
     * silent frames pass the input spectrum through unmodified (matches
     * the `float gf[FREQ_SIZE]={1}` initialiser in upstream). */
    float gf[MODEL_FREQ_SIZE];
    for (int i = 0; i < MODEL_FREQ_SIZE; ++i) {
        gf[i] = 1.0f;
    }

    /* 1.  Dequantise the 22 INT8 band gains and the VAD probability to float.
     *     This is the mirror of the quantisation step at the end of
     *     preprocess_frame(): both live at the stage boundary. */
    float gains[MODEL_NB_BANDS];
    for (int i = 0; i < MODEL_NB_BANDS; ++i)
    {
        gains[i] = dequantise_int8_to_f32(p_scratch->gains_int8[i],
                                          MODEL_OUTPUT_GAINS_SCALE,
                                          MODEL_OUTPUT_GAINS_ZP);
    }
    *p_vad_prob_out = dequantise_int8_to_f32(p_scratch->vad_prob_int8,
                                             MODEL_OUTPUT_VAD_PROB_SCALE,
                                             MODEL_OUTPUT_VAD_PROB_ZP);

    if (!p_scratch->silence) {
        /* 2.  Pitch-aware comb filter -- modifies scratch->X in place. */
        pitch_filter(p_scratch->X, p_scratch->P,
                     p_scratch->Ex, p_scratch->Ep, p_scratch->Exp,
                     gains);

        /* 3.  Gain smoothing across frames (denoise.c@1cbdbcf:506-510):
         *     g[i] = max(g[i], 0.6 * lastg[i]); lastg[i] = g[i]; */
        for (int i = 0; i < MODEL_NB_BANDS; ++i) {
            const float alpha = 0.6f;
            gains[i] = fmaxf(gains[i], alpha * p_st->lastg[i]);
            p_st->lastg[i] = gains[i];
        }

        /* 4.  22 band gains -> 481 FFT-bin gains. */
        interp_band_gain(gf, gains);

        /* 5.  Apply gains to the analysis spectrum. */
        for (int i = 0; i < MODEL_FREQ_SIZE; ++i) {
            p_scratch->X[i].r *= gf[i];
            p_scratch->X[i].i *= gf[i];
        }
    }

    /* 6.  IFFT + windowed overlap-add. Emits FRAME_SIZE float samples. */
    float out_f[MODEL_FRAME_SIZE];
    frame_synthesis(p_st, out_f, p_scratch->X);

    /* 7.  float -> int16 with saturation (host-side reference writes
     *     16-bit WAV via soundfile; we match that range). */
    for (int i = 0; i < MODEL_FRAME_SIZE; ++i) {
        p_pcm_out[i] = float_to_int16_saturate(out_f[i]);
    }
}
