/*
 * Copyright (c) 2020 - 2026 Renesas Electronics Corporation and/or its affiliates
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
/*
 * preprocessing.c
 *
 * Pre-NN DSP entry point for a single 10 ms RNNoise frame.
 * Converts PCM input into features and frame scratch state.
 */

#include <stdint.h>
#include <math.h>    /* roundf() */

#include "preprocessing.h"
#include "rnnoise_dsp/denoise_internal.h"  /* biquad, compute_frame_features */

/* quantise_f32_to_int8() — symmetric per-tensor INT8 quantisation
 * matching TFLite's reference formula:
 *   q = clip( round(x / scale) + zero_point, -128, 127 ) */
int8_t quantise_f32_to_int8(float x, float scale, int32_t zero_point)
{
    float q = roundf(x / scale) + (float)zero_point;

    if (-128.0f > q) { q = -128.0f; }
    if ( 127.0f < q) { q =  127.0f; }
    return (int8_t)q;
}

/* preprocess_frame() — one-shot front-end DSP for a single 10 ms frame.
 * Converts 480 int16 PCM samples into 42 float features and per-frame
 * DSP scratch state needed by postprocess_frame(). */
void preprocess_frame(DenoiseState *p_st,
                      const int16_t p_pcm_in[MODEL_FRAME_SIZE],
                      float p_features[MODEL_NB_FEATURES],
                      preprocess_scratch_t *p_scratch)
{
    static const float a_hp[2] = { -1.99599f,  0.99600f };
    static const float b_hp[2] = { -2.0f,      1.0f      };

    /* 1.  int16 -> float (raw amplitudes, no normalisation). */
    float x_in[MODEL_FRAME_SIZE];
    for (int i = 0; i < MODEL_FRAME_SIZE; ++i) {
        x_in[i] = (float)p_pcm_in[i];
    }

    /* 2.  HP-filter — produces the float frame that the feature
     *     extractor sees. Output goes into a separate buffer because
     *     compute_frame_features() reads its input as `const`. The
     *     2-tap filter memory `st->mem_hp_x` is carried across calls
     *     by upstream's DenoiseState contract. */
    float x_hp[MODEL_FRAME_SIZE];
    biquad(x_hp, p_st->mem_hp_x, x_in, b_hp, a_hp, MODEL_FRAME_SIZE);

    /* 3.  Full DSP front-end. Writes into scratch->{X,P,Ex,Ep,Exp},
     *     fills `features`, and returns the silence flag. */
    p_scratch->silence = compute_frame_features(
        p_st,
        p_scratch->X,
        p_scratch->P,
        p_scratch->Ex,
        p_scratch->Ep,
        p_scratch->Exp,
        p_features,
        x_hp
    );

    /* 4.  Quantise the 42 float features to INT8 so () can
     *     pass them directly to compute_sub_0000() without a second
     *     quantisation pass inside the inference wrapper. */
    for (int i = 0; i < MODEL_INPUT_FEATURES_SIZE; ++i)
    {
        p_scratch->features_int8[i] = quantise_f32_to_int8(
            p_features[i],
            MODEL_INPUT_FEATURES_SCALE,
            MODEL_INPUT_FEATURES_ZP);
    }
}

