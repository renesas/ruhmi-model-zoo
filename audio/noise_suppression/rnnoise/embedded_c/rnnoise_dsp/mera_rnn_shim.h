/*
 * Copyright (c) 2020 - 2026 Renesas Electronics Corporation and/or its affiliates
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
/**
 * @file mera_rnn_shim.h
 * @brief Shim header replacing upstream rnn.h/rnn_data.c with the MERA INT8 back-end.
 * @details
 *   Upstream rnnoise's denoise.c reaches into its own NN runtime
 *   (rnn.c / rnn.h / rnn_data.c) via three symbols:
 *
 *     - struct RNNState          per-stream GRU INT8 activations
 *     - struct RNNModel          read-only layer-size metadata
 *     - rnnoise_model_orig       default model instance
 * 
 *   both call compute_sub_0000() directly.
 *
 *   This header re-declares RNNState / RNNModel so denoise.c compiles
 *   unchanged below the MOD G.5 include-swap line.
 *
 *   rnnoise_model_orig is defined in ruhmi_perf_eval.c.
 *
 *   Nothing outside io_processing/rnnoise_dsp/ should include this
 *   header — it is the private cut-over point between the vendored DSP
 *   and the MERA NN back-end.
 *
 * @note SPDX-License-Identifier: BSD-3-Clause
 *       Renesas RA8P1 port: Copyright (c) 2020 - 2026 Renesas Electronics
 *       Corporation and/or its affiliates.
 */

#ifndef MERA_RNN_SHIM_H_
#define MERA_RNN_SHIM_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Read-only NN layer-size descriptor for the RNNoise GRU model.
 * @details Upstream's RNNModel carries weight/bias pointers; on this port
 *          only the three hidden-unit counts are kept. compute_sub_0000()
 *          bakes all shapes at code-generation time.
 */
typedef struct RNNModel {
    int vad_gru_size;       /**< @brief VAD GRU hidden units (24).    */
    int noise_gru_size;     /**< @brief Noise GRU hidden units (48).  */
    int denoise_gru_size;   /**< @brief Denoise GRU hidden units (96).*/
} RNNModel;

/* Standards-compliance alias (_t suffix per GSCE Section 7.1a). */
typedef RNNModel rnn_model_t;

/**
 * @brief Per-stream GRU hidden-state activations (INT8).
 * @details Upstream's RNNState uses float buffers; this port stores the
 *          three GRU states in INT8 (the native compute_sub_0000() I/O
 *          type) so DenoiseState layout is bit-identical to upstream.
 *          Buffer sizes are fixed by the model (see model_metadata.h §3).
 */
typedef struct RNNState {
    const RNNModel *model;                      /**< @brief Pointer to the model descriptor.      */
    int8_t vad_gru_state    [24];               /**< @brief VAD GRU hidden state    (INT8, 24 B). */
    int8_t noise_gru_state  [48];               /**< @brief Noise GRU hidden state  (INT8, 48 B). */
    int8_t denoise_gru_state[96];               /**< @brief Denoise GRU hidden state(INT8, 96 B). */
} RNNState;

/* Standards-compliance alias (_t suffix per GSCE Section 7.1a). */
typedef RNNState rnn_state_t;

/* ---- Built-in model — defined in ruhmi_perf_eval.c ----------------
 * Upstream ships a single global `rnnoise_model_orig` inside
 * `rnn_data.c`. We don't take rnn_data.c; the definition lives in
 * `ruhmi_perf_eval.c` (Phase G.6) instead, which is the firmware
 * TU that owns all model-side data. */
extern const RNNModel rnnoise_model_orig;

/* ---- NN inference entry point — defined in ruhmi_perf_eval.c ------
 * compute_rnn_int8() performs the INT8 NPU inference: copies quantised
 * inputs into model tensors, calls RunModel(), copies next-state outputs
 * back into rnn->*_gru_state, and returns raw INT8 gains/vad_prob.
 * denoise.c quantises features before calling and dequantises afterwards. */
void compute_rnn_int8(RNNState     *rnn,
                      int8_t       *p_gains_int8_out,
                      int8_t       *p_vad_prob_int8_out,
                      const int8_t *p_main_input_int8);

#ifdef __cplusplus
}
#endif

#endif /* MERA_RNN_SHIM_H_ */
