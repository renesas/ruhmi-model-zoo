/*
* Copyright (c) 2020 - 2026 Renesas Electronics Corporation and/or its affiliates
*
* SPDX-License-Identifier: BSD-3-Clause
*/
/* Mel filterbank weights for YAMNet, matching tf.signal.linear_to_mel_weight_matrix with
 *   num_mel_bins         = 64
 *   num_spectrogram_bins = 257
 *   sample_rate          = 16000
 *   lower_edge_hertz     = 125.0
 *   upper_edge_hertz     = 7500.0
 *
 * Layout: mel_matrix[spectrogram_bin][mel_bin].
 * Post-multiply the magnitude spectrogram by this table:
 *   mel_energy[m] = sum_over_k (mag[k] * mel_matrix[k][m])
 *
 * Storage for g_mel_matrix lives in mel_filterbank.c; this header exposes
 * only the extern declaration so no variable storage is declared in a
 * header file.
 */
#ifndef MEL_FILTERBANK_H
#define MEL_FILTERBANK_H

#define YAMNET_NUM_MEL_BINS       (64)
#define YAMNET_NUM_SPECTROGRAM_BINS (257)

extern const float g_mel_matrix[YAMNET_NUM_SPECTROGRAM_BINS][YAMNET_NUM_MEL_BINS];


#endif /* MEL_FILTERBANK_H */
