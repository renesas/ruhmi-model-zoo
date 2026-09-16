/*
* Copyright (c) 2020 - 2026 Renesas Electronics Corporation and/or its affiliates
*
* SPDX-License-Identifier: BSD-3-Clause
*/
/* Periodic Hann window of length 400 (matches
 * tf.signal.stft default and features.py::_tflite_stft_magnitude).
 *
 * Storage for g_yamnet_hann lives in hann_window.c; this header exposes
 * only the extern declaration so no variable storage is declared in a
 * header file.
 */
#ifndef HANN_WINDOW_H
#define HANN_WINDOW_H

#define YAMNET_WINDOW_SAMPLES (400)

extern const float g_yamnet_hann[YAMNET_WINDOW_SAMPLES];


#endif /* HANN_WINDOW_H */
