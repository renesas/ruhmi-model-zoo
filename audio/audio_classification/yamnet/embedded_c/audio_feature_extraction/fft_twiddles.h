/*
* Copyright (c) 2020 - 2026 Renesas Electronics Corporation and/or its affiliates
*
* SPDX-License-Identifier: BSD-3-Clause
*/
/**********************************************************************************************************************
 * File Name    : fft_twiddles.h
 * Description  : Twiddle LUTs for a radix-2 DIT complex FFT of length 512.
 *                Entry i corresponds to W = exp(-j 2 pi i / 512), i = 0 .. 255.
 *                At stage m (m = 2, 4, ..., N) use index k * (N / m), k = 0 .. m/2 - 1.
 *                Storage for g_fft_cos and g_fft_sin lives in fft_twiddles.c; this header
 *                exposes only the extern declarations so no variable storage is declared
 *                in a header file.
 *********************************************************************************************************************/
#ifndef FFT_TWIDDLES_H
#define FFT_TWIDDLES_H

#define YAMNET_FFT_SIZE       (512)
#define YAMNET_FFT_HALF_SIZE  (256)
#define YAMNET_BIT_SHIFT_STEP (1U)

extern const float g_fft_cos[YAMNET_FFT_HALF_SIZE];
extern const float g_fft_sin[YAMNET_FFT_HALF_SIZE];


#endif /* FFT_TWIDDLES_H */
