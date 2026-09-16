/* Copyright (c) 2009-2010 Xiph.Org Foundation
   Written by Jean-Marc Valin */
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
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/**
 * @file celt_lpc.h
 * @brief CELT linear-prediction utility API: LPC compute, FIR/IIR filters, autocorrelation.
 * @details
 *   Vendored from xiph/rnnoise \@1cbdbcf (Xiph.Org Foundation, Jean-Marc Valin).
 *   Declares four utility functions used by the pitch-estimation layer:
 *     - celt_lpc_compute()       Levinson-Durbin LPC analysis
 *     - celt_fir()               Direct-form FIR filter
 *     - celt_iir()               Direct-form IIR filter
 *     - celt_autocorr_compute()  Windowed autocorrelation
 *
 * @note Upstream BSD-3-Clause license retained above.
 *       Renesas RA8P1 port: Copyright (c) 2020 - 2026 Renesas Electronics
 *       Corporation and/or its affiliates.
 */

#ifndef PLC_H
#define PLC_H

#include "arch.h"
#include "common.h"

#if defined(OPUS_X86_MAY_HAVE_SSE4_1)
#include "x86/celt_lpc_sse.h"
#endif

#define LPC_ORDER 24

/**
 * @brief Compute LPC coefficients from autocorrelation values (Levinson-Durbin).
 * @param[out] _lpc LPC predictor coefficients [0..p-1] (opus_val16 = float).
 * @param[in]  ac   Autocorrelation values [0..p] (opus_val32 = float).
 * @param[in]  p    LPC order (LPC_ORDER = 24).
 */
void celt_lpc_compute(opus_val16 *p_lpc, const opus_val32 *p_ac, int p);

/**
 * @brief Apply a direct-form FIR filter to a signal.
 * @param[in]  x   Input signal, length N (opus_val16 = float).
 * @param[in]  num FIR numerator coefficients, length ord (opus_val16 = float).
 * @param[out] y   Filtered output, length N (opus_val16 = float).
 * @param[in]  N   Number of samples to filter.
 * @param[in]  ord Filter order.
 */
void celt_fir(
         const opus_val16 *p_x,
         const opus_val16 *p_num,
         opus_val16 *p_y,
         int N,
         int ord);

/**
 * @brief Apply a direct-form IIR filter to a signal.
 * @param[in]     x   Input signal, length N (opus_val32 = float).
 * @param[in]     den IIR denominator coefficients, length ord (opus_val16 = float).
 * @param[out]    y   Filtered output, length N (opus_val32 = float).
 * @param[in]     N   Number of samples to filter.
 * @param[in]     ord Filter order.
 * @param[in,out] mem Filter state memory, length ord (opus_val16 = float).
 */
void celt_iir(const opus_val32 *p_x,
         const opus_val16 *p_den,
         opus_val32 *p_y,
         int N,
         int ord,
         opus_val16 *p_mem);

/**
 * @brief Compute a windowed autocorrelation sequence.
 * @param[in]  x       Input signal, length n (opus_val16 = float).
 * @param[out] ac      Autocorrelation values [0..lag] (opus_val32 = float).
 * @param[in]  window  Analysis window, length overlap (opus_val16 = float), or NULL.
 * @param[in]  overlap Number of samples to apply the window over.
 * @param[in]  lag     Maximum lag index to compute.
 * @param[in]  n       Total input signal length.
 * @return 0 on success.
 */
int celt_autocorr_compute(const opus_val16 *p_x, opus_val32 *p_ac,
         const opus_val16 *p_window, int overlap, int lag, int n);

#endif /* PLC_H */
