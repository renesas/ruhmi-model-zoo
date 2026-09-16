/* Copyright (c) 2007-2008 CSIRO
   Copyright (c) 2007-2009 Xiph.Org Foundation
   Written by Jean-Marc Valin */
/**
 * @file pitch.h
 * @brief Pitch analysis API: downsample, search, and octave-doubling removal.
 * @details
 *   Vendored from xiph/rnnoise \@1cbdbcf (CSIRO / Xiph.Org Foundation,
 *   Jean-Marc Valin). Public declarations for the three pitch-analysis
 *   entry points and the inline xcorr_kernel used by the search.
 *
 *   All arithmetic is floating-point (FIXED_POINT not defined for this port).
 *
 * @note Upstream BSD-3-Clause license retained above.
 *       Renesas RA8P1 port: Copyright (c) 2020 - 2026 Renesas Electronics
 *       Corporation and/or its affiliates.
 */

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

#ifndef PITCH_H
#define PITCH_H

//#include "modes.h"
//#include "cpu_support.h"
#include "arch.h"

/**
 * @brief Downsample a multi-channel PCM signal by 2x for pitch analysis.
 * @param[in]  x     Array of C channel pointers, each of length len (celt_sig = float).
 * @param[out] x_lp  Downsampled output, length len/2 (opus_val16 = float).
 * @param[in]  len   Number of input samples per channel.
 * @param[in]  C     Number of channels (1 for mono RNNoise use).
 */
void pitch_downsample(celt_sig *p_x[], opus_val16 *p_x_lp,
      int len, int C);

/**
 * @brief Search for the best pitch period by normalised cross-correlation.
 * @param[in]  x_lp      Downsampled analysis frame (opus_val16 = float).
 * @param[in]  y         Downsampled pitch search buffer (opus_val16 = float).
 * @param[in]  len       Length of the analysis frame.
 * @param[in]  max_pitch Maximum pitch period to search (in samples).
 * @param[out] pitch     Best pitch period found (in samples).
 */
void pitch_search(const opus_val16 *p_x_lp, opus_val16 *p_y,
                  int len, int max_pitch, int *p_pitch);

/**
 * @brief Remove octave doublings from a candidate pitch period.
 * @details Checks whether the proposed pitch T0 is an octave doubling of a
 *          shorter period and corrects it using the signal autocorrelation.
 * @param[in]     x           Pitch buffer (opus_val16 = float), length maxperiod.
 * @param[in]     maxperiod   Maximum allowed pitch period (samples).
 * @param[in]     minperiod   Minimum allowed pitch period (samples).
 * @param[in]     N           Analysis frame length (samples).
 * @param[in,out] T0          On entry: initial pitch estimate; on exit: corrected period.
 * @param[in]     prev_period Previous frame's pitch period.
 * @param[in]     prev_gain   Previous frame's pitch gain.
 * @return Estimated pitch gain for the corrected period.
 */
opus_val16 remove_doubling(opus_val16 *p_x, int maxperiod, int minperiod,
      int N, int *p_T0, int prev_period, opus_val16 prev_gain);


/* OPT: This is the kernel you really want to optimize. It gets used a lot
   by the prefilter and by the PLC. */
static OPUS_INLINE void xcorr_kernel(const opus_val16 *p_x, const opus_val16 *p_y, opus_val32 p_sum[4], int len)
{
   int j;
   opus_val16 y_0, y_1, y_2, y_3;
   celt_assert(len>=3);
   y_3=0; /* gcc doesn't realize that y_3 can't be used uninitialized */
   y_0=*p_y++;
   y_1=*p_y++;
   y_2=*p_y++;
   for (j=0;j<len-3;j+=4)
   {
      opus_val16 tmp;
      tmp = *p_x++;
      y_3=*p_y++;
      p_sum[0] = MAC16_16(p_sum[0],tmp,y_0);
      p_sum[1] = MAC16_16(p_sum[1],tmp,y_1);
      p_sum[2] = MAC16_16(p_sum[2],tmp,y_2);
      p_sum[3] = MAC16_16(p_sum[3],tmp,y_3);
      tmp=*p_x++;
      y_0=*p_y++;
      p_sum[0] = MAC16_16(p_sum[0],tmp,y_1);
      p_sum[1] = MAC16_16(p_sum[1],tmp,y_2);
      p_sum[2] = MAC16_16(p_sum[2],tmp,y_3);
      p_sum[3] = MAC16_16(p_sum[3],tmp,y_0);
      tmp=*p_x++;
      y_1=*p_y++;
      p_sum[0] = MAC16_16(p_sum[0],tmp,y_2);
      p_sum[1] = MAC16_16(p_sum[1],tmp,y_3);
      p_sum[2] = MAC16_16(p_sum[2],tmp,y_0);
      p_sum[3] = MAC16_16(p_sum[3],tmp,y_1);
      tmp=*p_x++;
      y_2=*p_y++;
      p_sum[0] = MAC16_16(p_sum[0],tmp,y_3);
      p_sum[1] = MAC16_16(p_sum[1],tmp,y_0);
      p_sum[2] = MAC16_16(p_sum[2],tmp,y_1);
      p_sum[3] = MAC16_16(p_sum[3],tmp,y_2);
   }
   if (j++<len)
   {
      opus_val16 tmp = *p_x++;
      y_3=*p_y++;
      p_sum[0] = MAC16_16(p_sum[0],tmp,y_0);
      p_sum[1] = MAC16_16(p_sum[1],tmp,y_1);
      p_sum[2] = MAC16_16(p_sum[2],tmp,y_2);
      p_sum[3] = MAC16_16(p_sum[3],tmp,y_3);
   }
   if (j++<len)
   {
      opus_val16 tmp=*p_x++;
      y_0=*p_y++;
      p_sum[0] = MAC16_16(p_sum[0],tmp,y_1);
      p_sum[1] = MAC16_16(p_sum[1],tmp,y_2);
      p_sum[2] = MAC16_16(p_sum[2],tmp,y_3);
      p_sum[3] = MAC16_16(p_sum[3],tmp,y_0);
   }
   if (j<len)
   {
      opus_val16 tmp=*p_x++;
      y_1=*p_y++;
      p_sum[0] = MAC16_16(p_sum[0],tmp,y_2);
      p_sum[1] = MAC16_16(p_sum[1],tmp,y_3);
      p_sum[2] = MAC16_16(p_sum[2],tmp,y_0);
      p_sum[3] = MAC16_16(p_sum[3],tmp,y_1);
   }
}

static OPUS_INLINE void dual_inner_prod(const opus_val16 *p_x, const opus_val16 *p_y01, const opus_val16 *p_y02,
      int N, opus_val32 *p_xy1, opus_val32 *p_xy2)
{
   int i;
   opus_val32 xy01=0;
   opus_val32 xy02=0;
   for (i=0;i<N;i++)
   {
      xy01 = MAC16_16(xy01, p_x[i], p_y01[i]);
      xy02 = MAC16_16(xy02, p_x[i], p_y02[i]);
   }
   *p_xy1 = xy01;
   *p_xy2 = xy02;
}

/*We make sure a C version is always available for cases where the overhead of
  vectorization and passing around an arch flag aren't worth it.*/
static OPUS_INLINE opus_val32 celt_inner_prod(const opus_val16 *p_x,
      const opus_val16 *p_y, int N)
{
   int i;
   opus_val32 xy=0;
   for (i=0;i<N;i++)
      xy = MAC16_16(xy, p_x[i], p_y[i]);
   return xy;
}

void celt_pitch_xcorr(const opus_val16 *p_x, const opus_val16 *p_y,
      opus_val32 *p_xcorr, int len, int max_pitch);

#endif
