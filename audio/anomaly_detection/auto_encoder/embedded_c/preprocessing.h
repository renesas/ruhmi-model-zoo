/**
 * @file    preprocessing.h
 * @brief   Log-mel spectrogram feature extraction from raw int16 PCM (embedded).
 *
 * DSP pipeline (no heap allocation — all static buffers):
 *
 *   int16 PCM ──► STFT (n_fft=1024, hop=512, Hann periodic)
 *             ──► Mel filterbank (128 bins, Slaney norm)
 *             ──► log-dB : 10.0f * log10f(max(mel, 1e-10))
 *             ──► central crop columns [50, 250)
 *             ──► sliding window width=5 → 196 × 640 float32 vectors
 *
 * @author  Renesas Electronics
 * @date    2025
 * @version 2.0  Revised for MISRA C 2025 (ISO/IEC 9899:2018) compliance
 */

#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include <stdint.h>

/**
 * @brief  Extract sliding-window log-mel spectrogram feature vectors from raw
 *         int16 PCM audio.
 *
 * Implements the full DSP pipeline (STFT → Mel filterbank → log-dB → crop →
 * sliding-window vectorisation) with static-only buffers — no malloc/free.
 * Output vectors are written into the caller-provided buffer.
 *
 * @param[in]  p_pcm            Mono int16_t PCM samples
 * @param[in]  num_samples    Number of PCM samples
 * @param[in]  sample_rate    Sample rate in Hz (e.g. 16000)
 * @param[out] p_output_vectors Pre-allocated float buffer [NUM_VECTORS_PER_CLIP][INPUT_DIM]
 * @param[out] p_output_count   Number of feature vectors produced
 * @return     0 on success, -1 on error (audio too short or too long)
 */
int preprocess(const int16_t *p_pcm,
               int            num_samples,
               int            sample_rate,
               float         *p_output_vectors,
               int           *p_output_count);

#endif /* PREPROCESSING_H */
