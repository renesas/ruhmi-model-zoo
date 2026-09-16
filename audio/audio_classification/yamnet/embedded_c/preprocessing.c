/*
* Copyright (c) 2020 - 2026 Renesas Electronics Corporation and/or its affiliates
*
* SPDX-License-Identifier: BSD-3-Clause
*/
/**
 * @file preprocessing.c
 * @brief YAMNet pre-processing implementation for waveform framing, FFT, mel filtering,
 * patch extraction, and model-boundary quantization.
 */

#include <math.h>

#include "preprocessing.h"
#include "model_metadata.h"
#include "audio_feature_extraction/frontend.h"

/* Per-patch float scratch used by preprocess() to hold one extracted 96x64
 * log-mel patch before it is quantized into the caller's output buffer.
 * Kept file-static (not on the stack) because YAMNET_PATCH_ELEMENTS * 4 B is
 * 24 KB. preprocess() is not re-entrant. */
static float s_patch_scratch[YAMNET_PATCH_ELEMENTS];

/**
 * @brief Quantize a float patch into int16 values using the model input scale and zero point.
 * @param[in] p_src_patch Pointer to the float source patch.
 * @param[out] p_dst_q_patch Pointer to the destination int16 buffer.
 * @param[in] elem_count Number of elements to process.
 */
static void yamnet_quantize_patch_int16(const float *p_src_patch,
                                        int16_t     *p_dst_q_patch,
                                        size_t       _elem_count)
{
    size_t      _elem_idx;
    float       _scaled;
    int32_t     _quant_val;

    for (_elem_idx = 0; _elem_idx < _elem_count; _elem_idx++)
    {
        _scaled    = p_src_patch[_elem_idx] / MODEL_INPUT_SCALE + (float)MODEL_INPUT_ZERO_POINT;
        _quant_val = (int32_t)nearbyintf(_scaled);

        if (_quant_val < -32768)
        {
            _quant_val = -32768;
        }
        else if (_quant_val > 32767)
        {
            _quant_val = 32767;
        }
        else
        {
            /* _quant_val is already within [-32768, 32767]; no clipping required. */
        }

        p_dst_q_patch[_elem_idx] = (int16_t)_quant_val;
    }
}

/**
 * @brief Run the full YAMNet pre-processing pipeline on an input waveform.
 * @param[in] p_waveform Pointer to int16 PCM input samples.
 * @param[in] num_samples Length of `p_waveform` in samples.
 * @param[out] p_log_mel_scratch Caller-owned scratch buffer for log-mel rows.
 * @param[in] max_rows Capacity of `p_log_mel_scratch` expressed in rows.
 * @param[out] p_quantized_patches Caller-owned output buffer for quantized int16 patches.
 * @param[in] max_patches Capacity of `p_quantized_patches` expressed in patches.
 * @return Number of quantized patches produced, or `0` on invalid input or a short clip.
 */
size_t preprocess(const int16_t *p_waveform,
                  size_t         _num_samples,
                  float         *p_log_mel_scratch,
                  size_t         _max_rows,
                  int16_t       *p_quantized_patches,
                  size_t         _max_patches)
{
    size_t _num_rows;
    size_t _num_patches;
    size_t _patch_idx;

    if ((NULL == p_waveform) || (NULL == p_log_mel_scratch) || (NULL == p_quantized_patches))
    {
        return 0U;
    }

    /* Steps 1-6: raw int16 waveform -> log-mel row stream. */
    _num_rows = yamnet_frontend_compute_log_mel(p_waveform,
                                                _num_samples,
                                                p_log_mel_scratch,
                                                _max_rows);
    if (0U == _num_rows)
    {
        return 0U;
    }

    /* Step 7: slice into 96-row patches with hop 48. */
    _num_patches = yamnet_frontend_num_patches(_num_rows);
    if (_num_patches > _max_patches)
    {
        _num_patches = _max_patches;
    }

    /* Step 8: per-patch model-boundary quantize into the caller's buffer. */
    for (_patch_idx = 0U; _patch_idx < _num_patches; _patch_idx++)
    {
        (void)yamnet_frontend_extract_patch(p_log_mel_scratch,
                                            _num_rows,
                                            _patch_idx,
                                            s_patch_scratch);

        yamnet_quantize_patch_int16(s_patch_scratch,
                                    &p_quantized_patches[_patch_idx * YAMNET_PATCH_ELEMENTS],
                                    (size_t)YAMNET_PATCH_ELEMENTS);
    }

    return _num_patches;
}
