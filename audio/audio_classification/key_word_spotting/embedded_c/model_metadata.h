/**
 * @file    model_metadata.h
 * @brief   Compile-time metadata for the KWS DS-CNN TFLite models (FP32 & INT8).
 *
 * Generated from:
 *   Model  : kws_ref_model_INT8.tflite / kws_ref_model_FP32.tflite
 *   Source : MLCommons Tiny benchmark — DS-CNN keyword spotting
 *   Task   : 12-class keyword classification (10 keywords + silence + unknown)
 *   Date   : 2026-04-07
 *
 * I/O details verified with:
 *   interpreter.get_input_details()[0]   → shape [1, 49, 10, 1]  int8  (INT8 model)
 *                                        → shape [1, 49, 10, 1]  float32 (FP32 model)
 *   interpreter.get_output_details()[0]  → shape [1, 12]         int8  (INT8 model)
 *                                        → shape [1, 12]         float32 (FP32 model)
 *
 * MERA compilation:
 *   CPU (quantized) : embedded_c/src_mcu/kws_ref_model_INT8_CPU/
 *   NPU (quantized) : embedded_c/src_mcu_npu/kws_ref_model_INT8_NPU/
 *
 * ──────────────────────────────────────────────────────────────────────────
 * QUICK-START (bare-metal / TFLite Micro — INT8 model)
 * ──────────────────────────────────────────────────────────────────────────
 *
 *  1. Capture  : 1 second of 16 kHz mono PCM audio → 16 000 float32 samples
 *  2. Normalise: divide every sample by max(|audio|) → range [-1, 1]
 *  3. Zero-pad or truncate to exactly 16 000 samples
 *  4. STFT     : frame_length=480, frame_step=320, Hann window → 49 frames
 *  5. Mel FB   : 40 mel bins, 20 Hz – 4 000 Hz → shape (49, 40)
 *  6. Log-mel  : log(mel + 1e-6)
 *  7. MFCC     : DCT, keep first 10 coefficients → shape (49, 10)
 *  8. Reshape  : add channel dim → [1, 49, 10, 1]
 *  9. Quantize : q = clamp(round(mfcc / INPUT_SCALE) + INPUT_ZP, -128, 127)
 * 10. Run inference
 * 11. Dequantize output: prob[i] = (out_q[i] - OUTPUT_ZP) * OUTPUT_SCALE
 * 12. argmax(prob) → predicted class index → KWS_LABEL_*
 *
 * ──────────────────────────────────────────────────────────────────────────
 *
 * Copyright 2026 Renesas Electronics Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef MODEL_METADATA_H
#define MODEL_METADATA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif


/* ══════════════════════════════════════════════════════════════════════
 * 1.  MODEL IDENTITY
 * ══════════════════════════════════════════════════════════════════════ */

/** Human-readable model name. */
#define MODEL_NAME              "KWS DS-CNN"

/** Model source / benchmark. */
#define MODEL_SOURCE            "MLCommons Tiny benchmark — DS-CNN"

/** Relative paths to TFLite flatbuffers (from project root). */
#define MODEL_FILE_INT8         "model/kws_ref_model_INT8.tflite"
#define MODEL_FILE_FP32         "model/kws_ref_model_FP32.tflite"

/** Task performed by the model. */
#define MODEL_TASK              "Keyword Spotting (12-class audio classification)"

/** Model file sizes (bytes). */
#define CPU_RAM_TENSOR_ARENA_BYTES   20465    /* ~20 KB  */
#define NPU_RAM_TENSOR_ARENA_BYTES   16000    /* ~16 KB  */

/** Number of output classes. */
#define MODEL_NUM_CLASSES       12       /* 10 keywords + silence + unknown */


/* ══════════════════════════════════════════════════════════════════════
 * 2.  INPUT TENSOR
 *     Verified with: interpreter.get_input_details()[0]
 *     Name  : "serving_default_input_1_0"
 *     Index : 0
 *     Shape : [1, 49, 10, 1]  (batch, frames, MFCC coefficients, channel)
 *     DType : int8  (INT8 model) / float32 (FP32 model)
 * ══════════════════════════════════════════════════════════════════════ */

/** Number of MFCC time frames. */
#define MODEL_INPUT_FRAMES      49

/** Number of MFCC DCT coefficients per frame. */
#define MODEL_INPUT_COEFFS      10

/** Number of input channels (mono audio = 1). */
#define MODEL_INPUT_CHANNELS    1

/** Flat element count of one input tensor. */
#define MODEL_INPUT_SIZE        (MODEL_INPUT_FRAMES * MODEL_INPUT_COEFFS \
                                 * MODEL_INPUT_CHANNELS)  /* = 490 */

/** Memory layout of the input tensor. */
#define MODEL_INPUT_LAYOUT      "NHWC"  /* N=1, H=frames, W=coeffs, C=1 */

/**
 * INT8 input quantization scale.
 * Source: interpreter.get_input_details()[0]['quantization_parameters']['scales'][0]
 *
 * Maps int8 value q → float MFCC via:  f = (q - INPUT_ZP) * INPUT_SCALE
 * To quantize float MFCC → int8:       q = clamp(round(f / INPUT_SCALE) + INPUT_ZP, -128, 127)
 *
 * Note: MFCC values are NOT normalised to a fixed range — they can span
 * roughly [-30, 30] depending on the audio content. The scale/zp here
 * are the calibrated quantization parameters from the INT8 model.
 */
#define INPUT_SCALE             0.5821282267570496f

/**
 * INT8 input quantization zero-point.
 * Source: interpreter.get_input_details()[0]['quantization_parameters']['zero_points'][0]
 */
#define INPUT_ZP                84

/** Byte size of one INT8 input tensor buffer. */
#define MODEL_INPUT_BUF_BYTES   (MODEL_INPUT_SIZE * sizeof(int8_t))   /* 490 B */

/** Byte size of one FP32 input tensor buffer. */
#define MODEL_INPUT_FP32_BUF_BYTES  (MODEL_INPUT_SIZE * sizeof(float))  /* 1 960 B */


/* ══════════════════════════════════════════════════════════════════════
 * 3.  OUTPUT TENSOR
 *     Verified with: interpreter.get_output_details()[0]
 *     Name  : "StatefulPartitionedCall_0_70033"
 *     Index : 0
 *     Shape : [1, 12]
 *     DType : int8  (INT8 model) / float32 (FP32 model)
 * ══════════════════════════════════════════════════════════════════════ */

/**
 * INT8 output dequantization scale.
 * Source: interpreter.get_output_details()[0]['quantization_parameters']['scales'][0]
 * Dequantize: prob = (out_q - OUTPUT_ZP) * OUTPUT_SCALE
 *
 * The model's final layer is softmax, so dequantized outputs are already
 * probabilities in [0, ~1] that sum to ≈ 1.0. Do NOT apply softmax again.
 */
#define OUTPUT_SCALE            0.00390625f   /* = 1/256 */

/**
 * INT8 output dequantization zero-point.
 * Source: interpreter.get_output_details()[0]['quantization_parameters']['zero_points'][0]
 */
#define OUTPUT_ZP               (-128)

/** Dequantized probability range. */
#define OUTPUT_FLOAT_MIN        0.0f          /* q=-128 → (−128−(−128)) * 1/256 = 0.0    */
#define OUTPUT_FLOAT_MAX        0.99609375f   /* q= 127 → ( 127−(−128)) * 1/256 ≈ 0.996 */

/** 1 = softmax already baked into graph — do NOT re-apply after dequantization. */
#define OUTPUT_HAS_SOFTMAX      1

/** Byte size of one INT8 output tensor buffer. */
#define MODEL_OUTPUT_BUF_BYTES  (MODEL_NUM_CLASSES * sizeof(int8_t))   /* 12 B */


/* ══════════════════════════════════════════════════════════════════════
 * 4.  AUDIO & MFCC PARAMETERS  (must match training exactly)
 * ══════════════════════════════════════════════════════════════════════ */

/** Input sample rate (Hz). */
#ifndef SAMPLE_RATE
#define SAMPLE_RATE             16000
#endif

/** Required number of input samples (1 second @ 16 kHz). */
#define DESIRED_SAMPLES         16000

/** STFT window length in samples (30 ms @ 16 kHz). */
#define WINDOW_SIZE_SAMPLES     480

/** STFT hop / stride in samples (20 ms @ 16 kHz). */
#define WINDOW_STRIDE_SAMPLES   320

/** Number of STFT frequency bins = WINDOW_SIZE_SAMPLES/2 + 1. */
#define NUM_SPECTROGRAM_BINS    241    /* = 480/2 + 1 */

/** Number of mel filterbank bins. */
#define NUM_MEL_BINS            40

/** Lower frequency boundary of mel filterbank (Hz). */
#define MEL_LOWER_HZ            20.0f

/** Upper frequency boundary of mel filterbank (Hz). */
#define MEL_UPPER_HZ            4000.0f

/** Number of MFCC DCT coefficients retained (= MODEL_INPUT_COEFFS). */
#define DCT_COEFFICIENT_COUNT   10

/**
 * Number of MFCC time frames output by the STFT.
 * = ceil((DESIRED_SAMPLES - WINDOW_SIZE_SAMPLES) / WINDOW_STRIDE_SAMPLES) + 1
 * = ceil((16000 - 480) / 320) + 1 = 48 + 1 = 49
 */
#define NUM_MFCC_FRAMES         49     /* = MODEL_INPUT_FRAMES */


/* ══════════════════════════════════════════════════════════════════════
 * 5.  12-CLASS LABEL TABLE
 *     Order MUST match the MLCommons Tiny training label map exactly.
 * ══════════════════════════════════════════════════════════════════════ */

#define KWS_LABEL_DOWN          0
#define KWS_LABEL_GO            1
#define KWS_LABEL_LEFT          2
#define KWS_LABEL_NO            3
#define KWS_LABEL_OFF           4
#define KWS_LABEL_ON            5
#define KWS_LABEL_RIGHT         6
#define KWS_LABEL_STOP          7
#define KWS_LABEL_UP            8
#define KWS_LABEL_YES           9
#define KWS_LABEL_SILENCE      10
#define KWS_LABEL_UNKNOWN      11

/**
 * Null-terminated string array — index matches KWS_LABEL_* above.
 * Usage (C):
 *   const char *label = KWS_CLASS_NAMES[predicted_class];
 */
#define KWS_CLASS_NAMES  \
    { "down", "go", "left", "no", "off", "on", \
      "right", "stop", "up", "yes", "silence", "unknown" }


/* ══════════════════════════════════════════════════════════════════════
 * 6.  POST-PROCESSING PARAMETERS
 * ══════════════════════════════════════════════════════════════════════ */

/**
 * Minimum softmax probability to report a "positive" keyword detection.
 * Applied after dequantization (INT8) or directly on float output (FP32).
 * Increase to reduce false positives; decrease for higher sensitivity.
 */
#define KWS_CONFIDENCE_THRESHOLD  0.5f   /* 50 % confidence — tune per use-case */

/**
 * Class index considered "silence" — detections below threshold are
 * treated as silence rather than an error.
 */
#define KWS_SILENCE_CLASS_IDX   KWS_LABEL_SILENCE   /* = 10 */

/**
 * Class index for "unknown" — spoken words not in the 10-keyword set.
 */
#define KWS_UNKNOWN_CLASS_IDX   KWS_LABEL_UNKNOWN   /* = 11 */


/* ══════════════════════════════════════════════════════════════════════
 * 7.  MERA COMPILATION TARGETS
 * ══════════════════════════════════════════════════════════════════════ */

/**
 * CPU MERA compiled C-code (quantized via TFLite quantizer).
 *   Source tree: embedded_c/src_mcu/kws_ref_model_INT8_CPU/
 */
#define MERA_CPU_SOURCE_DIR   "embedded_c/src_mcu/kws_ref_model_INT8_CPU/"

/**
 * NPU MERA compiled C-code (ARM Ethos-U55, quantized via TFLite quantizer).
 *   Source tree: embedded_c/src_mcu_npu/kws_ref_model_INT8_NPU/
 */
#define MERA_NPU_SOURCE_DIR   "embedded_c/src_mcu_npu/kws_ref_model_INT8_NPU/"


#ifdef __cplusplus
}
#endif

#endif /* MODEL_METADATA_H */
