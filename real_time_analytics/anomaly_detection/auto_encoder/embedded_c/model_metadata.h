/**
 * @file    model_metadata.h
 * @brief   Compile-time metadata for Anomaly Detection AD01 model (CPU-only).
 *
 *
 * Generated from:
 *   Model  : ad01_INT8.tflite  (MLCommons Tiny AD01 autoencoder)
 *   Task   : Anomaly Detection (reconstruction-based)
 *   Target : RA8P1 CPU (MERA C-codegen, float32 I/O)
 *
 * ----------------------------------------------------------------------
 * QUICK-START (bare-metal)
 * ----------------------------------------------------------------------
 *
 *  1. Load  : raw int16 PCM audio from header file
 *  2. Preprocess : STFT -> Mel filterbank -> log-dB -> crop -> sliding window
 *  3. Run inference via compute_sub_0000()  (float32 I/O)
 *  4. Compute MSE between preprocessed input and reconstructed output
 *  5. Anomaly score = mean MSE across all sliding-window vectors
 *  6. Compare score against ANOMALY_THRESHOLD
 *
 * ----------------------------------------------------------------------
 */

#ifndef MODEL_METADATA_H
#define MODEL_METADATA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif


/* ======================================================================
 * 1.  MODEL IDENTITY
 * ====================================================================== */

/** Human-readable model name. */
#define MODEL_NAME              "AD01 Anomaly Detection Autoencoder"

/** Task performed by the model. */
#define MODEL_TASK              "Anomaly Detection"

/** Model data type (MERA compiled - float32 I/O). */
#define MODEL_DTYPE             "float32"


/* ======================================================================
 * 2.  INPUT TENSOR
 *     Shape : [1, 640]
 *     DType : float32 (MERA compiled)
 * ====================================================================== */

/** Number of mel bins in the spectrogram. */
#define N_MELS                  (128)

/** Number of sliding-window frames concatenated. */
#define FRAMES                  (5)

/** Flat feature dimension (N_MELS x FRAMES). */
#define INPUT_DIM               (N_MELS * FRAMES)   /* 640 */


/* ======================================================================
 * 3.  OUTPUT TENSOR
 *     Shape : [1, 640]
 *     DType : float32 (MERA compiled - autoencoder reconstruction)
 * ====================================================================== */

/** Flat output dimension (equals INPUT_DIM). */
#define OUTPUT_DIM              (640)


/* ======================================================================
 * 4.  DSP / PREPROCESSING PARAMETERS
 * ====================================================================== */

/** FFT size for STFT. */
#define PREP_N_FFT              (1024)

/** Hop length for STFT. */
#define PREP_HOP                (512)

/** Number of mel filter-bank bins. */
#define PREP_N_MELS             N_MELS   /* 128 */

/** Power exponent for mel spectrogram (2.0 = power). */
#define PREP_POWER              (2.0f)

/** Start column of the central crop (inclusive). */
#define PREP_CROP_START         (50)

/** End column of the central crop (exclusive). */
#define PREP_CROP_END           (250)

/** Number of sliding-window frames. */
#define PREP_FRAMES             FRAMES   /* 5 */

/** Small constant to avoid log(0). */
#define PREP_EPSILON            (1e-10f)

/** PCM normalization factor for int16_t samples (-32768 to 32767). */
#define PCM_NORM_FACTOR         (32768.0f)

/** Number of spectrogram columns after crop. */
#define PREP_CROP_COLS          (PREP_CROP_END - PREP_CROP_START)  /* 200 */

/** Number of feature vectors produced per audio clip. */
#define NUM_VECTORS_PER_CLIP    (PREP_CROP_COLS - PREP_FRAMES + 1) /* 196 */

/** Number of real-valued FFT output bins used in preprocessing. */
#define LOCAL_PREP_N_BINS          ((PREP_N_FFT / 2) + 1)

/** Maximum supported STFT frame count for preprocessing buffers. */
#define LOCAL_PREP_MAX_STFT_FRAMES (360)

#ifndef M_PI
#  define M_PI 3.14159265358979323846f
#endif

/** Slaney mel conversion constant: minimum frequency in Hz. */
#define MEL_F_MIN       0.0f

/** Slaney mel conversion constant: linear-frequency scaling factor. */
#define MEL_F_SP        (200.0f / 3.0f)

/** Slaney mel conversion constant: transition frequency to log region in Hz. */
#define MEL_MIN_LOG_HZ  1000.0f

/** Slaney mel conversion constant: mel value at MEL_MIN_LOG_HZ. */
#define MEL_MIN_LOG_MEL ((MEL_MIN_LOG_HZ - MEL_F_MIN) / MEL_F_SP)

/** Slaney mel conversion constant: logarithmic step value. */
#define MEL_LOGSTEP     (0.06875177742094912f)    /* log(6.4)/27 */


/* ======================================================================
 * 5.  POST-PROCESSING PARAMETERS
 * ====================================================================== */

/**
 * Anomaly score = mean MSE across all sliding-window vectors of one clip.
 * Scores ABOVE this threshold are classified as anomalous.
 */
#define ANOMALY_THRESHOLD       (10.7f)


/* ======================================================================
 * 6.  TEST CONFIGURATION
 * ====================================================================== */

/** Number of test audio clips (normal + anomaly). */
#define NUM_TEST_CLIPS          (4U)


#ifdef __cplusplus
}
#endif

#endif /* MODEL_METADATA_H */
