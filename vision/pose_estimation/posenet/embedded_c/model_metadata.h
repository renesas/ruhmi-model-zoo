/**
 * @file    model_metadata.h
 * @brief   Compile-time metadata for the PoseNet (MobileNetV1-050 @ 257x257)
 *          model deployed on the RA8P1 CPU target.
 *
 * Source model : posenet_mbv1_050_257_INT8.tflite
 *                (Google TF.js PoseNet MobileNet-050, 17 COCO keypoints, full INT8)
 *
 * Boundary tensors exposed to host code:
 *   Input  : serving_default_input_0          shape=[1,257,257,3]  dtype=int8
 *            scale=0.007843137718737125  zero_point=-1
 *            (host preprocessor: bilinear resize -> normalize -> quantize to int8)
 *   Outputs (4 heads, all NHWC on a 17x17 grid, output_stride = 16):
 *     PartitionedCall_0_70090  heatmap          shape=[1,17,17,17]  dtype=int8
 *                              scale=0.00390625  zero_point=-128
 *                              (sigmoid already applied in graph; dequant range [0, 1])
 *     PartitionedCall_1_70091  offset           shape=[1,17,17,34]  dtype=int8
 *                              scale=0.5891178846359253  zero_point=-1
 *                              (channels 0..16 = dy per keypoint,
 *                               channels 17..33 = dx per keypoint, in input-pixel units)
 *     PartitionedCall_2_70092  displacement_fwd shape=[1,17,17,32]  dtype=int8
 *                              scale=1.3544772863388062  zero_point=-63
 *                              (used by the PersonLab multi-pose decoder only)
 *     PartitionedCall_3_70093  displacement_bwd shape=[1,17,17,32]  dtype=int8
 *                              scale=1.1400184631347656  zero_point=41
 *                              (used by the PersonLab multi-pose decoder only)
 *
 * The on-MCU decoder ignores the two displacement heads and performs a
 * single-person argmax decode over the heatmap, refined by the offset head.
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
#define MODEL_NAME           ("PoseNet-MobileNetV1-050")
#define MODEL_FILE_INT8      ("posenet_mbv1_050_257_INT8.tflite")
#define MODEL_TASK           ("Pose Estimation (single-person)")
#define MODEL_DATASET        ("COCO val2017")
#define MODEL_NUM_KEYPOINTS  (17)

/* ======================================================================
 * 2.  INPUT TENSOR  [1, 257, 257, 3] float32
 * ====================================================================== */
#define MODEL_INPUT_H        (257)
#define MODEL_INPUT_W        (257)

#ifndef MODEL_CHANNELS
#define MODEL_INPUT_C        (3)
#else
#define MODEL_INPUT_C        (MODEL_CHANNELS)
#endif

#ifndef MODEL_INPUT_SIZE
#define MODEL_INPUT_SIZE     (MODEL_INPUT_H * MODEL_INPUT_W * MODEL_INPUT_C)
#endif
/* = 257 * 257 * 3 = 198 147 floats = 792 588 bytes */

#define MODEL_INPUT_LAYOUT   "NHWC"

/* Largest output-axis extent (MODEL_INPUT_H == MODEL_INPUT_W for PoseNet). */
#define MAX_DST_DIM         ((MODEL_INPUT_H > MODEL_INPUT_W) ? MODEL_INPUT_H : MODEL_INPUT_W)

/* Compile-time normalisation constant.  Use multiplication (not division
 * by 127.5f) so the float32 rounding remains ULP-stable. */
#define INPUT_NORM_MUL      (2.0f / 255.0f)


/* ------------------------------------------------------------------------- */
/* NHWC strides for each output head                                          */
/* ------------------------------------------------------------------------- */
#define HM_STRIDE_W    (MODEL_HEATMAP_C)                              /* 17  */
#define HM_STRIDE_H    (MODEL_OUTPUT_GW * MODEL_HEATMAP_C)            /* 289 */
#define OFF_STRIDE_W   (MODEL_OFFSET_C)                               /* 34  */
#define OFF_STRIDE_H   (MODEL_OUTPUT_GW * MODEL_OFFSET_C)             /* 578 */
#define DISP_STRIDE_W  (MODEL_DISP_C)                                 /* 32  */
#define DISP_STRIDE_H  (MODEL_OUTPUT_GW * MODEL_DISP_C)               /* 544 */

/* Input preprocessing pipeline:
 *   1. Normalize: f = raw_pixel * (2.0f / 255.0f) - 1.0f     range [-1, 1]
 *      (Multiplication, not division by 127.5f, for ULP-stable float32.)
 *   2. Quantize:  q = clamp(round(f / INPUT_QUANT_SCALE) + INPUT_QUANT_ZP, -128, 127) */

#define INPUT_PIXEL_MIN      (0)
#define INPUT_PIXEL_MAX      (255)

/* Normalization multiplier (step 1 above). */
/* INPUT_NORM_MUL is already defined above via the compile-time constant below. */

/* INT8 quantization parameters for the model input tensor. */
#define INPUT_QUANT_SCALE    (0.007843137718737125f)   /* 1/127.5 */
#define INPUT_QUANT_ZP       (-1)

#define INPUT_RESIZE_METHOD  ("Direct bilinear to 257x257 (no aspect-preserving crop)")

/* ======================================================================
 * 3.  OUTPUT GRID GEOMETRY  (shared by all 4 heads)
 * ====================================================================== */
#define MODEL_OUTPUT_STRIDE  (16)
#define MODEL_OUTPUT_GH      (17)    /* (257 - 1) / 16 + 1 = 17 */
#define MODEL_OUTPUT_GW      (17)

/* ======================================================================
 * 4.  OUTPUT TENSOR SIZES  (float32 elements)
 * ====================================================================== */
/* Heatmap : [1, 17, 17, 17]  -- one sigmoid score per keypoint per cell. */
#define MODEL_HEATMAP_C      (MODEL_NUM_KEYPOINTS)
#define MODEL_HEATMAP_SIZE   (MODEL_OUTPUT_GH * MODEL_OUTPUT_GW * MODEL_HEATMAP_C)
/* = 17 * 17 * 17 = 4 913 */

/* Offset  : [1, 17, 17, 34]  -- (dy[0..16], dx[0..16]) per cell, input-pixel units. */
#define MODEL_OFFSET_C       (2 * MODEL_NUM_KEYPOINTS)
#define MODEL_OFFSET_SIZE    (MODEL_OUTPUT_GH * MODEL_OUTPUT_GW * MODEL_OFFSET_C)
/* = 17 * 17 * 34 = 9 826 */

/* Displacement (fwd / bwd) : [1, 17, 17, 32] -- consumed by the multi-pose
 * PersonLab decoder. Channels 0..15 = dy per skeleton edge, 16..31 = dx. */
#define MODEL_DISP_C         (32)
#define MODEL_DISP_FWD_SIZE  (MODEL_OUTPUT_GH * MODEL_OUTPUT_GW * MODEL_DISP_C)
#define MODEL_DISP_BWD_SIZE  (MODEL_OUTPUT_GH * MODEL_OUTPUT_GW * MODEL_DISP_C)
/* = 17 * 17 * 32 = 9 248 each */

/* The heatmap already includes Sigmoid in the exported graph. */
#define HEATMAP_HAS_SIGMOID  (1)

/* INT8 quantization parameters for the heatmap output tensor.
 * Dequantize: score = ((float)q - HEATMAP_QUANT_ZP) * HEATMAP_QUANT_SCALE
 * Range maps int8 [-128, 127] -> float [0.0, 0.996]. */
#define HEATMAP_QUANT_SCALE  (0.00390625f)   /* 1/256 */
#define HEATMAP_QUANT_ZP     (-128)

/* INT8 quantization parameters for the offset output tensor.
 * Dequantize: offset_px = ((float)q - OFFSET_QUANT_ZP) * OFFSET_QUANT_SCALE */
#define OFFSET_QUANT_SCALE   (0.5891178846359253f)
#define OFFSET_QUANT_ZP      (-1)

/* INT8 quantization parameters for the displacement_fwd output tensor.
 * Dequantize: disp_fwd_px = ((float)q - DISP_FWD_QUANT_ZP) * DISP_FWD_QUANT_SCALE */
#define DISP_FWD_QUANT_SCALE (1.3544772863388062f)
#define DISP_FWD_QUANT_ZP    (-63)

/* INT8 quantization parameters for the displacement_bwd output tensor.
 * Dequantize: disp_bwd_px = ((float)q - DISP_BWD_QUANT_ZP) * DISP_BWD_QUANT_SCALE */
#define DISP_BWD_QUANT_SCALE (1.1400184631347656f)
#define DISP_BWD_QUANT_ZP    (41)

/* Heatmap dequantized range (informational). */
#define HEATMAP_FLOAT_MIN    (0.0f)
#define HEATMAP_FLOAT_MAX    (1.0f)

/* ======================================================================
 * 5.  DECODER PARAMETERS  (multi-pose PersonLab decoder)
 * ====================================================================== */
/* Per-keypoint score threshold used by accuracy_eval.c to count "visible"
 * keypoints. Matches DEFAULT_MIN_PART_SCORE on the host. */
#define MIN_PART_SCORE       (0.5f)

/* Maximum number of poses (persons) the decoder returns per image. */
#define MAX_POSE_DETECTIONS  (10)

/* Root-candidate score threshold: only heatmap peaks >= this seed a pose. */
#define POSE_ROOT_SCORE_THRESHOLD  (0.5f)

/* Minimum instance score for a decoded pose to be kept. */
#define POSE_MIN_SCORE       (0.5f)

/* Non-max-suppression radius (model-input pixels) between roots / same-part
 * keypoints across accepted poses. Squared internally in postprocessing. */
#define POSE_NMS_RADIUS_PX   (20.0f)

/* Half-side of the 3x3 window used by the local-max heatmap peak finder. */
#define LOCAL_MAX_RADIUS     (1)

/* Number of skeleton edges used by the PersonLab chain walk
 * (parent -> child pairs). Must equal MODEL_DISP_C / 2. */
#define NUM_POSE_EDGES       (16)

/* ======================================================================
 * 6.  MEMORY FOOTPRINT
 * ====================================================================== */
#define MODEL_INPUT_BUF_BYTES     (MODEL_INPUT_SIZE     * (uint32_t)sizeof(int8_t))  /* 198 147 B */
#define MODEL_HEATMAP_BUF_BYTES   (MODEL_HEATMAP_SIZE   * (uint32_t)sizeof(int8_t))  /*   4 913 B */
#define MODEL_OFFSET_BUF_BYTES    (MODEL_OFFSET_SIZE    * (uint32_t)sizeof(int8_t))  /*   9 826 B */
#define MODEL_DISP_FWD_BUF_BYTES  (MODEL_DISP_FWD_SIZE  * (uint32_t)sizeof(int8_t))  /*   9 248 B */
#define MODEL_DISP_BWD_BUF_BYTES  (MODEL_DISP_BWD_SIZE  * (uint32_t)sizeof(int8_t))  /*   9 248 B */

/* ======================================================================
 * 7.  VISUALIZATION BUFFER
 * ======================================================================
 * Maximum source-image size we draw skeleton overlays onto. The on-MCU
 * visualiser memcpy's the OSPI-resident test image into an SDRAM-backed
 * scratch buffer of this size, rasterises keypoints + skeleton edges into
 * it, then logs the buffer address via RTT so the host can dump it through
 * JLink for inspection.
 *
 * Sized for the largest test image (640 x 480 x 3 RGB888 = 921 600 B).
 * Bumped to 1 MiB for headroom in case larger samples are added later.
 * ====================================================================== */
#define OUTPUT_IMAGE_MAX_SIZE     (1048576U)   /* 1 MiB */

/* ======================================================================
 * 8.  PIPELINE REFERENCE
 * ======================================================================
 * Input  : bilinear resize to 257x257 -> normalize (* 2/255 - 1) ->
 *          quantize to int8 (INPUT_QUANT_SCALE / INPUT_QUANT_ZP) -> int8 tensor
 * Model  : compute_sub_0000(storage, int8_input, int8_heatmap, int8_offset,
 *                           int8_disp_fwd, int8_disp_bwd)
 * Output : PersonLab multi-pose decoder consumes all 4 int8 heads:
 *            1. Find local-max heatmap peaks -> root-candidate seeds
 *            2. For each seed (highest first), NMS vs already-accepted roots,
 *               then walk POSE_CHAIN forward + backward using disp_fwd / disp_bwd
 *               (dequantized on-the-fly) to fill all 17 keypoints per person.
 *            3. Instance score = mean of non-overlapping keypoint scores.
 *            4. Rescale accepted poses to source-image pixels.
 * Visual : memcpy src -> g_output_image -> draw skeleton + circles + labels
 *          for every accepted pose.
 * ====================================================================== */

#ifdef __cplusplus
}
#endif

#endif /* MODEL_METADATA_H */
