/**********************************************************************************************************************
 * Copyright (c) 2020 - 2026 Renesas Electronics Corporation and/or its affiliates
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * DISCLAIMER
 * This software is supplied by Renesas Electronics Corporation and is only intended for use with Renesas products.
 * No other uses are authorized.
 *********************************************************************************************************************/
/**********************************************************************************************************************
 * File Name    : model_metadata.h
 * Description  : Compile-time metadata for the MediaPipe face-landmark
 *                (face_landmark_INT8.tflite @ 192x192) model deployed on the
 *                RA8P1 CPU target.
 *
 *                Source model : face_landmark_INT8.tflite
 *                               (MediaPipe 468-point face mesh, fully-quantized INT8)
 *
 *                Boundary tensors exposed to host code:
 *                  Input  : input_1  shape=[1,192,192,3]  dtype=int8
 *                           scale=0.007843137718737125 zero_point=-1  (float range [0,1])
 *                  Outputs:
 *                    conv2d_20  shape=[1,1,1,1404]  dtype=int8    -- 468 landmarks (x,y,z)
 *                               scale=0.9023275971412659 zero_point=-84  (pixel coords in 192x192)
 *                    conv2d_30  shape=[1,1,1,1]     dtype=int8    -- face-presence logit
 *                               scale=0.31730183959007263 zero_point=-107
 **********************************************************************************************************************/

#ifndef MODEL_METADATA_H
#define MODEL_METADATA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ======================================================================
 * 1.  MODEL IDENTITY
 * ====================================================================== */
#define MODEL_NAME              ("MediaPipe-FaceLandmark-192")
#define MODEL_FILE_INT8         ("face_landmark_INT8.tflite")
#define MODEL_TASK              ("Facial Landmark Detection (468 points)")
#define MODEL_DATASET           ("MediaPipe FaceMesh training set")
#define MODEL_NUM_LANDMARKS     (468U)
#define MODEL_LANDMARK_DIMS     (3U)      /* (x, y, z) per landmark */

/* ======================================================================
 * 2.  INPUT TENSOR  [1, 192, 192, 3] int8
 * ====================================================================== */
#define MODEL_INPUT_H           (192U)
#define MODEL_INPUT_W           (192U)

#ifndef MODEL_CHANNELS
#define MODEL_INPUT_C           (3U)
#else
#define MODEL_INPUT_C           (MODEL_CHANNELS)
#endif

#ifndef MODEL_INPUT_SIZE
#define MODEL_INPUT_SIZE        (((MODEL_INPUT_H) * (MODEL_INPUT_W) * (MODEL_INPUT_C)))
#endif
/* = 192 * 192 * 3 = 110 592 int8 elements = 110 592 bytes */

#define MODEL_INPUT_LAYOUT      ("NHWC")
#define MODEL_INPUT_COLOR_ORDER ("RGB888")

/* Input pixel normalization applied by the host preprocessor before quantization:
 *     normalized = raw_pixel * (1.0f / 255.0f)               ->  range [0.0, 1.0]
 *
 * Input tensor quantization (float -> int8):
 *     q = round(normalized / INPUT_QUANT_SCALE + INPUT_QUANT_ZERO_POINT)
 *     q clipped to [-128, 127]
 * INPUT_QUANT_SCALE was extracted from the tflite flatbuffer and equals 2/255
 * exactly at float32 precision. */
#define INPUT_PIXEL_MIN         (0U)
#define INPUT_PIXEL_MAX         (255U)
#define INPUT_QUANT_SCALE       (0.007843137718737125f)
#define INPUT_QUANT_ZERO_POINT  (-1)

#define INPUT_RESIZE_METHOD     ("Direct 2-tap bilinear (half-pixel-centre) to 192x192")

/* ======================================================================
 * 3.  OUTPUT TENSORS
 * ====================================================================== */
/* Landmark output : [1, 1, 1, 1404] int8 -- flattened (x0,y0,z0,x1,y1,z1,...)
 * in pixels of the 192x192 model input frame. */
#define MODEL_LANDMARK_SIZE     (((MODEL_NUM_LANDMARKS) * (MODEL_LANDMARK_DIMS))) /* 1404 */
#define LANDMARK_QUANT_SCALE    (0.9023275971412659f)
#define LANDMARK_QUANT_ZERO_POINT (-84)

/* Face-presence output : [1, 1, 1, 1] int8 -- pre-sigmoid logit. */
#define MODEL_FACE_FLAG_SIZE    (1U)
#define FACE_FLAG_QUANT_SCALE   (0.31730183959007263f)
#define FACE_FLAG_QUANT_ZERO_POINT (-107)

/* Face-presence sigmoid threshold used by postprocessing. */
#define FACE_DETECTION_THRESHOLD (0.5f)

/* ======================================================================
 * 4.  MEMORY FOOTPRINT
 * ====================================================================== */
#define MODEL_INPUT_BUF_BYTES     (((MODEL_INPUT_SIZE) * ((uint32_t)sizeof(int8_t))))  /* 110 592 B */
#define MODEL_LANDMARK_BUF_BYTES  (((MODEL_LANDMARK_SIZE) * ((uint32_t)sizeof(int8_t))))  /*  1 404 B */
#define MODEL_FACE_FLAG_BUF_BYTES (((MODEL_FACE_FLAG_SIZE) * ((uint32_t)sizeof(int8_t))))  /*      1 B */

/* ======================================================================
 * 5.  VISUALIZATION BUFFER
 * ======================================================================
 * Maximum source-image size we draw landmark overlays onto. The on-MCU
 * visualiser rasterises landmark dots + optional mesh edges into the
 * SDRAM-resident source image buffer, then logs the buffer address via RTT
 * so the host can dump it through JLink for inspection.
 *
 * Sized for the largest expected cropped-face sample (e.g. 512x512x3 = 786 432 B).
 * Bumped to 1 MiB for headroom.
 * ====================================================================== */
#define OUTPUT_IMAGE_MAX_SIZE     (1048576U)   /* 1 MiB */

/* ======================================================================
 * 6.  PIPELINE REFERENCE
 * ======================================================================
 * Input  : RGB888 cropped face (any size) -> 2-tap bilinear to 192x192 (RGB order)
 *          -> normalize *(1/255) -> quantize (q=round(f/scale+zp), clip [-128,127])
 *          -> int8 tensor
 * Model  : compute_sub_0000(storage, input_int8, face_flag_int8, landmarks_int8)
 * Output : dequantize landmarks -> reshape to (468, 3) -> rescale from 192x192
 *          pixel space to source-image pixel space
 *          dequantize face-flag logit -> sigmoid -> compare to FACE_DETECTION_THRESHOLD
 * Visual : rasterise landmark dots (+ optional mesh edges) into source image buffer
 * ====================================================================== */

#ifdef __cplusplus
}
#endif

#endif /* MODEL_METADATA_H */
