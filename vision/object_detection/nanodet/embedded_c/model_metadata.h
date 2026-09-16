/**
 * @file    model_metadata.h
 * @brief   Compile-time metadata for NanoDet-Plus-m model (320x320 input).
 */

#ifndef MODEL_METADATA_H
#define MODEL_METADATA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MODEL_NAME              "NanoDet-Plus-m"
#define MODEL_DATASET           "COCO"
#define MODEL_NUM_CLASSES       (80U)
#define MODEL_DTYPE             "int8"

/* Kernel input tensor: [1, 320, 320, 3], int8 NHWC.
 * The application preprocesses in float and quantizes before compute_sub_0000. */
#define MODEL_INPUT_H           (320)
#define MODEL_INPUT_W           (320)
#define MODEL_INPUT_C           (3)
#define MODEL_INPUT_SIZE        (MODEL_INPUT_H * MODEL_INPUT_W * MODEL_INPUT_C) /* 307200 */

/* Kept for compatibility; NanoDet path uses warp resize (no padding). */
#define INPUT_LETTERBOX_PAD     (114)

/* Quant metadata for int8 input/output bridge around RUHMI kernel.
 * Values sourced from nanodet-plus-m_320_INT8.tflite tensor metadata. */
#define MODEL_INPUT_SCALE       (0.018658448f)
#define MODEL_INPUT_ZERO_POINT  (-14)

/* Kernel output tensor: [1, 2125, 112], int8.
 * The application dequantizes to float before NanoDet decode/NMS. */
#define MODEL_NUM_ANCHORS       (2125U)   /* 40*40 + 20*20 + 10*10 + 5*5 */
#define MODEL_OUTPUT_CHANNELS   (112U)    /* 80 class + 32 reg logits */
#define MODEL_OUTPUT_SIZE       (MODEL_NUM_ANCHORS * MODEL_OUTPUT_CHANNELS) /* 238000 */
#define MODEL_OUTPUT_SCALE      (0.166149095f)
#define MODEL_OUTPUT_ZERO_POINT (76)

/* FPN levels */
#define MODEL_STRIDE_S8         (8U)
#define MODEL_STRIDE_S16        (16U)
#define MODEL_STRIDE_S32        (32U)
#define MODEL_STRIDE_S64        (64U)
#define MODEL_GRID_S8           (40U)
#define MODEL_GRID_S16          (20U)
#define MODEL_GRID_S32          (10U)
#define MODEL_GRID_S64          (5U)

/* Post-process thresholds matching python/inference.py defaults */
#define POSTPROC_SCORE_THRESH   (0.35f)
#define POSTPROC_NMS_THRESH     (0.60f)
#define POSTPROC_MAX_DETS       (100U)
#define LOG_MAX_DETS            (20U)

/* GFL distribution regression max index (NanoDet-Plus architecture) */
#define MODEL_REG_MAX               (7U)

/* FPN architecture: number of stride levels */
#define MODEL_NUM_FPN_LEVELS        (4U)

/* Post-process internal raw detection buffer capacity */
#define MODEL_MAX_RAW_DETS          (4096U)

/* Preprocessing normalization constants in BGR output order (mirroring python/inference.py).
 * Applied after RGB→BGR channel swap; index 0=B, 1=G, 2=R. */
#define MODEL_MEAN_B                (103.530F)
#define MODEL_MEAN_G                (116.280F)
#define MODEL_MEAN_R                (123.675F)
#define MODEL_STD_B                 (57.375F)
#define MODEL_STD_G                 (57.120F)
#define MODEL_STD_R                 (58.395F)

#define NUM_TEST_IMAGES         (3U)
#define OUTPUT_IMAGE_MAX_SIZE   (921600U)

#ifdef __cplusplus
}
#endif

#endif /* MODEL_METADATA_H */
