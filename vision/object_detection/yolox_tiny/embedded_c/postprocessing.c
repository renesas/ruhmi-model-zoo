/*
* Copyright (c) 2020 - 2026 Renesas Electronics Corporation and/or its affiliates
*
* SPDX-License-Identifier: BSD-3-Clause
*/
/**
 ******************************************************************************
 * @file    postprocessing.c
 * @brief   YOLOX-Tiny INT8 anchor-free decode + per-class NMS - MCU portable.
 *
 * Pipeline (mirrors Python inference.py for TFLite INT8):
 *   1. dequantize()   — int8 → float32 using output quant params
 *   2. build_grid()   — build (x,y) grid offsets + stride per anchor (once)
 *   3. score_filter()  — score = obj × cls[argmax] > thresh
 *   4. decode_boxes() — cx=(cx_off+gx)*stride, w=exp(w_log)*stride
 *   5. undo_letterbox() — map model coords back to original image pixels
 *   6. nms_class()    — greedy IoU NMS per class
 *
 * All buffers are statically allocated — no malloc/free.
 ******************************************************************************
 */

#include "postprocessing.h"

#include <stddef.h>
#include <string.h>
#include <math.h>

/* ── File-scope constants, types, and static storage ────────────────── */

/* Store grid entries as uint8 to save RAM. */
typedef struct
{
    uint8_t gx;
    uint8_t gy;
    uint8_t stride;
} grid_entry_t;

typedef struct
{
    float    x1;
    float    y1;
    float    x2;
    float    y2;
    float    score;
    uint32_t cls_id;
} raw_det_t;

#define MAX_RAW_DETS  (512U)

static grid_entry_t s_grid[NUM_ANCHORS];
static int32_t      s_grid_ready = 0;
static raw_det_t    s_raw[MAX_RAW_DETS];

/* ── Inline helpers ──────────────────────────────────────────────────── */

/**
 * @brief Return maximum of two float values.
 * @param[in] first_value  First input value.
 * @param[in] second_value Second input value.
 * @return Maximum float value.
 */

static inline float get_maximum_float_value(float first_value, float second_value)
{
    /* Return larger of two values without side effects. */
    return (first_value > second_value) ? first_value : second_value;
}

/**
 * @brief Return minimum of two float values.
 * @param[in] first_value  First input value.
 * @param[in] second_value Second input value.
 * @return Minimum float value.
 */

static inline float get_minimum_float_value(float first_value, float second_value)
{
    /* Return smaller of two values without side effects. */
    return (first_value < second_value) ? first_value : second_value;
}

/* ── Dequantize helper ───────────────────────────────────────────────── */

/**
 * @brief Dequantize int8 value to float32.
 *        float = (int8_val - zero_point) * scale
 */
static inline float dequantize_int8_to_float(int8_t raw_int8_value)
{
    /* Affine dequantization for model output tensor. */
    return ((float)raw_int8_value - (float)MODEL_OUTPUT_ZERO_POINT) * MODEL_OUTPUT_SCALE;
}

/**
 * @brief Build anchor grid lookup table once.
 * @return void
 */

static void build_anchor_grid(void)
{
    /* Flat index that advances through all FPN levels. */
    uint32_t grid_entry_idx = 0U;
    uint32_t grid_row;
    uint32_t grid_col;

    if (s_grid_ready != 0)
    {
        /* Grid is cached after first call, so avoid rebuilding. */
        return;
    }

    /* Stride-8 : 28 × 28 feature map */
    for (grid_row = 0U; grid_row < GRID_S8; grid_row++)
    {
        for (grid_col = 0U; grid_col < GRID_S8; grid_col++)
        {
            s_grid[grid_entry_idx].gx     = (uint8_t)grid_col;   /* Cell x-index at stride 8. */
            s_grid[grid_entry_idx].gy     = (uint8_t)grid_row;   /* Cell y-index at stride 8. */
            s_grid[grid_entry_idx].stride = (uint8_t)STRIDE_S8;  /* Corresponding stride value. */
            grid_entry_idx++;
        }
    }

    /* Stride-16 : 14 × 14 feature map */
    for (grid_row = 0U; grid_row < GRID_S16; grid_row++)
    {
        for (grid_col = 0U; grid_col < GRID_S16; grid_col++)
        {
            s_grid[grid_entry_idx].gx     = (uint8_t)grid_col;    /* Cell x-index at stride 16. */
            s_grid[grid_entry_idx].gy     = (uint8_t)grid_row;    /* Cell y-index at stride 16. */
            s_grid[grid_entry_idx].stride = (uint8_t)STRIDE_S16;  /* Corresponding stride value. */
            grid_entry_idx++;
        }
    }

    /* Stride-32 : 7 × 7 feature map */
    for (grid_row = 0U; grid_row < GRID_S32; grid_row++)
    {
        for (grid_col = 0U; grid_col < GRID_S32; grid_col++)
        {
            s_grid[grid_entry_idx].gx     = (uint8_t)grid_col;    /* Cell x-index at stride 32. */
            s_grid[grid_entry_idx].gy     = (uint8_t)grid_row;    /* Cell y-index at stride 32. */
            s_grid[grid_entry_idx].stride = (uint8_t)STRIDE_S32;  /* Corresponding stride value. */
            grid_entry_idx++;
        }
    }

    s_grid_ready = 1;
}

/* ── IoU computation ─────────────────────────────────────────────────── */

/**
 * @brief Compute IoU between two detections in model-space coordinates.
 * @param[in] p_first_detection  First detection.
 * @param[in] p_second_detection Second detection.
 * @return IoU value in range [0.0, 1.0].
 */

static float compute_iou(const raw_det_t *p_first_detection, const raw_det_t *p_second_detection)
{
    /* Compute intersection box corners. */
    float intersection_left_x = get_maximum_float_value(p_first_detection->x1, p_second_detection->x1);
    float intersection_top_y = get_maximum_float_value(p_first_detection->y1, p_second_detection->y1);
    float intersection_right_x = get_minimum_float_value(p_first_detection->x2, p_second_detection->x2);
    float intersection_bottom_y = get_minimum_float_value(p_first_detection->y2, p_second_detection->y2);

    /* Compute non-negative intersection size. */
    float intersection_width = get_maximum_float_value(0.0F, intersection_right_x - intersection_left_x);
    float intersection_height = get_maximum_float_value(0.0F, intersection_bottom_y - intersection_top_y);
    float intersection_area = intersection_width * intersection_height;

    if (intersection_area == 0.0F)
    {
        return 0.0F;
    }

    /* Compute areas of both boxes for IoU denominator. */
    float first_detection_area = (p_first_detection->x2 - p_first_detection->x1) *
                                 (p_first_detection->y2 - p_first_detection->y1);
    float second_detection_area = (p_second_detection->x2 - p_second_detection->x1) *
                                  (p_second_detection->y2 - p_second_detection->y1);

    /* IoU = intersection / union; epsilon prevents divide-by-zero. */
    return intersection_area /
           (first_detection_area + second_detection_area - intersection_area + 1e-6F);
}

/* ── Per-class greedy NMS ────────────────────────────────────────────── */

/**
 * @brief Apply class-wise greedy NMS and write retained detections.
 * @param[in,out] p_raw_dets                Raw detection buffer.
 * @param[in]     num_raw_dets              Number of raw detections.
 * @param[in]     target_class_id           Class id being processed.
 * @param[in]     nms_thresh                IoU threshold for suppression.
 * @param[in]     p_letterbox_params        Letterbox parameters from preprocess.
 * @param[in]     orig_img_width            Original image width.
 * @param[in]     orig_img_height           Original image height.
 * @param[out]    p_out_dets                Output detection buffer.
 * @param[in]     out_capacity              Maximum writable detections.
 * @return Number of detections written to p_out_dets.
 */

static int32_t apply_nms_for_class(raw_det_t           *p_raw_dets,
                         int32_t              num_raw_dets,
                         uint32_t             target_class_id,
                         float                nms_thresh,
                         const letterbox_params_t *p_letterbox_params,
                         uint32_t             orig_img_width,
                         uint32_t             orig_img_height,
                         Detection_t         *p_out_dets,
                         int32_t              out_capacity)
{
    static int32_t sorted_indices[MAX_RAW_DETS];
    int32_t matching_detection_count = 0;
    int32_t selected_detection_index;
    int32_t comparison_detection_index;

    for (selected_detection_index = 0; selected_detection_index < num_raw_dets; selected_detection_index++)
    {
        /* Keep only detections of the current class and not already suppressed. */
        if ((p_raw_dets[selected_detection_index].cls_id == target_class_id) &&
            (p_raw_dets[selected_detection_index].score >= 0.0F))
        {
            sorted_indices[matching_detection_count] = selected_detection_index;
            matching_detection_count++;
        }
    }

    /* Insertion sort descending by score */
    for (selected_detection_index = 1; selected_detection_index < matching_detection_count; selected_detection_index++)
    {
        /* Classic insertion-sort pass on score descending order. */
        int32_t sorted_key_index = sorted_indices[selected_detection_index];
        comparison_detection_index = selected_detection_index - 1;
        while ((comparison_detection_index >= 0) &&
               (p_raw_dets[sorted_indices[comparison_detection_index]].score < p_raw_dets[sorted_key_index].score))
        {
            sorted_indices[comparison_detection_index + 1] = sorted_indices[comparison_detection_index];
            comparison_detection_index--;
        }
        sorted_indices[comparison_detection_index + 1] = sorted_key_index;
    }

    int32_t written_detection_count = 0;

    for (selected_detection_index = 0;
         (selected_detection_index < matching_detection_count) && (written_detection_count < out_capacity);
         selected_detection_index++)
    {
        int32_t current_detection_index = sorted_indices[selected_detection_index];
        if (p_raw_dets[current_detection_index].score < 0.0F)
        {
            /* Skip suppressed entries. */
            continue;
        }

        /* Map back to original image coordinates (undo letterbox) */
        /* Undo letterbox transform: remove padding then divide by scale. */
        float rescaled_x1 = (p_raw_dets[current_detection_index].x1 - (float)p_letterbox_params->pad_x) /
                    p_letterbox_params->scale;
        float rescaled_y1 = (p_raw_dets[current_detection_index].y1 - (float)p_letterbox_params->pad_y) /
                    p_letterbox_params->scale;
        float rescaled_x2 = (p_raw_dets[current_detection_index].x2 - (float)p_letterbox_params->pad_x) /
                    p_letterbox_params->scale;
        float rescaled_y2 = (p_raw_dets[current_detection_index].y2 - (float)p_letterbox_params->pad_y) /
                    p_letterbox_params->scale;

        /* Clamp to image boundaries */
        /* Clamp boxes to valid original image coordinate range. */
        rescaled_x1 = get_maximum_float_value(0.0F, get_minimum_float_value(rescaled_x1, (float)(orig_img_width  - 1U)));
        rescaled_y1 = get_maximum_float_value(0.0F, get_minimum_float_value(rescaled_y1, (float)(orig_img_height - 1U)));
        rescaled_x2 = get_maximum_float_value(0.0F, get_minimum_float_value(rescaled_x2, (float)(orig_img_width  - 1U)));
        rescaled_y2 = get_maximum_float_value(0.0F, get_minimum_float_value(rescaled_y2, (float)(orig_img_height - 1U)));

        /* Emit current kept detection to output list. */
        p_out_dets[written_detection_count].x1     = rescaled_x1;
        p_out_dets[written_detection_count].y1     = rescaled_y1;
        p_out_dets[written_detection_count].x2     = rescaled_x2;
        p_out_dets[written_detection_count].y2     = rescaled_y2;
        p_out_dets[written_detection_count].score  = p_raw_dets[current_detection_index].score;
        p_out_dets[written_detection_count].cls_id = target_class_id;
        written_detection_count++;

        /* Suppress overlapping boxes of the same class */
        for (comparison_detection_index = selected_detection_index + 1;
             comparison_detection_index < matching_detection_count;
             comparison_detection_index++)
        {
            int32_t suppressed_detection_index = sorted_indices[comparison_detection_index];
            if (p_raw_dets[suppressed_detection_index].score < 0.0F)
            {
                continue;
            }
            /* Suppress lower-scored boxes that overlap strongly with current box. */
            if (compute_iou(&p_raw_dets[current_detection_index],
                            &p_raw_dets[suppressed_detection_index]) > nms_thresh)
            {
                p_raw_dets[suppressed_detection_index].score = -1.0F;
            }
        }
    }

    return written_detection_count;
}

/* ── Public API ──────────────────────────────────────────────────────── */

/**
 * @brief Decode model output and run postprocessing pipeline.
 * @param[in]  raw_output    Raw model output tensor.
 * @param[in]  p_params      Letterbox parameters from preprocess.
 * @param[in]  orig_w        Original image width.
 * @param[in]  orig_h        Original image height.
 * @param[in]  score_thresh  Score threshold.
 * @param[in]  nms_thresh    NMS IoU threshold.
 * @param[out] p_out_dets    Output detection list.
 * @return Number of output detections.
 */

int32_t postprocess(const int8_t           *raw_output,
                    const letterbox_params_t *p_params,
                    uint32_t                orig_w,
                    uint32_t                orig_h,
                    float                   score_thresh,
                    float                   nms_thresh,
                    Detection_t            *p_out_dets)
{
    uint32_t anchor_idx;
    uint32_t class_idx;
    int32_t  num_raw_detections = 0;

    /* Validate required buffers and metadata pointers. */
    if ((raw_output == NULL) || (p_params == NULL) || (p_out_dets == NULL))
    {
        return 0;
    }

    build_anchor_grid();

    /* ── Step 1: dequantize + decode + score filter ──────────────────── */
    for (anchor_idx = 0U; anchor_idx < NUM_ANCHORS; anchor_idx++)
    {
        /* Each anchor row is laid out as [x, y, w, h, obj, class0..classN]. */
        const int8_t *anchor_output_row = raw_output + (anchor_idx * OUTPUT_CHANNELS);

        /* Fast early exit on raw objectness (avoid dequantize).
         * dequantize is monotonic, so compare in int8 domain:
         *   dequant(val) >= thresh  ⟺  val >= (thresh / scale) + zp       */
        int8_t objectness_raw_int8 = anchor_output_row[4];
        /* Pre-compute threshold in int8 space (could be compile-time const) */
        {
            /* Early reject by objectness threshold to avoid extra work. */
            float objectness_score = dequantize_int8_to_float(objectness_raw_int8);
            if (objectness_score < score_thresh)
            {
                continue;
            }
        }

        /* Find class with highest raw int8 logit — skip 80 dequantize calls.
         * Argmax is identical in int8 vs float because dequant is monotonic. */
        /* Find argmax class in raw int8 domain (order preserved after dequant). */
        uint32_t best_class_id = 0U;
        int8_t   best_class_raw = anchor_output_row[5];
        for (class_idx = 1U; class_idx < NUM_CLASSES; class_idx++)
        {
            if (anchor_output_row[5U + class_idx] > best_class_raw)
            {
                best_class_raw = anchor_output_row[5U + class_idx];
                best_class_id = class_idx;
            }
        }

        /* Dequantize only the values we actually need */
        /* Dequantize only fields needed for decode and score computation. */
        float objectness_score  = dequantize_int8_to_float(objectness_raw_int8);
        float best_class_logit  = dequantize_int8_to_float(best_class_raw);
        float center_x_offset   = dequantize_int8_to_float(anchor_output_row[0]);
        float center_y_offset   = dequantize_int8_to_float(anchor_output_row[1]);
        float width_log_encoded = dequantize_int8_to_float(anchor_output_row[2]);
        float height_log_encoded = dequantize_int8_to_float(anchor_output_row[3]);

        /* Score = obj × cls_logit (matches Python TFLite inference) */
        /* Final detection score combines objectness and class confidence. */
        float detection_score = objectness_score * best_class_logit;
        if (detection_score < score_thresh)
        {
            continue;
        }

        /* Anchor-free box decoding */
        /* Decode anchor-free box from offsets/log-width/log-height. */
        float anchor_stride = (float)s_grid[anchor_idx].stride;
        float box_center_x = (center_x_offset + (float)s_grid[anchor_idx].gx) * anchor_stride;
        float box_center_y = (center_y_offset + (float)s_grid[anchor_idx].gy) * anchor_stride;
        float box_width = expf(width_log_encoded) * anchor_stride;
        float box_height = expf(height_log_encoded) * anchor_stride;

        if ((uint32_t)num_raw_detections >= MAX_RAW_DETS)
        {
            break;
        }

        /* Convert center-format box to corner-format box in model coordinates. */
        s_raw[num_raw_detections].x1 = box_center_x - (box_width * 0.5F);
        s_raw[num_raw_detections].y1 = box_center_y - (box_height * 0.5F);
        s_raw[num_raw_detections].x2 = box_center_x + (box_width * 0.5F);
        s_raw[num_raw_detections].y2 = box_center_y + (box_height * 0.5F);
        s_raw[num_raw_detections].score  = detection_score;
        s_raw[num_raw_detections].cls_id = best_class_id;
        num_raw_detections++;
    }

    if (num_raw_detections == 0)
    {
        return 0;
    }

    /* ── Step 2: per-class NMS → original image coordinates ───────────── */
    /* Total detections written after class-wise NMS. */
    int32_t total_output_detections = 0;

    int32_t classes_seen[NUM_CLASSES];
    (void)memset(classes_seen, 0, sizeof(classes_seen));
    {
        int32_t raw_det_idx;
        for (raw_det_idx = 0; raw_det_idx < num_raw_detections; raw_det_idx++)
        {
            /* Mark class present so we run NMS only for observed classes. */
            classes_seen[s_raw[raw_det_idx].cls_id] = 1;
        }
    }

    for (class_idx = 0U; class_idx < NUM_CLASSES; class_idx++)
    {
        if (classes_seen[class_idx] == 0)
        {
            continue;
        }

        /* Keep output within configured maximum detection capacity. */
        int32_t remaining_capacity = (int32_t)MAX_DETECTIONS - total_output_detections;
        if (remaining_capacity <= 0)
        {
            break;
        }

        int32_t num_class_dets = apply_nms_for_class(s_raw, num_raw_detections, class_idx,
                              nms_thresh,
                      p_params, orig_w, orig_h,
                      &p_out_dets[total_output_detections], remaining_capacity);
        total_output_detections += num_class_dets;
    }

    return total_output_detections;
}
