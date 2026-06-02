/**********************************************************************************************************************
 * Copyright (c) 2020 - 2026 Renesas Electronics Corporation and/or its affiliates
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * File Name    : postprocessing.c
 * Description  : BlazeFace postprocessing implementation.
 *********************************************************************************************************************/

#include "postprocessing.h"
#include "../common/model_metadata.h"

#include <math.h>
#include <string.h>

/* ── Constants (sourced from model_metadata.h where possible) ────────── */
#define NUM_COORDS     OUTPUT_BOX_COORDS    /* 16: 4 bbox + 6 kp × 2 */
#define NUM_KP         NUM_KEYPOINTS        /* 6 facial keypoints     */
#define XY_SCALE       ((float)MODEL_INPUT_W)  /* 128.0f              */
#define SCORE_CLIP     (88.0f)  /* avoid exp overflow in sigmoid */

/**
 * @brief Dequantize a score tensor.
 * @param[in] p_quantized_scores Quantized score values.
 * @param[in] anchor_count Number of anchors in tensor.
 * @param[in] scale Quantization scale.
 * @param[in] zero_point Quantization zero-point.
 * @param[out] p_dequantized_scores Output dequantized score values.
 * @return void
 */

static void dequantize_scores_tensor(const int8_t *p_quantized_scores, int32_t anchor_count,
                                     float scale, int zero_point,
                                     float *p_dequantized_scores)
{
    for (int anchor_index = 0; anchor_index < anchor_count; anchor_index++) {
        p_dequantized_scores[anchor_index] = ((float)p_quantized_scores[anchor_index] - (float)zero_point) * scale;
    }
}

/**
 * @brief Dequantize a box tensor.
 * @param[in] p_quantized_boxes Quantized box tensor.
 * @param[in] anchor_count Number of anchors in tensor.
 * @param[in] scale Quantization scale.
 * @param[in] zero_point Quantization zero-point.
 * @param[out] p_dequantized_boxes Output dequantized box tensor.
 * @return void
 */
static void dequantize_boxes_tensor(const int8_t *p_quantized_boxes, int32_t anchor_count,
                                    float scale, int zero_point,
                                    float p_dequantized_boxes[][NUM_COORDS])
{
    for (int anchor_index = 0; anchor_index < anchor_count; anchor_index++) {
        for (int coord_index = 0; coord_index < NUM_COORDS; coord_index++) {
            p_dequantized_boxes[anchor_index][coord_index] = ((float)p_quantized_boxes[anchor_index * NUM_COORDS + coord_index] - (float)zero_point) * scale;
        }
    }
}

/**
 * @brief Compute sigmoid with clipping for numerical stability.
 * @param[in] logit Input logit.
 * @return Sigmoid(logit), clipped at SCORE_CLIP bounds.
 */

static inline float fast_sigmoid(float logit)
{
    if (logit > SCORE_CLIP) {
        return 1.0f;
    }
    if (logit < -SCORE_CLIP) {
        return 0.0f;
    }
    return 1.0f / (1.0f + expf(-logit));
}

/**
 * @brief Decode one raw anchor prediction into a detection structure.
 * @param[in] p_raw_box Raw decoded coordinates (NUM_COORDS values).
 * @param[in] p_anchor Anchor in [cx, cy, w, h] format.
 * @param[out] p_decoded_detection Output decoded detection.
 * @param[in] confidence Detection confidence score.
 * @return void
 */

static void decode_anchor_prediction(const float p_raw_box[NUM_COORDS],
                                     const float p_anchor[4],
                                     BfDetection *p_decoded_detection,
                                     float confidence)
{
    float anchor_center_x = p_anchor[0];
    float anchor_center_y = p_anchor[1];
    float anchor_width  = p_anchor[2];
    float anchor_height  = p_anchor[3];

    float box_center_x = p_raw_box[0] / XY_SCALE * anchor_width + anchor_center_x;
    float box_center_y = p_raw_box[1] / XY_SCALE * anchor_height + anchor_center_y;
    float box_width = p_raw_box[2] / XY_SCALE * anchor_width;
    float box_height = p_raw_box[3] / XY_SCALE * anchor_height;

    p_decoded_detection->ymin  = box_center_y - box_height * 0.5f;
    p_decoded_detection->xmin  = box_center_x - box_width * 0.5f;
    p_decoded_detection->ymax  = box_center_y + box_height * 0.5f;
    p_decoded_detection->xmax  = box_center_x + box_width * 0.5f;
    p_decoded_detection->score = confidence;

    for (int keypoint_index = 0; keypoint_index < NUM_KP; keypoint_index++) {
        int keypoint_offset = 4 + (keypoint_index * 2);
        p_decoded_detection->kp[keypoint_index][0] = p_raw_box[keypoint_offset]     / XY_SCALE * anchor_width + anchor_center_x;  /* x */
        p_decoded_detection->kp[keypoint_index][1] = p_raw_box[keypoint_offset + 1] / XY_SCALE * anchor_height + anchor_center_y;  /* y */
    }
}

/**
 * @brief Compute intersection-over-union for two detections.
 * @param[in] p_left_box First detection.
 * @param[in] p_right_box Second detection.
 * @return IoU value in [0, 1].
 */

static float compute_iou(const BfDetection *p_left_box, const BfDetection *p_right_box)
{
    float intersection_ymin = p_left_box->ymin > p_right_box->ymin ? p_left_box->ymin : p_right_box->ymin;
    float intersection_xmin = p_left_box->xmin > p_right_box->xmin ? p_left_box->xmin : p_right_box->xmin;
    float intersection_ymax = p_left_box->ymax < p_right_box->ymax ? p_left_box->ymax : p_right_box->ymax;
    float intersection_xmax = p_left_box->xmax < p_right_box->xmax ? p_left_box->xmax : p_right_box->xmax;

    float intersection_height = intersection_ymax - intersection_ymin;
    float intersection_width = intersection_xmax - intersection_xmin;
    if ((intersection_height <= 0.0f) || (intersection_width <= 0.0f)) {
        return 0.0f;
    }
    float intersection_area = intersection_height * intersection_width;

    float left_area = (p_left_box->ymax - p_left_box->ymin) * (p_left_box->xmax - p_left_box->xmin);
    float right_area = (p_right_box->ymax - p_right_box->ymin) * (p_right_box->xmax - p_right_box->xmin);
    return intersection_area / (left_area + right_area - intersection_area + 1.0e-6f);
}

/**
 * @brief Sort detections by descending score (insertion sort).
 * @param[in,out] p_detections Detection array to sort.
 * @param[in] detection_count Number of valid detections.
 * @return void
 */
static void sort_detections_by_score_desc(BfDetection *p_detections, int32_t detection_count)
{
    for (int current_index = 1; current_index < detection_count; current_index++) {
        BfDetection current_detection = p_detections[current_index];
        int insert_index = current_index - 1;
        while (insert_index >= 0 && p_detections[insert_index].score < current_detection.score) {
            p_detections[insert_index + 1] = p_detections[insert_index];
            insert_index--;
        }
        p_detections[insert_index + 1] = current_detection;
    }
}

/**
 * @brief Apply weighted non-maximum suppression.
 * @param[in,out] p_candidate_detections Candidate detections before NMS.
 * @param[in] candidate_count Number of candidates.
 * @param[in] nms_iou_threshold IoU threshold for grouping.
 * @param[out] p_output_detections Final detections after weighted merge.
 * @return void
 */
static void apply_weighted_nms(BfDetection *p_candidate_detections, int32_t candidate_count,
                               float nms_iou_threshold,
                               BfDetections *p_output_detections)
{
    p_output_detections->n = 0;
    if (0 == candidate_count) {
        return;
    }

    sort_detections_by_score_desc(p_candidate_detections, candidate_count);

    static int8_t is_active[TOTAL_ANCHORS];
    if (candidate_count > TOTAL_ANCHORS) {
        candidate_count = TOTAL_ANCHORS;
    }
    memset(is_active, 1, (size_t)candidate_count);

    for (int base_index = 0; base_index < candidate_count && p_output_detections->n < BF_MAX_DETECTIONS; base_index++) {
        if (!is_active[base_index]) {
            continue;
        }

        static int merge_group_indices[TOTAL_ANCHORS];
        int   group_count = 0;
        merge_group_indices[group_count++] = base_index;
        is_active[base_index] = 0;

        for (int compare_index = base_index + 1; compare_index < candidate_count; compare_index++) {
            if (!is_active[compare_index]) {
                continue;
            }
            if (compute_iou(&p_candidate_detections[base_index], &p_candidate_detections[compare_index]) > nms_iou_threshold) {
                merge_group_indices[group_count++] = compare_index;
                is_active[compare_index] = 0;
            }
        }

        if (1 == group_count) {
            p_output_detections->dets[p_output_detections->n++] = p_candidate_detections[base_index];
        } else {
            float sum_scores = 0.0f;
            for (int group_index = 0; group_index < group_count; group_index++) {
                sum_scores += p_candidate_detections[merge_group_indices[group_index]].score;
            }

            BfDetection merged;
            memset(&merged, 0, sizeof(merged));

            for (int group_index = 0; group_index < group_count; group_index++) {
                const BfDetection *p_current_detection = &p_candidate_detections[merge_group_indices[group_index]];
                float normalized_weight = p_current_detection->score / sum_scores;
                merged.ymin  += normalized_weight * p_current_detection->ymin;
                merged.xmin  += normalized_weight * p_current_detection->xmin;
                merged.ymax  += normalized_weight * p_current_detection->ymax;
                merged.xmax  += normalized_weight * p_current_detection->xmax;
                for (int keypoint_index = 0; keypoint_index < NUM_KP; keypoint_index++) {
                    merged.kp[keypoint_index][0] += normalized_weight * p_current_detection->kp[keypoint_index][0];
                    merged.kp[keypoint_index][1] += normalized_weight * p_current_detection->kp[keypoint_index][1];
                }
            }
            merged.score = sum_scores / (float)group_count;
            p_output_detections->dets[p_output_detections->n++] = merged;
        }
    }
}

/**
 * @brief Execute full postprocessing pipeline for BlazeFace outputs.
 * @param[in] p_scores_stride8_quantized Quantized scores for stride-8 anchors.
 * @param[in] p_scores_stride16_quantized Quantized scores for stride-16 anchors.
 * @param[in] p_boxes_stride8_quantized Quantized boxes for stride-8 anchors.
 * @param[in] p_boxes_stride16_quantized Quantized boxes for stride-16 anchors.
 * @param[in] p_anchors Anchor definitions [896][4].
 * @param[in] score_threshold Confidence threshold.
 * @param[in] nms_iou_threshold Weighted NMS IoU threshold.
 * @param[out] p_output_detections Output detections.
 * @return void
 */

void postprocess(const int8_t  *p_scores_stride8_quantized,
                 const int8_t  *p_scores_stride16_quantized,
                 const int8_t  *p_boxes_stride8_quantized,
                 const int8_t  *p_boxes_stride16_quantized,
                 const float    p_anchors[896][4],
                 float          score_threshold,
                 float          nms_iou_threshold,
                 BfDetections  *p_output_detections)
{
    /* Step 1: Dequantise */
    static float dequantized_scores[TOTAL_ANCHORS];
    dequantize_scores_tensor(p_scores_stride8_quantized,  ANCHORS_S8,
                             OUTPUT_SCORES_S8_SCALE,  OUTPUT_SCORES_S8_ZP,
                             dequantized_scores);
    dequantize_scores_tensor(p_scores_stride16_quantized, ANCHORS_S16,
                             OUTPUT_SCORES_S16_SCALE, OUTPUT_SCORES_S16_ZP,
                             dequantized_scores + ANCHORS_S8);

    static float dequantized_boxes[TOTAL_ANCHORS][NUM_COORDS];
    dequantize_boxes_tensor(p_boxes_stride8_quantized,  ANCHORS_S8,
                            OUTPUT_BOXES_S8_SCALE,  OUTPUT_BOXES_S8_ZP,
                            dequantized_boxes);
    dequantize_boxes_tensor(p_boxes_stride16_quantized, ANCHORS_S16,
                            OUTPUT_BOXES_S16_SCALE, OUTPUT_BOXES_S16_ZP,
                            dequantized_boxes + ANCHORS_S8);

    /* Steps 2-4: Sigmoid + threshold + decode */
    static BfDetection pre_nms_detections[TOTAL_ANCHORS];
    int pre_nms_count = 0;

    for (int anchor_index = 0; anchor_index < TOTAL_ANCHORS; anchor_index++) {
        float confidence = fast_sigmoid(dequantized_scores[anchor_index]);
        if (confidence < score_threshold) {
            continue;
        }
        decode_anchor_prediction(dequantized_boxes[anchor_index], p_anchors[anchor_index], &pre_nms_detections[pre_nms_count], confidence);
        pre_nms_count++;
    }

    /* Step 5: Weighted NMS */
    apply_weighted_nms(pre_nms_detections, pre_nms_count, nms_iou_threshold, p_output_detections);
}
