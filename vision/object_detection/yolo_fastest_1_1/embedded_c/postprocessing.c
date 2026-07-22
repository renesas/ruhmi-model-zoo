/*
* Copyright (c) 2020 - 2026 Renesas Electronics Corporation and/or its affiliates
*
* SPDX-License-Identifier: BSD-3-Clause
*/
/**
 ******************************************************************************
 * @file    postprocessing.c
 * @brief   YOLO-Fastest 1.1 INT8 anchor-based decode + per-class NMS - MCU portable.
 *
 * Pipeline (mirrors Python inference.py for TFLite INT8):
 *   1. dequantize()   — int8 → float32 using per-head output quant params
 *   2. For each head, each grid cell, each anchor:
 *        sigmoid(tx,ty) → centre offsets
 *        exp(tw,th) × anchor_wh → box size
 *        sigmoid(objectness) × sigmoid(class_logits[argmax]) → score
 *   3. score_filter() — combined score > thresh
 *   4. undo_letterbox() — map model coords back to original image pixels
 *   5. nms_class()    — greedy IoU NMS per class
 *
 * All buffers are statically allocated — no malloc/free.
 ******************************************************************************
 */

#include "postprocessing.h"

#include <stddef.h>
#include <string.h>
#include <math.h>

/* ── File-scope types and static storage ────────────────────────────── */

typedef struct
{
    float    x1;
    float    y1;
    float    x2;
    float    y2;
    float    score;
    uint32_t cls_id;
} raw_det_t;

static raw_det_t s_raw[MAX_RAW_DETS];

/* ── Anchor tables ───────────────────────────────────────────────────── */

typedef struct
{
    uint32_t aw;   /**< Anchor width in input pixels  */
    uint32_t ah;   /**< Anchor height in input pixels */
} anchor_t;

/* Head 0: stride-32 (large objects) — anchors from yolo-fastest-1.1.cfg mask 3,4,5 */
static const anchor_t s_head0_anchors[MODEL_NUM_ANCHORS_PER_CELL] = {
    { MODEL_HEAD0_AW0, MODEL_HEAD0_AH0 },
    { MODEL_HEAD0_AW1, MODEL_HEAD0_AH1 },
    { MODEL_HEAD0_AW2, MODEL_HEAD0_AH2 },
};

/* Head 1: stride-16 (small objects) — anchors from yolo-fastest-1.1.cfg mask 0,1,2 */
static const anchor_t s_head1_anchors[MODEL_NUM_ANCHORS_PER_CELL] = {
    { MODEL_HEAD1_AW0, MODEL_HEAD1_AH0 },
    { MODEL_HEAD1_AW1, MODEL_HEAD1_AH1 },
    { MODEL_HEAD1_AW2, MODEL_HEAD1_AH2 },
};

/* ── Inline helpers ──────────────────────────────────────────────────── */

/**
 * @brief Return the larger of two float values.
 *
 * @param[in] first_value  First operand.
 * @param[in] second_value Second operand.
 *
 * @return Maximum of first_value and second_value.
 */
static inline float get_maximum_float_value(float first_value, float second_value)
{
    return (first_value > second_value) ? first_value : second_value;
}

/**
 * @brief Return the smaller of two float values.
 *
 * @param[in] first_value  First operand.
 * @param[in] second_value Second operand.
 *
 * @return Minimum of first_value and second_value.
 */
static inline float get_minimum_float_value(float first_value, float second_value)
{
    return (first_value < second_value) ? first_value : second_value;
}

/**
 * @brief Numerically stable sigmoid.
 *
 * @param[in] x Input value.
 *
 * @return Sigmoid activation value in range (0,1).
 */
static inline float sigmoid_f(float x)
{
    if (x >= 0.0F)
    {
        float _ez = expf(-x);
        return SIGMOID_ONE_F / (SIGMOID_ONE_F + _ez);
    }
    else
    {
        float _ez = expf(x);
        return _ez / (SIGMOID_ONE_F + _ez);
    }
}

/* ── Dequantize helpers ──────────────────────────────────────────────── */

/**
 * @brief Dequantize one int8 value from head0 output tensor.
 *
 * @param[in] raw_val Quantized int8 value.
 *
 * @return Dequantized float value.
 */
static inline float dequantize_head0(int8_t raw_val)
{
    return ((float)raw_val - (float)MODEL_HEAD0_ZERO_POINT) * MODEL_HEAD0_SCALE;
}

/**
 * @brief Dequantize one int8 value from head1 output tensor.
 *
 * @param[in] raw_val Quantized int8 value.
 *
 * @return Dequantized float value.
 */
static inline float dequantize_head1(int8_t raw_val)
{
    return ((float)raw_val - (float)MODEL_HEAD1_ZERO_POINT) * MODEL_HEAD1_SCALE;
}

/* ── Decode one YOLO head ────────────────────────────────────────────── */

/**
 * @brief Decode one anchor-based YOLO head into raw detections.
 *
 * @param[in]  p_raw_head       Flat int8 output tensor [grid_h × grid_w × 255].
 * @param[in]  grid_h           Grid height.
 * @param[in]  grid_w           Grid width.
 * @param[in]  stride           Feature map stride (16 or 32).
 * @param[in]  p_anchors        Array of 3 anchors for this head.
 * @param[in]  p_dequant_fn     Dequantize function for this head.
 * @param[in]  score_thresh     Minimum detection score.
 * @param[in,out] p_raw_dets    Output raw detection buffer.
 * @param[in]  max_raw          Maximum capacity of p_raw_dets.
 * @param[in]  current_count    Current number of detections already in buffer.
 *
 * @return Updated total count of raw detections.
 */
static int32_t decode_yolo_head(const int8_t    *p_raw_head,
                                uint32_t         grid_h,
                                uint32_t         grid_w,
                                uint32_t         stride,
                                const anchor_t  *p_anchors,
                                float          (*p_dequant_fn)(int8_t),
                                float            score_thresh,
                                raw_det_t       *p_raw_dets,
                                uint32_t         max_raw,
                                int32_t          current_count)
{
    uint32_t _row;
    uint32_t _col;
    uint32_t _a_idx;
    uint32_t _cls_idx;
    int32_t _num_dets = current_count;

    for (_row = 0U; _row < grid_h; _row++)
    {
        for (_col = 0U; _col < grid_w; _col++)
        {
            /* Base offset into the flat [H, W, 255] tensor for this cell */
            uint32_t _cell_offset = (_row * grid_w + _col) * MODEL_OUTPUT_CHANNELS;

            for (_a_idx = 0U; _a_idx < MODEL_NUM_ANCHORS_PER_CELL; _a_idx++)
            {
                /* Each anchor has 85 values: [tx, ty, tw, th, obj, cls0..cls79] */
                uint32_t _anchor_offset = _cell_offset + _a_idx * MODEL_FIELDS_PER_ANCHOR;
                const int8_t *_p_anchor = p_raw_head + _anchor_offset;

                /* Dequantize objectness for early exit */
                float _obj_score = sigmoid_f(p_dequant_fn(_p_anchor[YOLO_OBJECTNESS_INDEX]));
                if (_obj_score < score_thresh)
                {
                    continue;
                }

                /* Find argmax class — work in int8 domain (monotonic) */
                uint32_t _best_class_id = 0U;
                int8_t _best_class_raw = _p_anchor[YOLO_CLASS_LOGITS_START_INDEX];
                for (_cls_idx = 1U; _cls_idx < MODEL_NUM_CLASSES; _cls_idx++)
                {
                    if (_p_anchor[YOLO_CLASS_LOGITS_START_INDEX + _cls_idx] > _best_class_raw)
                    {
                        _best_class_raw = _p_anchor[YOLO_CLASS_LOGITS_START_INDEX + _cls_idx];
                        _best_class_id = _cls_idx;
                    }
                }

                /* Dequantize and sigmoid the best class logit */
                float _best_class_score = sigmoid_f(p_dequant_fn(_best_class_raw));

                /* Combined score = objectness × class confidence */
                float _detection_score = _obj_score * _best_class_score;
                if (_detection_score < score_thresh)
                {
                    continue;
                }

                if ((uint32_t)_num_dets >= max_raw)
                {
                    return _num_dets;
                }

                /* Dequantize box regression outputs */
                float _tx = p_dequant_fn(_p_anchor[YOLO_BOX_TX_INDEX]);
                float _ty = p_dequant_fn(_p_anchor[YOLO_BOX_TY_INDEX]);
                float _tw = p_dequant_fn(_p_anchor[YOLO_BOX_TW_INDEX]);
                float _th = p_dequant_fn(_p_anchor[YOLO_BOX_TH_INDEX]);

                /* Anchor-based box decoding:
                 *   cx = (sigmoid(tx) + grid_x) * stride
                 *   cy = (sigmoid(ty) + grid_y) * stride
                 *   w  = exp(tw) * anchor_w
                 *   h  = exp(th) * anchor_h
                 */
                float _box_center_x = (sigmoid_f(_tx) + (float)_col) * (float)stride;
                float _box_center_y = (sigmoid_f(_ty) + (float)_row) * (float)stride;
                float _box_width = expf(_tw) * (float)p_anchors[_a_idx].aw;
                float _box_height = expf(_th) * (float)p_anchors[_a_idx].ah;

                /* Convert center-format to corner-format */
                p_raw_dets[_num_dets].x1 = _box_center_x - (_box_width * BOX_HALF_FACTOR);
                p_raw_dets[_num_dets].y1 = _box_center_y - (_box_height * BOX_HALF_FACTOR);
                p_raw_dets[_num_dets].x2 = _box_center_x + (_box_width * BOX_HALF_FACTOR);
                p_raw_dets[_num_dets].y2 = _box_center_y + (_box_height * BOX_HALF_FACTOR);
                p_raw_dets[_num_dets].score = _detection_score;
                p_raw_dets[_num_dets].cls_id = _best_class_id;
                _num_dets++;
            }
        }
    }

    return _num_dets;
}

/* ── IoU computation ─────────────────────────────────────────────────── */

/**
 * @brief Compute intersection-over-union between two detections.
 *
 * @param[in] p_first_detection  First detection box.
 * @param[in] p_second_detection Second detection box.
 *
 * @return IoU value in range [0,1].
 */
static float compute_iou(const raw_det_t *p_first_detection, const raw_det_t *p_second_detection)
{
    float _intersection_left_x = get_maximum_float_value(p_first_detection->x1, p_second_detection->x1);
    float _intersection_top_y = get_maximum_float_value(p_first_detection->y1, p_second_detection->y1);
    float _intersection_right_x = get_minimum_float_value(p_first_detection->x2, p_second_detection->x2);
    float _intersection_bottom_y = get_minimum_float_value(p_first_detection->y2, p_second_detection->y2);

    float _intersection_width = get_maximum_float_value(0.0F, _intersection_right_x - _intersection_left_x);
    float _intersection_height = get_maximum_float_value(0.0F, _intersection_bottom_y - _intersection_top_y);
    float _intersection_area = _intersection_width * _intersection_height;

    if (0.0F == _intersection_area)
    {
        return 0.0F;
    }

    float _first_area  = (p_first_detection->x2  - p_first_detection->x1) *
                        (p_first_detection->y2  - p_first_detection->y1);
    float _second_area = (p_second_detection->x2 - p_second_detection->x1) *
                        (p_second_detection->y2 - p_second_detection->y1);

    return _intersection_area / (_first_area + _second_area - _intersection_area + IOU_EPSILON_F);
}

/* ── Per-class greedy NMS ────────────────────────────────────────────── */

/**
 * @brief Apply class-wise greedy NMS and write surviving boxes.
 *
 * @param[in,out] p_raw_dets         Raw detection buffer; suppressed entries are marked.
 * @param[in]     num_raw_dets       Number of valid raw detections in p_raw_dets.
 * @param[in]     target_class_id    Class ID to process.
 * @param[in]     nms_thresh         IoU suppression threshold.
 * @param[in]     p_letterbox_params Letterbox scale and padding from preprocessing.
 * @param[in]     orig_img_width     Original image width in pixels.
 * @param[in]     orig_img_height    Original image height in pixels.
 * @param[out]    p_out_dets         Output destination for kept detections.
 * @param[in]     out_capacity       Maximum number of entries writable to p_out_dets.
 *
 * @return Number of detections written for target_class_id.
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
    static int32_t _sorted_indices[MAX_RAW_DETS];
    int32_t _matching_detection_count = 0;
    int32_t _i;
    int32_t _j;

    for (_i = 0; _i < num_raw_dets; _i++)
    {
        if ((p_raw_dets[_i].cls_id == target_class_id) && (p_raw_dets[_i].score >= 0.0F))
        {
            _sorted_indices[_matching_detection_count] = _i;
            _matching_detection_count++;
        }
    }

    /* Insertion sort descending by score */
    for (_i = 1; _i < _matching_detection_count; _i++)
    {
        int32_t _key_idx = _sorted_indices[_i];
        _j = _i - 1;
        while ((_j >= 0) && (p_raw_dets[_sorted_indices[_j]].score < p_raw_dets[_key_idx].score))
        {
            _sorted_indices[_j + 1] = _sorted_indices[_j];
            _j--;
        }
        _sorted_indices[_j + 1] = _key_idx;
    }

    int32_t _written = 0;

    for (_i = 0; (_i < _matching_detection_count) && (_written < out_capacity); _i++)
    {
        int32_t _cur_idx = _sorted_indices[_i];
        if (p_raw_dets[_cur_idx].score < 0.0F)
        {
            continue;
        }

        /* Undo letterbox transform: remove padding then divide by scale. */
        float _rescaled_x1 = (p_raw_dets[_cur_idx].x1 - (float)p_letterbox_params->pad_x) / p_letterbox_params->scale;
        float _rescaled_y1 = (p_raw_dets[_cur_idx].y1 - (float)p_letterbox_params->pad_y) / p_letterbox_params->scale;
        float _rescaled_x2 = (p_raw_dets[_cur_idx].x2 - (float)p_letterbox_params->pad_x) / p_letterbox_params->scale;
        float _rescaled_y2 = (p_raw_dets[_cur_idx].y2 - (float)p_letterbox_params->pad_y) / p_letterbox_params->scale;

        /* Clamp to image boundaries */
        _rescaled_x1 = get_maximum_float_value(0.0F, get_minimum_float_value(_rescaled_x1, (float)(orig_img_width  - 1U)));
        _rescaled_y1 = get_maximum_float_value(0.0F, get_minimum_float_value(_rescaled_y1, (float)(orig_img_height - 1U)));
        _rescaled_x2 = get_maximum_float_value(0.0F, get_minimum_float_value(_rescaled_x2, (float)(orig_img_width  - 1U)));
        _rescaled_y2 = get_maximum_float_value(0.0F, get_minimum_float_value(_rescaled_y2, (float)(orig_img_height - 1U)));

        p_out_dets[_written].x1 = _rescaled_x1;
        p_out_dets[_written].y1 = _rescaled_y1;
        p_out_dets[_written].x2 = _rescaled_x2;
        p_out_dets[_written].y2 = _rescaled_y2;
        p_out_dets[_written].score = p_raw_dets[_cur_idx].score;
        p_out_dets[_written].cls_id = target_class_id;
        _written++;

        /* Suppress overlapping boxes of the same class */
        for (_j = _i + 1; _j < _matching_detection_count; _j++)
        {
            int32_t _sup_idx = _sorted_indices[_j];
            if (p_raw_dets[_sup_idx].score < 0.0F)
            {
                continue;
            }
            if (compute_iou(&p_raw_dets[_cur_idx], &p_raw_dets[_sup_idx]) > nms_thresh)
            {
                p_raw_dets[_sup_idx].score = SUPPRESSED_SCORE_F;
            }
        }
    }

    return _written;
}

/* ── Public API ──────────────────────────────────────────────────────── */

/**
 * @brief Decode both YOLO heads, undo letterbox scaling, and apply per-class NMS.
 *
 * @param[in]  p_raw_head0   Flat int8 array for the stride-32 head.
 * @param[in]  p_raw_head1   Flat int8 array for the stride-16 head.
 * @param[in]  p_params      Letterbox parameters from preprocess().
 * @param[in]  orig_w        Original image width in pixels.
 * @param[in]  orig_h        Original image height in pixels.
 * @param[in]  score_thresh  Minimum detection score to keep.
 * @param[in]  nms_thresh    IoU threshold for per-class NMS.
 * @param[out] p_out_dets    Output array for final detections.
 *
 * @return Number of detections written to p_out_dets.
 */
int32_t postprocess(const int8_t           *p_raw_head0,
                    const int8_t           *p_raw_head1,
                    const letterbox_params_t *p_params,
                    uint32_t                orig_w,
                    uint32_t                orig_h,
                    float                   score_thresh,
                    float                   nms_thresh,
                    Detection_t            *p_out_dets)
{
    uint32_t _class_idx;
    int32_t _num_raw_detections = 0;

    if ((NULL == p_raw_head0) || (NULL == p_raw_head1) ||
        (NULL == p_params)  || (NULL == p_out_dets))
    {
        return 0;
    }

    /* ── Step 1: Decode both heads ───────────────────────────────────── */

    /* Head 0: stride-32, 10×10 grid, large-object anchors */
    _num_raw_detections = decode_yolo_head(p_raw_head0,
                                          MODEL_HEAD0_GRID_H, MODEL_HEAD0_GRID_W,
                                          MODEL_HEAD0_STRIDE,
                                          s_head0_anchors,
                                          dequantize_head0,
                                          score_thresh,
                                          s_raw, MAX_RAW_DETS,
                                          _num_raw_detections);

    /* Head 1: stride-16, 20×20 grid, small-object anchors */
    _num_raw_detections = decode_yolo_head(p_raw_head1,
                                          MODEL_HEAD1_GRID_H, MODEL_HEAD1_GRID_W,
                                          MODEL_HEAD1_STRIDE,
                                          s_head1_anchors,
                                          dequantize_head1,
                                          score_thresh,
                                          s_raw, MAX_RAW_DETS,
                                          _num_raw_detections);

    if (0 == _num_raw_detections)
    {
        return 0;
    }

    /* ── Step 2: per-class NMS → original image coordinates ───────────── */
    int32_t _total_output_detections = 0;

    int32_t _classes_seen[MODEL_NUM_CLASSES];
    (void)memset(_classes_seen, 0, sizeof(_classes_seen));
    {
        int32_t _raw_det_idx;
        for (_raw_det_idx = 0; _raw_det_idx < _num_raw_detections; _raw_det_idx++)
        {
            _classes_seen[s_raw[_raw_det_idx].cls_id] = 1;
        }
    }

    for (_class_idx = 0U; _class_idx < MODEL_NUM_CLASSES; _class_idx++)
    {
        if (_classes_seen[_class_idx] == 0)
        {
            continue;
        }

        int32_t _remaining_capacity = (int32_t)MAX_DETECTIONS - _total_output_detections;
        if (_remaining_capacity <= 0)
        {
            break;
        }

        int32_t _num_class_dets = apply_nms_for_class(s_raw, _num_raw_detections, _class_idx,
                                                      nms_thresh,
                                                      p_params, orig_w, orig_h,
                                                      &p_out_dets[_total_output_detections],
                                                      _remaining_capacity);
        _total_output_detections += _num_class_dets;
    }

    return _total_output_detections;
}
