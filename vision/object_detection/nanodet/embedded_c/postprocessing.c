/*
 * The Renesas Electronics Corporation
 * DISCLAIMER
 * This software is supplied by Renesas Electronics Corporation and is only intended for use with Renesas products. No
 * other uses are authorized. This software is owned by Renesas Electronics Corporation and is protected under all
 * applicable laws, including copyright laws.
 * THIS SOFTWARE IS PROVIDED 'AS IS' AND RENESAS MAKES NO WARRANTIES REGARDING
 * THIS SOFTWARE, WHETHER EXPRESS, IMPLIED OR STATUTORY, INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. ALL SUCH WARRANTIES ARE EXPRESSLY DISCLAIMED. TO THE MAXIMUM
 * EXTENT PERMITTED NOT PROHIBITED BY LAW, NEITHER RENESAS ELECTRONICS CORPORATION NOR ANY OF ITS AFFILIATED COMPANIES
 * SHALL BE LIABLE FOR ANY DIRECT, INDIRECT, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES FOR ANY REASON RELATED TO THIS
 * SOFTWARE, EVEN IF RENESAS OR ITS AFFILIATES HAVE BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
 * Renesas reserves the right, without notice, to make changes to this software and to discontinue the availability of
 * this software. By using this software, you agree to the additional terms and conditions found by accessing the
 * following link:
 * http://www.renesas.com/disclaimer
 */

/*
* Copyright (c) 2020 - 2026 Renesas Electronics Corporation and/or its affiliates
*
* SPDX-License-Identifier: BSD-3-Clause
*/
/**
 ******************************************************************************
 * @file    postprocessing.c
 * @brief   NanoDet output postprocessing
 *
 * @author  Renesas Electronics
 * @date    2026
 * @version 1.0
 ******************************************************************************
 */

#include "postprocessing.h"

#include <math.h>
#include <stddef.h>
#include <string.h>



typedef struct
{
    float _center_x;
    float _center_y;
    float _stride;
} center_prior_t;

typedef struct
{
    float _box_left;
    float _box_top;
    float _box_right;
    float _box_bottom;
    float _score;
    uint32_t _class_id;
} raw_det_t;

static center_prior_t _g_center_priors[MODEL_NUM_ANCHORS];
static int32_t _g_center_priors_ready = 0;

static raw_det_t _g_raw_detections[MODEL_MAX_RAW_DETS];

/**
 * @brief  Return the larger of two floating-point values.
 *
 * @param[in]  _left_value   First value.
 * @param[in]  _right_value  Second value.
 * @return                  Larger value.
 */
static inline float max_f32(float _left_value, float _right_value)
{
    return (_left_value > _right_value) ? _left_value : _right_value;
}

/**
 * @brief  Return the smaller of two floating-point values.
 *
 * @param[in]  _left_value   First value.
 * @param[in]  _right_value  Second value.
 * @return                  Smaller value.
 */
static inline float min_f32(float _left_value, float _right_value)
{
    return (_left_value < _right_value) ? _left_value : _right_value;
}

/**
 * @brief  Convert a quantized model output value into floating-point form.
 *
 * @param[in]  _quantized_value  Quantized int8 model output value.
 * @return                     Dequantized floating-point value.
 */
static float dequantize_output_to_float(int8_t _quantized_value)
{
    return ((float) ((int32_t) _quantized_value - MODEL_OUTPUT_ZERO_POINT)) * MODEL_OUTPUT_SCALE;
}

/**
 * @brief  Build the center-prior table used to decode NanoDet detections.
 */
static void build_center_priors(void)
{
    static const uint32_t _k_strides[MODEL_NUM_FPN_LEVELS] = {
        MODEL_STRIDE_S8, MODEL_STRIDE_S16, MODEL_STRIDE_S32, MODEL_STRIDE_S64
    };
    uint32_t _prior_index = 0U;
    uint32_t l;

    if (0 != _g_center_priors_ready)
    {
        return;
    }

    for (l = 0U; l < MODEL_NUM_FPN_LEVELS; l++)
    {
        const uint32_t _stride_value = _k_strides[l];
        const uint32_t _feature_map_height = (uint32_t) MODEL_INPUT_H / _stride_value;
        const uint32_t _feature_map_width = (uint32_t) MODEL_INPUT_W / _stride_value;
        uint32_t gy;

        for (gy = 0U; gy < _feature_map_height; gy++)
        {
            uint32_t _gx;
            for (_gx = 0U; _gx < _feature_map_width; _gx++)
            {
                if (_prior_index >= MODEL_NUM_ANCHORS)
                {
                    _g_center_priors_ready = 1;
                    return;
                }

                _g_center_priors[_prior_index]._center_x = (float) (_gx * _stride_value);
                _g_center_priors[_prior_index]._center_y = (float) (gy * _stride_value);
                _g_center_priors[_prior_index]._stride = (float) _stride_value;
                _prior_index++;
            }
        }
    }

    _g_center_priors_ready = 1;
}

/**
 * @brief  Compute softmax probabilities over the 8-bin regression distribution.
 *
 * @param[in]  p_input_values          Input logits.
 * @param[out] p_output_probabilities  Output softmax probabilities.
 */
static void softmax_8(const float *p_input_values, float *p_output_probabilities)
{
    float _maximum_value = p_input_values[0];
    float _probability_sum = 0.0F;
    int32_t i;

    for (i = 1; i < REG_MAX_PLUS_ONE; i++)
    {
        if (p_input_values[i] > _maximum_value)
        {
            _maximum_value = p_input_values[i];
        }
    }

    for (i = 0; i < REG_MAX_PLUS_ONE; i++)
    {
        p_output_probabilities[i] = expf(p_input_values[i] - _maximum_value);
        _probability_sum += p_output_probabilities[i];
    }

    if (_probability_sum <= 0.0F)
    {
        return;
    }

    for (i = 0; i < REG_MAX_PLUS_ONE; i++)
    {
        p_output_probabilities[i] /= _probability_sum;
    }
}

/**
 * @brief  Project four regression distributions into scalar distances.
 *
 * @param[in]  p_regression_logits     Regression logits for four box sides.
 * @param[out] p_projected_distances   Projected distances for left, top, right, bottom.
 */
static void distribution_project_4(const float *p_regression_logits, float *p_projected_distances)
{
    int32_t _side;

    for (_side = 0; _side < NANODET_BBOX_DIMS; _side++)
    {
        float _class_probabilities[REG_MAX_PLUS_ONE];
        float _distance_accumulator = 0.0F;
        int32_t b;

        softmax_8(&p_regression_logits[_side * REG_MAX_PLUS_ONE], _class_probabilities);
        for (b = 0; b < REG_MAX_PLUS_ONE; b++)
        {
            _distance_accumulator += _class_probabilities[b] * (float) b;
        }
        p_projected_distances[_side] = _distance_accumulator;
    }
}

/**
 * @brief  Compute the intersection-over-union between two raw detections.
 *
 * @param[in]  p_left_detection   First detection.
 * @param[in]  p_right_detection  Second detection.
 * @return                       Intersection-over-union value.
 */
static float compute_iou(const raw_det_t *p_left_detection, const raw_det_t *p_right_detection)
{
    const float _intersection_left = max_f32(p_left_detection->_box_left, p_right_detection->_box_left);
    const float _intersection_top = max_f32(p_left_detection->_box_top, p_right_detection->_box_top);
    const float _intersection_right = min_f32(p_left_detection->_box_right, p_right_detection->_box_right);
    const float _intersection_bottom = min_f32(p_left_detection->_box_bottom, p_right_detection->_box_bottom);

    const float _intersection_width = max_f32(0.0F, _intersection_right - _intersection_left);
    const float _intersection_height = max_f32(0.0F, _intersection_bottom - _intersection_top);
    const float _intersection_area = _intersection_width * _intersection_height;

    if (_intersection_area <= 0.0F)
    {
        return 0.0F;
    }

    {
        const float _left_detection_area =
            max_f32(0.0F, p_left_detection->_box_right - p_left_detection->_box_left) *
            max_f32(0.0F, p_left_detection->_box_bottom - p_left_detection->_box_top);
        const float _right_detection_area =
            max_f32(0.0F, p_right_detection->_box_right - p_right_detection->_box_left) *
            max_f32(0.0F, p_right_detection->_box_bottom - p_right_detection->_box_top);
        return _intersection_area /
               (_left_detection_area + _right_detection_area - _intersection_area + 1e-6F);
    }
}

/**
 * @brief  Apply per-class NMS and write filtered detections to the output buffer.
 *
 * @param[in,out] p_detections           Raw detections for all classes.
 * @param[in]     _detection_count        Number of raw detections.
 * @param[in]     _class_id               Class to filter.
 * @param[in]     _nms_iou_threshold      IoU threshold for suppression.
 * @param[out]    p_output_detections    Output detection buffer.
 * @param[in]     _output_capacity        Maximum output detections to write.
 * @return                             Number of detections written.
 */
static int32_t apply_nms_for_class(raw_det_t *p_detections,
                                   int32_t _detection_count,
                                   uint32_t _class_id,
                                   float _nms_iou_threshold,
                                   Detection_t *p_output_detections,
                                   int32_t _output_capacity)
{
    int32_t _detection_indices[MODEL_MAX_RAW_DETS];
    int32_t _matching_detection_count = 0;
    int32_t i;
    int32_t _written_detection_count = 0;

    for (i = 0; i < _detection_count; i++)
    {
        if ((p_detections[i]._class_id == _class_id) && (p_detections[i]._score >= 0.0F))
        {
            _detection_indices[_matching_detection_count++] = i;
        }
    }

    for (i = 1; i < _matching_detection_count; i++)
    {
        const int32_t _current_detection_index = _detection_indices[i];
        int32_t j = i - 1;

        while ((j >= 0) &&
               (p_detections[_detection_indices[j]]._score < p_detections[_current_detection_index]._score))
        {
            _detection_indices[j + 1] = _detection_indices[j];
            j--;
        }
        _detection_indices[j + 1] = _current_detection_index;
    }

    for (i = 0; (i < _matching_detection_count) && (_written_detection_count < _output_capacity); i++)
    {
        const int32_t _selected_detection_index = _detection_indices[i];
        int32_t j;

        if (p_detections[_selected_detection_index]._score < 0.0F)
        {
            continue;
        }

        p_output_detections[_written_detection_count]._x1 = p_detections[_selected_detection_index]._box_left;
        p_output_detections[_written_detection_count]._y1 = p_detections[_selected_detection_index]._box_top;
        p_output_detections[_written_detection_count]._x2 = p_detections[_selected_detection_index]._box_right;
        p_output_detections[_written_detection_count]._y2 = p_detections[_selected_detection_index]._box_bottom;
        p_output_detections[_written_detection_count]._score = p_detections[_selected_detection_index]._score;
        p_output_detections[_written_detection_count]._cls_id = _class_id;
        _written_detection_count++;

        for (j = i + 1; j < _matching_detection_count; j++)
        {
            const int32_t _suppressed_detection_index = _detection_indices[j];
            if (p_detections[_suppressed_detection_index]._score < 0.0F)
            {
                continue;
            }
            if (compute_iou(&p_detections[_selected_detection_index],
                            &p_detections[_suppressed_detection_index]) > _nms_iou_threshold)
            {
                p_detections[_suppressed_detection_index]._score = -1.0F;
            }
        }
    }

    return _written_detection_count;
}

/**
 * @brief  Decode NanoDet raw output and apply per-class non-maximum suppression.
 *
 * @param[in]  p_raw_output            Flat int8 model output buffer.
 * @param[in]  p_letterbox_params      Resize metadata from preprocessing.
 * @param[in]  _original_width_pixels   Original image width in pixels.
 * @param[in]  _original_height_pixels  Original image height in pixels.
 * @param[in]  _score_threshold         Minimum score required to keep a detection.
 * @param[in]  _nms_iou_threshold       IoU threshold for suppression.
 * @param[out] p_output_detections     Output detection array.
 * @return                             Number of valid detections written.
 */
int32_t postprocess(const int8_t *p_raw_output,
                    const letterbox_params_t *p_letterbox_params,
                    uint32_t _original_width_pixels,
                    uint32_t _original_height_pixels,
                    float _score_threshold,
                    float _nms_iou_threshold,
                    Detection_t *p_output_detections)
{
    int32_t _raw_detection_count = 0;
    int32_t a;
    int32_t _total_output_count = 0;
    int32_t _class_presence_flags[MODEL_NUM_CLASSES];

    if ((NULL == p_raw_output) || (NULL == p_letterbox_params) || (NULL == p_output_detections))
    {
        return 0;
    }

    (void) _original_width_pixels;
    (void) _original_height_pixels;

    if (MODEL_OUTPUT_CHANNELS <= (NANODET_BBOX_DIMS * REG_MAX_PLUS_ONE))
    {
        return 0;
    }

    build_center_priors();

    for (a = 0; a < (int32_t) MODEL_NUM_ANCHORS; a++)
    {
        const int8_t *p_raw_output_row = p_raw_output + ((size_t) a * MODEL_OUTPUT_CHANNELS);
        const int32_t _class_count = (int32_t) MODEL_OUTPUT_CHANNELS - (NANODET_BBOX_DIMS * REG_MAX_PLUS_ONE);
        int32_t _cls = 0;
        int32_t _best_class_index = 0;
        float _best_score;
        float _dequantized_output_row[MODEL_OUTPUT_CHANNELS];
        float _regression_logits[NANODET_BBOX_DIMS * REG_MAX_PLUS_ONE];
        float _projected_distances[NANODET_BBOX_DIMS];
        float _decoded_box_left;
        float _decoded_box_top;
        float _decoded_box_right;
        float _decoded_box_bottom;

        if (_class_count > (int32_t) MODEL_NUM_CLASSES)
        {
            return 0;
        }

        /* Dequantize the full per-anchor row once, then reuse it for class and box decoding. */
        for (_cls = 0; _cls < (int32_t) MODEL_OUTPUT_CHANNELS; _cls++)
        {
            _dequantized_output_row[_cls] = dequantize_output_to_float(p_raw_output_row[_cls]);
        }

        _best_score = _dequantized_output_row[0];
        for (_cls = 1; _cls < _class_count; _cls++)
        {
            const float _class_score = _dequantized_output_row[_cls];
            if (_class_score > _best_score)
            {
                _best_score = _class_score;
                _best_class_index = _cls;
            }
        }

        if (_best_score <= _score_threshold)
        {
            continue;
        }

        /* Split the regression logits from the class scores before projecting distances. */
        for (_cls = 0; _cls < NANODET_BBOX_DIMS * REG_MAX_PLUS_ONE; _cls++)
        {
            _regression_logits[_cls] = _dequantized_output_row[_class_count + _cls];
        }

        distribution_project_4(_regression_logits, _projected_distances);

        _projected_distances[0] *= _g_center_priors[a]._stride;
        _projected_distances[1] *= _g_center_priors[a]._stride;
        _projected_distances[2] *= _g_center_priors[a]._stride;
        _projected_distances[3] *= _g_center_priors[a]._stride;

        _decoded_box_left = _g_center_priors[a]._center_x - _projected_distances[0];
        _decoded_box_top = _g_center_priors[a]._center_y - _projected_distances[1];
        _decoded_box_right = _g_center_priors[a]._center_x + _projected_distances[2];
        _decoded_box_bottom = _g_center_priors[a]._center_y + _projected_distances[3];

        _decoded_box_left = _decoded_box_left / p_letterbox_params->_scale_w;
        _decoded_box_top = _decoded_box_top / p_letterbox_params->_scale_h;
        _decoded_box_right = _decoded_box_right / p_letterbox_params->_scale_w;
        _decoded_box_bottom = _decoded_box_bottom / p_letterbox_params->_scale_h;

        if (_raw_detection_count < MODEL_MAX_RAW_DETS)
        {
            /* best_cls and _best_score are identical to what was already found
             * above; collapse into the same values — no second scan needed. */
            _g_raw_detections[_raw_detection_count]._box_left = _decoded_box_left;
            _g_raw_detections[_raw_detection_count]._box_top = _decoded_box_top;
            _g_raw_detections[_raw_detection_count]._box_right = _decoded_box_right;
            _g_raw_detections[_raw_detection_count]._box_bottom = _decoded_box_bottom;
            _g_raw_detections[_raw_detection_count]._score = _best_score;
            _g_raw_detections[_raw_detection_count]._class_id = (uint32_t) _best_class_index;
            _raw_detection_count++;
        }
    }

    if (0 == _raw_detection_count)
    {
        return 0;
    }

    (void) memset(_class_presence_flags, 0, sizeof(_class_presence_flags));
    for (a = 0; a < _raw_detection_count; a++)
    {
        if (_g_raw_detections[a]._class_id < MODEL_NUM_CLASSES)
        {
            _class_presence_flags[_g_raw_detections[a]._class_id] = 1;
        }
    }

    for (a = 0; a < (int32_t) MODEL_NUM_CLASSES; a++)
    {
        int32_t _remaining_output_capacity;
        int32_t _written_detection_count;
        if (0 == _class_presence_flags[a])
        {
            continue;
        }

        _remaining_output_capacity = (int32_t) POSTPROC_MAX_DETS - _total_output_count;
        if (_remaining_output_capacity <= 0)
        {
            break;
        }

        /* NMS is applied per class so that overlapping boxes from different classes are preserved. */
        _written_detection_count = apply_nms_for_class(_g_raw_detections,
                                                      _raw_detection_count,
                                                      (uint32_t) a,
                                                      _nms_iou_threshold,
                                                      &p_output_detections[_total_output_count],
                                                      _remaining_output_capacity);
        _total_output_count += _written_detection_count;
    }

    return _total_output_count;
}
