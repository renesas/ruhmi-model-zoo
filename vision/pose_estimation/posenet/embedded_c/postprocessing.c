/*
 * Copyright (C) 2026 Renesas Electronics Corporation. All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

/***********************************************************************************************************************
 * File Name    : postprocessing.c
 * Description  : PoseNet output postprocessing implementation - PersonLab
 *                multi-person pose decoding, keypoint detection, NMS,
 *                instance scoring, and coordinate rescaling.
 **********************************************************************************************************************/


#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include "postprocessing.h"
#include "model_metadata.h"



/* ------------------------------------------------------------------------- */
/* Skeleton edge list used by the chain walk.                                 */
/* Each row is (parent_id, child_id), i.e. { source, target } for forward     */
/* traversal. Backward traversal treats { child, parent } as { source, target}*/
/* Order must match the model's displacement_fwd / displacement_bwd channel   */
/* layout (edge index = channel index for dy; edge index + NUM_POSE_EDGES for */
/* dx).                                                                       */
/* ------------------------------------------------------------------------- */
static const uint8_t s_pose_chain[NUM_POSE_EDGES][2] =
{
    /*  0 */ { 0,  1},   /* nose            -> leftEye        */
    /*  1 */ { 1,  3},   /* leftEye         -> leftEar        */
    /*  2 */ { 0,  2},   /* nose            -> rightEye       */
    /*  3 */ { 2,  4},   /* rightEye        -> rightEar       */
    /*  4 */ { 0,  5},   /* nose            -> leftShoulder   */
    /*  5 */ { 5,  7},   /* leftShoulder    -> leftElbow      */
    /*  6 */ { 7,  9},   /* leftElbow       -> leftWrist      */
    /*  7 */ { 5, 11},   /* leftShoulder    -> leftHip        */
    /*  8 */ {11, 13},   /* leftHip         -> leftKnee       */
    /*  9 */ {13, 15},   /* leftKnee        -> leftAnkle      */
    /* 10 */ { 0,  6},   /* nose            -> rightShoulder  */
    /* 11 */ { 6,  8},   /* rightShoulder   -> rightElbow     */
    /* 12 */ { 8, 10},   /* rightElbow      -> rightWrist     */
    /* 13 */ { 6, 12},   /* rightShoulder   -> rightHip       */
    /* 14 */ {12, 14},   /* rightHip        -> rightKnee      */
    /* 15 */ {14, 16},   /* rightKnee       -> rightAnkle     */
};

/* ------------------------------------------------------------------------- */
/* Dequantization helpers                                                     */
/* ------------------------------------------------------------------------- */

/**
 * @brief Dequantize an int8 heatmap sample to float using HEATMAP_QUANT_*.
 *
 * @param[in] _quantized_value Quantized heatmap sample.
 * @return Dequantized heatmap value.
 * @note Variable use:
 *       - _quantized_value : Input int8 tensor value before zero-point/scale conversion.
 */
static inline float dequant_heatmap(int8_t _quantized_value)
{
    return ((float)_quantized_value - (float)HEATMAP_QUANT_ZP) * HEATMAP_QUANT_SCALE;
}

/**
 * @brief Dequantize an int8 offset sample to float using OFFSET_QUANT_*.
 *
 * @param[in] _quantized_value Quantized offset sample.
 * @return Dequantized offset value.
 * @note Variable use:
 *       - _quantized_value : Input int8 tensor value before zero-point/scale conversion.
 */
static inline float dequant_offset(int8_t _quantized_value)
{
    return ((float)_quantized_value - (float)OFFSET_QUANT_ZP) * OFFSET_QUANT_SCALE;
}

/**
 * @brief Dequantize an int8 forward displacement sample using DISP_FWD_QUANT_*.
 *
 * @param[in] _quantized_value Quantized forward displacement sample.
 * @return Dequantized forward displacement value.
 * @note Variable use:
 *       - _quantized_value : Input int8 tensor value before zero-point/scale conversion.
 */
static inline float dequant_disp_fwd(int8_t _quantized_value)
{
    return ((float)_quantized_value - (float)DISP_FWD_QUANT_ZP) * DISP_FWD_QUANT_SCALE;
}

/**
 * @brief Dequantize an int8 backward displacement sample using DISP_BWD_QUANT_*.
 *
 * @param[in] _quantized_value Quantized backward displacement sample.
 * @return Dequantized backward displacement value.
 * @note Variable use:
 *       - _quantized_value : Input int8 tensor value before zero-point/scale conversion.
 */
static inline float dequant_disp_bwd(int8_t _quantized_value)
{
    return ((float)_quantized_value - (float)DISP_BWD_QUANT_ZP) * DISP_BWD_QUANT_SCALE;
}

/* ------------------------------------------------------------------------- */
/* Tensor accessors                                                           */
/* ------------------------------------------------------------------------- */

/**
 * @brief Read and dequantize the heatmap value at grid (_grid_y, _grid_x)
 *        for keypoint _keypoint_id.
 *
 * @param[in] p_hm Heatmap tensor pointer.
 * @param[in] _grid_y     Grid y index.
 * @param[in] _grid_x     Grid x index.
 * @param[in] _keypoint_id Keypoint channel index.
 * @return Dequantized heatmap score at (_grid_y, _grid_x, _keypoint_id).
 * @note Variable use:
 *       - _grid_y/_grid_x : Spatial location in output heatmap grid.
 *       - _keypoint_id    : Keypoint channel index.
 */
static inline float hm_at(const int8_t *p_hm, int _grid_y, int _grid_x, int _keypoint_id)
{
    return dequant_heatmap(p_hm[_grid_y * HM_STRIDE_H + _grid_x * HM_STRIDE_W + _keypoint_id]);
}

/**
 * @brief Read the offset (dy, dx) at grid (_grid_y, _grid_x) for keypoint _keypoint_id.
 *
 * @param[in]  p_off    Offset tensor pointer.
 * @param[in]  _grid_y  Grid y index.
 * @param[in]  _grid_x  Grid x index.
 * @param[in]  _keypoint_id Keypoint channel index.
 * @param[out] p_out_dy Output offset along y (model-input pixels).
 * @param[out] p_out_dx Output offset along x (model-input pixels).
 * @note Variable use:
 *       - _base_index : Flattened tensor base index for the selected grid cell.
 */
static inline void off_at(const int8_t *p_off, int _grid_y, int _grid_x, int _keypoint_id,
                          float *p_out_dy, float *p_out_dx)
{
    const int _base_index = _grid_y * OFF_STRIDE_H + _grid_x * OFF_STRIDE_W;
    *p_out_dy = dequant_offset(p_off[_base_index + _keypoint_id]);
    *p_out_dx = dequant_offset(p_off[_base_index + _keypoint_id + MODEL_NUM_KEYPOINTS]);
}

/**
 * @brief Read the forward displacement (dy, dx) at grid (_grid_y, _grid_x) for edge.
 *
 * @param[in]  p_df     Forward displacement tensor pointer.
 * @param[in]  _grid_y  Grid y index.
 * @param[in]  _grid_x  Grid x index.
 * @param[in]  _edge_id Pose-chain edge/channel index.
 * @param[out] p_out_dy Output forward displacement along y.
 * @param[out] p_out_dx Output forward displacement along x.
 * @note Variable use:
 *       - _base_index : Flattened tensor base index for the selected grid cell.
 */
static inline void disp_fwd_at(const int8_t *p_df, int _grid_y, int _grid_x, int _edge_id,
                               float *p_out_dy, float *p_out_dx)
{
    const int _base_index = _grid_y * DISP_STRIDE_H + _grid_x * DISP_STRIDE_W;
    *p_out_dy = dequant_disp_fwd(p_df[_base_index + _edge_id]);
    *p_out_dx = dequant_disp_fwd(p_df[_base_index + _edge_id + NUM_POSE_EDGES]);
}

/**
 * @brief Read the backward displacement (dy, dx) at grid (_grid_y, _grid_x) for edge.
 *
 * @param[in]  p_db     Backward displacement tensor pointer.
 * @param[in]  _grid_y  Grid y index.
 * @param[in]  _grid_x  Grid x index.
 * @param[in]  _edge_id Pose-chain edge/channel index.
 * @param[out] p_out_dy Output backward displacement along y.
 * @param[out] p_out_dx Output backward displacement along x.
 * @note Variable use:
 *       - _base_index : Flattened tensor base index for the selected grid cell.
 */
static inline void disp_bwd_at(const int8_t *p_db, int _grid_y, int _grid_x, int _edge_id,
                               float *p_out_dy, float *p_out_dx)
{
    const int _base_index = _grid_y * DISP_STRIDE_H + _grid_x * DISP_STRIDE_W;
    *p_out_dy = dequant_disp_bwd(p_db[_base_index + _edge_id]);
    *p_out_dx = dequant_disp_bwd(p_db[_base_index + _edge_id + NUM_POSE_EDGES]);
}

/**
 * @brief Round a model-input pixel coordinate to the nearest grid index and
 *        clamp to [0, _max_idx].
 *
 * @param[in] _pos     Coordinate in model-input pixel space.
 * @param[in] _max_idx Maximum valid grid index.
 * @return Clamped grid index.
 * @note Variable use:
 *       - _idx : Rounded/clamped index used for safe tensor lookup.
 */
static inline int coord_to_grid(float _pos, int _max_idx)
{
    int _idx = (int)lroundf(_pos / (float)MODEL_OUTPUT_STRIDE);
    if (0 > _idx)        { _idx = 0;        }
    if (_max_idx < _idx) { _idx = _max_idx; }
    return _idx;
}

/**
 * @brief Scan the heatmap for local-max peaks with score >= threshold.
 *        Populates ``p_out`` sorted in descending score order.
 *
 * @param[in]  p_hm  Quantized heatmap tensor.
 * @param[out] p_out Output candidate array (up to MAX_PEAK_CANDIDATES).
 * @return Number of candidates found (up to MAX_PEAK_CANDIDATES).
 * @note Variable use:
 *       - _count : Number of accepted peaks written into p_out.
 *       - _score : Candidate keypoint score from heatmap.
 *       - _is_max: Local 3x3 non-maximum suppression flag.
 */
static int build_peak_candidates(const int8_t *p_hm, peak_candidate_t *p_out)
{
    int _count = 0;
    int _keypoint_id;
    int _grid_y;
    int _grid_x;

    for (_keypoint_id = 0; _keypoint_id < MODEL_NUM_KEYPOINTS; _keypoint_id++)
    {
        for (_grid_y = 0; _grid_y < MODEL_OUTPUT_GH; _grid_y++)
        {
            for (_grid_x = 0; _grid_x < MODEL_OUTPUT_GW; _grid_x++)
            {
                float _score = hm_at(p_hm, _grid_y, _grid_x, _keypoint_id);
                if (POSE_ROOT_SCORE_THRESHOLD > _score)
                {
                    continue;
                }

                /* 3x3 local-max check (radius = LOCAL_MAX_RADIUS). */
                int _is_max = 1;
                int _neighbor_dy;
                int _neighbor_dx;
                for (_neighbor_dy = -LOCAL_MAX_RADIUS;
                     (_neighbor_dy <= LOCAL_MAX_RADIUS) && _is_max;
                     _neighbor_dy++)
                {
                    int _neighbor_y = _grid_y + _neighbor_dy;
                    if ((0 > _neighbor_y) || (MODEL_OUTPUT_GH <= _neighbor_y))
                    {
                        continue;
                    }
                    for (_neighbor_dx = -LOCAL_MAX_RADIUS;
                         _neighbor_dx <= LOCAL_MAX_RADIUS;
                         _neighbor_dx++)
                    {
                        int _neighbor_x = _grid_x + _neighbor_dx;
                        if ((0 > _neighbor_x) || (MODEL_OUTPUT_GW <= _neighbor_x))
                        {
                            continue;
                        }
                        if ((0 == _neighbor_dy) && (0 == _neighbor_dx))
                        {
                            continue;
                        }
                        if (hm_at(p_hm, _neighbor_y, _neighbor_x, _keypoint_id) > _score)
                        {
                            _is_max = 0;
                            break;
                        }
                    }
                }
                if (!_is_max)
                {
                    continue;
                }

                if (MAX_PEAK_CANDIDATES > (unsigned)_count)
                {
                    p_out[_count].gy     = (int16_t)_grid_y;
                    p_out[_count].gx     = (int16_t)_grid_x;
                    p_out[_count].kp     = (int16_t)_keypoint_id;
                    p_out[_count]._pad   = 0;
                    p_out[_count].score  = _score;
                    _count++;
                }
            }
        }
    }

    /* Insertion sort by descending score. Candidate count is typically
     * small (< 60) so O(n^2) is acceptable. */
    for (int _insert_idx = 1; _insert_idx < _count; _insert_idx++)
    {
        peak_candidate_t _current = p_out[_insert_idx];
        int _scan_idx = _insert_idx - 1;
        while ((0 <= _scan_idx) && (p_out[_scan_idx].score < _current.score))
        {
            p_out[_scan_idx + 1] = p_out[_scan_idx];
            _scan_idx--;
        }
        p_out[_scan_idx + 1] = _current;
    }

    return _count;
}

/**
 * @brief Follow one skeleton edge from a source keypoint to its target.
 *
 * @param[in]  p_hm             Heatmap tensor.
 * @param[in]  p_off            Offset tensor.
 * @param[in]  p_disp           Displacement tensor (fwd or bwd, per direction).
 * @param[in]  _edge_id         Index into POSE_CHAIN / displacement channels.
 * @param[in]  _target_kp       Keypoint index of the target part.
 * @param[in]  _src_y, _src_x   Source keypoint position in model-input pixels.
 * @param[in]  _use_bwd         Non-zero to dequantize with disp_bwd params.
 * @param[out] p_out_score      Target-part heatmap score at the landed grid cell.
 * @param[out] p_out_y, p_out_x Target position in model-input pixels.
 * @note Variable use:
 *       - _src_grid_y/_src_grid_x : Grid cell nearest to source keypoint.
 *       - _ddy/_ddx               : Displacement vector sampled on source grid.
 *       - _tgy/_tgx               : Target grid cell after displacement.
 *       - _ody/_odx               : Offset-based sub-grid refinement.
 */
static void traverse_edge(const int8_t *p_hm,
                          const int8_t *p_off,
                          const int8_t *p_disp,
                          int _edge_id,
                          int _target_kp,
                          float _src_y,
                          float _src_x,
                          int _use_bwd,
                          float *p_out_score,
                          float *p_out_y,
                          float *p_out_x)
{
    /* Snap the source position to the nearest grid cell. */
    int _src_grid_y = coord_to_grid(_src_y, MODEL_OUTPUT_GH - 1);
    int _src_grid_x = coord_to_grid(_src_x, MODEL_OUTPUT_GW - 1);

    /* Look up the displacement vector at that source cell. */
    float _ddy;
    float _ddx;
    if (_use_bwd)
    {
        disp_bwd_at(p_disp, _src_grid_y, _src_grid_x, _edge_id, &_ddy, &_ddx);
    }
    else
    {
        disp_fwd_at(p_disp, _src_grid_y, _src_grid_x, _edge_id, &_ddy, &_ddx);
    }

    /* Displaced position, then snap back to grid at target. */
    float _dy_pos = _src_y + _ddy;
    float _dx_pos = _src_x + _ddx;
    int _tgy = coord_to_grid(_dy_pos, MODEL_OUTPUT_GH - 1);
    int _tgx = coord_to_grid(_dx_pos, MODEL_OUTPUT_GW - 1);

    /* Read target score and refine with the offset head. */
    float _ody;
    float _odx;
    off_at(p_off, _tgy, _tgx, _target_kp, &_ody, &_odx);
    *p_out_score = hm_at(p_hm, _tgy, _tgx, _target_kp);
    *p_out_y     = (float)_tgy * (float)MODEL_OUTPUT_STRIDE + _ody;
    *p_out_x     = (float)_tgx * (float)MODEL_OUTPUT_STRIDE + _odx;
}

/**
 * @brief Fill 17 keypoints of one pose starting from a root, using both
 *        directions of the POSE_CHAIN. Coordinates are in model-input pixels.
 *
 * @param[in]  p_hm        Heatmap tensor.
 * @param[in]  p_off       Offset tensor.
 * @param[in]  p_df        Forward displacement tensor.
 * @param[in]  p_db        Backward displacement tensor.
 * @param[in]  _root_id    Root keypoint index.
 * @param[in]  _root_score Root keypoint score.
 * @param[in]  _root_y     Root y coordinate in model-input pixels.
 * @param[in]  _root_x     Root x coordinate in model-input pixels.
 * @param[out] p_kp        Decoded keypoint array for one pose.
 * @note Variable use:
 *       - _source/_target : Pose-chain source and target keypoint indices.
 *       - _edge_idx       : Pose-chain edge index used to select displacement channels.
 */
static void decode_pose(const int8_t *p_hm,
                        const int8_t *p_off,
                        const int8_t *p_df,
                        const int8_t *p_db,
                        int   _root_id,
                        float _root_score,
                        float _root_y,
                        float _root_x,
                        pose_keypoint_t *p_kp)
{
    int _keypoint_idx;
    for (_keypoint_idx = 0; _keypoint_idx < MODEL_NUM_KEYPOINTS; _keypoint_idx++)
    {
        p_kp[_keypoint_idx].part_id = (int32_t)_keypoint_idx;
        p_kp[_keypoint_idx].x       = 0.0f;
        p_kp[_keypoint_idx].y       = 0.0f;
        p_kp[_keypoint_idx].score   = 0.0f;
    }
    p_kp[_root_id].score = _root_score;
    p_kp[_root_id].y     = _root_y;
    p_kp[_root_id].x     = _root_x;

    /* Backward walk: iterate edges 15 -> 0 using disp_bwd.
     * For each edge, treat (parent, child) as (target, source). */
    int _edge_idx;
    for (_edge_idx = NUM_POSE_EDGES - 1; _edge_idx >= 0; _edge_idx--)
    {
        int _target = s_pose_chain[_edge_idx][0];  /* parent becomes target */
        int _source = s_pose_chain[_edge_idx][1];  /* child  becomes source */
        if ((0.0f < p_kp[_source].score) && (0.0f == p_kp[_target].score))
        {
            float _target_score;
            float _target_y;
            float _target_x;
            traverse_edge(p_hm, p_off, p_db, _edge_idx, _target,
                          p_kp[_source].y, p_kp[_source].x, 1,
                          &_target_score, &_target_y, &_target_x);
            p_kp[_target].score = _target_score;
            p_kp[_target].y     = _target_y;
            p_kp[_target].x     = _target_x;
        }
    }

    /* Forward walk: iterate edges 0 -> 15 using disp_fwd.
     * (parent, child) = (source, target). */
    for (_edge_idx = 0; _edge_idx < NUM_POSE_EDGES; _edge_idx++)
    {
        int _source = s_pose_chain[_edge_idx][0];
        int _target = s_pose_chain[_edge_idx][1];
        if ((0.0f < p_kp[_source].score) && (0.0f == p_kp[_target].score))
        {
            float _target_score;
            float _target_y;
            float _target_x;
            traverse_edge(p_hm, p_off, p_df, _edge_idx, _target,
                          p_kp[_source].y, p_kp[_source].x, 0,
                          &_target_score, &_target_y, &_target_x);
            p_kp[_target].score = _target_score;
            p_kp[_target].y     = _target_y;
            p_kp[_target].x     = _target_x;
        }
    }
}

/**
 * @brief Return non-zero if candidate root (_root_y, _root_x) is within
 *        ``_sq_radius`` of the same keypoint index in any accepted pose.
 *        Coordinates and radius are in model-input pixels.
 *
 * @param[in] p_accepted    Accepted pose-keypoint array.
 * @param[in] _num_accepted Number of accepted poses.
 * @param[in] _root_id      Root keypoint index to compare.
 * @param[in] _root_y       Candidate root y coordinate.
 * @param[in] _root_x       Candidate root x coordinate.
 * @param[in] _sq_radius    Squared NMS radius threshold.
 * @return 1 if overlapping with accepted pose root of same part; otherwise 0.
 * @note Variable use:
 *       - _dy/_dx : Root coordinate deltas used in squared-distance NMS test.
 */
static int within_nms(const pose_keypoint_t p_accepted[][MODEL_NUM_KEYPOINTS],
                      int   _num_accepted,
                      int   _root_id,
                      float _root_y,
                      float _root_x,
                      float _sq_radius)
{
    int _pose_idx;
    for (_pose_idx = 0; _pose_idx < _num_accepted; _pose_idx++)
    {
        float _dy = p_accepted[_pose_idx][_root_id].y - _root_y;
        float _dx = p_accepted[_pose_idx][_root_id].x - _root_x;
        if ((_dy * _dy + _dx * _dx) <= _sq_radius)
        {
            return 1;
        }
    }
    return 0;
}

/**
 * @brief Instance score: mean of keypoint scores whose position is not
 *        within ``_sq_radius`` of the corresponding keypoint in any
 *        previously accepted pose.
 *
 * @param[in] p_accepted    Accepted pose-keypoint array.
 * @param[in] _num_accepted Number of accepted poses.
 * @param[in] p_cand        Candidate pose keypoints to score.
 * @param[in] _sq_radius    Squared overlap radius threshold.
 * @return Instance score for the candidate pose.
 * @note Variable use:
 *       - _overlap : Per-keypoint overlap flag against accepted poses.
 *       - _sum     : Accumulator of non-overlapping keypoint scores.
 */
static float compute_instance_score(const pose_keypoint_t p_accepted[][MODEL_NUM_KEYPOINTS],
                                    int   _num_accepted,
                                    const pose_keypoint_t *p_cand,
                                    float _sq_radius)
{
    float _sum = 0.0f;
    int _keypoint_idx;
    for (_keypoint_idx = 0; _keypoint_idx < MODEL_NUM_KEYPOINTS; _keypoint_idx++)
    {
        int _overlap = 0;
        int _pose_idx;
        for (_pose_idx = 0; _pose_idx < _num_accepted; _pose_idx++)
        {
            float _dy = p_accepted[_pose_idx][_keypoint_idx].y - p_cand[_keypoint_idx].y;
            float _dx = p_accepted[_pose_idx][_keypoint_idx].x - p_cand[_keypoint_idx].x;
            if ((_dy * _dy + _dx * _dx) <= _sq_radius)
            {
                _overlap = 1;
                break;
            }
        }
        if (!_overlap)
        {
            _sum += p_cand[_keypoint_idx].score;
        }
    }
    return _sum / (float)MODEL_NUM_KEYPOINTS;
}

/* ------------------------------------------------------------------------- */
/* Public entry point                                                         */
/* ------------------------------------------------------------------------- */
/**
 * @brief Decode multi-person poses from quantized PoseNet output tensors.
 *
 * @param[in]  p_heatmap_q     Quantized heatmap tensor.
 * @param[in]  p_offset_q      Quantized offset tensor.
 * @param[in]  p_disp_fwd_q    Quantized forward displacement tensor.
 * @param[in]  p_disp_bwd_q    Quantized backward displacement tensor.
 * @param[in]  _source_width   Original source image width in pixels.
 * @param[in]  _source_height  Original source image height in pixels.
 * @param[out] p_keypoints_out Output poses (keypoints rescaled to source image).
 * @param[out] p_pose_scores   Output per-pose confidence scores.
 * @param[in]  _max_poses      Maximum number of poses to emit.
 * @return Number of accepted poses written to the outputs.
 * @note Variable use:
 *       - s_candidates : Temporary root-peak candidate buffer.
 *       - _sq_r        : Squared NMS radius used for overlap checks.
 *       - _num_cand    : Number of detected candidate roots.
 *       - _num_poses   : Number of accepted output poses.
 */
int postprocess(const int8_t   *p_heatmap_q,
                const int8_t   *p_offset_q,
                const int8_t   *p_disp_fwd_q,
                const int8_t   *p_disp_bwd_q,
                int              _source_width,
                int              _source_height,
                pose_keypoint_t (*p_keypoints_out)[MODEL_NUM_KEYPOINTS],
                float           *p_pose_scores,
                int              _max_poses)
{
    /* Candidate peaks are large enough for the whole heatmap so allocate
     * statically to keep the stack shallow. */
    static peak_candidate_t s_candidates[MAX_PEAK_CANDIDATES];

    if ((0 >= _max_poses) || (NULL == p_keypoints_out) ||
        (NULL == p_pose_scores))
    {
        return 0;
    }

    const int   _num_cand = build_peak_candidates(p_heatmap_q, s_candidates);
    const float _sq_r     = POSE_NMS_RADIUS_PX * POSE_NMS_RADIUS_PX;
    int _num_poses = 0;

    /* Decode candidates in score order. */
    int _candidate_idx;
    for (_candidate_idx = 0;
         (_candidate_idx < _num_cand) && (_num_poses < _max_poses);
         _candidate_idx++)
    {
        const peak_candidate_t *p_c = &s_candidates[_candidate_idx];

        /* Root position refined by the offset head. */
        float _ody;
        float _odx;
        off_at(p_offset_q, p_c->gy, p_c->gx, p_c->kp, &_ody, &_odx);
        const float _root_y = (float)p_c->gy * (float)MODEL_OUTPUT_STRIDE + _ody;
        const float _root_x = (float)p_c->gx * (float)MODEL_OUTPUT_STRIDE + _odx;

        /* NMS: skip if this root is too close to an already-accepted same-part
         * keypoint (in model-input pixels). */
        if (within_nms(p_keypoints_out, _num_poses,
                       p_c->kp, _root_y, _root_x, _sq_r))
        {
            continue;
        }

        /* Decode all 17 keypoints from this root (model-input pixels). */
        pose_keypoint_t _tmp_kp[MODEL_NUM_KEYPOINTS];
        decode_pose(p_heatmap_q, p_offset_q,
                    p_disp_fwd_q, p_disp_bwd_q,
                    p_c->kp, p_c->score, _root_y, _root_x,
                    _tmp_kp);

        /* Instance score (non-overlapping keypoint average). */
        float _pose_score = compute_instance_score(p_keypoints_out, _num_poses,
                                                   _tmp_kp, _sq_r);
        if (POSE_MIN_SCORE > _pose_score)
        {
            continue;
        }

        /* Accept: copy into output slot (still in model-input pixels). */
        int _keypoint_idx;
        for (_keypoint_idx = 0; _keypoint_idx < MODEL_NUM_KEYPOINTS; _keypoint_idx++)
        {
            p_keypoints_out[_num_poses][_keypoint_idx] = _tmp_kp[_keypoint_idx];
        }
        p_pose_scores[_num_poses] = _pose_score;
        _num_poses++;
    }

    /* Rescale accepted poses from model-input pixels -> source-image pixels. */
    const float _scale_x = (float)_source_width  / (float)MODEL_INPUT_W;
    const float _scale_y = (float)_source_height / (float)MODEL_INPUT_H;
    int _pose_idx;
    for (_pose_idx = 0; _pose_idx < _num_poses; _pose_idx++)
    {
        int _keypoint_idx;
        for (_keypoint_idx = 0; _keypoint_idx < MODEL_NUM_KEYPOINTS; _keypoint_idx++)
        {
            p_keypoints_out[_pose_idx][_keypoint_idx].x *= _scale_x;
            p_keypoints_out[_pose_idx][_keypoint_idx].y *= _scale_y;
        }
    }

    return _num_poses;
}
