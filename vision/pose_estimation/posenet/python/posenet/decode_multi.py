# Copyright 2018 Ross Wightman  (rwightman/posenet-pytorch, Apache-2.0)
# Copyright 2026 Renesas Electronics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Multi-pose decoder.

Algorithm (Papandreou et al., PersonLab):
  1. Pick local-max heatmap peaks above ``score_threshold`` as candidate roots.
  2. For each candidate root (highest score first):
       a. Skip if too close to an already accepted pose's root keypoint
          (radius NMS).
       b. Walk the POSE_CHAIN tree forwards and backwards using the
          displacement heads to recover all 17 keypoints.
       c. Compute an instance score (mean of non-overlapping keypoint scores)
          and keep if >= ``min_pose_score``.
  3. Return up to ``max_pose_detections`` poses.
"""
import numpy as np
import torch
import torch.nn.functional as F

from posenet.constants import LOCAL_MAXIMUM_RADIUS, NUM_KEYPOINTS, PARENT_CHILD_TUPLES


def _build_part_with_score(score_threshold, local_max_radius, scores):
    lmd = 2 * local_max_radius + 1
    max_vals = F.max_pool2d(scores, lmd, stride=1, padding=1)
    max_loc  = (scores == max_vals) & (scores >= score_threshold)
    max_loc_idx = max_loc.nonzero()
    scores_vec  = scores[max_loc]
    sort_idx    = torch.argsort(scores_vec, descending=True)
    return scores_vec[sort_idx], max_loc_idx[sort_idx]


def _within_nms_radius(pose_coords, squared_nms_radius, point):
    if not pose_coords.shape[0]:
        return False
    return np.any(np.sum((pose_coords - point) ** 2, axis=1) <= squared_nms_radius)


def _instance_score(exist_pose_coords, squared_nms_radius,
                    keypoint_scores, keypoint_coords):
    if exist_pose_coords.shape[0]:
        s = np.sum((exist_pose_coords - keypoint_coords) ** 2, axis=2) > squared_nms_radius
        ok = np.sum(keypoint_scores[np.all(s, axis=0)])
    else:
        ok = np.sum(keypoint_scores)
    return ok / len(keypoint_scores)


def _traverse(edge_id, source_keypoint, target_keypoint_id, scores, offsets,
              output_stride, displacements):
    height, width = scores.shape[1], scores.shape[2]
    src_idx = np.clip(
        np.round(source_keypoint / output_stride),
        a_min=0, a_max=[height - 1, width - 1]).astype(np.int32)

    displaced = source_keypoint + displacements[edge_id, src_idx[0], src_idx[1]]
    disp_idx  = np.clip(
        np.round(displaced / output_stride),
        a_min=0, a_max=[height - 1, width - 1]).astype(np.int32)

    score = scores[target_keypoint_id, disp_idx[0], disp_idx[1]]
    coord = disp_idx * output_stride + offsets[target_keypoint_id, disp_idx[0], disp_idx[1]]
    return score, coord


def _decode_pose(root_score, root_id, root_image_coord,
                 scores, offsets, output_stride,
                 displacements_fwd, displacements_bwd):
    num_parts = scores.shape[0]
    num_edges = len(PARENT_CHILD_TUPLES)

    kp_scores = np.zeros(num_parts)
    kp_coords = np.zeros((num_parts, 2))
    kp_scores[root_id] = root_score
    kp_coords[root_id] = root_image_coord

    # Walk backwards along the tree
    for edge in reversed(range(num_edges)):
        target_id, source_id = PARENT_CHILD_TUPLES[edge]
        if kp_scores[source_id] > 0.0 and kp_scores[target_id] == 0.0:
            s, c = _traverse(edge, kp_coords[source_id], target_id,
                             scores, offsets, output_stride, displacements_bwd)
            kp_scores[target_id] = s
            kp_coords[target_id] = c

    # And forwards
    for edge in range(num_edges):
        source_id, target_id = PARENT_CHILD_TUPLES[edge]
        if kp_scores[source_id] > 0.0 and kp_scores[target_id] == 0.0:
            s, c = _traverse(edge, kp_coords[source_id], target_id,
                             scores, offsets, output_stride, displacements_fwd)
            kp_scores[target_id] = s
            kp_coords[target_id] = c

    return kp_scores, kp_coords


def decode_multiple_poses(
        scores, offsets, displacements_fwd, displacements_bwd,
        output_stride,
        max_pose_detections=10,
        score_threshold=0.5,
        nms_radius=20,
        min_pose_score=0.5):
    """Decode up to ``max_pose_detections`` poses from the 4 PoseNet heads.

    Parameters
    ----------
    scores, offsets, displacements_fwd, displacements_bwd : torch.Tensor
        Per-image CHW tensors from the model (drop the batch dim before calling).
    output_stride : int
        16 for all standard PoseNet variants.

    Returns
    -------
    pose_scores      : (max_pose_detections,)              instance scores
    keypoint_scores  : (max_pose_detections, 17)           per-keypoint score
    keypoint_coords  : (max_pose_detections, 17, 2)        (y, x) in model-input pixels
    """
    part_scores, part_idx = _build_part_with_score(
        score_threshold, LOCAL_MAXIMUM_RADIUS, scores)
    part_scores = part_scores.cpu().numpy()
    part_idx    = part_idx.cpu().numpy()

    scores = scores.cpu().numpy()
    height, width = scores.shape[1], scores.shape[2]

    # (16, H, W, 2) layout for vector look-ups
    displacements_fwd = displacements_fwd.cpu().numpy().reshape(2, -1, height, width).transpose((1, 2, 3, 0))
    displacements_bwd = displacements_bwd.cpu().numpy().reshape(2, -1, height, width).transpose((1, 2, 3, 0))
    offsets_arr       = offsets.cpu().numpy().reshape(2, -1, height, width).transpose((1, 2, 3, 0))

    squared_nms_radius = nms_radius ** 2

    pose_scores         = np.zeros(max_pose_detections)
    pose_keypoint_scores = np.zeros((max_pose_detections, NUM_KEYPOINTS))
    pose_keypoint_coords = np.zeros((max_pose_detections, NUM_KEYPOINTS, 2))
    pose_count = 0

    for root_score, (root_id, y, x) in zip(part_scores, part_idx):
        root_coord = np.array([y, x])
        root_img_coord = root_coord * output_stride + offsets_arr[root_id, y, x]

        if _within_nms_radius(
                pose_keypoint_coords[:pose_count, root_id, :],
                squared_nms_radius, root_img_coord):
            continue

        kp_scores, kp_coords = _decode_pose(
            root_score, root_id, root_img_coord,
            scores, offsets_arr, output_stride,
            displacements_fwd, displacements_bwd)

        ps = _instance_score(
            pose_keypoint_coords[:pose_count, :, :], squared_nms_radius,
            kp_scores, kp_coords)

        if min_pose_score == 0.0 or ps >= min_pose_score:
            pose_scores[pose_count] = ps
            pose_keypoint_scores[pose_count, :]    = kp_scores
            pose_keypoint_coords[pose_count, :, :] = kp_coords
            pose_count += 1

        if pose_count >= max_pose_detections:
            break

    return pose_scores, pose_keypoint_scores, pose_keypoint_coords
