# Copyright 2018 Ross Wightman  (rwightman/posenet-pytorch, Apache-2.0)
# Copyright 2026 Renesas Electronics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Image I/O, preprocessing and skeleton-drawing utilities for PoseNet."""
import cv2
import numpy as np

from posenet.constants import CONNECTED_PART_INDICES


def valid_resolution(width, height, output_stride=16):
    """Round (w, h) UP to the next (stride * N + 1) compatible size."""
    target_width  = (int(width)  // output_stride) * output_stride + 1
    target_height = (int(height) // output_stride) * output_stride + 1
    return target_width, target_height


def preprocess_image(source_bgr, target_size, *, layout="NHWC"):
    """Resize, BGR->RGB, scale to [-1, 1].

    Algorithm: 2-tap separable bilinear (half-pixel centres) +
    ``x * (2/255) - 1`` normalisation.

    Parameters
    ----------
    source_bgr : np.ndarray (H, W, 3) uint8
    target_size : (int, int)        (W, H)
    layout : 'NHWC' (TFLite-style)  or 'NCHW' (PyTorch-style)
    """
    target_w, target_h = target_size
    # Cast to float32 BEFORE cv2.resize so cv2 uses pure 2-tap float bilinear
    # (its uint8 fast-path uses INT11 fixed-point arithmetic, which would
    # produce slightly different rounded values).
    img = source_bgr.astype(np.float32)
    img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img * (2.0 / 255.0) - 1.0
    if layout == "NCHW":
        img = img.transpose((2, 0, 1))                       # CHW
        return img[None]                                     # NCHW
    return img[None]                                         # NHWC


def read_imgfile(path, target_size, *, layout="NHWC"):
    src = cv2.imread(path)
    if src is None:
        raise IOError(f"cannot read {path}")
    return preprocess_image(src, target_size, layout=layout), src


def _adjacent_keypoints(keypoint_scores, keypoint_coords, min_confidence=0.1):
    out = []
    for a, b in CONNECTED_PART_INDICES:
        if keypoint_scores[a] < min_confidence or keypoint_scores[b] < min_confidence:
            continue
        out.append(np.array([keypoint_coords[a][::-1],
                             keypoint_coords[b][::-1]]).astype(np.int32))
    return out


def draw_skel_and_kp(img, pose_scores, keypoint_scores, keypoint_coords,
                     min_pose_score=0.25, min_part_score=0.25,
                     color=(76, 230, 76), keypoint_radius=4):
    """Draw the detected poses on a copy of ``img``.

    Default skeleton colour is the same vibrant lime-green (BGR ``(76, 230, 76)``
    ≈ ``#4CE64C``) used by the MoveNet inference script, so both pose models
    produce visually consistent overlays.
    """
    out = img.copy()
    edges_to_draw = []
    cv_keypoints  = []

    for ii, score in enumerate(pose_scores):
        if score < min_pose_score:
            continue
        edges_to_draw.extend(_adjacent_keypoints(
            keypoint_scores[ii, :], keypoint_coords[ii, :, :], min_part_score))
        for ks, kc in zip(keypoint_scores[ii, :], keypoint_coords[ii, :, :]):
            if ks < min_part_score:
                continue
            cv_keypoints.append(cv2.KeyPoint(float(kc[1]), float(kc[0]),
                                             keypoint_radius * 2.5 * float(ks)))

    if cv_keypoints:
        out = cv2.drawKeypoints(
            out, cv_keypoints, outImage=np.array([]), color=color,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    if edges_to_draw:
        out = cv2.polylines(out, edges_to_draw, isClosed=False, color=color, thickness=2)
    return out
