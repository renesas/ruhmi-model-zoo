# Copyright 2018 Ross Wightman  (rwightman/posenet-pytorch, Apache-2.0)
# Copyright 2026 Renesas Electronics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Constants for the 17-keypoint COCO PoseNet topology."""

PART_NAMES = [
    "nose", "leftEye", "rightEye", "leftEar", "rightEar",
    "leftShoulder", "rightShoulder", "leftElbow", "rightElbow",
    "leftWrist", "rightWrist", "leftHip", "rightHip",
    "leftKnee", "rightKnee", "leftAnkle", "rightAnkle",
]
NUM_KEYPOINTS = len(PART_NAMES)
PART_IDS = {pn: pid for pid, pn in enumerate(PART_NAMES)}

# Edges drawn as part of the skeleton overlay (not used by the decoder).
CONNECTED_PART_NAMES = [
    ("leftHip", "leftShoulder"), ("leftElbow", "leftShoulder"),
    ("leftElbow", "leftWrist"),  ("leftHip", "leftKnee"),
    ("leftKnee", "leftAnkle"),   ("rightHip", "rightShoulder"),
    ("rightElbow", "rightShoulder"), ("rightElbow", "rightWrist"),
    ("rightHip", "rightKnee"),   ("rightKnee", "rightAnkle"),
    ("leftShoulder", "rightShoulder"), ("leftHip", "rightHip"),
]
CONNECTED_PART_INDICES = [(PART_IDS[a], PART_IDS[b]) for a, b in CONNECTED_PART_NAMES]

# Local-maximum radius for the heatmap NMS peak finder.
LOCAL_MAXIMUM_RADIUS = 1

# Tree used by the multi-pose decoder to walk from a root keypoint to all
# other 16 via the displacement_fwd / displacement_bwd heads.
POSE_CHAIN = [
    ("nose", "leftEye"),       ("leftEye", "leftEar"),       ("nose", "rightEye"),
    ("rightEye", "rightEar"),  ("nose", "leftShoulder"),
    ("leftShoulder", "leftElbow"),   ("leftElbow", "leftWrist"),
    ("leftShoulder", "leftHip"),     ("leftHip", "leftKnee"),
    ("leftKnee", "leftAnkle"),       ("nose", "rightShoulder"),
    ("rightShoulder", "rightElbow"), ("rightElbow", "rightWrist"),
    ("rightShoulder", "rightHip"),   ("rightHip", "rightKnee"),
    ("rightKnee", "rightAnkle"),
]
PARENT_CHILD_TUPLES = [(PART_IDS[parent], PART_IDS[child]) for parent, child in POSE_CHAIN]
