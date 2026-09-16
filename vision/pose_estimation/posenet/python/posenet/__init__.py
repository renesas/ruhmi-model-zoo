# Copyright 2018 Ross Wightman  (rwightman/posenet-pytorch, Apache-2.0)
# Copyright 2026 Renesas Electronics Corporation
# SPDX-License-Identifier: Apache-2.0
"""PoseNet (MobileNetV1) PyTorch implementation for MCU pose estimation.

Adapted from rwightman/posenet-pytorch and michellelychan/posenet-pytorch.
"""
from posenet.constants import (
    PART_NAMES,
    NUM_KEYPOINTS,
    PART_IDS,
    CONNECTED_PART_NAMES,
    CONNECTED_PART_INDICES,
    LOCAL_MAXIMUM_RADIUS,
    POSE_CHAIN,
    PARENT_CHILD_TUPLES,
)
from posenet.models.mobilenet_v1 import MobileNetV1, MOBILENET_V1_CHECKPOINTS
from posenet.models.model_factory import load_model
from posenet.decode_multi import decode_multiple_poses
from posenet.utils import (
    valid_resolution,
    preprocess_image,
    read_imgfile,
    draw_skel_and_kp,
)
