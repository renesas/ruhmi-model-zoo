# Copyright 2018 Ross Wightman  (rwightman/posenet-pytorch, Apache-2.0)
# Copyright 2026 Renesas Electronics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Load (or auto-build) a PoseNet PyTorch checkpoint."""
import os
import torch

from posenet.models.mobilenet_v1 import MobileNetV1, MOBILENET_V1_CHECKPOINTS

DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "..", "..", "pretrained")


def load_model(model_id, output_stride=16, model_dir=DEFAULT_MODEL_DIR):
    """Return a PoseNet MobileNetV1 model with weights loaded.

    On first run, downloads the corresponding TFJS PoseNet checkpoint and
    converts it to a PyTorch ``.pth`` cached in ``model_dir``.
    """
    ckpt_name = MOBILENET_V1_CHECKPOINTS[model_id]
    model_path = os.path.join(model_dir, f"{ckpt_name}.pth")

    if not os.path.exists(model_path):
        os.makedirs(model_dir, exist_ok=True)
        print(f"  {ckpt_name}.pth not found, converting from TFJS weights ...")
        from posenet.converter.tfjs2pytorch import convert
        convert(model_id, model_dir, check=False)
        assert os.path.exists(model_path), f"conversion failed: {model_path}"

    model = MobileNetV1(model_id, output_stride=output_stride)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model
