# Copyright 2018 Ross Wightman  (rwightman/posenet-pytorch, Apache-2.0)
# Copyright 2026 Renesas Electronics Corporation
# SPDX-License-Identifier: Apache-2.0
"""TFJS -> PyTorch weight converter for PoseNet MobileNetV1."""
import json
import os
import struct
import tempfile

import numpy as np
import torch

from posenet.models.mobilenet_v1 import MobileNetV1, MOBILENET_V1_CHECKPOINTS

BASE_DIR = os.path.join(tempfile.gettempdir(), "_posenet_tfjs_weights")


def _to_torch_name(tf_name):
    tf_name = tf_name.lower()
    parts = tf_name.split("/")
    layer_split = parts[1].split("_")
    var_type = parts[2]

    if var_type in ("weights", "depthwise_weights"):
        postfix = ".weight"
    elif var_type == "biases":
        postfix = ".bias"
    else:
        postfix = ""

    if layer_split[0] == "conv2d":
        name = "features.conv" + layer_split[1]
        name += ("." + layer_split[2]) if len(layer_split) > 2 else ".conv"
        return name + postfix

    # Heads: heatmap_2 / offset_2 / displacement_(fwd|bwd)_2 — skip the _1 helpers
    if layer_split[0] in ("offset", "displacement", "heatmap") and layer_split[-1] == "2":
        return "_".join(layer_split[:-1]) + postfix
    return ""


def _load_variables(chkpoint, base_dir=BASE_DIR):
    manifest_path = os.path.join(base_dir, chkpoint, "manifest.json")
    if not os.path.exists(manifest_path):
        print(f"  TFJS weights for {chkpoint} not present, downloading to {base_dir}")
        from posenet.converter.wget import download
        download(chkpoint, base_dir)
        assert os.path.exists(manifest_path)

    with open(manifest_path) as f:
        variables = json.load(f)

    state_dict = {}
    for x in variables:
        torch_name = _to_torch_name(x)
        if not torch_name:
            continue
        filename = variables[x]["filename"]
        with open(os.path.join(base_dir, chkpoint, filename), "rb") as f:
            byte = f.read()
        fmt = f"{int(len(byte) / struct.calcsize('f'))}f"
        d = np.array(struct.unpack(fmt, byte), dtype=np.float32)
        shape = variables[x]["shape"]
        if len(shape) == 4:
            # depthwise: HWIO -> OIHW(=I*1HW logically), pointwise: HWIO -> OIHW
            tpt = (2, 3, 0, 1) if "depthwise" in filename else (3, 2, 0, 1)
            d = np.reshape(d, shape).transpose(tpt)
        state_dict[torch_name] = torch.from_numpy(d)
    return state_dict


def convert(model_id, model_dir, output_stride=16, check=False):
    """Materialise the PyTorch checkpoint for one TFJS PoseNet variant.

    Writes ``model_dir/<checkpoint>.pth`` containing the converted state-dict.
    """
    checkpoint_name = MOBILENET_V1_CHECKPOINTS[model_id]
    os.makedirs(model_dir, exist_ok=True)

    state_dict = _load_variables(checkpoint_name)
    m = MobileNetV1(model_id, output_stride=output_stride)
    m.load_state_dict(state_dict)
    out_path = os.path.join(model_dir, f"{checkpoint_name}.pth")
    torch.save(m.state_dict(), out_path)
    print(f"  Wrote PyTorch checkpoint -> {out_path}")
    return out_path
