# Copyright 2018 Ross Wightman  (rwightman/posenet-pytorch, Apache-2.0)
# Copyright 2026 Renesas Electronics Corporation
# SPDX-License-Identifier: Apache-2.0
"""PoseNet MobileNetV1 backbone with 4 detection heads.

Heads:
  heatmap          : 17 channels (per-keypoint sigmoid confidence)
  offset           : 34 channels (dy, dx) per keypoint, refines grid coord
  displacement_fwd : 32 channels (16 POSE_CHAIN edges x 2)  parent -> child
  displacement_bwd : 32 channels                            child  -> parent
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_output_strided_layers(convolution_def, output_stride):
    """Inject dilation/stride so the network ends at exactly ``output_stride``."""
    current_stride = 1
    rate = 1
    block_id = 0
    buff = []
    for c in convolution_def:
        conv_type, inp, outp, stride = c

        if current_stride == output_stride:
            layer_stride = 1
            layer_rate = rate
            rate *= stride
        else:
            layer_stride = stride
            layer_rate = 1
            current_stride *= stride

        buff.append(dict(
            block_id=block_id, conv_type=conv_type,
            inp=inp, outp=outp,
            stride=layer_stride, rate=layer_rate,
            output_stride=current_stride))
        block_id += 1
    return buff


def _get_padding(kernel_size, stride, dilation):
    return ((stride - 1) + dilation * (kernel_size - 1)) // 2


class InputConv(nn.Module):
    def __init__(self, inp, outp, k=3, stride=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(inp, outp, k, stride,
                              padding=_get_padding(k, stride, dilation),
                              dilation=dilation)

    def forward(self, x):
        return F.relu6(self.conv(x))


class SeperableConv(nn.Module):
    def __init__(self, inp, outp, k=3, stride=1, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv2d(inp, inp, k, stride,
                                   padding=_get_padding(k, stride, dilation),
                                   dilation=dilation, groups=inp)
        self.pointwise = nn.Conv2d(inp, outp, 1, 1)

    def forward(self, x):
        x = F.relu6(self.depthwise(x))
        x = F.relu6(self.pointwise(x))
        return x


# Depth-multiplier checkpoints supported by TFJS PoseNet.
MOBILENET_V1_CHECKPOINTS = {
    50:  "mobilenet_v1_050",
    75:  "mobilenet_v1_075",
    100: "mobilenet_v1_100",
    101: "mobilenet_v1_101",
}

MOBILE_NET_V1_100 = [
    (InputConv,    3,    32, 2),
    (SeperableConv, 32,   64, 1),
    (SeperableConv, 64,  128, 2),
    (SeperableConv, 128, 128, 1),
    (SeperableConv, 128, 256, 2),
    (SeperableConv, 256, 256, 1),
    (SeperableConv, 256, 512, 2),
    (SeperableConv, 512, 512, 1),
    (SeperableConv, 512, 512, 1),
    (SeperableConv, 512, 512, 1),
    (SeperableConv, 512, 512, 1),
    (SeperableConv, 512, 512, 1),
    (SeperableConv, 512, 1024, 2),
    (SeperableConv, 1024, 1024, 1),
]

MOBILE_NET_V1_75 = [
    (InputConv,     3,   24, 2),
    (SeperableConv, 24,  48, 1),
    (SeperableConv, 48,  96, 2),
    (SeperableConv, 96,  96, 1),
    (SeperableConv, 96,  192, 2),
    (SeperableConv, 192, 192, 1),
    (SeperableConv, 192, 384, 2),
    (SeperableConv, 384, 384, 1),
    (SeperableConv, 384, 384, 1),
    (SeperableConv, 384, 384, 1),
    (SeperableConv, 384, 384, 1),
    (SeperableConv, 384, 384, 1),
    (SeperableConv, 384, 384, 1),
    (SeperableConv, 384, 384, 1),
]

MOBILE_NET_V1_50 = [
    (InputConv,     3,   16, 2),
    (SeperableConv, 16,  32, 1),
    (SeperableConv, 32,  64, 2),
    (SeperableConv, 64,  64, 1),
    (SeperableConv, 64,  128, 2),
    (SeperableConv, 128, 128, 1),
    (SeperableConv, 128, 256, 2),
    (SeperableConv, 256, 256, 1),
    (SeperableConv, 256, 256, 1),
    (SeperableConv, 256, 256, 1),
    (SeperableConv, 256, 256, 1),
    (SeperableConv, 256, 256, 1),
    (SeperableConv, 256, 256, 1),
    (SeperableConv, 256, 256, 1),
]


class MobileNetV1(nn.Module):
    def __init__(self, model_id, output_stride=16):
        super().__init__()
        assert model_id in MOBILENET_V1_CHECKPOINTS, f"Unknown model_id {model_id}"
        self.output_stride = output_stride

        if model_id == 50:
            arch = MOBILE_NET_V1_50
        elif model_id == 75:
            arch = MOBILE_NET_V1_75
        else:
            arch = MOBILE_NET_V1_100   # 100 & 101 share the same architecture

        conv_def = _to_output_strided_layers(arch, output_stride)
        conv_list = [
            (f"conv{c['block_id']}",
             c["conv_type"](c["inp"], c["outp"], 3,
                            stride=c["stride"], dilation=c["rate"]))
            for c in conv_def
        ]
        last_depth = conv_def[-1]["outp"]

        self.features = nn.Sequential(OrderedDict(conv_list))
        self.heatmap          = nn.Conv2d(last_depth, 17, 1, 1)
        self.offset           = nn.Conv2d(last_depth, 34, 1, 1)
        self.displacement_fwd = nn.Conv2d(last_depth, 32, 1, 1)
        self.displacement_bwd = nn.Conv2d(last_depth, 32, 1, 1)

    def forward(self, x):
        x = self.features(x)
        heatmap = torch.sigmoid(self.heatmap(x))     # sigmoid baked into the graph
        return heatmap, self.offset(x), self.displacement_fwd(x), self.displacement_bwd(x)
