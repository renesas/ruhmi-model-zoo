#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Self-contained YOLOX-Tiny model builder.
# Extracted from the official YOLOX repository so that download_model.py
# can build the model architecture and load weights without installing
# the yolox package.
#
# Source: https://github.com/Megvii-BaseDetection/YOLOX
"""
Minimal, self-contained YOLOX-Tiny model definition.
=====================================================
Contains all building blocks needed to construct the YOLOX-Tiny network
and export it to ONNX — no external ``yolox`` package required.

Modules included (all from the YOLOX repo):
  - network blocks : SiLU, BaseConv, DWConv, Focus, SPPBottleneck,
                     Bottleneck, CSPLayer, ResLayer
  - backbone       : CSPDarknet
  - neck           : YOLOPAFPN
  - head           : YOLOXHead
  - top-level      : YOLOX
  - utility        : replace_module, build_yolox_tiny
"""

import math

import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════════════════════
# Network blocks
# ══════════════════════════════════════════════════════════════════════════════

class SiLU(nn.Module):
    """Export-friendly version of ``nn.SiLU()``."""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name="silu", inplace=True):
    if name == "silu":
        return nn.SiLU(inplace=inplace)
    elif name == "relu":
        return nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        return nn.LeakyReLU(0.1, inplace=inplace)
    raise AttributeError(f"Unsupported act type: {name}")


class BaseConv(nn.Module):
    """Conv2d → BatchNorm → Activation."""

    def __init__(self, in_channels, out_channels, ksize, stride,
                 groups=1, bias=False, act="silu"):
        super().__init__()
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize,
                              stride=stride, padding=pad, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DWConv(nn.Module):
    """Depthwise Conv + Pointwise Conv."""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(in_channels, in_channels, ksize=ksize,
                              stride=stride, groups=in_channels, act=act)
        self.pconv = BaseConv(in_channels, out_channels, ksize=1,
                              stride=1, groups=1, act=act)

    def forward(self, x):
        return self.pconv(self.dconv(x))


class Bottleneck(nn.Module):
    """Standard bottleneck block."""

    def __init__(self, in_channels, out_channels, shortcut=True,
                 expansion=0.5, depthwise=False, act="silu"):
        super().__init__()
        hidden = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden, 1, stride=1, act=act)
        self.conv2 = Conv(hidden, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return y + x if self.use_add else y


class ResLayer(nn.Module):
    """Residual layer (Darknet-style)."""

    def __init__(self, in_channels):
        super().__init__()
        mid = in_channels // 2
        self.layer1 = BaseConv(in_channels, mid, ksize=1, stride=1, act="lrelu")
        self.layer2 = BaseConv(mid, in_channels, ksize=3, stride=1, act="lrelu")

    def forward(self, x):
        return x + self.layer2(self.layer1(x))


class SPPBottleneck(nn.Module):
    """Spatial Pyramid Pooling (YOLOv3-SPP)."""

    def __init__(self, in_channels, out_channels,
                 kernel_sizes=(5, 9, 13), activation="silu"):
        super().__init__()
        hidden = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden, 1, stride=1, act=activation)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
             for ks in kernel_sizes]
        )
        self.conv2 = BaseConv(hidden * (len(kernel_sizes) + 1), out_channels,
                              1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(torch.cat([x] + [m(x) for m in self.m], dim=1))


class CSPLayer(nn.Module):
    """C3 / CSP Bottleneck with 3 convolutions."""

    def __init__(self, in_channels, out_channels, n=1, shortcut=True,
                 expansion=0.5, depthwise=False, act="silu"):
        super().__init__()
        hidden = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden, out_channels, 1, stride=1, act=act)
        self.m = nn.Sequential(
            *[Bottleneck(hidden, hidden, shortcut, 1.0, depthwise, act=act)
              for _ in range(n)]
        )

    def forward(self, x):
        x_1 = self.m(self.conv1(x))
        x_2 = self.conv2(x)
        return self.conv3(torch.cat((x_1, x_2), dim=1))


class Focus(nn.Module):
    """Focus width/height into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        return self.conv(torch.cat([
            x[..., ::2, ::2],
            x[..., 1::2, ::2],
            x[..., ::2, 1::2],
            x[..., 1::2, 1::2],
        ], dim=1))


# ══════════════════════════════════════════════════════════════════════════════
# Backbone — CSPDarknet
# ══════════════════════════════════════════════════════════════════════════════

class CSPDarknet(nn.Module):
    def __init__(self, dep_mul, wid_mul,
                 out_features=("dark3", "dark4", "dark5"),
                 depthwise=False, act="silu"):
        super().__init__()
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv

        base_channels = int(wid_mul * 64)
        base_depth = max(round(dep_mul * 3), 1)

        self.stem = Focus(3, base_channels, ksize=3, act=act)

        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(base_channels * 2, base_channels * 2,
                     n=base_depth, depthwise=depthwise, act=act),
        )
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(base_channels * 4, base_channels * 4,
                     n=base_depth * 3, depthwise=depthwise, act=act),
        )
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(base_channels * 8, base_channels * 8,
                     n=base_depth * 3, depthwise=depthwise, act=act),
        )
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(base_channels * 16, base_channels * 16,
                     n=base_depth, shortcut=False, depthwise=depthwise, act=act),
        )

    def forward(self, x):
        outputs = {}
        x = self.stem(x);    outputs["stem"] = x
        x = self.dark2(x);   outputs["dark2"] = x
        x = self.dark3(x);   outputs["dark3"] = x
        x = self.dark4(x);   outputs["dark4"] = x
        x = self.dark5(x);   outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}


# ══════════════════════════════════════════════════════════════════════════════
# Neck — YOLOPAFPN
# ══════════════════════════════════════════════════════════════════════════════

class YOLOPAFPN(nn.Module):
    def __init__(self, depth=1.0, width=1.0,
                 in_features=("dark3", "dark4", "dark5"),
                 in_channels=[256, 512, 1024],
                 depthwise=False, act="silu"):
        super().__init__()
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # top-down
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width), int(in_channels[1] * width),
            round(3 * depth), False, depthwise=depthwise, act=act)

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width), int(in_channels[0] * width),
            round(3 * depth), False, depthwise=depthwise, act=act)

        # bottom-up
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act)
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width), int(in_channels[1] * width),
            round(3 * depth), False, depthwise=depthwise, act=act)

        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act)
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width), int(in_channels[2] * width),
            round(3 * depth), False, depthwise=depthwise, act=act)

    def forward(self, input):
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = features

        fpn_out0 = self.lateral_conv0(x0)
        f_out0 = torch.cat([self.upsample(fpn_out0), x1], 1)
        f_out0 = self.C3_p4(f_out0)

        fpn_out1 = self.reduce_conv1(f_out0)
        f_out1 = torch.cat([self.upsample(fpn_out1), x2], 1)
        pan_out2 = self.C3_p3(f_out1)

        p_out1 = torch.cat([self.bu_conv2(pan_out2), fpn_out1], 1)
        pan_out1 = self.C3_n3(p_out1)

        p_out0 = torch.cat([self.bu_conv1(pan_out1), fpn_out0], 1)
        pan_out0 = self.C3_n4(p_out0)

        return (pan_out2, pan_out1, pan_out0)


# ══════════════════════════════════════════════════════════════════════════════
# Head — YOLOXHead  (inference-only subset, no loss computation)
# ══════════════════════════════════════════════════════════════════════════════

class YOLOXHead(nn.Module):
    def __init__(self, num_classes, width=1.0, strides=[8, 16, 32],
                 in_channels=[256, 512, 1024], act="silu", depthwise=False):
        super().__init__()
        self.num_classes = num_classes
        self.decode_in_inference = True  # set False for raw-output export

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        Conv = DWConv if depthwise else BaseConv

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(int(in_channels[i] * width), int(256 * width),
                         ksize=1, stride=1, act=act))
            self.cls_convs.append(nn.Sequential(
                Conv(int(256 * width), int(256 * width), 3, 1, act=act),
                Conv(int(256 * width), int(256 * width), 3, 1, act=act),
            ))
            self.reg_convs.append(nn.Sequential(
                Conv(int(256 * width), int(256 * width), 3, 1, act=act),
                Conv(int(256 * width), int(256 * width), 3, 1, act=act),
            ))
            self.cls_preds.append(nn.Conv2d(int(256 * width), num_classes, 1, 1, 0))
            self.reg_preds.append(nn.Conv2d(int(256 * width), 4, 1, 1, 0))
            self.obj_preds.append(nn.Conv2d(int(256 * width), 1, 1, 1, 0))

        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(1, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = nn.Parameter(b.view(-1), requires_grad=True)
        for conv in self.obj_preds:
            b = conv.bias.view(1, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, xin, labels=None, imgs=None):
        outputs = []
        for k, (cls_conv, reg_conv, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, xin)
        ):
            x = self.stems[k](x)
            cls_feat = cls_conv(x)
            cls_output = self.cls_preds[k](cls_feat)
            reg_feat = reg_conv(x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            # inference path: sigmoid on obj and cls
            output = torch.cat(
                [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
            )
            outputs.append(output)

        self.hw = [x.shape[-2:] for x in outputs]
        # [batch, n_anchors_all, 85]
        outputs = torch.cat(
            [x.flatten(start_dim=2) for x in outputs], dim=2
        ).permute(0, 2, 1)

        if self.decode_in_inference:
            return self.decode_outputs(outputs, dtype=xin[0].type())
        return outputs

    def decode_outputs(self, outputs, dtype):
        grids, strides = [], []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = torch.meshgrid(torch.arange(hsize), torch.arange(wsize),
                                    indexing="ij")
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            strides.append(torch.full((*grid.shape[:2], 1), stride))
        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)
        outputs = torch.cat([
            (outputs[..., 0:2] + grids) * strides,
            torch.exp(outputs[..., 2:4]) * strides,
            outputs[..., 4:],
        ], dim=-1)
        return outputs


# ══════════════════════════════════════════════════════════════════════════════
# Top-level model — YOLOX
# ══════════════════════════════════════════════════════════════════════════════

class YOLOX(nn.Module):
    def __init__(self, backbone=None, head=None):
        super().__init__()
        self.backbone = backbone if backbone is not None else YOLOPAFPN()
        self.head = head if head is not None else YOLOXHead(80)

    def forward(self, x, targets=None):
        fpn_outs = self.backbone(x)
        return self.head(fpn_outs)


# ══════════════════════════════════════════════════════════════════════════════
# Utility — replace_module  (for nn.SiLU → export-friendly SiLU)
# ══════════════════════════════════════════════════════════════════════════════

def replace_module(module, replaced_module_type, new_module_type,
                   replace_func=None):
    """Recursively replace *replaced_module_type* with *new_module_type*."""
    if replace_func is None:
        replace_func = lambda old_t, new_t: new_t()

    if isinstance(module, replaced_module_type):
        return replace_func(replaced_module_type, new_module_type)

    for name, child in module.named_children():
        new_child = replace_module(child, replaced_module_type, new_module_type,
                                   replace_func)
        if new_child is not child:
            module.add_module(name, new_child)
    return module


# ══════════════════════════════════════════════════════════════════════════════
# Builder — build_yolox_tiny
# ══════════════════════════════════════════════════════════════════════════════

def build_yolox_tiny(num_classes=80):
    """
    Build the YOLOX-Tiny model with the official config.

    YOLOX-Tiny config (from exps/default/yolox_tiny.py):
      depth = 0.33, width = 0.375, depthwise = False, act = 'silu'

    Returns
    -------
    model : YOLOX
        Ready for weight loading and export.
    """
    depth = 0.33
    width = 0.375
    in_channels = [256, 512, 1024]
    act = "silu"

    backbone = YOLOPAFPN(depth, width, in_channels=in_channels,
                         act=act, depthwise=False)
    head = YOLOXHead(num_classes, width, in_channels=in_channels,
                     act=act, depthwise=False)
    model = YOLOX(backbone, head)

    # Init BN eps/momentum (matches official init_yolo)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eps = 1e-3
            m.momentum = 0.03

    model.head.initialize_biases(1e-2)
    return model
