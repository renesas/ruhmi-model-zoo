#!/usr/bin/env python3
# Copyright 2026 Renesas Electronics Corporation
# SPDX-License-Identifier: Apache-2.0
"""
convert_to_onnx.py
==================
Convert a Yolo-Fastest Darknet model (.cfg + .weights) to ONNX.

This script is **self-contained** — it implements its own Darknet parser and
PyTorch model builder.  No third-party darknet2onnx package is required.

Architecture handled
--------------------
All layer types present in yolo-fastest-1.1.cfg are supported:
  - convolutional  (1×1 and k×k, regular and depthwise groups, BN + leaky / linear)
  - shortcut       (residual add with optional activation)
  - dropout        (identity at inference, weight loading skipped)
  - maxpool        (SPP multi-scale pooling)
  - route          (concatenation / passthrough)
  - upsample       (nearest ×2 upsampling)
  - yolo           (anchor / detection head — outputs raw tensor, no decode)

Exported ONNX model
-------------------
  Input  : "images"   float32  [1, 3, 320, 320]  (NCHW, normalised [0, 1])
  Output : two raw YOLO head tensors (pre-sigmoid, pre-NMS):
      "output0"  float32  [1, 255, 10, 10]  (large objects, stride 32)
      "output1"  float32  [1, 255, 20, 20]  (small objects, stride 16)

Usage
-----
    # Convert base model with default paths (downloads must have run first)
    python convert_to_onnx.py

    # Convert XL variant
    python convert_to_onnx.py --variant xl

    # Specify fully custom paths
    python convert_to_onnx.py \\
        --cfg     model/yolo-fastest-1.1.cfg \\
        --weights model/yolo-fastest-1.1.weights \\
        --output  model/yolo_fastest_1.1.onnx

    # Verify exported ONNX immediately after conversion
    python convert_to_onnx.py --verify
    python convert_to_onnx.py --variant xl --verify

After conversion run inference.py with the generated ONNX file:
    python inference.py --image sample.jpg --model model/yolo_fastest_1.1.onnx
    python inference.py --image sample.jpg --model model/yolo_fastest_1.1_xl.onnx
"""

import argparse
import os
import struct
import sys

import numpy as np

# ── Defaults ──────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR = os.path.join(_HERE, "model")

_VARIANT_DEFAULTS = {
    "base": {
        "cfg":     "yolo-fastest-1.1.cfg",
        "weights": "yolo-fastest-1.1.weights",
        "output":  "yolo_fastest_1.1.onnx",
    },
    "xl": {
        "cfg":     "yolo-fastest-1.1-xl.cfg",
        "weights": "yolo-fastest-1.1-xl.weights",
        "output":  "yolo_fastest_1.1_xl.onnx",
    },
}

DEFAULT_CFG     = os.path.join(_MODEL_DIR, "yolo-fastest-1.1.cfg")
DEFAULT_WEIGHTS = os.path.join(_MODEL_DIR, "yolo-fastest-1.1.weights")
DEFAULT_OUTPUT  = os.path.join(_MODEL_DIR, "yolo_fastest_1.1.onnx")

INPUT_W = 320
INPUT_H = 320


# ══════════════════════════════════════════════════════════════════════════════
# 1. CFG PARSER
# ══════════════════════════════════════════════════════════════════════════════

def parse_cfg(cfg_path: str) -> list:
    """
    Parse a Darknet .cfg file into an ordered list of block dicts.

    Parameters
    ----------
    cfg_path : str
        Path to the Darknet configuration file.

    Returns
    -------
    list of dict
        Each dict has at minimum a ``"type"`` key with the block name and the
        remaining keys from the ``key=value`` lines in that block.
    """
    blocks = []
    block  = {}
    with open(cfg_path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("["):
                if block:
                    blocks.append(block)
                block = {"type": line[1:-1].strip()}
            else:
                key, _, val = line.partition("=")
                block[key.strip()] = val.strip()
    if block:
        blocks.append(block)
    return blocks


# ══════════════════════════════════════════════════════════════════════════════
# 2. PYTORCH LAYER MODULES
# ══════════════════════════════════════════════════════════════════════════════

def _build_modules(blocks: list, net_info: dict):
    """
    Build a ``torch.nn.ModuleList`` from parsed cfg blocks.

    Parameters
    ----------
    blocks : list of dict
        Layer blocks returned by :func:`parse_cfg` (excluding the [net] block).
    net_info : dict
        The [net] block with global training hyper-parameters.

    Returns
    -------
    torch.nn.ModuleList
        One module per non-[net] block.
    list of int
        Cumulative output channel counts (index 0 = network input channels).
    """
    import torch.nn as nn

    module_list  = nn.ModuleList()
    out_channels = [int(net_info.get("channels", 3))]  # [0] = network input

    for idx, block in enumerate(blocks):
        btype = block["type"]
        mod   = nn.Sequential()

        if btype == "convolutional":
            in_ch   = out_channels[-1]
            filters = int(block["filters"])
            ksize   = int(block["size"])
            stride  = int(block["stride"])
            pad     = int(block.get("pad", 0))
            groups  = int(block.get("groups", 1))
            has_bn  = int(block.get("batch_normalize", 0)) == 1
            act     = block.get("activation", "linear")

            # Darknet "pad=1" → symmetric padding to keep spatial size
            padding = (ksize - 1) // 2 if pad else 0

            conv = nn.Conv2d(in_ch, filters, ksize, stride=stride,
                             padding=padding, groups=groups, bias=not has_bn)
            mod.add_module("conv", conv)

            if has_bn:
                bn = nn.BatchNorm2d(filters, eps=1e-4, momentum=0.03)
                mod.add_module("bn", bn)

            if act == "leaky":
                mod.add_module("act", nn.LeakyReLU(0.1, inplace=True))
            # "linear" → no activation

            out_channels.append(filters)

        elif btype == "shortcut":
            m = nn.Identity()
            m.block_type = "shortcut"
            m.from_idx   = int(block["from"])   # relative index (always negative)
            mod = m
            out_channels.append(out_channels[-1])

        elif btype == "dropout":
            m = nn.Identity()
            m.block_type  = "dropout"
            m.probability = float(block.get("probability", 0.0))
            mod = m
            out_channels.append(out_channels[-1])

        elif btype == "maxpool":
            size   = int(block["size"])
            stride = int(block.get("stride", 1))
            pad    = (size - 1) // 2
            mod.add_module("maxpool",
                           nn.MaxPool2d(size, stride=stride, padding=pad))
            out_channels.append(out_channels[-1])

        elif btype == "route":
            m = nn.Identity()
            m.block_type   = "route"
            m.route_layers = [int(x.strip()) for x in block["layers"].split(",")]
            # Compute output channels by summing referenced layer outputs
            ch = 0
            for ri in m.route_layers:
                ref = ri if ri >= 0 else idx + ri   # relative → absolute
                ch += out_channels[ref + 1]          # +1: out_channels[0] = net input
            mod = m
            out_channels.append(ch)

        elif btype == "upsample":
            scale = int(block.get("stride", 2))
            mod.add_module("upsample",
                           nn.Upsample(scale_factor=scale, mode="nearest"))
            out_channels.append(out_channels[-1])

        elif btype == "yolo":
            m = nn.Identity()
            m.block_type = "yolo"
            mod = m
            out_channels.append(out_channels[-1])

        else:
            raise ValueError(f"Unsupported layer type at index {idx}: {btype!r}")

        module_list.append(mod)

    return module_list, out_channels


# ══════════════════════════════════════════════════════════════════════════════
# 3. DARKNET MODEL  (nn.Module)
# ══════════════════════════════════════════════════════════════════════════════

import torch.nn as nn   # noqa: E402 (needed for DarknetModel base class)


class DarknetModel(nn.Module):
    """
    Pure-PyTorch reconstruction of a Darknet network from a .cfg file.

    Parameters
    ----------
    cfg_path : str
        Path to the Darknet .cfg file.
    """

    def __init__(self, cfg_path: str):
        super().__init__()
        blocks = parse_cfg(cfg_path)

        assert blocks[0]["type"] == "net", "First block must be [net]"
        self.net_info     = blocks[0]
        self.layer_blocks = blocks[1:]

        self.module_list, self.out_channels = _build_modules(
            self.layer_blocks, self.net_info
        )

    def forward(self, x):
        import torch
        layer_outputs = []   # layer_outputs[i] = output tensor of layer i
        yolo_outputs  = []

        for idx, (block, module) in enumerate(
                zip(self.layer_blocks, self.module_list)):
            btype = block["type"]

            if btype in ("convolutional", "maxpool", "upsample"):
                x = module(x)

            elif btype == "shortcut":
                from_abs = idx + module.from_idx   # from_idx is always negative
                x = x + layer_outputs[from_abs]
                # activation is always "linear" for shortcut in this cfg

            elif btype == "dropout":
                pass   # identity at inference

            elif btype == "route":
                parts = []
                for ri in module.route_layers:
                    ref = ri if ri >= 0 else idx + ri
                    parts.append(layer_outputs[ref])
                x = torch.cat(parts, dim=1)

            elif btype == "yolo":
                yolo_outputs.append(x)
                # x unchanged so layer_outputs[idx] carries the raw head tensor

            else:
                raise ValueError(f"Unknown layer type {btype!r}")

            layer_outputs.append(x)

        return yolo_outputs   # list of 2 tensors


# ══════════════════════════════════════════════════════════════════════════════
# 4. WEIGHT LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_darknet_weights(model: DarknetModel, weights_path: str) -> None:
    """
    Load Darknet binary weights into a :class:`DarknetModel`.

    Binary format::

        [header: major(i32) minor(i32) revision(i32) seen(i64)]
        [weights: float32 ...]

    For each convolutional layer the weight order is:
      - With batch-norm : bn.bias, bn.weight, bn.running_mean, bn.running_var
      - Without         : conv.bias
      - Always          : conv.weight  (row-major, shape [out, in/g, kh, kw])

    Parameters
    ----------
    model : DarknetModel
        The model whose parameters will be filled in-place.
    weights_path : str
        Path to the .weights binary file.
    """
    import torch

    with open(weights_path, "rb") as fh:
        major, minor, revision = struct.unpack("3i", fh.read(12))
        if (major * 10 + minor) >= 2 and major < 1000 and minor < 1000:
            fh.read(8)   # seen: int64
        else:
            fh.read(4)   # seen: int32
        raw = np.frombuffer(fh.read(), dtype=np.float32).copy()

    ptr = 0

    for idx, block in enumerate(model.layer_blocks):
        if block["type"] != "convolutional":
            continue

        module = model.module_list[idx]
        conv   = module.conv
        has_bn = "bn" in module._modules
        n_filt = conv.weight.data.shape[0]

        if has_bn:
            bn = module.bn
            bn.bias.data.copy_(
                torch.from_numpy(raw[ptr:ptr + n_filt]));          ptr += n_filt
            bn.weight.data.copy_(
                torch.from_numpy(raw[ptr:ptr + n_filt]));          ptr += n_filt
            bn.running_mean.copy_(
                torch.from_numpy(raw[ptr:ptr + n_filt]));          ptr += n_filt
            bn.running_var.copy_(
                torch.from_numpy(raw[ptr:ptr + n_filt]));          ptr += n_filt
        else:
            n_bias = conv.bias.data.numel()
            conv.bias.data.copy_(
                torch.from_numpy(raw[ptr:ptr + n_bias]));          ptr += n_bias

        n_w = conv.weight.data.numel()
        conv.weight.data.copy_(
            torch.from_numpy(raw[ptr:ptr + n_w]).view_as(conv.weight.data))
        ptr += n_w

    total = raw.size
    if ptr != total:
        print(f"  [WARN] Weight pointer mismatch: consumed {ptr}, "
              f"total {total} (diff={total - ptr})")
    else:
        print(f"  ✅ Weights loaded: {ptr:,} float32 values")


# ══════════════════════════════════════════════════════════════════════════════
# 5. ONNX EXPORT WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

class _YoloExportWrapper(nn.Module):
    """
    Wraps :class:`DarknetModel` so ``torch.onnx.export`` sees a tuple return.

    ``torch.onnx.export`` requires ``forward()`` to return a tensor or a tuple
    of tensors — not a Python list.  This wrapper converts the list of YOLO
    outputs to a tuple so the ONNX graph has properly named outputs.

    Parameters
    ----------
    model : DarknetModel
        The loaded Darknet model.
    """

    def __init__(self, model: DarknetModel):
        super().__init__()
        self.model = model

    def forward(self, x):
        return tuple(self.model(x))


# ══════════════════════════════════════════════════════════════════════════════
# 6. PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def convert(cfg_path: str, weights_path: str, output_path: str) -> None:
    """
    Convert a Darknet model (.cfg + .weights) to ONNX.

    Parameters
    ----------
    cfg_path : str
        Path to the Darknet .cfg file.
    weights_path : str
        Path to the Darknet .weights file.
    output_path : str
        Destination path for the exported ONNX model.

    Raises
    ------
    FileNotFoundError
        If ``cfg_path`` or ``weights_path`` do not exist.
    """
    import torch
    import onnx

    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"CFG not found: {cfg_path}")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    print("Darknet model:")
    print(f"  CFG     : {cfg_path}")
    print(f"  Weights : {weights_path}")
    print(f"  Output  : {output_path}\n")

    # ── Build & load ──────────────────────────────────────────────────────────
    print("Building PyTorch model from cfg …")
    model = DarknetModel(cfg_path)
    model.eval()

    print("Loading Darknet weights …")
    load_darknet_weights(model, weights_path)

    # ── Dry-run to confirm shapes ─────────────────────────────────────────────
    print("Dry-run forward pass …")
    dummy = torch.zeros(1, 3, INPUT_H, INPUT_W)
    with torch.no_grad():
        raw_outs = model(dummy)
    for i, t in enumerate(raw_outs):
        print(f"  Head {i}: {tuple(t.shape)}")

    # ── Export ────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    wrapper   = _YoloExportWrapper(model)
    wrapper.eval()
    out_names = [f"output{i}" for i in range(len(raw_outs))]

    print(f"\nExporting to ONNX (opset 13) …  outputs: {out_names}")
    # Use torch.onnx.export.  On torch >= 2.5 use torch.onnx.utils.export
    # to avoid onnxscript conflicts, but torch.onnx.export works for 1.x–2.4.
    # Static batch=1 — required by MERA for MCU deployment.
    # Opset 13 — required by onnxruntime per-channel INT8 quantization
    #            (DequantizeLinear with axis attribute needs opset ≥ 13).
    _export_fn = getattr(torch.onnx, "utils", torch.onnx)
    _export_fn = getattr(_export_fn, "export", torch.onnx.export)
    _export_fn(
        wrapper,
        dummy,
        output_path,
        input_names        = ["images"],
        output_names       = out_names,
        opset_version      = 13,
        do_constant_folding = True,
    )
    print(f"  ✅ ONNX saved: {output_path}")

    onnx.checker.check_model(onnx.load(output_path))
    print("  ✅ ONNX graph check passed (opset 13)")


def verify(onnx_path: str) -> None:
    """
    Verify an ONNX model with onnxruntime inference.

    Parameters
    ----------
    onnx_path : str
        Path to the .onnx file.
    """
    import onnx
    import onnxruntime as ort

    print(f"\nVerifying ONNX model: {onnx_path}")
    onnx.checker.check_model(onnx.load(onnx_path))
    print("  ✅ ONNX graph check passed")

    sess    = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    dummy   = np.random.rand(1, 3, INPUT_H, INPUT_W).astype(np.float32)
    outs    = sess.run(None, {in_name: dummy})
    print("  ✅ OnnxRuntime inference OK")
    for i, o in enumerate(outs):
        print(f"     output{i}: shape={o.shape}  dtype={o.dtype}  "
              f"min={o.min():.4f}  max={o.max():.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Convert Yolo-Fastest Darknet model to ONNX."
    )
    parser.add_argument(
        "--variant", choices=list(_VARIANT_DEFAULTS.keys()), default=None,
        help=(
            "Model variant shortcut: 'base' (yolo-fastest-1.1) or "
            "'xl' (yolo-fastest-1.1-xl). "
            "Sets default --cfg / --weights / --output paths. "
            "Explicit --cfg / --weights / --output flags override this."
        ),
    )
    parser.add_argument(
        "--cfg", default=None,
        help=f"Path to .cfg file  (default: model/yolo-fastest-1.1.cfg)"
    )
    parser.add_argument(
        "--weights", default=None,
        help=f"Path to .weights file  (default: model/yolo-fastest-1.1.weights)"
    )
    parser.add_argument(
        "--output", default=None,
        help=f"Output ONNX path  (default: model/yolo_fastest_1.1.onnx)"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Run onnxruntime verification after conversion"
    )
    args = parser.parse_args()

    # Resolve paths: variant defaults → explicit overrides
    vkey = args.variant or "base"
    vd = _VARIANT_DEFAULTS[vkey]
    cfg_path     = args.cfg     or os.path.join(_MODEL_DIR, vd["cfg"])
    weights_path = args.weights or os.path.join(_MODEL_DIR, vd["weights"])
    output_path  = args.output  or os.path.join(_MODEL_DIR, vd["output"])

    convert(cfg_path, weights_path, output_path)

    if args.verify:
        verify(output_path)

    print("\nDone.  Next step:")
    print(f"  python inference.py --image sample.jpg --model {output_path}")


if __name__ == "__main__":
    main()
