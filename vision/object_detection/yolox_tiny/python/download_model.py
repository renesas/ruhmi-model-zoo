#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""
YOLOX-Tiny -- Download, Export to ONNX, Quantize INT8, and Convert to TFLite
==============================================================================
Downloads the pre-trained YOLOX-Tiny checkpoint (.pth) from the official YOLOX
GitHub release, exports it to a single-file ONNX model, quantizes it to INT8
using ONNX Runtime static quantization, and converts to TFLite FP32/INT8 using
onnx2tf with COCO val2017 calibration images.

All outputs are stored under model/ within this project directory.

Pipeline:
  1. Download YOLOX-Tiny .pth from GitHub Releases
  2. PyTorch .pth -> ONNX FP32  (via torch.onnx.export, using local utils/)
  3. ONNX   FP32 -> ONNX INT8   (via onnxruntime static quantization)
  4. ONNX   FP32 -> TFLite FP32 (via onnx2tf)
  5. ONNX   FP32 -> TFLite INT8 (via onnx2tf + TFLite quantization)

Usage:
  python download_model.py                          # full pipeline (all models)
  python download_model.py --mode fp32              # FP32 ONNX only
  python download_model.py --mode int8              # FP32 + INT8 ONNX
  python download_model.py --mode tflite            # FP32 ONNX + TFLite FP32/INT8
  python download_model.py --mode all               # all models
  python download_model.py --input-size 416         # export at 416x416 (default)
  python download_model.py --input-size 224         # export at 224x224 (adds _224 suffix)
  python download_model.py --calib-dir /my/images   # custom calibration dir
  python download_model.py --calib-num 200          # number of calibration images
"""

import argparse
import os
import shutil
import sys
import urllib.request
import zipfile
from typing import List

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

OUTPUT_DIR  = "model"
PTH_PATH    = os.path.join(OUTPUT_DIR, "yolox_tiny.pth")

# YOLOX-Tiny model defaults
DEFAULT_INPUT_SIZE = 416


def _model_paths(input_size: int = DEFAULT_INPUT_SIZE):
    """Return (onnx_fp32, onnx_int8, tflite_fp32, tflite_int8) paths.

    When input_size == 416 (default) the names are unchanged for
    backward-compatibility.  For any other size a ``_<size>`` suffix
    is inserted, e.g. ``yolox_tiny_224.onnx``.
    """
    if input_size == DEFAULT_INPUT_SIZE:
        sfx = ""
    else:
        sfx = f"_{input_size}"
    return (
        os.path.join(OUTPUT_DIR, f"yolox_tiny{sfx}.onnx"),
        os.path.join(OUTPUT_DIR, f"yolox_tiny{sfx}_int8.onnx"),
        os.path.join(OUTPUT_DIR, f"yolox_tiny{sfx}_FP32.tflite"),
        os.path.join(OUTPUT_DIR, f"yolox_tiny{sfx}_INT8.tflite"),
    )


# Legacy global paths (default 416 — kept for imports)
ONNX_PATH, INT8_PATH, TFLITE_FP32, TFLITE_INT8 = _model_paths(DEFAULT_INPUT_SIZE)
NUM_CLASSES        = 80
PAD_VALUE          = 114

# Calibration defaults
CALIB_DIR          = os.path.join("Datasets", "val2017")
DEFAULT_CALIB_NUM  = 100

# Download URLs
YOLOX_TINY_PTH_URL = (
    "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/"
    "yolox_tiny.pth"
)
_COCO_VAL2017_URL  = "http://images.cocodataset.org/zips/val2017.zip"
_DATASET_DIR       = os.path.dirname(CALIB_DIR)          # "Datasets"
_COCO_VAL_ZIP      = os.path.join(_DATASET_DIR, "val2017.zip")


# --------------------------------------------------------------------------
# Helper utilities
# --------------------------------------------------------------------------
def _download_with_progress(url: str, dest: str) -> None:
    """Download *url* to *dest* with a live progress bar."""
    print(f"Downloading {url}\n         -> {dest} ...")
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    if _HAS_TQDM:
        class _Reporter:
            def __init__(self):
                self.pbar = None
            def __call__(self, block_num, block_size, total_size):
                if self.pbar is None:
                    self.pbar = tqdm(total=total_size, unit="B",
                                     unit_scale=True, desc="  download")
                self.pbar.update(min(block_size, max(0, total_size - self.pbar.n)))
                if block_num * block_size >= total_size:
                    self.pbar.close()
        urllib.request.urlretrieve(url, dest, reporthook=_Reporter())
    else:
        def _report(b, bs, total):
            sys.stdout.write(f"\r  {b*bs/1e6:.1f} / {total/1e6:.1f} MB")
            sys.stdout.flush()
        urllib.request.urlretrieve(url, dest, reporthook=_report)
        sys.stdout.write("\n")


def list_images(folder: str, need: int,
                exts=(".jpg", ".jpeg", ".png", ".bmp")) -> List[str]:
    """Return up to *need* image paths from *folder*."""
    if not os.path.isdir(folder):
        return []
    out = []
    for root, _, files in os.walk(folder):
        for f in sorted(files):
            if f.lower().endswith(exts):
                out.append(os.path.join(root, f))
                if len(out) >= need:
                    return out
    return out


def ensure_dataset(calib_dir: str = CALIB_DIR) -> str:
    """
    Return a valid calibration directory, downloading COCO val2017 if necessary.
    """
    if os.path.isdir(calib_dir) and list_images(calib_dir, need=1):
        print(f"[OK] Dataset found: {calib_dir}")
        return calib_dir

    print(f"[!!] Dataset not found at {calib_dir}")
    os.makedirs(_DATASET_DIR, exist_ok=True)

    if not os.path.isfile(_COCO_VAL_ZIP):
        _download_with_progress(_COCO_VAL2017_URL, _COCO_VAL_ZIP)
    else:
        print(f"[OK] ZIP already present: {_COCO_VAL_ZIP}")

    print(f"Extracting {_COCO_VAL_ZIP} -> {_DATASET_DIR} ...")
    with zipfile.ZipFile(_COCO_VAL_ZIP, "r") as zf:
        members = zf.namelist()
        if _HAS_TQDM:
            for m in tqdm(members, desc="  extract", unit="file"):
                zf.extract(m, path=_DATASET_DIR)
        else:
            total = len(members)
            for i, m in enumerate(members, 1):
                zf.extract(m, path=_DATASET_DIR)
                sys.stdout.write(f"\r  Extracting {i}/{total}")
                sys.stdout.flush()
            sys.stdout.write("\n")

    print(f"[OK] Dataset ready: {calib_dir}")
    return calib_dir


# --------------------------------------------------------------------------
# Preprocessing (matches inference.py exactly)
# --------------------------------------------------------------------------
def preprocess_image(image_path: str, input_size: int = DEFAULT_INPUT_SIZE):
    """Load and preprocess a single image for YOLOX-Tiny.

    YOLOX convention: letterbox resize with gray (114) padding,
    float32 0-255 range (no /255 normalisation), HWC -> NCHW.
    """
    import cv2
    import numpy as np

    img = cv2.imread(image_path)
    if img is None:
        return None

    ih, iw = img.shape[:2]
    scale = min(input_size / iw, input_size / ih)
    new_w, new_h = int(iw * scale), int(ih * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    padded = np.full((input_size, input_size, 3), PAD_VALUE, dtype=np.uint8)
    pad_w = (input_size - new_w) // 2
    pad_h = (input_size - new_h) // 2
    padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

    blob = padded.astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...]  # NCHW
    return blob


# --------------------------------------------------------------------------
# Pipeline functions
# --------------------------------------------------------------------------
def download_checkpoint():
    """Download the pre-trained YOLOX-Tiny checkpoint from GitHub Releases."""
    print("\n" + "=" * 60)
    print("  Download YOLOX-Tiny .pth")
    print("=" * 60)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if os.path.isfile(PTH_PATH):
        size_mb = os.path.getsize(PTH_PATH) / 1024 / 1024
        print(f"[OK] Checkpoint already exists: {PTH_PATH} ({size_mb:.1f} MB)")
        return

    _download_with_progress(YOLOX_TINY_PTH_URL, PTH_PATH)
    size_mb = os.path.getsize(PTH_PATH) / 1024 / 1024
    print(f"[OK] Downloaded: {PTH_PATH} ({size_mb:.1f} MB)")


def export_onnx(input_size: int = DEFAULT_INPUT_SIZE):
    """Export the YOLOX-Tiny PyTorch model to ONNX format.

    Uses the self-contained model builder in utils/yolox_model.py.
    No dependency on the external yolox package.
    """
    onnx_path = _model_paths(input_size)[0]

    print("\n" + "=" * 60)
    print(f"  Export PyTorch .pth -> ONNX FP32  ({input_size}x{input_size})")
    print("=" * 60)

    if os.path.isfile(onnx_path):
        size_mb = os.path.getsize(onnx_path) / 1024 / 1024
        print(f"[OK] ONNX already exists: {onnx_path} ({size_mb:.1f} MB)")
        return

    import torch
    from torch import nn
    from utils.yolox_model import build_yolox_tiny, replace_module, SiLU

    # Build model from scratch (YOLOX-Tiny: depth=0.33, width=0.375, depthwise=False)
    print("  Building YOLOX-Tiny architecture ...")
    model = build_yolox_tiny(num_classes=NUM_CLASSES)

    # Load checkpoint
    print(f"  Loading weights from {PTH_PATH} ...")
    ckpt = torch.load(PTH_PATH, map_location="cpu")
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt)
    model.eval()

    # Replace nn.SiLU with export-friendly SiLU for ONNX compatibility
    model = replace_module(model, nn.SiLU, SiLU)
    model.head.decode_in_inference = False

    # Export to ONNX
    dummy_input = torch.randn(1, 3, input_size, input_size)
    print(f"  Exporting ONNX (input: 1x3x{input_size}x{input_size}) ...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["images"],
        output_names=["output"],
        opset_version=18,
    )
    print(f"  ONNX exported: {onnx_path}")

    # Simplify with onnxsim if available
    try:
        import onnx
        from onnxsim import simplify
        onnx_model = onnx.load(onnx_path)
        original_ir = onnx_model.ir_version
        if onnx_model.ir_version > 4:
            onnx_model.ir_version = 4
        model_simp, check = simplify(onnx_model)
        if check:
            model_simp.ir_version = original_ir
            onnx.save(model_simp, onnx_path)
            print("  ONNX simplified with onnxsim")
        else:
            print("  (onnxsim check failed -- keeping unsimplified model)")
    except ImportError:
        print("  (onnxsim not installed -- skipping simplification)")
    except Exception as e:
        print(f"  (onnxsim failed: {e} -- skipping simplification)")

    # Merge external data into a single .onnx file if needed
    try:
        import onnx as _onnx
        _m = _onnx.load(onnx_path, load_external_data=True)
        _onnx.save_model(_m, onnx_path, save_as_external_data=False)
        ext_data = onnx_path + ".data"
        if os.path.exists(ext_data):
            os.remove(ext_data)
            print("  Merged external data into single .onnx file")
    except Exception:
        pass

    size_mb = os.path.getsize(onnx_path) / 1024 / 1024
    print(f"[OK] ONNX FP32 saved: {onnx_path} ({size_mb:.2f} MB)")


def convert_int8(input_size: int = DEFAULT_INPUT_SIZE,
                 calib_dir: str = CALIB_DIR,
                 calib_num: int = DEFAULT_CALIB_NUM):
    """Quantize the FP32 ONNX model to INT8 using ONNX Runtime static quantization."""
    onnx_path, int8_path, _, _ = _model_paths(input_size)

    print("\n" + "=" * 60)
    print(f"  Quantize ONNX FP32 -> ONNX INT8  ({input_size}x{input_size})")
    print("=" * 60)

    if os.path.isfile(int8_path):
        size_mb = os.path.getsize(int8_path) / 1024 / 1024
        print(f"[OK] INT8 ONNX already exists: {int8_path} ({size_mb:.1f} MB)")
        return

    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(
            f"FP32 ONNX model not found: {onnx_path}. Export FP32 first."
        )

    import numpy as np
    from onnxruntime.quantization import (
        CalibrationDataReader,
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    # Auto-download dataset if needed
    calib_dir = ensure_dataset(calib_dir)

    # Collect calibration image paths
    calib_paths = list_images(calib_dir, need=calib_num)
    if len(calib_paths) < 10:
        raise FileNotFoundError(
            f"Not enough calibration images in {calib_dir} "
            f"(found {len(calib_paths)}, need at least 10). "
            "Check your dataset or internet connection."
        )
    calib_paths = calib_paths[:calib_num]
    print(f"  Using {len(calib_paths)} calibration images from {calib_dir}")

    # ── Identify detection-head nodes to keep in FP32 ────────
    import onnx as _onnx
    _model = _onnx.load(onnx_path)

    _head_concat_names = {"node_cat_14", "node_cat_15", "node_cat_16", "node_cat_17"}
    _head_inputs = set()
    for _n in _model.graph.node:
        if _n.name in _head_concat_names:
            _head_inputs.update(_n.input)

    _nodes_to_exclude = list(_head_concat_names)
    for _n in _model.graph.node:
        for _o in _n.output:
            if _o in _head_inputs:
                _nodes_to_exclude.append(_n.name)
    for _n in _model.graph.node:
        if _n.name.startswith("node_view"):
            _nodes_to_exclude.append(_n.name)
    del _model
    print(f"  Excluding {len(_nodes_to_exclude)} detection-head nodes from quantization")

    # Calibration data reader
    class _YOLOXCalibrationReader(CalibrationDataReader):
        def __init__(self, image_paths, size):
            self.image_paths = image_paths
            self.size = size
            self.index = 0
            self.total = len(image_paths)
            if _HAS_TQDM:
                self.pbar = tqdm(total=self.total, desc="  calibrating",
                                 unit="img")
            else:
                self.pbar = None

        def get_next(self):
            if self.index >= len(self.image_paths):
                if self.pbar is not None:
                    self.pbar.close()
                return None
            img_path = self.image_paths[self.index]
            self.index += 1

            blob = preprocess_image(img_path, self.size)
            if blob is None:
                if self.pbar is not None:
                    self.pbar.update(1)
                return self.get_next()

            if self.pbar is not None:
                self.pbar.update(1)
            elif self.index % 100 == 0 or self.index == self.total:
                sys.stdout.write(f"\r  calibrating {self.index}/{self.total}")
                sys.stdout.flush()
                if self.index == self.total:
                    sys.stdout.write("\n")

            return {"images": blob}

    # Pre-process the ONNX model for quantization (shape inference)
    preprocessed_path = onnx_path.replace(".onnx", "_preprocessed.onnx")
    print("  Running shape inference on FP32 model ...")
    quant_pre_process(onnx_path, preprocessed_path)

    print(f"  Running INT8 static quantization ({len(calib_paths)} images) ...")
    calib_reader = _YOLOXCalibrationReader(calib_paths, input_size)
    quantize_static(
        model_input=preprocessed_path,
        model_output=int8_path,
        calibration_data_reader=calib_reader,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        calibrate_method=CalibrationMethod.MinMax,
        nodes_to_exclude=_nodes_to_exclude,
    )

    if os.path.isfile(preprocessed_path):
        os.remove(preprocessed_path)

    size_mb = os.path.getsize(int8_path) / 1024 / 1024
    print(f"[OK] INT8 ONNX saved: {int8_path} ({size_mb:.2f} MB)")


# --------------------------------------------------------------------------
# Preprocessing for TFLite (NHWC layout)
# --------------------------------------------------------------------------
def preprocess_image_nhwc(image_path: str, input_size: int = DEFAULT_INPUT_SIZE):
    """Load and letterbox-preprocess a single image for TFLite (NHWC).

    Returns np.ndarray shape (1, H, W, 3) float32 0-255, or None if unreadable.
    """
    import cv2
    import numpy as np

    img = cv2.imread(image_path)
    if img is None:
        return None

    ih, iw = img.shape[:2]
    scale = min(input_size / iw, input_size / ih)
    new_w, new_h = int(iw * scale), int(ih * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    padded = np.full((input_size, input_size, 3), PAD_VALUE, dtype=np.uint8)
    pad_w = (input_size - new_w) // 2
    pad_h = (input_size - new_h) // 2
    padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

    blob = padded.astype(np.float32)[np.newaxis, ...]  # (1, H, W, 3) NHWC
    return blob


# --------------------------------------------------------------------------
# Convert ONNX -> TFLite FP32
# --------------------------------------------------------------------------
def convert_tflite_fp32(input_size: int = DEFAULT_INPUT_SIZE):
    """Convert the ONNX FP32 model to TFLite FP32 using onnx2tf."""
    onnx_path, _, tflite_fp32, _ = _model_paths(input_size)

    print("\n" + "=" * 60)
    print(f"  Convert ONNX FP32 -> TFLite FP32  ({input_size}x{input_size})")
    print("=" * 60)

    if os.path.isfile(tflite_fp32):
        size_mb = os.path.getsize(tflite_fp32) / 1024 / 1024
        print(f"[OK] TFLite FP32 already exists: {tflite_fp32} ({size_mb:.1f} MB)")
        return

    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    import numpy as np
    import onnx2tf

    # onnx2tf internally needs a small test array for verification
    _test_npy = "calibration_image_sample_data_20x128x128x3_float32.npy"
    if not os.path.isfile(_test_npy):
        np.save(_test_npy, np.random.RandomState(0).rand(20, 128, 128, 3).astype(np.float32))

    saved_model_dir = os.path.join(OUTPUT_DIR, "saved_model_fp32")

    print(f"  Converting {onnx_path} -> TFLite FP32 ...")
    onnx2tf.convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=saved_model_dir,
        copy_onnx_input_output_names_to_tflite=True,
        non_verbose=True,
    )

    # Find the generated .tflite file
    tflite_candidates = []
    for root, _, files in os.walk(saved_model_dir):
        for f in files:
            if f.endswith(".tflite") and "float32" in f.lower():
                tflite_candidates.append(os.path.join(root, f))
    if not tflite_candidates:
        for root, _, files in os.walk(saved_model_dir):
            for f in files:
                if f.endswith(".tflite"):
                    tflite_candidates.append(os.path.join(root, f))

    if not tflite_candidates:
        raise FileNotFoundError(
            f"No .tflite file found in {saved_model_dir}. "
            "onnx2tf conversion may have failed."
        )

    shutil.copy2(tflite_candidates[0], tflite_fp32)
    shutil.rmtree(saved_model_dir, ignore_errors=True)

    size_mb = os.path.getsize(tflite_fp32) / 1024 / 1024
    print(f"[OK] TFLite FP32 saved: {tflite_fp32} ({size_mb:.2f} MB)")


# --------------------------------------------------------------------------
# Convert ONNX -> TFLite INT8 (fully quantized)
# --------------------------------------------------------------------------
def convert_tflite_int8(input_size: int = DEFAULT_INPUT_SIZE,
                        calib_dir: str = CALIB_DIR,
                        calib_num: int = DEFAULT_CALIB_NUM):
    """Convert ONNX FP32 to a fully quantized INT8 TFLite model."""
    onnx_path, _, _, tflite_int8 = _model_paths(input_size)

    print("\n" + "=" * 60)
    print(f"  Convert ONNX FP32 -> TFLite INT8  ({input_size}x{input_size})")
    print("=" * 60)

    if os.path.isfile(tflite_int8):
        size_mb = os.path.getsize(tflite_int8) / 1024 / 1024
        print(f"[OK] TFLite INT8 already exists: {tflite_int8} ({size_mb:.1f} MB)")
        return

    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    import numpy as np
    import tensorflow as tf
    import onnx2tf

    _test_npy = "calibration_image_sample_data_20x128x128x3_float32.npy"
    if not os.path.isfile(_test_npy):
        np.save(_test_npy, np.random.RandomState(0).rand(20, 128, 128, 3).astype(np.float32))

    # Ensure calibration dataset
    calib_dir = ensure_dataset(calib_dir)
    calib_paths = list_images(calib_dir, need=calib_num)
    if len(calib_paths) < 10:
        raise FileNotFoundError(
            f"Not enough calibration images in {calib_dir} "
            f"(found {len(calib_paths)}, need at least 10)."
        )
    calib_paths = calib_paths[:calib_num]
    print(f"  Using {len(calib_paths)} calibration images from {calib_dir}")

    # Convert ONNX -> SavedModel
    saved_model_dir = os.path.join(OUTPUT_DIR, "saved_model_int8")

    print(f"  Converting {onnx_path} -> SavedModel ...")
    onnx2tf.convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=saved_model_dir,
        copy_onnx_input_output_names_to_tflite=True,
        non_verbose=True,
    )

    # Determine input layout from SavedModel
    loaded = tf.saved_model.load(saved_model_dir)
    concrete_func = loaded.signatures["serving_default"]
    input_spec = list(concrete_func.structured_input_signature[1].values())[0]
    input_shape = input_spec.shape
    is_nhwc = (len(input_shape) == 4 and input_shape[-1] == 3)

    print(f"  SavedModel input shape: {input_shape}")

    def representative_dataset():
        items = calib_paths
        if _HAS_TQDM:
            items = tqdm(items, desc="  calibrating", unit="img")
        for path in items:
            blob = preprocess_image_nhwc(path, input_size)
            if blob is None:
                continue
            if not is_nhwc:
                blob = blob.transpose(0, 3, 1, 2)
            yield [blob.astype(np.float32)]

    # TFLite INT8 conversion
    print("  Running TFLite INT8 quantization ...")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    with open(tflite_int8, "wb") as f:
        f.write(tflite_model)

    shutil.rmtree(saved_model_dir, ignore_errors=True)

    size_mb = os.path.getsize(tflite_int8) / 1024 / 1024
    print(f"[OK] TFLite INT8 saved: {tflite_int8} ({size_mb:.2f} MB)")


# --------------------------------------------------------------------------
# Verify TFLite models
# --------------------------------------------------------------------------
def verify_tflite(path: str, tag: str):
    """Load a TFLite model and print its IO details."""
    import numpy as np
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    print(f"  [{tag}] input  -> shape={list(inp['shape'])}, "
          f"dtype={inp['dtype'].__name__}")
    print(f"  [{tag}] output -> shape={list(out['shape'])}, "
          f"dtype={out['dtype'].__name__}")

    if inp["dtype"] != np.float32:
        s, z = inp["quantization"]
        print(f"  [{tag}] input  quant: scale={s:.6f}, zero_point={z}")
    if out["dtype"] != np.float32:
        s, z = out["quantization"]
        print(f"  [{tag}] output quant: scale={s:.6f}, zero_point={z}")


# --------------------------------------------------------------------------
# Main entry-point
# --------------------------------------------------------------------------
def main(mode: str = "all",
         input_size: int = DEFAULT_INPUT_SIZE,
         calib_dir: str = CALIB_DIR,
         calib_num: int = DEFAULT_CALIB_NUM):
    onnx_path, int8_path, tflite_fp32, tflite_int8 = _model_paths(input_size)

    # Download checkpoint
    download_checkpoint()

    # Export FP32 ONNX (always needed)
    export_onnx(input_size)

    # Quantize to INT8 ONNX
    if mode in ("int8", "both", "all"):
        convert_int8(input_size, calib_dir, calib_num)

    # Convert to TFLite FP32
    if mode in ("tflite", "all"):
        convert_tflite_fp32(input_size)

    # Convert to TFLite INT8
    if mode in ("tflite", "all"):
        convert_tflite_int8(input_size, calib_dir, calib_num)

    # Verify TFLite models
    if mode in ("tflite", "all"):
        print("\n" + "=" * 60)
        print("  TFLite Verification")
        print("=" * 60)
        if os.path.isfile(tflite_fp32):
            verify_tflite(tflite_fp32, "FP32")
        if os.path.isfile(tflite_int8):
            verify_tflite(tflite_int8, "INT8")

    # Summary
    print("\n" + "=" * 60)
    print(f"  All done!  YOLOX-Tiny Model Files ({input_size}x{input_size}):")
    print("=" * 60)
    for label, path in [
        ("PTH  (PyTorch)", PTH_PATH),
        ("ONNX (FP32)   ", onnx_path),
        ("ONNX (INT8)   ", int8_path),
        ("TFLite (FP32) ", tflite_fp32),
        ("TFLite (INT8) ", tflite_int8),
    ]:
        if os.path.exists(path):
            sz = os.path.getsize(path) / 1024 / 1024
            print(f"  {label} : {path}  ({sz:.2f} MB)")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YOLOX-Tiny -- Download .pth, export ONNX FP32, quantize INT8, convert TFLite."
    )
    parser.add_argument(
        "--mode",
        choices=["fp32", "int8", "tflite", "both", "FP32", "INT8", "BOTH"],
        default="both",
        help="Export mode: fp32 (FP32 only), int8 (FP32 + INT8 ONNX), "
             "tflite (FP32 ONNX + TFLite FP32/INT8), all (everything). Default: all.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=224,
        help=f"Input square size for ONNX/TFLite export (default: {DEFAULT_INPUT_SIZE}). "
             f"Non-default sizes add a suffix, e.g. yolox_tiny_224.onnx.",
    )
    parser.add_argument(
        "--calib-dir",
        type=str,
        default=CALIB_DIR,
        help=f"Path to calibration images directory (default: {CALIB_DIR}). "
             "Auto-downloaded if not found.",
    )
    parser.add_argument(
        "--calib-num",
        type=int,
        default=DEFAULT_CALIB_NUM,
        help=f"Number of calibration images for INT8 quantization (default: {DEFAULT_CALIB_NUM}).",
    )
    args = parser.parse_args()
    main(args.mode, args.input_size, args.calib_dir, args.calib_num)
