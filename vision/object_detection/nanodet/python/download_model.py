#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright 2026 Renesas Electronics Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
# NanoDet-Plus model is from the NanoDet repository:
#   https://github.com/RangiLyu/nanodet
# Licensed under the Apache License, Version 2.0.
"""
NanoDet-Plus-m -- Download ONNX, Quantize INT8, and Convert to TFLite
======================================================================
Downloads the pre-built NanoDet-Plus-m ONNX model from the official GitHub
release, quantizes it to INT8 using ONNX Runtime static quantization, and
converts to TFLite FP32/INT8 using onnx2tf.

All outputs are stored under model/ within this project directory.

Pipeline:
    1. Download NanoDet-Plus-m .onnx from GitHub Releases
    2. ONNX FP32 -> ONNX INT8  (via onnxruntime static quantization)
    3. ONNX FP32 -> TFLite FP32 (via onnx2tf)
    4. ONNX FP32 -> TFLite INT8 Static  (int8 I/O)

Supported input sizes: 320 (default), 416 — both have pre-built ONNX models.

Usage:
  python download_model.py                          # full pipeline (all models, 320x320)
  python download_model.py --mode fp32              # FP32 ONNX only
  python download_model.py --mode int8              # FP32 + INT8 ONNX
  python download_model.py --mode tflite            # FP32 ONNX + TFLite FP32/INT8
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

OUTPUT_DIR = "model"

# Supported input sizes → download URLs
DEFAULT_INPUT_SIZE = 320

_ONNX_URLS = {
    320: "https://github.com/RangiLyu/nanodet/releases/download/v1.0.0-alpha-1/nanodet-plus-m_320.onnx",
    416: "https://github.com/RangiLyu/nanodet/releases/download/v1.0.0-alpha-1/nanodet-plus-m_416.onnx",
}

# Normalization params (BGR order) — needed for calibration preprocessing
MEAN = [103.53, 116.28, 123.675]
STD = [57.375, 57.12, 58.395]
NUM_CLASSES = 80


def _model_paths(input_size: int = DEFAULT_INPUT_SIZE):
    """Return (onnx_fp32, onnx_int8, tflite_fp32, tflite_int8_static) paths."""
    return (
        os.path.join(OUTPUT_DIR, f"nanodet-plus-m_{input_size}.onnx"),
        os.path.join(OUTPUT_DIR, f"nanodet-plus-m_{input_size}_int8.onnx"),
        os.path.join(OUTPUT_DIR, f"nanodet-plus-m_{input_size}_FP32.tflite"),
        os.path.join(OUTPUT_DIR, f"nanodet-plus-m_{input_size}_INT8.tflite"),
    )


# Legacy global paths
ONNX_PATH, INT8_PATH, TFLITE_FP32, TFLITE_INT8 = _model_paths(DEFAULT_INPUT_SIZE)

# Calibration defaults
CALIB_DIR = os.path.join("Datasets", "val2017")
DEFAULT_CALIB_NUM = 100

_COCO_VAL2017_URL = "http://images.cocodataset.org/zips/val2017.zip"
_DATASET_DIR = os.path.dirname(CALIB_DIR)
_COCO_VAL_ZIP = os.path.join(_DATASET_DIR, "val2017.zip")


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
    """Return a valid calibration directory, downloading COCO val2017 if necessary."""
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


def ensure_onnx2tf_test_npy(path: str = "calibration_image_sample_data_20x128x128x3_float32.npy") -> str:
    """Ensure the onnx2tf sample .npy exists and can be loaded."""
    import numpy as np

    if os.path.isfile(path):
        try:
            arr = np.load(path)
            if arr.shape == (20, 128, 128, 3) and arr.dtype == np.float32:
                return path
            print(f"[!] Existing sample file has unexpected contents: {path} ({arr.shape}, {arr.dtype})")
        except Exception as e:
            print(f"[!] Existing sample file is unreadable: {path} ({e})")

    np.save(path, np.random.RandomState(0).rand(20, 128, 128, 3).astype(np.float32))
    arr = np.load(path)
    print(f"[OK] Recreated sample file: {path} ({arr.shape}, {arr.dtype})")
    return path


# --------------------------------------------------------------------------
# Preprocessing for calibration (matches inference.py)
# --------------------------------------------------------------------------
def preprocess_image(image_path: str, input_size: int = DEFAULT_INPUT_SIZE):
    """Load and preprocess a single image for NanoDet-Plus-m (NCHW).

    NanoDet convention: warp resize (direct resize), mean/std normalize, HWC→NCHW.
    """
    import cv2
    import numpy as np

    img = cv2.imread(image_path)
    if img is None:
        return None

    resized = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    blob = resized.astype(np.float32)
    mean = np.array(MEAN, dtype=np.float32)
    std = np.array(STD, dtype=np.float32)
    blob = (blob - mean) / std
    blob = blob.transpose(2, 0, 1)[np.newaxis, ...]  # NCHW
    return blob


# --------------------------------------------------------------------------
# Pipeline functions
# --------------------------------------------------------------------------
def download_onnx(input_size: int = DEFAULT_INPUT_SIZE):
    """Download the pre-built NanoDet-Plus-m ONNX model from GitHub Releases."""
    onnx_path = _model_paths(input_size)[0]

    print("\n" + "=" * 60)
    print(f"  Download NanoDet-Plus-m ONNX ({input_size}x{input_size})")
    print("=" * 60)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if os.path.isfile(onnx_path):
        size_mb = os.path.getsize(onnx_path) / 1024 / 1024
        print(f"[OK] ONNX already exists: {onnx_path} ({size_mb:.1f} MB)")
        return

    if input_size not in _ONNX_URLS:
        raise ValueError(
            f"No pre-built ONNX available for input size {input_size}. "
            f"Supported sizes: {sorted(_ONNX_URLS.keys())}"
        )

    url = _ONNX_URLS[input_size]
    _download_with_progress(url, onnx_path)

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
            print("  (onnxsim check failed -- keeping original)")
    except ImportError:
        print("  (onnxsim not installed -- skipping simplification)")
    except Exception as e:
        print(f"  (onnxsim failed: {e} -- skipping)")

    size_mb = os.path.getsize(onnx_path) / 1024 / 1024
    print(f"[OK] ONNX FP32 saved: {onnx_path} ({size_mb:.2f} MB)")


def _ensure_min_opset(onnx_path: str, min_opset: int = 13):
    """Upgrade an ONNX model in-place to at least `min_opset` for ai.onnx domain.

    Per-channel quantize_static emits DequantizeLinear/QuantizeLinear with an
    `axis` attribute, which is only defined from opset 13 onwards.
    """
    import onnx
    from onnx import version_converter

    model = onnx.load(onnx_path)
    current = next(
        (o.version for o in model.opset_import if (o.domain or "ai.onnx") == "ai.onnx"),
        None,
    )
    if current is None or current >= min_opset:
        return

    print(f"  Upgrading ONNX opset {current} -> {min_opset} for per-channel Q/DQ ...")
    upgraded = version_converter.convert_version(model, min_opset)
    onnx.save(upgraded, onnx_path)


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
        raise FileNotFoundError(f"FP32 ONNX not found: {onnx_path}. Download first.")

    import numpy as np
    from onnxruntime.quantization import (
        CalibrationDataReader,
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    calib_dir = ensure_dataset(calib_dir)
    calib_paths = list_images(calib_dir, need=calib_num)
    if len(calib_paths) < 10:
        raise FileNotFoundError(
            f"Not enough calibration images in {calib_dir} "
            f"(found {len(calib_paths)}, need at least 10)."
        )
    calib_paths = calib_paths[:calib_num]
    print(f"  Using {len(calib_paths)} calibration images from {calib_dir}")

    # Determine input name from ONNX model
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    del sess

    class _NanoDetCalibrationReader(CalibrationDataReader):
        def __init__(self, image_paths, size, input_name):
            self.image_paths = image_paths
            self.size = size
            self.input_name = input_name
            self.index = 0
            self.total = len(image_paths)
            if _HAS_TQDM:
                self.pbar = tqdm(total=self.total, desc="  calibrating", unit="img")
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

            return {self.input_name: blob}

    # Pre-process the ONNX model for quantization
    preprocessed_path = onnx_path.replace(".onnx", "_preprocessed.onnx")
    print("  Running shape inference on FP32 model ...")
    quant_pre_process(onnx_path, preprocessed_path)

    # Ensure opset >= 13 so per-channel Q/DQ nodes may carry the `axis`
    # attribute (DequantizeLinear.axis was introduced in opset 13).
    _ensure_min_opset(preprocessed_path, min_opset=13)

    print(f"  Running INT8 static quantization ({len(calib_paths)} images) ...")
    calib_reader = _NanoDetCalibrationReader(calib_paths, input_size, input_name)

    quantize_static(
        model_input=preprocessed_path,
        model_output=int8_path,
        calibration_data_reader=calib_reader,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        calibrate_method=CalibrationMethod.MinMax,
    )

    if os.path.isfile(preprocessed_path):
        os.remove(preprocessed_path)

    size_mb = os.path.getsize(int8_path) / 1024 / 1024
    print(f"[OK] INT8 ONNX saved: {int8_path} ({size_mb:.2f} MB)")


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

    import onnx2tf

    ensure_onnx2tf_test_npy()

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
# Convert ONNX -> TFLite INT8 Static using onnx2tf's OWN built-in
# quantizer (fixes the accuracy collapse seen with the manual
# tf.lite.TFLiteConverter + representative_dataset approach).
# --------------------------------------------------------------------------
def convert_tflite_int8_static_onnx2tf_builtin(input_size: int = DEFAULT_INPUT_SIZE,
                                               calib_dir: str = CALIB_DIR,
                                               calib_num: int = DEFAULT_CALIB_NUM,
                                               calib_paths=None):
    """Produce a full-integer (int8 I/O) TFLite model using onnx2tf's own
    integrated `output_integer_quantized_tflite` quantizer.

    """
    import numpy as np
    import onnx
    import onnx2tf

    onnx_path, _, _, tflite_int8_static = _model_paths(input_size)

    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    if calib_paths is None:
        calib_dir = ensure_dataset(calib_dir)
        calib_paths = list_images(calib_dir, need=calib_num)
        if len(calib_paths) < 10:
            raise FileNotFoundError(
                f"Not enough calibration images in {calib_dir} "
                f"(found {len(calib_paths)}, need at least 10)."
            )
        calib_paths = calib_paths[:calib_num]

    print(f"  Using {len(calib_paths)} calibration images for onnx2tf built-in quantizer")

    # onnx2tf's custom calibration data must be pre-normalized to [0, 1]
    # (NOT mean/std normalized) -- it applies (value - mean) / std
    # internally. Scale NanoDet's 0-255-range MEAN/STD down to match.
    import cv2

    calib_arr = np.zeros((len(calib_paths), input_size, input_size, 3), dtype=np.float32)
    n_valid = 0
    for path in calib_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        resized = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
        calib_arr[n_valid] = resized.astype(np.float32) / 255.0
        n_valid += 1
    calib_arr = calib_arr[:n_valid]

    calib_npy_path = os.path.join(OUTPUT_DIR, f"onnx2tf_calib_{input_size}.npy")
    np.save(calib_npy_path, calib_arr)

    mean_01 = (np.array(MEAN, dtype=np.float32) / 255.0).reshape(1, 1, 1, 3)
    std_01 = (np.array(STD, dtype=np.float32) / 255.0).reshape(1, 1, 1, 3)

    onnx_model = onnx.load(onnx_path)
    input_op_name = onnx_model.graph.input[0].name

    saved_model_dir = os.path.join(OUTPUT_DIR, "saved_model_int8_onnx2tf_builtin")
    if os.path.isdir(saved_model_dir):
        shutil.rmtree(saved_model_dir)

    onnx2tf.convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=saved_model_dir,
        copy_onnx_input_output_names_to_tflite=True,
        output_integer_quantized_tflite=True,
        quant_type="per-channel",
        input_quant_dtype="int8",
        output_quant_dtype="int8",
        custom_input_op_name_np_data_path=[
            [input_op_name, calib_npy_path, mean_01.tolist(), std_01.tolist()],
        ],
        non_verbose=True,
    )

    produced = [
        f for f in os.listdir(saved_model_dir)
        if f.endswith("_full_integer_quant.tflite")
    ]
    if not produced:
        raise FileNotFoundError(
            f"onnx2tf did not produce a *_full_integer_quant.tflite file in {saved_model_dir}"
        )

    shutil.copy2(os.path.join(saved_model_dir, produced[0]), tflite_int8_static)
    shutil.rmtree(saved_model_dir, ignore_errors=True)
    os.remove(calib_npy_path)

    size_mb = os.path.getsize(tflite_int8_static) / 1024 / 1024
    print(f"[OK] TFLite INT8 static (onnx2tf built-in) saved: {tflite_int8_static} ({size_mb:.2f} MB)")
    return tflite_int8_static


# --------------------------------------------------------------------------
# Convert ONNX -> TFLite INT8 (fully quantized)
# --------------------------------------------------------------------------
def convert_tflite_int8(input_size: int = DEFAULT_INPUT_SIZE,
                        calib_dir: str = CALIB_DIR,
                        calib_num: int = DEFAULT_CALIB_NUM):
    """Convert ONNX FP32 to a static INT8 (int8 I/O) TFLite model."""
    onnx_path, _, _, tflite_int8_static = _model_paths(input_size)

    print("\n" + "=" * 60)
    print(f"  Convert ONNX FP32 -> TFLite INT8  ({input_size}x{input_size})")
    print("=" * 60)

    if os.path.isfile(tflite_int8_static):
        size_sta = os.path.getsize(tflite_int8_static) / 1024 / 1024
        print(f"[OK] TFLite INT8 static already exists: {tflite_int8_static} ({size_sta:.1f} MB)")
        return

    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    ensure_onnx2tf_test_npy()

    calib_dir = ensure_dataset(calib_dir)
    calib_paths = list_images(calib_dir, need=calib_num)
    if len(calib_paths) < 10:
        raise FileNotFoundError(
            f"Not enough calibration images in {calib_dir} "
            f"(found {len(calib_paths)}, need at least 10)."
        )
    calib_paths = calib_paths[:calib_num]
    print(f"  Using {len(calib_paths)} calibration images from {calib_dir}")

    # TFLite INT8 full static conversion (int8 input/output).
    #
    # IMPORTANT: this intentionally uses onnx2tf's built-in quantizer.
    # A separate, external tf.lite.TFLiteConverter pass was found to
    # silently corrupt accuracy to near-zero (~0.03% AP) for this model:
    # onnx2tf internally synthesizes an explicit Pad (with an extreme
    # sentinel value, clamped to roughly +-1e10) to implement MaxPool's
    # explicit ONNX padding semantics in TensorFlow. Our own external
    # TFLiteConverter pass has no knowledge of this convention and
    # calibrates/quantizes it naively, corrupting scales throughout the
    # whole backbone. onnx2tf's OWN integrated quantizer
    # (output_integer_quantized_tflite=True) handles this correctly,
    # since it generated the pattern in the first place. See
    # INT8_QUANTIZATION_FINDINGS.md for the full investigation.
    print("  Running TFLite INT8 quantization (full static int8 I/O, onnx2tf built-in quantizer) ...")
    convert_tflite_int8_static_onnx2tf_builtin(input_size, calib_dir, calib_num, calib_paths=calib_paths)

    size_sta = os.path.getsize(tflite_int8_static) / 1024 / 1024
    print(f"[OK] TFLite INT8 static saved: {tflite_int8_static} ({size_sta:.2f} MB)")



# --------------------------------------------------------------------------
# Verify TFLite models
# --------------------------------------------------------------------------
def verify_tflite(path: str, tag: str):
    """Load a TFLite model and print its IO details."""
    import numpy as np
    import tensorflow as tf

    try:
        interpreter = tf.lite.Interpreter(model_path=path, num_threads=1)
        interpreter.allocate_tensors()
    except RuntimeError:
        # Fallback: disable XNNPACK delegate
        interpreter = tf.lite.Interpreter(
            model_path=path, num_threads=1,
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
        )
        interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    outputs = interpreter.get_output_details()

    print(f"  [{tag}] input  -> shape={list(inp['shape'])}, "
          f"dtype={inp['dtype'].__name__}")
    if inp["dtype"] != np.float32:
        s, z = inp["quantization"]
        print(f"  [{tag}] input  quant: scale={s:.6f}, zero_point={z}")

    for out in outputs:
        name = out.get("name", "")
        print(f"  [{tag}] output '{name}' -> shape={list(out['shape'])}, "
              f"dtype={out['dtype'].__name__}")
        if out["dtype"] != np.float32:
            s, z = out["quantization"]
            print(f"  [{tag}] output '{name}' quant: scale={s:.6f}, zero_point={z}")



# --------------------------------------------------------------------------
# Main entry-point
# --------------------------------------------------------------------------
def main(mode: str = "all",
         input_size: int = DEFAULT_INPUT_SIZE,
         calib_dir: str = CALIB_DIR,
         calib_num: int = DEFAULT_CALIB_NUM):
    onnx_path, int8_path, tflite_fp32, tflite_int8_static = _model_paths(input_size)

    # Download pre-built ONNX
    download_onnx(input_size)

    # Quantize to INT8 ONNX
    if mode in ("int8", "all"):
        convert_int8(input_size, calib_dir, calib_num)

    # Convert to TFLite FP32
    if mode in ("tflite", "all"):
        convert_tflite_fp32(input_size)

    # Convert to TFLite INT8 static (int8 I/O)
    if mode in ("tflite", "all"):
        convert_tflite_int8(input_size, calib_dir, calib_num)

    # Verify TFLite models
    if mode in ("tflite", "all"):
        print("\n" + "=" * 60)
        print("  TFLite Verification")
        print("=" * 60)
        if os.path.isfile(tflite_fp32):
            verify_tflite(tflite_fp32, "FP32")
        if os.path.isfile(tflite_int8_static):
            verify_tflite(tflite_int8_static, "INT8_STATIC")

    # Summary
    print("\n" + "=" * 60)
    print(f"  All done!  NanoDet-Plus-m Model Files ({input_size}x{input_size}):")
    print("=" * 60)
    for label, path in [
        ("ONNX (FP32)   ", onnx_path),
        ("ONNX (INT8)   ", int8_path),
        ("TFLite (FP32) ", tflite_fp32),
        ("TFLite (INT8) ", tflite_int8_static),
    ]:
        if os.path.exists(path):
            sz = os.path.getsize(path) / 1024 / 1024
            print(f"  {label} : {path}  ({sz:.2f} MB)")
    print("=" * 60)

    # Clean up onnx2tf sample calibration file if it was created.
    _sample_npy = "calibration_image_sample_data_20x128x128x3_float32.npy"
    if os.path.isfile(_sample_npy):
        try:
            os.remove(_sample_npy)
        except OSError:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NanoDet-Plus-m -- Download ONNX, quantize INT8, convert TFLite."
    )
    parser.add_argument(
        "--mode",
        choices=["fp32", "int8", "tflite", "all"],
        default="all",
        help="Export mode: fp32 (FP32 only), int8 (FP32 + INT8 ONNX), "
             "tflite (FP32 ONNX + TFLite FP32/INT8), all (everything). Default: all.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=DEFAULT_INPUT_SIZE,
        help=f"Input square size (default: {DEFAULT_INPUT_SIZE}). "
             f"Supported: {sorted(_ONNX_URLS.keys())}.",
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
