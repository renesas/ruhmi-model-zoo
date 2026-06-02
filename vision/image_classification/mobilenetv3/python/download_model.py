# SPDX-License-Identifier: BSD-3-Clause

# This file uses the Keras MobileNetV3Small model provided by TensorFlow/Keras
# (https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV3Small).
# The TensorFlow and Keras code and model weights are distributed under the
# Apache License, Version 2.0. Users should ensure compliance with that
# license when redistributing or reusing the model or its weights.

# ──────────────────────────────────────────────────────────────────────────────
# NOTE: ImageNet images are NOT redistributable under a commercial licence.
# You must supply your own calibration images (e.g. from the ILSVRC 2012
# validation set obtained via https://image-net.org/download.php).
# Pass the directory with --calib-dir when running in INT8 mode.
# ──────────────────────────────────────────────────────────────────────────────

"""
Build MobileNetV3-Small (minimalistic) 192×192 from Keras and convert to TFLite (FP32 / INT8).

Usage:
    python download_model.py                           # both FP32 + INT8
    python download_model.py --mode fp32               # FP32 only
    python download_model.py --mode int8 --calib-dir /path/to/ILSVRC2012_img_val
"""
import argparse
import os
import sys
from typing import List

import numpy as np
import tensorflow as tf
import keras
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False
from PIL import Image

DEFAULT_INPUT_SIZE = 192


def list_images(folder: str, need: int, exts=(".jpg", ".jpeg", ".png", ".bmp")) -> List[str]:
    if not os.path.isdir(folder):
        return []
    out = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(exts):
                out.append(os.path.join(root, f))
                if len(out) >= need:
                    return out
    return out


def build_model(input_size: int = 192):
    """Build MobileNetV3-Small (minimalistic) with ImageNet weights."""
    base = keras.applications.MobileNetV3Small(
        input_shape=(input_size, input_size, 3),
        include_top=True,
        weights="imagenet",
        minimalistic=True,
        classifier_activation="softmax",
    )
    inputs = keras.Input(shape=(input_size, input_size, 3), batch_size=1)
    outputs = base(inputs)
    return keras.Model(inputs, outputs)


def preprocess(image_path: str,
               height: int = DEFAULT_INPUT_SIZE,
               width: int  = DEFAULT_INPUT_SIZE) -> np.ndarray:
    """Load and preprocess one image for MobileNetV3.

    MobileNetV3 expects raw pixel values in [0, 255] (float32).
    The model's internal Rescaling layer handles normalisation.
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((width, height), resample=Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


def make_representative_dataset(calib_paths: List[str], size: int = DEFAULT_INPUT_SIZE):
    def gen():
        total = len(calib_paths)
        if _HAS_TQDM:
            iterator = tqdm(calib_paths, desc="Generating calibration data")
        else:
            iterator = calib_paths
        for i, p in enumerate(iterator, start=1):
            if not _HAS_TQDM and total:
                sys.stdout.write(f"Calibration: {i}/{total}\r")
                sys.stdout.flush()
            yield [preprocess(p, size, size)]
        if not _HAS_TQDM:
            sys.stdout.write("\n")
    return gen


def convert_fp32(model: keras.Model) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    return converter.convert()


def convert_int8(model: keras.Model, calib_paths: List[str], size: int = DEFAULT_INPUT_SIZE) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    converter.representative_dataset = make_representative_dataset(calib_paths, size)
    return converter.convert()


def main():
    parser = argparse.ArgumentParser(
        description="Build MobileNetV3-Small (minimalistic) 192×192 from Keras and convert to TFLite."
    )
    parser.add_argument(
        "--mode",
        choices=["fp32", "int8", "both", "FP32", "INT8", "BOTH"],
        default="both",
        help="Export mode: fp32, int8 or both (default: both).",
    )
    parser.add_argument(
        "--calib-num",
        type=int,
        default=1000,
        help="Number of calibration images for INT8 quantization (default: 1000).",
    )
    parser.add_argument(
        "--calib-dir",
        type=str,
        default=None,
        help="Path to directory containing calibration images. "
             "Required for INT8 mode. Obtain ImageNet validation images from "
             "https://image-net.org/download.php (registration required).",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=192,
        choices=[128, 192, 224],
        help="Input image size (H=W). Supported: 128, 192, 224 (default: 192).",
    )
    args = parser.parse_args()

    input_size = args.input_size
    model = build_model(input_size)
    os.makedirs("model", exist_ok=True)

    mode = args.mode.upper()

    # ── FP32 ─────────────────────────────────────────────────────────────────
    if mode in ("FP32", "BOTH"):
        print("Converting to FP32 ...")
        tflite_bytes = convert_fp32(model)
        out_path = f"model/mobilenet_v3_small_{input_size}_FP32.tflite"
        with open(out_path, "wb") as f:
            f.write(tflite_bytes)
        print(f"Saved {out_path}")

    # ── INT8 ─────────────────────────────────────────────────────────────────
    if mode in ("INT8", "BOTH"):
        if not args.calib_dir:
            sys.exit(
                "[ERROR] --calib-dir is required for INT8 mode.\n"
                "        Download the ImageNet validation set from "
                "https://image-net.org/download.php and pass the directory.\n"
                "        Example: python download_model.py --mode int8 "
                "--calib-dir /path/to/ILSVRC2012_img_val"
            )
        calib_paths = list_images(args.calib_dir, need=args.calib_num)
        if len(calib_paths) < 100:
            sys.exit(
                f"[ERROR] Not enough calibration images in '{args.calib_dir}' "
                f"(found {len(calib_paths)}, need at least 100).\n"
                "        Check the path and make sure the directory contains JPEG/PNG images."
            )
        calib_paths = calib_paths[:args.calib_num]
        print(f"Using {len(calib_paths)} calibration images from {args.calib_dir}")
        print("Converting to INT8 ...")
        tflite_bytes = convert_int8(model, calib_paths, input_size)
        out_path = f"model/mobilenet_v3_small_{input_size}_INT8.tflite"
        with open(out_path, "wb") as f:
            f.write(tflite_bytes)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
