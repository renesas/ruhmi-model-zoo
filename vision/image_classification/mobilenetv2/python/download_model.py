
# This file uses the Keras MobileNetV2 model provided by TensorFlow/Keras
# (https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2).
# The TensorFlow and Keras code and model weights are distributed under the
# Apache License, Version 2.0. Users should ensure compliance with that
# license when redistributing or reusing the model or its weights.

# SPDX-License-Identifier: BSD-3-Clause

# ──────────────────────────────────────────────────────────────────────────────
# NOTE: ImageNet images are NOT redistributable under a commercial licence.
# You must supply your own calibration images (e.g. from the ILSVRC 2012
# validation set obtained via https://image-net.org/download.php).
# Pass the directory with --calib-dir when running in INT8 mode.
# ──────────────────────────────────────────────────────────────────────────────

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

INPUT_HEIGHT = 224
INPUT_WIDTH  = 224


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


def build_model():
    base = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        alpha=1.0,
        include_top=True,
        weights="imagenet",
        classifier_activation="softmax",
    )
    inputs = keras.Input(shape=(224, 224, 3), batch_size=1)
    outputs = base(inputs)
    return keras.Model(inputs, outputs)


def preprocess(image_path: str,
               height: int = INPUT_HEIGHT,
               width: int  = INPUT_WIDTH) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    img = img.resize((width, height), resample=Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    arr = arr / 127.5 - 1.0
    return np.expand_dims(arr, axis=0)


def make_representative_dataset(calib_paths: List[str]):
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
            yield [preprocess(p)]
        if not _HAS_TQDM:
            sys.stdout.write("\n")
    return gen


def convert_fp32(model: keras.Model) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    return converter.convert()


def convert_int8(model: keras.Model, calib_paths: List[str]) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    converter.representative_dataset = make_representative_dataset(calib_paths)
    return converter.convert()


def main():
    parser = argparse.ArgumentParser(
        description="Build MobileNetV2 (alpha=1.0) from Keras and convert to TFLite."
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
    args = parser.parse_args()

    model = build_model()
    os.makedirs("model", exist_ok=True)

    mode = args.mode.upper()

    # ── FP32 ──────────────────────────────────────────────────────────────────
    if mode in ("FP32", "BOTH"):
        print("Converting to FP32 ...")
        tflite_bytes = convert_fp32(model)
        out_path = "model/mobilenet_v2_FP32.tflite"
        with open(out_path, "wb") as f:
            f.write(tflite_bytes)
        print(f"Saved {out_path}")

    # ── INT8 ──────────────────────────────────────────────────────────────────
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
        tflite_bytes = convert_int8(model, calib_paths)
        out_path = "model/mobilenet_v2_INT8.tflite"
        with open(out_path, "wb") as f:
            f.write(tflite_bytes)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
