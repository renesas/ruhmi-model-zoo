# SPDX-License-Identifier: BSD-3-Clause
#
# Export SqueezeNet 1.1 (PyTorch → ONNX → TF SavedModel → TFLite).
#
# ──────────────────────────────────────────────────────────────────────────────
# NOTE: ImageNet images are NOT redistributable under a commercial licence.
# You must supply your own calibration images (e.g. from the ILSVRC 2012
# validation set obtained via https://image-net.org/download.php).
# Pass the directory with --calib-dir when running in INT8 mode.
# ──────────────────────────────────────────────────────────────────────────────

import argparse
import glob
import os
import sys
from typing import List

import cv2
import numpy as np
import torch
import torchvision.models as models
import onnx
import onnx2tf
import tensorflow as tf

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG  (paths are relative to this script's directory: python/)
# ──────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR     = "model"
ONNX_PATH      = os.path.join(OUTPUT_DIR, "squeezenet1_1.onnx")
TF_SAVED_MODEL = os.path.join(OUTPUT_DIR, "squeezenet1_1_saved_model")
FP32_OUT       = os.path.join(OUTPUT_DIR, "squeezenet1_1_FP32.tflite")
INT8_OUT       = os.path.join(OUTPUT_DIR, "squeezenet1_1_INT8.tflite")

INPUT_HEIGHT   = 224
INPUT_WIDTH    = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────────────────────────────────────
def preprocess_image(img_path: str) -> np.ndarray:
    """Load and preprocess a single image to (1, H, W, 3) float32 (NHWC)."""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT))
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return np.expand_dims(img, axis=0)   # (1, H, W, 3)


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────
def export_onnx():
    """Export PyTorch SqueezeNet1.1 → ONNX."""
    print("=" * 60)
    print("Exporting PyTorch SqueezeNet1.1 → ONNX")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = models.squeezenet1_1(pretrained=True)
    model.eval()

    dummy_input = torch.randn(1, 3, INPUT_HEIGHT, INPUT_WIDTH)

    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        verbose=False,
        input_names=['input'],
        output_names=['output'],
        opset_version=18,
        export_params=True,
        do_constant_folding=True,
    )
    print(f"SqueezeNet model successfully exported to {ONNX_PATH}")

    # Inline external data if the exporter created a .data sidecar file
    data_file = ONNX_PATH + ".data"
    if os.path.exists(data_file):
        print("External data file detected — inlining into the .onnx file...")
        loaded = onnx.load(ONNX_PATH, load_external_data=True)
        tmp_path = ONNX_PATH + ".tmp"
        onnx.save_model(loaded, tmp_path, save_as_external_data=False)
        os.replace(tmp_path, ONNX_PATH)
        os.remove(data_file)
        print(f"Done. Single-file model saved to {ONNX_PATH}")
    else:
        print("No external data file found — model is already self-contained.")


def onnx_to_savedmodel():
    """Convert ONNX → TensorFlow SavedModel."""
    print("\n" + "=" * 60)
    print("ONNX → TensorFlow SavedModel")
    print("=" * 60)

    # Create dummy npy that onnx2tf tries to download for validation
    _test_npy = "calibration_image_sample_data_20x128x128x3_float32.npy"
    if not os.path.isfile(_test_npy):
        np.save(_test_npy,
                np.random.RandomState(0).rand(20, 128, 128, 3).astype(np.float32))

    # Monkey-patch np.load to allow pickle (needed by onnx2tf's test image loader)
    _original_np_load = np.load
    np.load = lambda *args, **kwargs: _original_np_load(*args, **{**kwargs, 'allow_pickle': True})
    try:
        onnx2tf.convert(
            input_onnx_file_path=ONNX_PATH,
            output_folder_path=TF_SAVED_MODEL,
            non_verbose=True,
            copy_onnx_input_output_names_to_tflite=True,
            output_signaturedefs=True,
        )
    finally:
        np.load = _original_np_load
    print(f"✅ SavedModel written to: {TF_SAVED_MODEL}")


def convert_fp32():
    """Convert SavedModel → FP32 TFLite."""
    print("\n" + "=" * 60)
    print("SavedModel → FP32 TFLite")
    print("=" * 60)

    converter = tf.lite.TFLiteConverter.from_saved_model(TF_SAVED_MODEL)
    tflite_fp32 = converter.convert()
    with open(FP32_OUT, "wb") as f:
        f.write(tflite_fp32)
    print(f"✅ FP32 TFLite saved: {FP32_OUT}  ({len(tflite_fp32)/1e6:.2f} MB)")


def convert_int8(calib_dir: str, calib_num: int):
    """Convert SavedModel → INT8 TFLite via post-training quantization."""
    print("\n" + "=" * 60)
    print("SavedModel → INT8 TFLite (post-training quantization)")
    print("=" * 60)

    img_files = list_images(calib_dir, need=calib_num)

    if not img_files:
        sys.exit(
            f"[ERROR] No calibration images found in '{calib_dir}'.\n"
            "        Check the path and make sure it contains JPEG/PNG images."
        )

    print(f"Using {len(img_files)} images from '{calib_dir}' for calibration.")
    if _HAS_TQDM:
        calib_data = [
            preprocess_image(p)
            for p in tqdm(img_files, desc="Loading calib images", unit="img")
        ]
    else:
        calib_data = []
        for i, p in enumerate(tqdm(img_files, desc="Loading calib images", unit="img") if _HAS_TQDM else img_files, 1):
            calib_data.append(preprocess_image(p))
        if not _HAS_TQDM:
            print(f"  Loaded {len(calib_data)} calibration images.")

    def representative_dataset():
        if _HAS_TQDM:
            for sample in tqdm(calib_data, desc="Calibrating INT8", unit="sample"):
                yield [sample]
        else:
            for sample in calib_data:
                yield [sample]

    converter_int8 = tf.lite.TFLiteConverter.from_saved_model(TF_SAVED_MODEL)
    converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_int8.representative_dataset = representative_dataset
    converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter_int8.inference_input_type  = tf.int8
    converter_int8.inference_output_type = tf.int8

    tflite_int8 = converter_int8.convert()
    with open(INT8_OUT, "wb") as f:
        f.write(tflite_int8)
    print(f"✅ INT8 TFLite saved: {INT8_OUT}  ({len(tflite_int8)/1e6:.2f} MB)")


# ──────────────────────────────────────────────────────────────────────────────
# Main entry-point
# ──────────────────────────────────────────────────────────────────────────────
def main(mode: str = "both", calib_num: int = 1000, calib_dir: str = None):
    """
    Run the full pipeline or selected steps.

    Parameters
    ----------
    mode : str
        "fp32"  → export_onnx + onnx_to_savedmodel + convert_fp32.
        "int8"  → export_onnx + onnx_to_savedmodel + convert_int8.
        "both"   → all of the above (FP32 + INT8 TFLite).
    calib_num : int
        Number of calibration images for INT8 quantization.
    calib_dir : str
        Path to calibration images directory. Required for int8/all modes.
    """
    if mode in ("int8", "both") and not calib_dir:
        sys.exit(
            "[ERROR] calib_dir is required for INT8 / all mode.\n"
            "        Download the ImageNet validation set from "
            "https://image-net.org/download.php and pass the directory."
        )

    export_onnx()
    onnx_to_savedmodel()

    if mode in ("fp32", "both"):
        convert_fp32()

    if mode in ("int8", "both"):
        convert_int8(calib_dir, calib_num)

    print("\n" + "=" * 60)
    print("All done!")
    print(f"  ONNX  : {ONNX_PATH}")
    if mode in ("fp32", "both"):
        print(f"  FP32  : {FP32_OUT}")
    if mode in ("int8", "both"):
        print(f"  INT8  : {INT8_OUT}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export SqueezeNet1.1 (PyTorch → ONNX → TF → TFLite)."
    )
    parser.add_argument(
        "--mode",
        choices=["fp32", "int8", "both", "FP32", "INT8", "BOTH"],
        default="both",
        help="Export mode: fp32 (FP32 only), int8 (INT8 only), all (both). Default: all.",
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
             "Required for INT8/all mode. Obtain ImageNet validation images from "
             "https://image-net.org/download.php (registration required).",
    )
    args = parser.parse_args()
    main(args.mode, args.calib_num, args.calib_dir)
