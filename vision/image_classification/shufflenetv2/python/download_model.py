#!/usr/bin/env python3
"""
download_model.py — ShuffleNetV2 x0.5 → ONNX → TFLite (FP32 / INT8)

This script:
  1. Loads ShuffleNetV2 x0.5 pretrained weights from torchvision.
  2. Exports to ONNX.
  3. Converts ONNX → TFLite FP32.
  4. (Optional) Converts ONNX → TFLite INT8 with representative dataset
     calibration using ImageNet validation images.

Usage:
  python download_model.py --mode fp32
  python download_model.py --mode int8 --calib-num 1000
  python download_model.py --mode both
"""

import os
import sys
import argparse
from typing import List

import numpy as np
from PIL import Image

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

# ──────────────────────────────────────────────────────────────────────────────
# Paths / constants
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "Model")
ONNX_PATH  = os.path.join(MODEL_DIR, "shufflenet_v2_x0_5.onnx")
FP32_PATH  = os.path.join(MODEL_DIR, "shufflenet_v2_x0_5_FP32.tflite")
INT8_PATH  = os.path.join(MODEL_DIR, "shufflenet_v2_x0_5_INT8.tflite")

CALIB_DIR  = None  # Must be provided via --calib-dir for INT8 mode

INPUT_HEIGHT = 224
INPUT_WIDTH  = 224


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def list_images(folder: str, need: int,
                exts=(".jpg", ".jpeg", ".png", ".bmp")) -> List[str]:
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
# Preprocessing (ImageNet standard for torchvision models)
#   1. Resize to 256 (shortest edge)
#   2. Center-crop to 224×224
#   3. Normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
#
# For TFLite conversion we use the SAME preprocessing but output shape
# is (1, 224, 224, 3) NHWC float32.
# ──────────────────────────────────────────────────────────────────────────────
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_nhwc(image_path: str,
                    height: int = INPUT_HEIGHT,
                    width: int  = INPUT_WIDTH) -> np.ndarray:
    """
    Standard ImageNet preprocessing → (1, H, W, 3) float32 for TFLite.
    """
    img = Image.open(image_path).convert("RGB")

    # Resize shortest edge to 256, then center-crop 224×224
    w, h = img.size
    scale = 256.0 / min(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img = img.resize((new_w, new_h), resample=Image.BILINEAR)
    left = (new_w - width) // 2
    top  = (new_h - height) // 2
    img = img.crop((left, top, left + width, top + height))

    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return np.expand_dims(arr, axis=0)  # (1, H, W, 3)


def preprocess_nchw(image_path: str,
                    height: int = INPUT_HEIGHT,
                    width: int  = INPUT_WIDTH) -> np.ndarray:
    """
    Standard ImageNet preprocessing → (1, 3, H, W) float32 for PyTorch/ONNX.
    """
    nhwc = preprocess_nhwc(image_path, height, width)  # (1, H, W, 3)
    return np.transpose(nhwc, (0, 3, 1, 2))            # (1, 3, H, W)


# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Export PyTorch → ONNX
# ──────────────────────────────────────────────────────────────────────────────
def export_onnx(onnx_path: str = ONNX_PATH) -> str:
    import torch
    import torchvision.models as models

    print("Loading ShuffleNetV2 x0.5 from torchvision (pretrained=True) ...")
    model = models.shufflenet_v2_x0_5(weights=models.ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1)
    model.eval()

    dummy = torch.randn(1, 3, INPUT_HEIGHT, INPUT_WIDTH)

    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    print(f"Exporting to ONNX → {onnx_path}")
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        opset_version=13,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,  # fixed batch = 1
    )
    print(f"✅ ONNX saved: {onnx_path}  "
          f"({os.path.getsize(onnx_path) / 1e6:.2f} MB)")
    return onnx_path


# ──────────────────────────────────────────────────────────────────────────────
# Step 2: ONNX → TFLite FP32
# ──────────────────────────────────────────────────────────────────────────────
def _onnx_to_saved_model(onnx_path: str, saved_model_dir: str) -> str:
    """Convert ONNX → TF SavedModel using onnx2tf.

    Creates a dummy calibration .npy file that onnx2tf expects on disk,
    same approach used in NanoDet-Plus-m/download_model.py.
    """
    import onnx2tf

    # onnx2tf tries to download a test image .npy; create a dummy one locally
    _test_npy = "calibration_image_sample_data_20x128x128x3_float32.npy"
    if not os.path.isfile(_test_npy):
        np.save(_test_npy,
                np.random.RandomState(0).rand(20, 128, 128, 3).astype(np.float32))

    onnx2tf.convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=saved_model_dir,
        copy_onnx_input_output_names_to_tflite=True,
        output_signaturedefs=True,
        non_verbose=True,
    )
    return saved_model_dir


def convert_fp32(onnx_path: str = ONNX_PATH,
                 tflite_path: str = FP32_PATH) -> str:
    import tensorflow as tf

    print("Converting ONNX → TFLite FP32 via onnx2tf ...")
    saved_model_dir = os.path.join(MODEL_DIR, "_tf_saved_model_fp32")
    _onnx_to_saved_model(onnx_path, saved_model_dir)

    # onnx2tf may produce a .tflite automatically
    auto_tflite = os.path.join(saved_model_dir, "model_float32.tflite")
    if os.path.isfile(auto_tflite):
        import shutil
        shutil.copy2(auto_tflite, tflite_path)
    else:
        print("Converting SavedModel → TFLite FP32 ...")
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        tflite_bytes = converter.convert()
        with open(tflite_path, "wb") as f:
            f.write(tflite_bytes)

    print(f"✅ TFLite FP32 saved: {tflite_path}  "
          f"({os.path.getsize(tflite_path) / 1e6:.2f} MB)")
    return tflite_path


# ──────────────────────────────────────────────────────────────────────────────
# Step 3: ONNX → TFLite INT8 (full-integer quantization via onnx2tf)
# ──────────────────────────────────────────────────────────────────────────────
def convert_int8(onnx_path: str = ONNX_PATH,
                 tflite_path: str = INT8_PATH,
                 calib_paths: List[str] = None) -> str:
    """Convert ONNX → TFLite INT8 using onnx2tf's built-in quantization.

    NOTE: We use onnx2tf's native INT8 quantization rather than
    tf.lite.TFLiteConverter because the TFLite converter produces broken
    INT8 models for ShuffleNetV2 — its channel-shuffle pattern
    (Reshape→Transpose→Reshape) is not quantized correctly by TFLite's
    MLIR quantizer, resulting in 0% accuracy.  onnx2tf handles this
    correctly by quantizing during the ONNX→TF conversion itself.
    """
    import onnx2tf

    # Build calibration npy: stack N images into (N, 224, 224, 3) float32
    print(f"Preparing {len(calib_paths)} calibration images ...")
    calib_npy_path = os.path.join(MODEL_DIR, "_calib_data.npy")
    imgs = []
    iterator = tqdm(calib_paths, desc="Preprocessing") if _HAS_TQDM else calib_paths
    for i, p in enumerate(iterator, 1):
        if not _HAS_TQDM:
            sys.stdout.write(f"\rPreprocessing: {i}/{len(calib_paths)}")
            sys.stdout.flush()
        imgs.append(preprocess_nhwc(p)[0])  # drop batch dim → (H, W, 3)
    if not _HAS_TQDM:
        sys.stdout.write("\n")
    np.save(calib_npy_path, np.stack(imgs).astype(np.float32))
    print(f"  Calibration data saved: {calib_npy_path}")

    # Dummy npy that onnx2tf may try to download
    _test_npy = "calibration_image_sample_data_20x128x128x3_float32.npy"
    if not os.path.isfile(_test_npy):
        np.save(_test_npy,
                np.random.RandomState(0).rand(20, 128, 128, 3).astype(np.float32))

    out_dir = os.path.join(MODEL_DIR, "_onnx2tf_int8_out")
    print("Converting ONNX → TFLite INT8 via onnx2tf (built-in quantization) ...")
    onnx2tf.convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=out_dir,
        copy_onnx_input_output_names_to_tflite=True,
        output_signaturedefs=True,
        non_verbose=True,
        output_integer_quantized_tflite=True,
        quant_type="per-channel",
        custom_input_op_name_np_data_path=[
            ["input", calib_npy_path, [0, 0, 0], [1, 1, 1]],
        ],
    )

    # onnx2tf generates *_full_integer_quant.tflite (INT8 I/O)
    model_stem = os.path.splitext(os.path.basename(onnx_path))[0]
    auto_int8 = os.path.join(out_dir, f"{model_stem}_full_integer_quant.tflite")
    if not os.path.isfile(auto_int8):
        raise FileNotFoundError(
            f"Expected onnx2tf to produce {auto_int8} but file not found. "
            f"Available: {os.listdir(out_dir)}"
        )

    import shutil
    shutil.copy2(auto_int8, tflite_path)
    print(f"✅ TFLite INT8 saved: {tflite_path}  "
          f"({os.path.getsize(tflite_path) / 1e6:.2f} MB)")
    return tflite_path


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="ShuffleNetV2 x0.5: PyTorch → ONNX → TFLite (FP32 / INT8)."
    )
    parser.add_argument(
        "--mode",
        choices=["fp32", "int8", "both", "FP32", "INT8", "BOTH"],
        default="both",
        help="Export mode (default: fp32).",
    )
    parser.add_argument(
        "--calib-num", type=int, default=1000,
        help="Number of calibration images for INT8 (default: 1000).",
    )
    parser.add_argument(
        "--calib-dir", type=str, default=None,
        help="Path to directory with calibration images (required for INT8 mode).",
    )
    args = parser.parse_args()
    mode = args.mode.upper()

    # Always export ONNX first (if not already present)
    if not os.path.isfile(ONNX_PATH):
        export_onnx()
    else:
        print(f"✅ ONNX already exists: {ONNX_PATH}")

    if mode in ("FP32", "BOTH"):
        convert_fp32()

    if mode in ("INT8", "BOTH"):
        if not args.calib_dir:
            sys.exit(
                "[ERROR] --calib-dir is required for INT8 mode.\n"
                "ImageNet validation images cannot be auto-downloaded due to licensing.\n"
                "See https://image-net.org/download.php for access, then run:\n"
                "  python download_model.py --mode int8 "
                "--calib-dir /path/to/ILSVRC2012_img_val"
            )
        calib_paths = list_images(args.calib_dir, need=args.calib_num)
        if len(calib_paths) < 100:
            raise FileNotFoundError(
                f"Not enough calibration images in {args.calib_dir} "
                f"(found {len(calib_paths)}, need ≥100)."
            )
        calib_paths = calib_paths[:args.calib_num]
        print(f"Using {len(calib_paths)} calibration images from {args.calib_dir}")
        convert_int8(calib_paths=calib_paths)

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
