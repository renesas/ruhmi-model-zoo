# SPDX-License-Identifier: BSD-3-Clause
"""Download & Convert MCUNet-in0 to TFLite (MERA-quantizer-compatible variant).

Here the model's forward method is patched to fuse the
global-average-pool (x.mean(3).mean(2)) into a single mean(dim=(2, 3))
so the MERA MCU quantizer can lower it.

Model: mcunet-in0 (mcunet-10fps_imagenet)
  - Input resolution: 48x48
  - Top-1 accuracy: ~41.5% (fp32) / ~40.4% (int8)

Pipeline:
  1. Load pretrained PyTorch model from MIT HAN Lab (mcunet package)
  2. Patch forward() to use fused global-avg-pool
  3. Export to ONNX
  4. Convert ONNX -> TFLite FP32 (via onnx2tf)
  5. Quantize ONNX -> TFLite INT8 (via onnx2tf, using calibration images)

Usage:
    python download_model_mera_fix.py                 # FP32 + INT8
    python download_model_mera_fix.py --mode fp32     # FP32 only
    python download_model_mera_fix.py --mode int8     # INT8 only
    python download_model_mera_fix.py --calib-num 500 # 500 calibration images
"""

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import types
import urllib.request
from typing import List

import numpy as np
from PIL import Image

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
CALIB_DIR = os.path.join(BASE_DIR, "Datasets", "ILSVRC2012_img_val")

MCUNET_NET_ID = "mcunet-in0"
INPUT_RESOLUTION = 48

_IMAGENET_VAL_URL = "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar"
_DATASET_DIR = os.path.dirname(CALIB_DIR)
_IMAGENET_TAR = os.path.join(_DATASET_DIR, "ILSVRC2012_img_val.tar")

MCUNET_GIT_URL = "https://github.com/mit-han-lab/mcunet.git"
MCUNET_LOCAL_REPO = os.path.join(BASE_DIR, "third_party", "mcunet")


# Download helpers
def _download_with_progress(url: str, dest: str) -> None:
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


# Dataset helpers
def list_images(folder: str, need: int,
                exts=(".jpg", ".jpeg", ".png", ".bmp")) -> List[str]:
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


def ensure_dataset(calib_dir: str = CALIB_DIR) -> str:
    """Return a valid calibration directory, downloading ImageNet val if needed."""
    if os.path.isdir(calib_dir) and list_images(calib_dir, need=1):
        print(f"Dataset found: {calib_dir}")
        return calib_dir

    print(f"Dataset not found at {calib_dir}")
    os.makedirs(calib_dir, exist_ok=True)

    if not os.path.isfile(_IMAGENET_TAR):
        _download_with_progress(_IMAGENET_VAL_URL, _IMAGENET_TAR)
    else:
        print(f"TAR already present: {_IMAGENET_TAR}")

    print(f"Extracting {_IMAGENET_TAR} -> {calib_dir} ...")
    with tarfile.open(_IMAGENET_TAR, "r:") as tar:
        members = tar.getmembers()
        if _HAS_TQDM:
            for m in tqdm(members, desc="  extract", unit="file"):
                tar.extract(m, path=calib_dir)
        else:
            total = len(members)
            for i, m in enumerate(members, 1):
                tar.extract(m, path=calib_dir)
                sys.stdout.write(f"\r  Extracting {i}/{total}")
                sys.stdout.flush()
            sys.stdout.write("\n")

    print(f"Dataset ready: {calib_dir}")
    return calib_dir


# Preprocessing: 48x48 center-crop, normalize to [-1, 1]
def preprocess_image(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    target_short = int(round(INPUT_RESOLUTION * 256 / 224))
    ratio = target_short / min(w, h)
    new_w, new_h = int(round(w * ratio)), int(round(h * ratio))
    img = img.resize((new_w, new_h), resample=Image.BILINEAR)
    left = (new_w - INPUT_RESOLUTION) // 2
    top = (new_h - INPUT_RESOLUTION) // 2
    img = img.crop((left, top, left + INPUT_RESOLUTION, top + INPUT_RESOLUTION))
    arr = np.array(img, dtype=np.float32)
    arr = arr / 127.5 - 1.0
    return np.expand_dims(arr, axis=0)


# mcunet repo management
def _ensure_mcunet_repo() -> str:
    if os.path.isdir(os.path.join(MCUNET_LOCAL_REPO, "mcunet")):
        print(f"  mcunet repo already present: {MCUNET_LOCAL_REPO}")
        return MCUNET_LOCAL_REPO

    print(f"  Cloning mcunet repo into {MCUNET_LOCAL_REPO} ...")
    os.makedirs(os.path.dirname(MCUNET_LOCAL_REPO), exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", MCUNET_GIT_URL, MCUNET_LOCAL_REPO],
        check=True,
    )
    return MCUNET_LOCAL_REPO


def _cleanup_mcunet_repo() -> None:
    third_party_dir = os.path.dirname(MCUNET_LOCAL_REPO)
    if os.path.isdir(MCUNET_LOCAL_REPO):
        print(f"  Removing cloned mcunet repo: {MCUNET_LOCAL_REPO}")
        shutil.rmtree(MCUNET_LOCAL_REPO, ignore_errors=True)
    if os.path.isdir(third_party_dir) and not os.listdir(third_party_dir):
        os.rmdir(third_party_dir)
    for mod_name in list(sys.modules.keys()):
        if mod_name == "mcunet" or mod_name.startswith("mcunet."):
            del sys.modules[mod_name]


# Build model: PyTorch -> patch forward -> ONNX
def build_model():
    """Load MCUNet-in0, patch forward() with fused global-avg-pool, export ONNX."""
    import torch

    mcunet_repo = _ensure_mcunet_repo()
    if mcunet_repo not in sys.path:
        sys.path.insert(0, mcunet_repo)

    from mcunet.model_zoo import build_model as mcunet_build

    print("Loading pretrained MCUNet-in0 from MIT HAN Lab...")
    model, resolution, description = mcunet_build(
        net_id=MCUNET_NET_ID, pretrained=True
    )
    model.eval()
    print(f"  Model loaded: {description}")
    print(f"  Resolution: {resolution}x{resolution}, "
          f"Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    def _patched_forward(self, x):
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        if self.feature_mix_layer is not None:
            x = self.feature_mix_layer(x)
        x = x.mean(dim=(2, 3))
        x = self.classifier(x)
        return x

    model.forward = types.MethodType(_patched_forward, model)
    print("  Patched forward(): fused global-avg-pool")

    os.makedirs(MODEL_DIR, exist_ok=True)
    onnx_path = os.path.join(MODEL_DIR, "mcunet_in0.onnx")
    dummy_input = torch.randn(1, 3, resolution, resolution)

    print(f"Exporting to ONNX: {onnx_path}")
    torch.onnx.export(
        model, dummy_input, onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
        dynamic_axes=None,
    )
    print(f"  ONNX model saved ({os.path.getsize(onnx_path)/1e6:.2f} MB)")

    del model
    _cleanup_mcunet_repo()

    return onnx_path, resolution


# TFLite conversion (via onnx2tf)
def _onnx_to_saved_model(onnx_path: str, saved_model_dir: str) -> str:
    import onnx2tf

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


def convert_fp32(onnx_path: str) -> str:
    import tensorflow as tf

    saved_model_dir = os.path.join(MODEL_DIR, "_tf_saved_model_fp32")
    print("\nConverting ONNX -> TFLite FP32 via onnx2tf ...")
    _onnx_to_saved_model(onnx_path, saved_model_dir)

    auto_tflite = os.path.join(saved_model_dir, "model_float32.tflite")
    fp32_path = os.path.join(MODEL_DIR, "mcunet_in0_FP32.tflite")
    if os.path.isfile(auto_tflite):
        shutil.copy2(auto_tflite, fp32_path)
    else:
        print("  Converting SavedModel -> TFLite FP32 ...")
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        with open(fp32_path, "wb") as f:
            f.write(converter.convert())

    print(f"  Saved {fp32_path} ({os.path.getsize(fp32_path)/1e6:.2f} MB)")
    return fp32_path


def convert_int8(onnx_path: str, calib_paths: List[str]) -> str:
    import onnx2tf

    print(f"\nPreparing {len(calib_paths)} calibration images ...")
    calib_npy_path = os.path.join(MODEL_DIR, "_calib_data.npy")
    imgs = []
    iterator = tqdm(calib_paths, desc="  Preprocessing") if _HAS_TQDM else calib_paths
    for i, p in enumerate(iterator, 1):
        if not _HAS_TQDM:
            sys.stdout.write(f"\r  Preprocessing: {i}/{len(calib_paths)}")
            sys.stdout.flush()
        imgs.append(preprocess_image(p)[0])
    if not _HAS_TQDM:
        sys.stdout.write("\n")
    np.save(calib_npy_path, np.stack(imgs).astype(np.float32))
    print(f"  Calibration data saved: {calib_npy_path}")

    _test_npy = "calibration_image_sample_data_20x128x128x3_float32.npy"
    if not os.path.isfile(_test_npy):
        np.save(_test_npy,
                np.random.RandomState(0).rand(20, 128, 128, 3).astype(np.float32))

    out_dir = os.path.join(MODEL_DIR, "_onnx2tf_int8_out")
    print("  Converting ONNX -> TFLite INT8 via onnx2tf ...")
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

    int8_path = os.path.join(MODEL_DIR, "mcunet_in0_INT8.tflite")
    model_stem = os.path.splitext(os.path.basename(onnx_path))[0]
    auto_int8 = os.path.join(out_dir, f"{model_stem}_full_integer_quant.tflite")
    if not os.path.isfile(auto_int8):
        candidates = [f for f in os.listdir(out_dir) if "integer_quant" in f]
        if candidates:
            auto_int8 = os.path.join(out_dir, candidates[0])
        else:
            raise FileNotFoundError(
                f"Expected onnx2tf to produce INT8 tflite but not found. "
                f"Available: {os.listdir(out_dir)}"
            )

    shutil.copy2(auto_int8, int8_path)
    print(f"  Saved {int8_path} ({os.path.getsize(int8_path)/1e6:.2f} MB)")
    return int8_path


def main():
    parser = argparse.ArgumentParser(
        description="Download MCUNet-in0 and convert to TFLite FP32/INT8 "
                    "(MERA-quantizer-compatible variant with fused global-avg-pool)."
    )
    parser.add_argument(
        "--mode", choices=["fp32", "int8", "all", "FP32", "INT8", "ALL"],
        default="all",
        help="Export mode: fp32, int8, or all (default: all).",
    )
    parser.add_argument(
        "--calib-num", type=int, default=1000,
        help="Number of calibration images for INT8 quantization (default: 1000).",
    )
    parser.add_argument(
        "--calib-dir", type=str, default=CALIB_DIR,
        help=f"Path to calibration images directory (default: {CALIB_DIR}).",
    )
    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)
    mode = args.mode.upper()

    print("\n" + "=" * 64)
    print("  MCUNet-in0  ->  TFLite  (fused global-avg-pool variant)")
    print("=" * 64)

    onnx_path = os.path.join(MODEL_DIR, "mcunet_in0.onnx")
    if not os.path.isfile(onnx_path):
        onnx_path, _ = build_model()
    else:
        print(f"ONNX already exists: {onnx_path}")

    calib_paths = None
    if mode in ("INT8", "ALL"):
        calib_dir = ensure_dataset(args.calib_dir)
        calib_paths = list_images(calib_dir, need=args.calib_num)
        if len(calib_paths) < 100:
            raise FileNotFoundError(
                f"Not enough calibration images in {calib_dir} "
                f"(found {len(calib_paths)}, need at least 100)."
            )
        calib_paths = calib_paths[: args.calib_num]
        print(f"\nUsing {len(calib_paths)} calibration images from {calib_dir}")

    if mode in ("FP32", "ALL"):
        convert_fp32(onnx_path)
    if mode in ("INT8", "ALL"):
        convert_int8(onnx_path, calib_paths)

    print("\n" + "=" * 64)
    print("Done. Models saved in:", MODEL_DIR)
    print("=" * 64)


if __name__ == "__main__":
    main()