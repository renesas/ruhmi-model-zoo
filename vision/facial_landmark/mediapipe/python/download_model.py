#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download face_landmark.tflite and convert it to full INT8.


Pipeline:
1. Download FP32 TFLite from patlevin/face-detection-tflite
2. Quantize FP32 -> full INT8 TFLite using TensorFlow Lite calibrator

Quantization API details:
- Module: `tensorflow.lite.python.optimize.calibrator`
- Class: `calibrator.Calibrator(model_content)`
- Method: `calibrate_and_quantize(...)`
- This script forces full-integer quantization for all inputs, outputs, activations, and biases.
- Representative data is provided via `representative_dataset` and is used to
    calibrate activation ranges before int8 parameters are emitted.

Usage:
    python download_model.py                     # full pipeline (FP32 + INT8)
    python download_model.py --mode fp32         # FP32 TFLite only
    python download_model.py --mode int8         # FP32 + INT8 TFLite
    python download_model.py --calib-dir /my/images
    python download_model.py --calib-num 200
"""

import argparse
import os
import sys
import urllib.request
from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    from tqdm import tqdm

    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False


SCRIPT_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = SCRIPT_DIR / "model"
FP32_PATH = OUTPUT_DIR / "face_landmark_FP32.tflite"
INT8_PATH = OUTPUT_DIR / "face_landmark_INT8.tflite"

MODEL_URL = (
    "https://raw.githubusercontent.com/patlevin/face-detection-tflite/"
    "main/fdlite/data/face_landmark.tflite"
)


def _download_with_progress(url: str, dest: Path) -> None:
    print(f"Downloading {url}\n         -> {dest} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)

    if _HAS_TQDM:
        class _Reporter:
            def __init__(self):
                self.pbar = None

            def __call__(self, block_num, block_size, total_size):
                if self.pbar is None:
                    self.pbar = tqdm(
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        desc="  download",
                    )
                self.pbar.update(min(block_size, max(0, total_size - self.pbar.n)))
                if block_num * block_size >= total_size:
                    self.pbar.close()

        urllib.request.urlretrieve(url, str(dest), reporthook=_Reporter())
    else:
        def _report(block_num, block_size, total_size):
            current = block_num * block_size
            print(f"\r  {current/1e6:.1f} / {total_size/1e6:.1f} MB", end="")

        urllib.request.urlretrieve(url, str(dest), reporthook=_report)
        print()


def download_fp32_model(force: bool = False) -> Path:
    if FP32_PATH.exists() and not force:
        size_mb = FP32_PATH.stat().st_size / (1024 * 1024)
        print(f"[OK] FP32 model already exists: {FP32_PATH} ({size_mb:.2f} MB)")
        return FP32_PATH

    _download_with_progress(MODEL_URL, FP32_PATH)
    size_mb = FP32_PATH.stat().st_size / (1024 * 1024)
    print(f"[OK] Downloaded FP32 model: {FP32_PATH} ({size_mb:.2f} MB)")
    return FP32_PATH


def _normalize_shape(shape):
    return [int(d) if int(d) > 0 else 1 for d in list(shape)]


def _parse_calib_num(value: str) -> Optional[int]:
    v = value.strip().lower()
    if v == "all":
        return None
    try:
        n = int(v)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--calib-num must be a positive integer or 'all'"
        ) from exc
    if n <= 0:
        raise argparse.ArgumentTypeError(
            "--calib-num must be > 0 or 'all'"
        )
    return n


def _list_images(folder: Path, need: Optional[int]) -> List[Path]:
    if not folder.is_dir():
        return []
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    out = []
    for root, _, files in os.walk(folder):
        for name in sorted(files):
            p = Path(root) / name
            if p.suffix.lower() in exts:
                out.append(p)
                if need is not None and len(out) >= need:
                    return out
    return out


def quantize_fp32_to_int8(
    model_path: Path,
    output_path: Path,
    num_samples: Optional[int] = 100,
    calib_dir: Path | None = None,
) -> Path:
    """Convert FP32 TFLite to full INT8 using TF Lite low-level calibrator API.

    API used:
    - `tensorflow.lite.python.optimize.calibrator.Calibrator`
    - `Calibrator.calibrate_and_quantize(...)`

    Why this API:
    - It is compatible across the TensorFlow versions used in this repo.
    - It gives direct control over input/output/activation/bias quantized dtypes.
    """
    try:
        import tensorflow as tf
        from tensorflow.lite.python.optimize import calibrator
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required for TFLite int8 conversion. "
            "Install tensorflow in this environment."
        ) from exc

    print("[step] Quantizing FP32 -> full INT8 TFLite ...")
    model_content = model_path.read_bytes()

    probe = tf.lite.Interpreter(model_content=model_content)
    probe.allocate_tensors()
    input_details = probe.get_input_details()

    calib_samples = None
    random_count = 100 if num_samples is None else max(1, num_samples)
    if calib_dir is not None:
        image_paths = _list_images(calib_dir, None if num_samples is None else max(1, num_samples))
        if image_paths:
            try:
                from PIL import Image
            except ImportError as exc:
                raise RuntimeError(
                    "Pillow is required for --calib-dir image loading. Install pillow."
                ) from exc

            calib_samples = []
            use_paths = image_paths if num_samples is None else image_paths[:num_samples]
            for p in use_paths:
                # Match landmark model preprocessing used in inference: [-1, 1].
                img = Image.open(p).convert("RGB").resize((192, 192), Image.BILINEAR)
                arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
                calib_samples.append(arr[np.newaxis, ...])
            print(f"[step] Using calibration images from {calib_dir} ({len(calib_samples)} samples)")
        else:
            print(f"[warn] No images found in {calib_dir}; using random representative data")

    def representative_dataset():
        rng = np.random.default_rng(0)
        if calib_samples is not None:
            for sample in calib_samples:
                yield [sample]
            return

        for _ in range(random_count):
            batch = []
            for d in input_details:
                shape = _normalize_shape(d["shape"])
                dtype = d["dtype"]
                # The source model expects normalized float-like input.
                if dtype == np.float32:
                    arr = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
                else:
                    arr = np.zeros(shape, dtype=dtype)
                batch.append(arr)
            yield batch

    # Low-level API call that performs both calibration and full INT8 conversion.
    cal = calibrator.Calibrator(model_content)
    quantized = cal.calibrate_and_quantize(
        representative_dataset,
        input_type=tf.int8,
        output_type=tf.int8,
        allow_float=False,
        activations_type=tf.int8,
        bias_type=tf.int32,
        resize_input=True,
        disable_per_channel=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(quantized)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"[OK] Wrote INT8 model: {output_path} ({size_mb:.2f} MB)")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["fp32", "int8", "all"],
        default="all",
        help="Run mode: fp32 (download only), int8 (download + int8), all (same as int8).",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download the source FP32 model even if it already exists.",
    )
    parser.add_argument(
        "--calib-num",
        type=_parse_calib_num,
        default=100,
        help="Representative dataset sample count, or 'all' to use all images from --calib-dir.",
    )
    parser.add_argument(
        "--calib-dir",
        type=str,
        default=None,
        help="Optional directory of calibration images.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    fp32_model = download_fp32_model(force=args.force_download)
    if args.mode in ("int8", "all"):
        calib_dir = None
        if args.calib_dir:
            calib_dir = Path(args.calib_dir).expanduser().resolve()
        quantize_fp32_to_int8(
            fp32_model,
            INT8_PATH,
            num_samples=args.calib_num,
            calib_dir=calib_dir,
        )

    print("[DONE] Pipeline completed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
