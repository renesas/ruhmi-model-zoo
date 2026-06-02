# Copyright 2026 Renesas Electronics Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# BlazeFace model from Google MediaPipe, PINTO model zoo conversion:
#   https://github.com/PINTO0309/PINTO_model_zoo/tree/main/030_BlazeFace
# Licensed under the Apache License, Version 2.0.

# ──────────────────────────────────────────────────────────────────────────────
# NOTE: WIDER FACE images are NOT redistributable under a commercial licence.
# The dataset is released for non-commercial research use only.
# You must supply your own calibration images (e.g. from the WIDER FACE
# validation set obtained via http://shuoyang1213.me/WIDERFACE/).
# Pass the directory with --calib-dir when running in INT8 mode.
# ──────────────────────────────────────────────────────────────────────────────
"""
PINTOBlazeFace — Model Download & Conversion
=============================================
Self-contained download and conversion pipeline — no references to any
paths outside this directory (PINTOBlazeFace/).

Downloads the PINTO model zoo BlazeFace resource archive, extracts the
front-face SavedModel, converts it to FP32 and INT8 TFLite, and saves
the BlazeFace front-face anchors (896, 4).

The PINTO archive has a two-level structure:
    resources.tar.gz                               ← outer (Wasabi S3)
      └─ 030_BlazeFace/01_float32/
           └─ resources_new.tar.gz                 ← inner
                └─ saved_model_face_detection_front/  ← what we need

Output (into model/):
    blazeface_front_fp32.tflite   — 128×128 NHWC, 4-output PINTO split layout
    blazeface_front_int8.tflite   — same layout, fully quantized INT8
    anchors.npy                   — (896, 4) BlazeFace front anchors

INT8 calibration requires face images (e.g. WIDER FACE val set).
Pass the directory with --calib-dir when running in INT8 mode.
Obtain the WIDER FACE dataset from http://shuoyang1213.me/WIDERFACE/
(non-commercial research use only).

Usage:
    python3 download_model.py                   # FP32 + INT8 (requires --calib-dir)
    python3 download_model.py --mode fp32       # FP32 only (no calibration needed)
    python3 download_model.py --mode int8 --calib-dir /path/to/WIDER_val/images
    python3 download_model.py --calib-num 100 --calib-dir /path/to/WIDER_val/images
    python3 download_model.py --force-download  # re-download even if cached
"""

import argparse
import glob
import os
import sys
import tarfile
import urllib.request

import numpy as np

# ── All paths relative to THIS file ──────────────────────────────────────────
_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR    = os.path.join(_THIS_DIR, "model")
_DOWNLOAD_DIR = os.path.join(_THIS_DIR, "downloads")

# PINTO model zoo resource archive
_RESOURCES_URL        = (
    "https://s3.ap-northeast-2.wasabisys.com/"
    "pinto-model-zoo/030_BlazeFace/resources.tar.gz"
)
_INNER_TAR_INNER      = "030_BlazeFace/01_float32/resources_new.tar.gz"
_FRONT_SAVEDMODEL_INNER = "saved_model_face_detection_front"

INPUT_SIZE = 128


# ═══════════════════════════════════════════════════════════════════════════════
# Download & extract
# ═══════════════════════════════════════════════════════════════════════════════

def _download_and_extract(force: bool = False) -> str:
    """Download the PINTO resource tarball and extract the front SavedModel.

    The PINTO archive has a two-level structure:
        resources.tar.gz                                  (outer — from S3)
          └─ 030_BlazeFace/01_float32/
               └─ resources_new.tar.gz                   (inner)
                    └─ saved_model_face_detection_front/  ← what we need

    Returns the path to the extracted front-face SavedModel directory.
    Skips any step whose output already exists (unless force=True).
    """
    savedmodel_dir = os.path.join(_DOWNLOAD_DIR, _FRONT_SAVEDMODEL_INNER)

    if os.path.isdir(savedmodel_dir) and not force:
        print(f"[INFO] SavedModel already extracted: {savedmodel_dir}")
        return savedmodel_dir

    os.makedirs(_DOWNLOAD_DIR, exist_ok=True)
    outer_tar = os.path.join(_DOWNLOAD_DIR, "resources.tar.gz")

    # ── Step A: download outer tarball ────────────────────────────────────────
    if not os.path.isfile(outer_tar) or force:
        print(f"[INFO] Downloading PINTO BlazeFace resources ...")
        print(f"       URL  : {_RESOURCES_URL}")
        print(f"       Dest : {outer_tar}")

        from tqdm import tqdm

        class _TqdmProgress:
            """Adapter between urllib reporthook and tqdm."""
            def __init__(self):
                self._bar = None

            def __call__(self, block_num, block_size, total_size):
                if self._bar is None:
                    self._bar = tqdm(
                        total=total_size, unit="B", unit_scale=True,
                        unit_divisor=1024, desc="  resources.tar.gz",
                        dynamic_ncols=True,
                    )
                downloaded = block_num * block_size
                self._bar.n = min(downloaded, total_size)
                self._bar.refresh()
                if downloaded >= total_size:
                    self._bar.close()

        urllib.request.urlretrieve(_RESOURCES_URL, outer_tar, _TqdmProgress())
        size_mb = os.path.getsize(outer_tar) / (1024 * 1024)
        print(f"[INFO] Downloaded ({size_mb:.1f} MB) → {outer_tar}")
    else:
        print(f"[INFO] Outer tarball already present: {outer_tar}")

    # ── Step B: extract outer tarball → produces inner tar ───────────────────
    inner_tar = os.path.join(_DOWNLOAD_DIR, _INNER_TAR_INNER)
    if not os.path.isfile(inner_tar) or force:
        print(f"[INFO] Extracting outer archive ...")
        from tqdm import tqdm
        with tarfile.open(outer_tar, "r:gz") as tar:
            members = tar.getmembers()
            for member in tqdm(members, desc="  outer tar", unit="file",
                               dynamic_ncols=True):
                tar.extract(member, _DOWNLOAD_DIR)
        print(f"[INFO] Outer archive extracted → {_DOWNLOAD_DIR}")
    else:
        print(f"[INFO] Inner tarball already present: {inner_tar}")

    if not os.path.isfile(inner_tar):
        raise RuntimeError(
            f"Inner tarball not found after outer extraction:\n  {inner_tar}"
        )

    # ── Step C: extract inner tarball → saved_model_face_detection_front/ ────
    print(f"[INFO] Extracting inner archive (resources_new.tar.gz) ...")
    from tqdm import tqdm
    with tarfile.open(inner_tar, "r:gz") as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc="  inner tar", unit="file",
                           dynamic_ncols=True):
            tar.extract(member, _DOWNLOAD_DIR)
    print(f"[INFO] Inner archive extracted → {_DOWNLOAD_DIR}")

    if not os.path.isdir(savedmodel_dir):
        raise RuntimeError(
            f"SavedModel not found after inner extraction:\n  {savedmodel_dir}"
        )

    print(f"[INFO] SavedModel ready: {savedmodel_dir}")
    return savedmodel_dir


# ═══════════════════════════════════════════════════════════════════════════════
# Anchor generation
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_anchors_front_128() -> np.ndarray:
    """Generate 896 BlazeFace front-face anchors for 128×128 input.

    Layout:
        Stride-8:  feature map 16×16, 2 anchors per cell = 512
        Stride-16: feature map  8×8,  6 anchors per cell = 384   total = 896
    Each anchor = [cx, cy, 1.0, 1.0] in normalised [0,1] coords.
    """
    anchors = []
    for feat, stride, n in [(16, 8, 2), (8, 16, 6)]:
        for row in range(feat):
            for col in range(feat):
                for _ in range(n):
                    cx = (col + 0.5) * stride / INPUT_SIZE
                    cy = (row + 0.5) * stride / INPUT_SIZE
                    anchors.append([cx, cy, 1.0, 1.0])
    arr = np.array(anchors, dtype=np.float32)
    assert arr.shape == (896, 4), f"Expected (896,4), got {arr.shape}"
    return arr


# ═══════════════════════════════════════════════════════════════════════════════
# FP32 conversion
# ═══════════════════════════════════════════════════════════════════════════════

def convert_fp32(savedmodel_dir: str, output_path: str) -> None:
    """Convert SavedModel → FP32 TFLite (4-output PINTO split layout)."""
    import tensorflow as tf

    print(f"[INFO] Converting FP32 TFLite ...")
    print(f"       Source : {savedmodel_dir}")
    print(f"       Output : {output_path}")

    converter = tf.lite.TFLiteConverter.from_saved_model(savedmodel_dir)
    converter.optimizations = []
    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"[INFO] FP32 TFLite saved ({size_kb:.1f} KB) → {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# INT8 conversion
# ═══════════════════════════════════════════════════════════════════════════════

def _load_calib_images(calib_dir: str, n: int) -> list:
    """Load up to *n* calibration images (JPEG/PNG) from calib_dir."""
    import cv2
    from tqdm import tqdm

    patterns = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG"]
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(calib_dir, "**", pat),
                               recursive=True))
    files = sorted(files)[:n]

    if not files:
        raise FileNotFoundError(
            f"No JPEG/PNG calibration images found under: {calib_dir}\n"
            "Run python3 download_model.py first — it auto-downloads WIDER FACE\n"
            "val, or supply --calib-dir pointing to a folder of JPEG/PNG images."
        )
    print(f"[INFO] Loading {len(files)} calibration images from {calib_dir}")

    images = []
    for path in tqdm(files, desc="  loading calib", unit="img",
                     dynamic_ncols=True):
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        images.append(img)
    return images


def convert_int8(savedmodel_dir: str,
                 output_path: str,
                 calib_dir: str,
                 calib_num: int = 100) -> None:
    """Convert SavedModel → fully quantized INT8 TFLite."""
    import tensorflow as tf
    from tqdm import tqdm

    calib_images = _load_calib_images(calib_dir, calib_num)

    _pbar = tqdm(total=len(calib_images), desc="  calibrating",
                 unit="img", dynamic_ncols=True)

    def representative_dataset():
        for img in calib_images:
            _pbar.update(1)
            yield [img[np.newaxis, ...].astype(np.float32)]  # (1,128,128,3)

    print(f"[INFO] Converting INT8 TFLite ...")
    print(f"       Source      : {savedmodel_dir}")
    print(f"       Output      : {output_path}")
    print(f"       Calib images: {len(calib_images)}")

    converter = tf.lite.TFLiteConverter.from_saved_model(savedmodel_dir)
    converter.optimizations              = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset     = representative_dataset
    converter.target_spec.supported_ops  = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type       = tf.int8
    converter.inference_output_type      = tf.int8

    int8_model = converter.convert()
    _pbar.close()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(int8_model)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"[INFO] INT8 TFLite saved ({size_kb:.1f} KB) → {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Post-conversion verification
# ═══════════════════════════════════════════════════════════════════════════════

def verify_model(model_path: str) -> None:
    """Print input/output tensor details for a TFLite model."""
    import warnings
    import tensorflow as tf

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()

    inp  = interp.get_input_details()[0]
    outs = interp.get_output_details()

    qp = inp["quantization_parameters"]
    sc = f"  scale={qp['scales'][0]:.6f} zp={qp['zero_points'][0]}" \
         if qp["scales"].size else ""
    print(f"  Input:  {list(inp['shape'])} {inp['dtype'].__name__}{sc}")
    for i, o in enumerate(outs):
        qp = o["quantization_parameters"]
        sc = f"  scale={qp['scales'][0]:.6f} zp={qp['zero_points'][0]}" \
             if qp["scales"].size else ""
        print(f"  Out[{i}]: {list(o['shape'])} {o['dtype'].__name__}  {o['name']}{sc}")


# ═══════════════════════════════════════════════════════════════════════════════
# Anchor verification
# ═══════════════════════════════════════════════════════════════════════════════

def verify_anchors(anchors: np.ndarray) -> None:
    """Sanity-check anchor shape and value range."""
    assert anchors.shape == (896, 4), \
        f"Anchor shape error: expected (896,4), got {anchors.shape}"
    assert np.allclose(anchors[:, 2], 1.0) and np.allclose(anchors[:, 3], 1.0), \
        "Anchor w/h values should all be 1.0"
    assert anchors[:, 0].min() > 0 and anchors[:, 0].max() <= 1.0, \
        f"cx out of range: [{anchors[:,0].min():.3f}, {anchors[:,0].max():.3f}]"
    print(f"[INFO] Anchors verified: shape={anchors.shape} ✅")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(
        description="Download and convert PINTO BlazeFace front to FP32/INT8 TFLite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--mode", choices=["fp32", "int8", "both", "FP32", "INT8", "BOTH"], default="both",
        help="Which TFLite models to generate",
    )
    p.add_argument(
        "--calib-dir",
        default=None,
        help="Path to directory containing calibration images (JPEG/PNG with faces). "
             "Required for INT8 mode. Obtain WIDER FACE validation images from "
             "http://shuoyang1213.me/WIDERFACE/ (non-commercial research use only).",
    )
    p.add_argument(
        "--calib-num", type=int, default=100,
        help="Number of calibration images to use for INT8",
    )
    p.add_argument(
        "--out-dir", default=_MODEL_DIR,
        help="Output directory for TFLite models and anchors.npy",
    )
    p.add_argument(
        "--force-download", action="store_true",
        help="Re-download and re-extract even if archive is already cached",
    )
    p.add_argument(
        "--verify", action="store_true", default=True,
        help="Verify model I/O tensors after conversion",
    )
    p.add_argument(
        "--no-verify", dest="verify", action="store_false",
    )
    return p.parse_args()


def main():
    args = _parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    fp32_path = os.path.join(args.out_dir, "blazeface_front_fp32.tflite")
    int8_path = os.path.join(args.out_dir, "blazeface_front_int8.tflite")
    anc_path  = os.path.join(args.out_dir, "anchors.npy")

    # ── Step 1: Download & extract SavedModel ─────────────────────────────────
    savedmodel_dir = _download_and_extract(force=args.force_download)

    # ── Step 2: Anchors ───────────────────────────────────────────────────────
    print("[INFO] Generating front-face anchors (128×128) ...")
    anchors = _generate_anchors_front_128()
    verify_anchors(anchors)
    np.save(anc_path, anchors)
    print(f"[INFO] Anchors saved ({anchors.shape}) → {anc_path}")

    # ── Step 3: FP32 ─────────────────────────────────────────────────────────
    if args.mode in ("fp32", "both", "all"):
        convert_fp32(savedmodel_dir, fp32_path)
        if args.verify:
            print("[INFO] Verifying FP32 model:")
            verify_model(fp32_path)

    # ── Step 4: INT8 ─────────────────────────────────────────────────────────
    if args.mode in ("int8", "both", "all"):
        if not args.calib_dir:
            sys.exit(
                "[ERROR] --calib-dir is required for INT8 mode.\n"
                "        Download the WIDER FACE validation set from "
                "http://shuoyang1213.me/WIDERFACE/ and pass the images directory.\n"
                "        Example: python download_model.py --mode int8 "
                "--calib-dir /path/to/WIDER_val/images"
            )
        calib_dir = args.calib_dir
        if not os.path.isdir(calib_dir):
            sys.exit(
                f"[ERROR] Calibration directory not found: {calib_dir}\n"
                "        Check the path and make sure it contains JPEG/PNG face images."
            )
        convert_int8(savedmodel_dir, int8_path,
                     calib_dir=calib_dir,
                     calib_num=args.calib_num)
        if args.verify:
            print("[INFO] Verifying INT8 model:")
            verify_model(int8_path)

    # ── Done ──────────────────────────────────────────────────────────────────
    print("\n[INFO] Done!")
    if args.mode in ("fp32", "both", "all"):
        print(f"       FP32 : {fp32_path}")
    if args.mode in ("int8", "both", "all"):
        print(f"       INT8 : {int8_path}")
    print(f"       Anch : {anc_path}")


if __name__ == "__main__":
    main()
