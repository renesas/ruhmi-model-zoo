# Copyright 2026 Renesas Electronics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Download & convert PoseNet (MobileNetV1) to TFLite (FP32 + INT8).

Pipeline:
    1. Download Google TFJS PoseNet weights and convert to PyTorch state-dict.
    2. Export PyTorch model -> ONNX (opset 13).
    3. Convert ONNX -> TFLite FP32 via onnx2tf.
    4. Re-quantise to TFLite INT8 using TensorFlow's converter with a
       per-sample representative dataset built from COCO val2017 person crops.

Model: PoseNet MobileNetV1 (Google, original TFJS port)
  - Default variant:  --model 50 --input_size 257   (depth-0.5 backbone)
  - Output stride  :  16 (default) or 8
  - Input range    :  [-1, 1] float32  (normalised RGB)
  - 4 output heads :  heatmap(17) / offset(34) / displacement_fwd(32) / displacement_bwd(32)

Supported variants (see posenet/models/mobilenet_v1.py for the checkpoint map):
  --model         50  | 75 | 100 | 101         (MobileNetV1 depth multiplier)
                  50  = 0.50x channels  (smallest, default; recommended for MCU)
                  75  = 0.75x channels
                  100 = 1.00x channels  (TFJS "mobilenet_v1_100")
                  101 = 1.00x channels  (TFJS "mobilenet_v1_101", alt weights)

  --input_size    Any value satisfying (input_size - 1) %% output_stride == 0.
                  With the default --output_stride 16 the valid presets are
                  129, 145, 161, 177, 193, 209, 225, 241, 257 (default),
                  273, 289, 321, 353, 385, 417, 449, 481, 513.
                  Lower = faster + smaller, higher = more accurate.

  --output_stride 16 (default) or 8. With stride 8 valid input sizes
                  become 8N+1 (129, 137, 145, ..., 513). Stride 8 gives
                  a larger heatmap (more precise keypoints) at higher
                  compute and memory cost.

Example combinations (smallest -> largest):
  --model 50  --input_size 193                      smallest model overall
  --model 50  --input_size 257   [default]          small + fast
  --model 75  --input_size 257                      better accuracy
  --model 100 --input_size 353                      highest accuracy

Usage:
    python download_model.py                                # default: 050 @ 257
    python download_model.py --mode fp32                    # FP32 only
    python download_model.py --mode int8 --calib-num 200    # smaller calibration set
    python download_model.py --model 75 --input_size 193    # different variant
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import posenet                                          # noqa: E402
from utils.progress import spinner                      # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# Locations
# ──────────────────────────────────────────────────────────────────────────────
PRETRAINED_DIR = BASE_DIR / "pretrained"        # PyTorch .pth + .onnx
MODEL_DIR      = BASE_DIR / "model"             # final .tflite
DATASETS_DIR   = BASE_DIR / "Datasets"          # default COCO val2017 location

# onnx2tf requires this file in cwd; we pre-create a stub so its broken
# remote download never fires, if not present its auto created.
ONNX2TF_DUMMY = BASE_DIR / "calibration_image_sample_data_20x128x128x3_float32.npy"


# ──────────────────────────────────────────────────────────────────────────────
# COCO auto-download + calibration helpers
# ──────────────────────────────────────────────────────────────────────────────
COCO_VAL2017_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_VAL2017_ANN_URL    = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def _download_with_progress_bar(url: str, dest: Path) -> None:
    """Stream-download ``url`` to ``dest`` with a tqdm byte bar."""
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)

    pbar = None
    def _report(block_num, block_size, total_size):
        nonlocal pbar
        if pbar is None:
            pbar = tqdm(total=total_size, unit="B", unit_scale=True,
                        unit_divisor=1024, desc="      download",
                        ncols=80, leave=False)
        downloaded = min(block_num * block_size, total_size)
        pbar.update(downloaded - pbar.n)
        if total_size > 0 and downloaded >= total_size:
            pbar.close()

    print(f"      URL  : {url}")
    print(f"      Dest : {dest}")
    urllib.request.urlretrieve(url, dest, reporthook=_report)


def _download_and_extract_zip(url: str, extract_to: Path, label: str) -> None:
    """Download ``url`` as a zip into ``extract_to`` and extract in place."""
    import zipfile
    extract_to.mkdir(parents=True, exist_ok=True)
    zip_path = extract_to / Path(url).name
    _download_with_progress_bar(url, zip_path)
    with spinner(f"Extracting {label}"):
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_to)
    zip_path.unlink()


def ensure_coco_val2017(calib_dir: Path = DATASETS_DIR) -> Tuple[Path, Path]:
    """Ensure COCO val2017 images + person_keypoints json are present.
    
    Parameters
    ----------
    calib_dir : Path
        Root directory containing ``val2017/`` and ``annotations/``.
        Pass any pre-existing COCO root to skip the download.

    Returns
    -------
    img_dir, ann_file : Path
    """
    calib_dir = Path(calib_dir)
    img_dir   = calib_dir / "val2017"
    ann_file  = calib_dir / "annotations" / "person_keypoints_val2017.json"

    # ── Images (~780 MB) ─────────────────────────────────────────────────────
    have_imgs = img_dir.is_dir() and any(img_dir.glob("*.jpg"))
    if not have_imgs:
        print(f"      COCO val2017 images not found at {img_dir}")
        print( "      Downloading COCO val2017 images (~780 MB)…")
        _download_and_extract_zip(COCO_VAL2017_IMAGES_URL, calib_dir,
                                  "COCO val2017 images")
    n_img = sum(1 for _ in img_dir.glob("*.jpg"))
    print(f"      val2017 images : {img_dir}  ({n_img} images)")

    # ── Annotations (~241 MB; same zip contains both instances and keypoints)
    if not ann_file.is_file():
        print(f"      COCO keypoints annotations not found at {ann_file}")
        print( "      Downloading COCO annotations (~241 MB)…")
        _download_and_extract_zip(COCO_VAL2017_ANN_URL, calib_dir,
                                  "COCO annotations")
    if not ann_file.is_file():
        raise FileNotFoundError(
            f"person_keypoints_val2017.json not found at {ann_file} "
            f"after download attempt.")
    print(f"      keypoints ann  : {ann_file}")

    return img_dir, ann_file


def list_person_rich_images(num_images: int,
                            min_visible_kp: int = 8,
                            calib_dir: Path = DATASETS_DIR) -> List[Path]:
    """Return paths of the most person-rich COCO val2017 images.

    Selects images that contain at least one annotated person with
    ``min_visible_kp`` or more visible keypoints, ranked by best keypoint
    count per image.

    Auto-downloads COCO val2017 into ``calib_dir`` if not present.
    """
    img_dir, ann_file = ensure_coco_val2017(calib_dir)

    with spinner(f"Loading COCO annotations ({ann_file.name})"):
        with open(ann_file) as f:
            ann = json.load(f)

    with spinner(f"Ranking {len(ann['annotations']):,} annotations"):
        per_img: dict = {}
        for a in ann["annotations"]:
            if a.get("num_keypoints", 0) < min_visible_kp:
                continue
            per_img.setdefault(a["image_id"], 0)
            per_img[a["image_id"]] = max(per_img[a["image_id"]], a["num_keypoints"])
        id2file = {im["id"]: im["file_name"] for im in ann["images"]}
        ranked  = sorted(per_img.items(), key=lambda kv: -kv[1])

    chosen: List[Path] = []
    for img_id, _ in ranked:
        p = img_dir / id2file[img_id]
        if p.exists():
            chosen.append(p)
            if len(chosen) >= num_images:
                break
    if len(chosen) < num_images:
        raise SystemExit(
            f"Only found {len(chosen)} qualifying images (asked for {num_images}).")
    print(f"      Picked {len(chosen)} person-rich images (>={min_visible_kp} visible kp)")
    return chosen


def preprocess_image(image_path, input_size: int) -> np.ndarray:
    """Read+resize+normalise one image → (1, input_size, input_size, 3) float32.

    NHWC layout, values in [-1, 1]. Used by both the TFLite converter
    (representative_dataset) and the MERA quantiser.

    Algorithm: 2-tap separable bilinear with half-pixel centres,
    then normalised via ``x * (2/255) - 1`` to ``[-1, 1]``.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise IOError(f"cannot read {image_path}")
    # Cast to float32 BEFORE cv2.resize so cv2 uses pure 2-tap float bilinear
    # (its uint8 fast-path uses INT11 fixed-point arithmetic, which would
    # produce slightly different rounded values).
    img = img.astype(np.float32)
    img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img * (2.0 / 255.0) - 1.0
    return np.expand_dims(img, axis=0)


def make_representative_dataset(calib_paths: List[Path], input_size: int):
    """Lazy generator for ``TFLiteConverter.representative_dataset``.

    TFLiteConverter iterates the generator multiple times (one pass for
    statistics, one for refinement), so we restart the tqdm bar on each pass.
    Images are read fresh from disk on every pass.
    """
    pass_n = 0

    def gen():
        nonlocal pass_n
        pass_n += 1
        pbar = tqdm(calib_paths,
                    desc=f"      TFLite calib pass #{pass_n}",
                    unit="img", ncols=80, leave=False)
        for p in pbar:
            yield [preprocess_image(p, input_size)]
        pbar.close()
    return gen


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 — pretrained TFJS -> PyTorch
# ──────────────────────────────────────────────────────────────────────────────
def download_pytorch(model_id, output_stride):
    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n[1/4] Pretrained weights ({posenet.MOBILENET_V1_CHECKPOINTS[model_id]})")
    with spinner("Loading PoseNet MobileNetV1 (auto-downloads on first run)"):
        model = posenet.load_model(model_id=model_id,
                                   output_stride=output_stride,
                                   model_dir=str(PRETRAINED_DIR))
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 — PyTorch -> ONNX (with a parity check)
# ──────────────────────────────────────────────────────────────────────────────
def export_onnx(model, model_id, input_size) -> Path:
    onnx_path = PRETRAINED_DIR / f"posenet_mbv1_{model_id:03d}_{input_size}.onnx"
    print(f"\n[2/4] Export ONNX  ->  {onnx_path}")

    dummy = torch.zeros(1, 3, input_size, input_size, dtype=torch.float32)
    with spinner("torch.onnx.export"):
        torch.onnx.export(
            model, dummy, str(onnx_path),
            input_names=["input"],
            output_names=["heatmap", "offset", "displacement_fwd", "displacement_bwd"],
            opset_version=13,
            do_constant_folding=True,
            dynamic_axes=None,
        )

    try:
        import onnxruntime as ort
        with spinner("ONNX vs PyTorch parity check"):
            sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            with torch.no_grad():
                torch_out = model(dummy)
            onnx_out = sess.run(None, {"input": dummy.numpy()})
            diffs = [(name, float(np.abs(t.numpy() - o).max()))
                     for name, t, o in zip(["heatmap", "offset", "disp_fwd", "disp_bwd"],
                                           torch_out, onnx_out)]
        for name, diff in diffs:
            print(f"      ONNX vs PyTorch {name:<10s} max|diff| = {diff:.3e}")
    except ImportError:
        print("      (onnxruntime not installed, skipping parity check)")
    return onnx_path


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 — ONNX -> TFLite FP32 via onnx2tf
# ──────────────────────────────────────────────────────────────────────────────
def onnx_to_tflite_fp32(onnx_path, model_id, input_size):
    work_dir = PRETRAINED_DIR / f"_onnx2tf_{model_id:03d}_{input_size}"
    print(f"\n[3/4] ONNX -> TFLite FP32  ({work_dir})")

    if work_dir.exists():
        shutil.rmtree(work_dir)

    if not ONNX2TF_DUMMY.exists():
        np.save(ONNX2TF_DUMMY, np.zeros((20, 128, 128, 3), dtype=np.float32))

    cmd = [
        sys.executable, "-m", "onnx2tf",
        "-i", str(onnx_path),
        "-o", str(work_dir),
        "-nuo",                                  # skip unused onnx-graphsurgeon ops
    ]
    print("      " + " ".join(cmd))
    with spinner("onnx2tf converting"):
        env = os.environ.copy()
        env["TF_CPP_MIN_LOG_LEVEL"] = "3"
        res = subprocess.run(cmd, capture_output=True, text=True,
                             cwd=str(BASE_DIR), env=env)
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr, file=sys.stderr)
        raise SystemExit("onnx2tf failed")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    src = work_dir / f"posenet_mbv1_{model_id:03d}_{input_size}_float32.tflite"
    assert src.exists(), f"onnx2tf did not produce {src.name}"
    fp32_dst = MODEL_DIR / f"posenet_mbv1_{model_id:03d}_{input_size}_FP32.tflite"
    shutil.copy2(src, fp32_dst)
    print(f"      -> {fp32_dst}  ({fp32_dst.stat().st_size/1024:.1f} KB)")

    return work_dir


# ──────────────────────────────────────────────────────────────────────────────
# Step 4 — INT8 quantisation via TFLite converter
# ──────────────────────────────────────────────────────────────────────────────
def quantise_int8(work_dir, model_id, input_size, calib_num,
                  calib_dir: Path = DATASETS_DIR) -> Path:
    print(f"\n[4/4] Quantise -> TFLite INT8  (calib_num={calib_num})")

    saved_model_dir = work_dir                 # onnx2tf wrote a TF SavedModel here
    assert (saved_model_dir / "saved_model.pb").exists(), \
        "no SavedModel from onnx2tf — re-run step 3?"

    calib_paths = list_person_rich_images(calib_num, calib_dir=calib_dir)

    # Silence TF before importing
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = make_representative_dataset(calib_paths, input_size)
    converter.inference_input_type   = tf.int8
    converter.inference_output_type  = tf.int8
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    with spinner("TFLite full-INT8 converter (MLIR quantisation passes)"):
        tfl = converter.convert()

    int8_dst = MODEL_DIR / f"posenet_mbv1_{model_id:03d}_{input_size}_INT8.tflite"
    int8_dst.write_bytes(tfl)
    print(f"      -> {int8_dst}  ({int8_dst.stat().st_size/1024:.1f} KB)")

    with spinner("Verifying INT8 TFLite I/O"):
        it = tf.lite.Interpreter(model_path=str(int8_dst))
        it.allocate_tensors()
        inp = it.get_input_details()[0]
    print(f"      Final I/O    : in={inp['dtype'].__name__} shape={list(inp['shape'])}")
    return int8_dst


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model", type=int, default=50, choices=[50, 75, 100, 101],
                   help="MobileNetV1 depth multiplier")
    p.add_argument("--input_size", type=int, default=257,
                   help="Square input size (must satisfy (size-1) %% output_stride == 0)")
    p.add_argument("--output_stride", type=int, default=16, choices=[8, 16],
                   help="PoseNet output stride (8 or 16)")
    p.add_argument("--mode", choices=["all", "fp32", "int8"], default="all")
    p.add_argument("--calib-num", type=int, default=1000,
                   help="Number of COCO val2017 person-rich images used to "
                        "calibrate INT8.")
    p.add_argument("--calib-dir", type=Path, default=DATASETS_DIR,
                   help="Directory containing COCO val2017/ and annotations/. "
                        "If missing, COCO val2017 is auto-downloaded here from "
                        "cocodataset.org (~1 GB total). Pass an existing COCO "
                        "root to skip the download.")
    args = p.parse_args()

    assert (args.input_size - 1) % args.output_stride == 0, (
        f"input_size-1 ({args.input_size-1}) must divide output_stride ({args.output_stride})")

    # Steps 1-3 are always run; step 4 reuses the SavedModel that onnx2tf
    # writes in step 3, so we run step 3 unconditionally.
    model = download_pytorch(args.model, args.output_stride)
    onnx_path = export_onnx(model, args.model, args.input_size)
    work_dir = onnx_to_tflite_fp32(onnx_path, args.model, args.input_size)

    if args.mode in ("all", "int8"):
        quantise_int8(work_dir, args.model, args.input_size,
                      args.calib_num, calib_dir=args.calib_dir)

    print("\nDone. Models in:", MODEL_DIR)
    for path in sorted(MODEL_DIR.glob("*.tflite")):
        print(f"  {path.name:<60s} {path.stat().st_size/1024:8.1f} KB")


if __name__ == "__main__":
    main()
