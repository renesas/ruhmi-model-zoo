# Copyright 2026 Renesas Electronics Corporation
# SPDX-License-Identifier: Apache-2.0
#
# The pre-trained VWW model (vww_96.h5) is from the MLCommons Tiny benchmark:
#   https://github.com/mlcommons/tiny
# Licensed under the Apache License, Version 2.0.
#
# Calibration dataset: COCO val2014 (person/notperson subset).
# COCO images are made available under a Creative Commons Attribution 4.0
# License. Annotations are under a Creative Commons Attribution 4.0 License.
# See https://cocodataset.org/#termsofuse for details.
# The script will automatically download COCO val2014 if not present.
"""
Download the Visual Wake Words (VWW) pre-trained model from MLCommons Tiny and
convert it to TFLite (FP32 and/or INT8).

The H5 model is downloaded automatically from:
  https://github.com/mlcommons/tiny/blob/master/benchmark/training/visual_wake_words/trained_models/vww_96.h5

INT8 quantization uses the VWW calibration dataset (person + notperson images
derived from COCO val2014). The script will automatically:
  1. Check if the VWW dataset already exists  → use it
  2. If not, check if COCO val2014 images + annotations exist → create VWW subset
  3. If val2014 is missing too → download it, then create the VWW subset

All downloaded data and the derived VWW dataset are kept in datasets/.

Outputs (in model/):
  vww_96_FP32.tflite  — float32 model
  vww_96_INT8.tflite  — fully int8 quantised model (input/output int8)

Usage:
  python download_model.py                            # FP32 + INT8
  python download_model.py --mode fp32                # FP32 only
  python download_model.py --mode int8                # INT8 only
  python download_model.py --calib-num 500            # fewer calibration images
  python download_model.py --calib-dir datasets/vww_coco_val2014
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile

import h5py
import numpy as np

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR   = os.path.join(SCRIPT_DIR, "model")
FP32_OUT     = os.path.join(OUTPUT_DIR, "vww_96_FP32.tflite")
INT8_OUT     = os.path.join(OUTPUT_DIR, "vww_96_INT8.tflite")

DATASETS_DIR       = os.path.join(SCRIPT_DIR, "datasets")
DEFAULT_VWW_DIR    = os.path.join(DATASETS_DIR, "vww_coco_val2014")
DEFAULT_VAL2014    = os.path.join(DATASETS_DIR, "val2014")
DEFAULT_ANNOTATIONS = os.path.join(DATASETS_DIR, "annotations", "instances_val2014.json")

MODEL_URL      = (
    "https://github.com/mlcommons/tiny/raw/refs/heads/master/"
    "benchmark/training/visual_wake_words/trained_models/vww_96.h5"
)
COCO_IMAGES_URL = "http://images.cocodataset.org/zips/val2014.zip"
COCO_ANN_URL    = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"

INPUT_SIZE     = 96
MIN_AREA_RATIO = 0.005   # minimum bbox-area / image-area to count as "person"


# ──────────────────────────────────────────────────────────────────────────────
# Download helper
# ──────────────────────────────────────────────────────────────────────────────
def _download_with_progress(url: str, dest: str) -> None:
    print(f"Downloading {url}\n         → {dest} ...")
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


def _download_and_extract_zip(url: str, extract_to: str) -> None:
    os.makedirs(extract_to, exist_ok=True)
    zip_path = os.path.join(extract_to, os.path.basename(url))
    if not os.path.isfile(zip_path):
        _download_with_progress(url, zip_path)
    else:
        print(f"  Zip already present: {zip_path}")
    print(f"  Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print(f"  Extraction complete.")


# ──────────────────────────────────────────────────────────────────────────────
# Model download + Keras 3 patch
# ──────────────────────────────────────────────────────────────────────────────
def download_h5(dest_dir: str) -> str:
    """Download vww_96.h5 from MLCommons Tiny (Apache 2.0) if not present."""
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, "vww_96.h5")
    if os.path.isfile(dest):
        size_mb = os.path.getsize(dest) / 1e6
        print(f"  Model already exists: {dest} ({size_mb:.1f} MB) — skipping.")
        return dest
    _download_with_progress(MODEL_URL, dest)
    print(f"  Saved → {dest} ({os.path.getsize(dest)/1e6:.1f} MB)")
    return dest


def patch_h5(src_path: str) -> str:
    """
    Patch the H5 for Keras 3 compatibility.
    Older Keras saved 'groups' in DepthwiseConv2D config; Keras 3.x rejects it.
    Returns path to a patched temp copy.
    """
    patched = os.path.join(tempfile.gettempdir(), "vww_96_patched.h5")
    shutil.copy2(src_path, patched)
    with h5py.File(patched, "r+") as f:
        raw = f.attrs["model_config"]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        config = json.loads(raw)

        def _strip(obj):
            if isinstance(obj, dict):
                if obj.get("class_name") == "DepthwiseConv2D":
                    obj.get("config", {}).pop("groups", None)
                for v in obj.values():
                    _strip(v)
            elif isinstance(obj, list):
                for v in obj:
                    _strip(v)

        _strip(config)
        f.attrs["model_config"] = json.dumps(config)
    print(f"  Patched H5 (Keras 3 compat) → {patched}")
    return patched


# ──────────────────────────────────────────────────────────────────────────────
# VWW dataset (derived from COCO val2014, CC-BY 4.0)
# ──────────────────────────────────────────────────────────────────────────────
def _count_images(d: str) -> int:
    if not os.path.isdir(d):
        return 0
    return sum(1 for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png")))


def _find_annotations(val2014_dir: str):
    candidates = [
        DEFAULT_ANNOTATIONS,
        os.path.join(val2014_dir, "..", "annotations", "instances_val2014.json"),
    ]
    for c in candidates:
        p = os.path.normpath(c)
        if os.path.isfile(p):
            return p
    return None


def _build_vww_subset(val2014_dir: str, ann_path: str, output_dir: str) -> str:
    """
    Split COCO val2014 into person/ and notperson/ subdirectories.
    Images are symlinked/copied at original resolution.
    """
    print(f"  Loading annotations from {ann_path} ...")
    with open(ann_path) as f:
        coco = json.load(f)
    print(f"    {len(coco['images'])} images, {len(coco['annotations'])} annotations")

    person_cat_ids = {c["id"] for c in coco["categories"] if c["name"] == "person"}
    img_dims = {img["id"]: (img["width"], img["height"]) for img in coco["images"]}

    person_ids = set()
    for ann in coco["annotations"]:
        if ann["category_id"] not in person_cat_ids or ann.get("iscrowd", 0):
            continue
        iid = ann["image_id"]
        if iid in person_ids:
            continue
        w, h = img_dims.get(iid, (1, 1))
        bbox = ann["bbox"]
        if (bbox[2] * bbox[3]) / (w * h) >= MIN_AREA_RATIO:
            person_ids.add(iid)

    person_dir    = os.path.join(output_dir, "person")
    notperson_dir = os.path.join(output_dir, "notperson")
    os.makedirs(person_dir, exist_ok=True)
    os.makedirs(notperson_dir, exist_ok=True)

    n_p = n_np = n_skip = 0
    items = list(enumerate(coco["images"]))
    iterator = tqdm(items, desc="  creating VWW subset", unit="img") if _HAS_TQDM else items
    for idx, img_info in iterator:
        src = os.path.join(val2014_dir, img_info["file_name"])
        if not os.path.isfile(src):
            n_skip += 1
            continue
        if img_info["id"] in person_ids:
            shutil.copy2(src, os.path.join(person_dir, img_info["file_name"]))
            n_p += 1
        else:
            shutil.copy2(src, os.path.join(notperson_dir, img_info["file_name"]))
            n_np += 1
        if not _HAS_TQDM and (idx + 1) % 2000 == 0:
            print(f"    {idx+1}/{len(items)} ...")

    with open(os.path.join(output_dir, "labels.txt"), "w") as f:
        f.write("notperson\nperson\n")

    print(f"  VWW subset created: {n_p} person, {n_np} notperson, {n_skip} skipped")
    return output_dir


def ensure_vww_dataset(vww_dir: str = DEFAULT_VWW_DIR,
                       val2014_dir: str = DEFAULT_VAL2014) -> str:
    """
    Ensure the VWW calibration dataset (person/ + notperson/) is ready.

    The dataset is derived from COCO val2014, which is released under a
    Creative Commons Attribution 4.0 License (cocodataset.org/#termsofuse).
    The script will download COCO val2014 automatically if it is not present.

    Priority:
      1. VWW subset already present  → use it
      2. COCO val2014 images + annotations already present  → create subset
      3. Nothing present  → download COCO val2014 → create subset
    """
    n_person    = _count_images(os.path.join(vww_dir, "person"))
    n_notperson = _count_images(os.path.join(vww_dir, "notperson"))
    if n_person > 0 and n_notperson > 0:
        print(f"  VWW dataset found: {vww_dir}")
        print(f"    person: {n_person}, notperson: {n_notperson}")
        return vww_dir

    print(f"  VWW dataset not found at {vww_dir}")

    # Ensure COCO val2014 images
    if _count_images(val2014_dir) == 0:
        print(f"  COCO val2014 images not found — downloading (~13 GB) ...")
        _download_and_extract_zip(COCO_IMAGES_URL, DATASETS_DIR)
        val2014_dir = os.path.join(DATASETS_DIR, "val2014")

    # Ensure COCO annotations
    ann_path = _find_annotations(val2014_dir)
    if ann_path is None:
        print(f"  COCO annotations not found — downloading ...")
        _download_and_extract_zip(COCO_ANN_URL, DATASETS_DIR)
        ann_path = os.path.join(DATASETS_DIR, "annotations", "instances_val2014.json")
        if not os.path.isfile(ann_path):
            sys.exit(f"[ERROR] Expected annotations at {ann_path} after extraction.")

    print(f"  Using annotations: {ann_path}")
    print(f"  Creating VWW subset at {vww_dir} ...")
    _build_vww_subset(val2014_dir, ann_path, vww_dir)
    return vww_dir


# ──────────────────────────────────────────────────────────────────────────────
# Calibration dataset generator
# ──────────────────────────────────────────────────────────────────────────────
def build_representative_dataset(vww_dir: str, num_samples: int, img_size: int):
    """
    Return a generator function that yields float32 [0, 1] images from the
    VWW dataset, balanced across person/ and notperson/.
    """
    from PIL import Image as PILImage

    def _collect(d: str, n: int):
        paths = sorted(
            os.path.join(d, f)
            for f in os.listdir(d)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        step = max(1, len(paths) // n)
        return paths[::step][:n]

    half     = num_samples // 2
    selected = (
        _collect(os.path.join(vww_dir, "person"),    half) +
        _collect(os.path.join(vww_dir, "notperson"), num_samples - half)
    )
    print(f"  Calibration: {len(selected)} images ({half} person + {num_samples - half} notperson)")

    def gen():
        items = tqdm(selected, desc="  calibrating", unit="img") if _HAS_TQDM else selected
        for p in items:
            img = PILImage.open(p).convert("RGB").resize((img_size, img_size), PILImage.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            yield [arr[np.newaxis, ...]]

    return gen


# ──────────────────────────────────────────────────────────────────────────────
# TFLite conversion
# ──────────────────────────────────────────────────────────────────────────────
def convert_fp32(model, output_path: str) -> None:
    import tensorflow as tf
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    buf = converter.convert()
    with open(output_path, "wb") as f:
        f.write(buf)
    print(f"  FP32 TFLite saved: {output_path}  ({len(buf)/1024:.1f} KB)")


def convert_int8(model, output_path: str, rep_gen) -> None:
    import tensorflow as tf
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8
    buf = converter.convert()
    with open(output_path, "wb") as f:
        f.write(buf)
    print(f"  INT8 TFLite saved: {output_path}  ({len(buf)/1024:.1f} KB)")


def verify_tflite(path: str, tag: str) -> None:
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    print(f"  [{tag}] input  → shape={list(inp['shape'])}, dtype={inp['dtype'].__name__}")
    print(f"  [{tag}] output → shape={list(out['shape'])}, dtype={out['dtype'].__name__}")
    if inp["dtype"] != np.float32:
        s, z = inp["quantization"]
        print(f"  [{tag}] input  quant: scale={s:.6f}, zero_point={z}")
    if out["dtype"] != np.float32:
        s, z = out["quantization"]
        print(f"  [{tag}] output quant: scale={s:.6f}, zero_point={z}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Download VWW model from MLCommons Tiny & convert to TFLite."
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
        default=DEFAULT_VWW_DIR,
        help=f"Path to VWW calibration dataset (person/ + notperson/). "
             f"Auto-created from COCO val2014 if not found (default: {DEFAULT_VWW_DIR}).",
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    mode = args.mode

    # ── Download + patch H5 ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Downloading VWW model (H5) from MLCommons Tiny")
    print("=" * 60)
    h5_path     = download_h5(OUTPUT_DIR)
    patched_h5  = patch_h5(h5_path)

    # ── Load Keras model ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Loading Keras model")
    print("=" * 60)
    import tensorflow as tf
    model = tf.keras.models.load_model(patched_h5)
    print(f"  Input  shape : {model.input_shape}")
    print(f"  Output shape : {model.output_shape}")
    print(f"  Parameters   : {model.count_params():,}")

    # ── Ensure VWW calibration dataset ────────────────────────────────────────
    if mode in ("int8", "both"):
        print("\n" + "=" * 60)
        print("Ensuring VWW calibration dataset (COCO val2014, CC-BY 4.0)")
        print("=" * 60)
        vww_dir = ensure_vww_dataset(args.calib_dir, DEFAULT_VAL2014)

    # ── FP32 conversion ───────────────────────────────────────────────────────
    if mode in ("fp32", "both"):
        print("\n" + "=" * 60)
        print("Converting to FP32 TFLite")
        print("=" * 60)
        convert_fp32(model, FP32_OUT)
        verify_tflite(FP32_OUT, "FP32")

    # ── INT8 conversion ───────────────────────────────────────────────────────
    if mode in ("int8", "both"):
        print("\n" + "=" * 60)
        print("Converting to INT8 TFLite (calibrating with VWW dataset)")
        print("=" * 60)
        rep_gen = build_representative_dataset(vww_dir, args.calib_num, INPUT_SIZE)
        convert_int8(model, INT8_OUT, rep_gen)
        verify_tflite(INT8_OUT, "INT8")

    # ── Cleanup + summary ─────────────────────────────────────────────────────
    if os.path.exists(patched_h5):
        os.remove(patched_h5)

    print("\n" + "=" * 60)
    print("All done!")
    if mode in ("fp32", "both"):
        print(f"  FP32 : {FP32_OUT}")
    if mode in ("int8", "both"):
        print(f"  INT8 : {INT8_OUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
