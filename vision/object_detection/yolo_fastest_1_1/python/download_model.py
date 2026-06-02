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
# Yolo-Fastest model files are from:
#   https://github.com/dog-qiuqiu/Yolo-Fastest
# Licensed under the MIT License.
"""
Download Yolo-Fastest model files and convert to TFLite FP32 / INT8.

Pipeline
--------
  1. Download .cfg / .weights / .names from GitHub (dog-qiuqiu/Yolo-Fastest)
  2. Darknet → ONNX FP32  (delegates to convert_to_onnx.py in the same directory)
  3. ONNX FP32 → TFLite FP32  (via onnx2tf, handles NCHW→NHWC automatically)
  4. ONNX FP32 → TFLite INT8  (via onnx2tf with representative dataset,
                                full-integer quantization calibrated on COCO val2017)

All outputs land in model/ next to this script.

Output models
-------------
  model/yolo_fastest_1.1.tflite       — FP32 TFLite (float32 NHWC I/O)
  model/yolo_fastest_1.1_int8.tflite  — INT8 TFLite (int8 NHWC I/O, full-integer)

Dependencies
------------
    pip install onnx2tf tensorflow  (onnx2tf handles NCHW→NHWC transpose internally)

Usage
-----
    python download_model.py                      # full pipeline (FP32 + INT8), base model
    python download_model.py --variant xl         # XL model, full pipeline
    python download_model.py --mode fp32          # FP32 TFLite only
    python download_model.py --mode int8          # FP32 TFLite + INT8 TFLite
    python download_model.py --mode all           # same as int8 (default)
    python download_model.py --skip-convert       # Darknet files only, no conversion
    python download_model.py --force              # re-download / re-convert even if files exist
    python download_model.py --calib-num 200      # number of calibration images (default 100)
"""

import argparse
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
HERE         = Path(__file__).resolve().parent
MODEL_DIR    = HERE / "model"

# Per-variant configuration
VARIANTS = {
    "base": {
        "cfg":          "yolo-fastest-1.1.cfg",
        "weights":      "yolo-fastest-1.1.weights",
        "names":        "coco.names",
        "onnx_fp32":    "yolo_fastest_1.1.onnx",
        "tflite_fp32":  "yolo_fastest_1.1.tflite",
        "tflite_int8":  "yolo_fastest_1.1_int8.tflite",
        "remote_dir":   "yolo-fastest-1.1_coco",
    },
    "xl": {
        "cfg":          "yolo-fastest-1.1-xl.cfg",
        "weights":      "yolo-fastest-1.1-xl.weights",
        "names":        "coco.names",
        "onnx_fp32":    "yolo_fastest_1.1_xl.onnx",
        "tflite_fp32":  "yolo_fastest_1.1_xl.tflite",
        "tflite_int8":  "yolo_fastest_1.1_xl_int8.tflite",
        "remote_dir":   "yolo-fastest-1.1_coco",
    },
}

# Resolved at runtime (defaults to base); functions use these module-level vars
ONNX_FP32_PATH   = HERE / "model" / VARIANTS["base"]["onnx_fp32"]
TFLITE_FP32_PATH = MODEL_DIR / VARIANTS["base"]["tflite_fp32"]
TFLITE_INT8_PATH = MODEL_DIR / VARIANTS["base"]["tflite_int8"]
# onnx2tf places its saved_model output here temporarily during conversion
_ONNX2TF_SAVED_MODEL = MODEL_DIR / "_onnx2tf_saved_model"

# Name of the dummy test-image npy that onnx2tf downloads for verification.
# We generate it locally to avoid network failures inside dev containers.
_TEST_NPY = "calibration_image_sample_data_20x128x128x3_float32.npy"

INPUT_H = INPUT_W = 320
DEFAULT_CALIB_NUM = 100

# Calibration images — stored inside the running directory
DATASETS_DIR = HERE / "Datasets"
CALIB_DIR    = DATASETS_DIR / "val2017"

# COCO download URLs
COCO_VAL2017_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_VAL2017_ANN_URL    = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

_GITHUB_RAW = (
    "https://raw.githubusercontent.com/dog-qiuqiu/Yolo-Fastest/master/ModelZoo"
)


# ──────────────────────────────────────────────────────────────────────────────
# 1. DOWNLOAD HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def _progress_hook(count, block_size, total_size):
    """Simple ASCII progress bar."""
    if total_size <= 0:
        return
    pct  = min(count * block_size / total_size * 100, 100)
    done = int(pct / 5)
    bar  = "█" * done + "░" * (20 - done)
    print(f"\r  [{bar}] {pct:5.1f}%", end="", flush=True)


def download_file(url: str, dst: Path, force: bool = False) -> None:
    """Download *url* to *dst* with a progress bar.

    Parameters
    ----------
    url   : str   Full URL to download.
    dst   : Path  Destination file path.
    force : bool  Re-download even if *dst* already exists.
    """
    if dst.exists() and not force:
        size_mb = dst.stat().st_size / 1e6
        print(f"  [SKIP] Already exists: {dst.name}  ({size_mb:.1f} MB)")
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {dst.name} …")
    try:
        urllib.request.urlretrieve(url, dst, reporthook=_progress_hook)
        print()
        size_mb = dst.stat().st_size / 1e6
        print(f"  ✅ Saved: {dst.name}  ({size_mb:.1f} MB)")
    except Exception as exc:
        print()
        if dst.exists():
            dst.unlink()
        raise RuntimeError(f"Download failed for {url}:\n  {exc}") from exc


# ──────────────────────────────────────────────────────────────────────────────
# 2. ONNX FP32  (delegate to convert_to_onnx.py in the same directory)
# ──────────────────────────────────────────────────────────────────────────────
def ensure_onnx_fp32(force: bool = False) -> None:
    """Produce the FP32 ONNX model if not already present.

    Delegates to ``convert_to_onnx.py`` which reads the Darknet files from
    ``model/`` and writes ``model/yolo_fastest_1.1.onnx``.

    Parameters
    ----------
    force : bool
        Reconvert even if the ONNX already exists.
    """
    if ONNX_FP32_PATH.exists() and not force:
        size_mb = ONNX_FP32_PATH.stat().st_size / 1e6
        print(f"  [SKIP] FP32 ONNX already exists: {ONNX_FP32_PATH.name}"
              f"  ({size_mb:.1f} MB)")
        return

    # Identify which variant is active by matching ONNX_FP32_PATH
    vcfg = next(
        (v for v in VARIANTS.values()
         if (HERE / "model" / v["onnx_fp32"]) == ONNX_FP32_PATH),
        VARIANTS["base"],
    )
    cfg     = HERE / "model" / vcfg["cfg"]
    weights = HERE / "model" / vcfg["weights"]
    convert = HERE / "convert_to_onnx.py"

    if not cfg.exists() or not weights.exists():
        raise FileNotFoundError(
            f"Darknet files not found in {HERE / 'model'}.\n"
            "Run step 1 (download) first."
        )

    print("\nConverting Darknet → ONNX FP32 …")
    result = subprocess.run(
        [sys.executable, str(convert),
         "--cfg",     str(cfg),
         "--weights", str(weights),
         "--output",  str(ONNX_FP32_PATH),
         "--verify"],
    )
    if result.returncode != 0:
        print("\n[ERROR] ONNX FP32 conversion failed.")
        sys.exit(1)
    print(f"\n  ✅ FP32 ONNX ready: {ONNX_FP32_PATH}")


# ──────────────────────────────────────────────────────────────────────────────
# 3. TFLite FP32  (ONNX FP32 → TFLite float32 via onnx2tf)
# ──────────────────────────────────────────────────────────────────────────────
def convert_tflite_fp32(force: bool = False) -> None:
    """Convert the FP32 ONNX model to a float32 TFLite model using onnx2tf.

    onnx2tf automatically handles the NCHW→NHWC transposition that the original
    ONNX model (1,3,320,320) requires, producing a TFLite model with NHWC inputs
    (1,320,320,3) and NHWC outputs (1,grid_h,grid_w,255).

    Parameters
    ----------
    force : bool  Re-convert even if the TFLite FP32 file already exists.
    """
    if TFLITE_FP32_PATH.exists() and not force:
        size_mb = TFLITE_FP32_PATH.stat().st_size / 1e6
        print(f"  [SKIP] TFLite FP32 already exists: {TFLITE_FP32_PATH.name}"
              f"  ({size_mb:.1f} MB)")
        return

    if not ONNX_FP32_PATH.exists():
        raise FileNotFoundError(
            f"FP32 ONNX not found: {ONNX_FP32_PATH}\n"
            "Run with --mode fp32 first (or ensure model/yolo_fastest_1.1.onnx exists)."
        )

    print("\nConverting ONNX FP32 → TFLite FP32 (via onnx2tf) …")
    try:
        import shutil as _shutil
        import onnx2tf

        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        # onnx2tf downloads a test-image npy for internal verification.
        # The remote URL is unreachable in many dev environments, so we
        # generate it locally if missing.  This is NOT calibration data.
        _test_npy = HERE / _TEST_NPY
        if not _test_npy.exists():
            import numpy as _np
            _np.save(str(_test_npy),
                     _np.random.RandomState(0).rand(20, 128, 128, 3).astype(_np.float32))

        saved_model_dir = str(MODEL_DIR / "_saved_model_fp32")
        if Path(saved_model_dir).exists():
            _shutil.rmtree(saved_model_dir)

        onnx2tf.convert(
            input_onnx_file_path              = str(ONNX_FP32_PATH),
            output_folder_path                = saved_model_dir,
            copy_onnx_input_output_names_to_tflite = True,
            non_verbose                       = True,
            output_signaturedefs              = True,
        )

        # Find the generated .tflite file (FP32, no quant suffix)
        tflite_files = list(Path(saved_model_dir).rglob("*.tflite"))
        fp32_files   = [p for p in tflite_files
                        if "quant" not in p.name and "int8" not in p.name]
        src = fp32_files[0] if fp32_files else (tflite_files[0] if tflite_files else None)
        if src is None:
            raise FileNotFoundError(
                f"onnx2tf did not produce a .tflite file in {saved_model_dir}"
            )
        _shutil.copy2(str(src), str(TFLITE_FP32_PATH))
        _shutil.rmtree(saved_model_dir, ignore_errors=True)

        size_mb = TFLITE_FP32_PATH.stat().st_size / 1e6
        print(f"  ✅ TFLite FP32 saved: {TFLITE_FP32_PATH.name}  ({size_mb:.2f} MB)")

    except ImportError:
        print("\n[ERROR] onnx2tf not installed.  pip install onnx2tf tensorflow")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] TFLite FP32 conversion failed: {e}")
        raise


# ──────────────────────────────────────────────────────────────────────────────
# 4. TFLite INT8  (ONNX FP32 → TFLite INT8 via onnx2tf + representative dataset)
# ──────────────────────────────────────────────────────────────────────────────
def _resolve_calib_dir() -> Path:
    """Return a directory containing COCO val2017 images.

    All data must live inside this model's root directory (``Datasets/val2017``).
    If not present, download automatically from cocodataset.org.
    Never falls back to shared paths outside the model root.
    """
    if CALIB_DIR.is_dir() and any(CALIB_DIR.glob("*.jpg")):
        return CALIB_DIR
    # Auto-download into model root
    print("  COCO val2017 not found locally — downloading (~780 MB) …")
    import zipfile
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATASETS_DIR / "val2017.zip"
    urllib.request.urlretrieve(
        COCO_VAL2017_IMAGES_URL,
        zip_path, reporthook=_progress_hook,
    )
    print()
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATASETS_DIR)
    zip_path.unlink()
    return CALIB_DIR


def _list_images(folder: Path, need: int) -> list:
    """Return up to *need* .jpg/.png paths from *folder*, sorted."""
    exts = {".jpg", ".jpeg", ".png"}
    imgs = []
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in exts:
            imgs.append(p)
            if len(imgs) >= need:
                break
    return imgs


def convert_tflite_int8(calib_num: int = DEFAULT_CALIB_NUM,
                        force: bool = False) -> None:
    """Quantize the FP32 ONNX model to a full-integer INT8 TFLite model.

    Uses onnx2tf with a representative dataset drawn from COCO val2017 images.
    onnx2tf internally runs TFLiteConverter with TFLITE_BUILTINS_INT8 and
    the provided calibration data, producing int8 input / int8 output tensors.

    Parameters
    ----------
    calib_num : int  Number of calibration images (default 100).
    force     : bool Re-quantize even if the INT8 file already exists.
    """
    if TFLITE_INT8_PATH.exists() and not force:
        size_mb = TFLITE_INT8_PATH.stat().st_size / 1e6
        print(f"  [SKIP] TFLite INT8 already exists: {TFLITE_INT8_PATH.name}"
              f"  ({size_mb:.1f} MB)")
        return

    if not ONNX_FP32_PATH.exists():
        raise FileNotFoundError(
            f"FP32 ONNX not found: {ONNX_FP32_PATH}\n"
            "Run with --mode fp32 first (or ensure model/yolo_fastest_1.1.onnx exists)."
        )

    print("\n" + "─" * 60)
    print("  Quantize ONNX FP32 → TFLite INT8 (onnx2tf, full-integer)")
    print("─" * 60)

    try:
        import cv2
        import numpy as np
        import shutil as _shutil
        import onnx2tf
        import tensorflow as tf

        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        # Ensure onnx2tf test npy exists locally
        _test_npy = HERE / _TEST_NPY
        if not _test_npy.exists():
            np.save(str(_test_npy),
                    np.random.RandomState(0).rand(20, 128, 128, 3).astype(np.float32))

        # ── Calibration images ────────────────────────────────────────────────
        calib_dir   = _resolve_calib_dir()
        calib_paths = _list_images(calib_dir, need=calib_num)
        if len(calib_paths) < 10:
            raise FileNotFoundError(
                f"Not enough calibration images in {calib_dir} "
                f"(found {len(calib_paths)}, need ≥ 10)."
            )
        print(f"  Using {len(calib_paths)} calibration images from {calib_dir}")

        # ── Step A: ONNX → SavedModel ─────────────────────────────────────────
        saved_model_dir = str(MODEL_DIR / "_saved_model_int8")
        if Path(saved_model_dir).exists():
            _shutil.rmtree(saved_model_dir)

        print("  Converting ONNX FP32 → TF SavedModel (onnx2tf) …")
        onnx2tf.convert(
            input_onnx_file_path              = str(ONNX_FP32_PATH),
            output_folder_path                = saved_model_dir,
            copy_onnx_input_output_names_to_tflite = True,
            non_verbose                       = True,
            output_signaturedefs              = True,
        )

        # ── Step B: Determine SavedModel input shape / layout ─────────────────
        loaded        = tf.saved_model.load(saved_model_dir)
        concrete_func = loaded.signatures["serving_default"]
        in_specs      = list(concrete_func.structured_input_signature[1].values())
        input_shape   = in_specs[0].shape   # e.g. (1, 320, 320, 3) NHWC
        print(f"  SavedModel input shape: {input_shape}")
        is_nhwc = (len(input_shape) == 4 and input_shape[-1] == 3)

        # ── Step C: Representative dataset ────────────────────────────────────
        def _representative_dataset():
            pbar = (tqdm(calib_paths, desc="  calibrating", unit="img")
                    if _HAS_TQDM else calib_paths)
            for img_path in pbar:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                h, w   = img.shape[:2]
                scale  = min(INPUT_W / w, INPUT_H / h)
                rw, rh = int(round(w * scale)), int(round(h * scale))
                resized = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_LINEAR)
                pad_top  = (INPUT_H - rh) // 2
                pad_left = (INPUT_W - rw) // 2
                padded   = np.full((INPUT_H, INPUT_W, 3), 114, dtype=np.uint8)
                padded[pad_top:pad_top + rh, pad_left:pad_left + rw] = resized
                rgb  = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
                blob = rgb.astype(np.float32) / 255.0
                if is_nhwc:
                    blob = blob[np.newaxis, ...]              # (1, H, W, 3)
                else:
                    blob = blob.transpose(2, 0, 1)[np.newaxis, ...]   # (1, 3, H, W)
                yield [blob]

        # ── Step D: TFLite INT8 quantization ──────────────────────────────────
        print("  Running TFLite INT8 quantization (this may take a few minutes) …")
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        converter.optimizations          = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = _representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type   = tf.int8
        converter.inference_output_type  = tf.int8
        tflite_model = converter.convert()

        TFLITE_INT8_PATH.write_bytes(tflite_model)
        _shutil.rmtree(saved_model_dir, ignore_errors=True)

        size_mb = TFLITE_INT8_PATH.stat().st_size / 1e6
        print(f"\n  ✅ TFLite INT8 saved: {TFLITE_INT8_PATH.name}  ({size_mb:.2f} MB)")

        # Print quantization parameters
        interp = tf.lite.Interpreter(model_path=str(TFLITE_INT8_PATH))
        interp.allocate_tensors()
        inp = interp.get_input_details()[0]
        out = interp.get_output_details()[0]
        print(f"  Input  dtype={inp['dtype'].__name__}  "
              f"scale={inp['quantization'][0]:.6f}  "
              f"zero_point={int(inp['quantization'][1])}")
        print(f"  Output dtype={out['dtype'].__name__}  "
              f"scale={out['quantization'][0]:.6f}  "
              f"zero_point={int(out['quantization'][1])}")

    except ImportError as e:
        print(f"\n[ERROR] Missing dependency: {e}")
        print("        Install with:  pip install onnx2tf tensorflow opencv-python")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] TFLite INT8 quantization failed: {e}")
        raise


# ──────────────────────────────────────────────────────────────────────────────
# 6. CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Download Yolo-Fastest Darknet files, convert to TFLite FP32, "
            "and optionally quantize to TFLite INT8."
        )
    )
    parser.add_argument(
        "--variant", choices=list(VARIANTS.keys()), default="base",
        help=(
            "Model variant: 'base' = yolo-fastest-1.1 (default); "
            "'xl' = yolo-fastest-1.1-xl."
        ),
    )
    parser.add_argument(
        "--mode", choices=["fp32", "int8", "both", "FP32", "INT8", "BOTH"], default="both",
        help=(
            "'fp32' = TFLite FP32 only; "
            "'int8' = TFLite FP32 + INT8; "
            "'all'  = same as 'int8' (default)."
        ),
    )
    parser.add_argument(
        "--skip-convert", action="store_true",
        help="Download Darknet files only; skip all conversion steps.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download / re-convert even if output files already exist.",
    )
    parser.add_argument(
        "--calib-num", type=int, default=DEFAULT_CALIB_NUM,
        help=f"Number of calibration images for INT8 (default: {DEFAULT_CALIB_NUM}).",
    )
    args = parser.parse_args()

    # Resolve variant paths into module-level globals used by helper functions
    global ONNX_FP32_PATH, TFLITE_FP32_PATH, TFLITE_INT8_PATH
    vcfg = VARIANTS[args.variant]
    ONNX_FP32_PATH   = HERE / "model" / vcfg["onnx_fp32"]
    TFLITE_FP32_PATH = MODEL_DIR  / vcfg["tflite_fp32"]
    TFLITE_INT8_PATH = MODEL_DIR  / vcfg["tflite_int8"]

    label_variant = "Yolo-Fastest 1.1" if args.variant == "base" else "Yolo-Fastest 1.1-XL"

    # ── Step 1: Download Darknet files ────────────────────────────────────────
    remote_dir = vcfg["remote_dir"]
    download_list = [
        (f"{remote_dir}/{vcfg['cfg']}",     vcfg["cfg"]),
        (f"{remote_dir}/{vcfg['weights']}", vcfg["weights"]),
        (f"{remote_dir}/{vcfg['names']}",   vcfg["names"]),
    ]
    dest_dir = HERE / "model"
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {label_variant} model files → {dest_dir}\n")
    for remote_path, local_name in download_list:
        url = f"{_GITHUB_RAW}/{remote_path}"
        download_file(url, dest_dir / local_name, force=args.force)
    print(f"\n[OK] Darknet files in: {dest_dir}")

    if args.skip_convert:
        print("\n[SKIP] Conversion (--skip-convert).")
        return

    # ── Step 2: ONNX FP32 (needed as intermediate) ────────────────────────────
    print("\n" + "─" * 60)
    print("  Step 1/2: Darknet → ONNX FP32")
    print("─" * 60)
    ensure_onnx_fp32(force=args.force)

    # ── Step 3: TFLite FP32 ───────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  Step 2a/2: ONNX FP32 → TFLite FP32  (onnx2tf)")
    print("─" * 60)
    convert_tflite_fp32(force=args.force)

    # ── Step 4: TFLite INT8 ───────────────────────────────────────────────────
    if args.mode in ("int8", "both", "all"):
        print("\n" + "─" * 60)
        print("  Step 2b/2: ONNX FP32 → TFLite INT8  (onnx2tf + representative dataset)")
        print("─" * 60)
        convert_tflite_int8(calib_num=args.calib_num, force=args.force)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  Summary")
    print("═" * 60)
    for label, path in [
        ("Darknet CFG    ", dest_dir / vcfg["cfg"]),
        ("Darknet Weights", dest_dir / vcfg["weights"]),
        ("ONNX FP32      ", ONNX_FP32_PATH),
        ("TFLite FP32    ", TFLITE_FP32_PATH),
        ("TFLite INT8    ", TFLITE_INT8_PATH),
    ]:
        status = f"{path.stat().st_size / 1e6:.1f} MB" if path.exists() else "not built"
        print(f"  {label} : {path.name}  ({status})")
    print("═" * 60)
    print("\nRun inference:")
    print(f"  python inference.py --image sample.jpg --model model/{vcfg['tflite_fp32']}")
    print(f"  python inference.py --image sample.jpg --model model/{vcfg['tflite_int8']}")
    print("\nRun validation:")
    print(f"  python validate.py --variant {args.variant} --models tflite_fp32")
    print(f"  python validate.py --variant {args.variant} --models tflite_int8 mera_tflite_int8")


if __name__ == "__main__":
    main()
