# SPDX-License-Identifier: BSD-3-Clause
"""Download & Convert MobileFaceNet to TFLite (FP32 + INT8).

Pipeline:
  1. Auto-clone foamliu/MobileFaceNet to get the TorchScript checkpoint
     (third_party/MobileFaceNet/pretrained_model/mobilefacenet_scripted.pt)
  2. Load via torch.jit.load and export to ONNX (no model code needed)
  3. Convert ONNX -> TFLite FP32 via onnx2tf
  4. Quantize ONNX -> TFLite INT8 via onnx2tf using LFW face calibration data

Model: MobileFaceNet (foamliu/MobileFaceNet, v1.0)
  - Input  : (1, 3, 112, 112) NCHW float32, normalised to [-1, 1]
  - Output : (1, 128) L2-normalisable embedding

Usage:
    python download_model.py                   # FP32 + INT8
    python download_model.py --mode fp32       # FP32 only
    python download_model.py --mode int8       # INT8 only
    python download_model.py --calib-num 1000  # 1000 calibration faces
"""

import argparse
import contextlib
import io
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List

import numpy as np

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False


# ──────────────────────────────────────────────────────────────────────────────
# Spinner: lightweight live indicator for blocking ops
# ──────────────────────────────────────────────────────────────────────────────

# When non-None, spinner writes go straight to this fd instead of sys.stdout,
# so they survive even while a _capture_pipeline_stage() context has redirected
# the normal Python/C-level stdout streams.
_SPINNER_FD: "int | None" = None


def _spinner_write(text: str) -> None:
    if _SPINNER_FD is not None:
        try:
            os.write(_SPINNER_FD, text.encode("utf-8", errors="replace"))
            return
        except OSError:
            pass
    sys.stdout.write(text)
    sys.stdout.flush()


@contextlib.contextmanager
def spinner(msg: str, status_fn=None):
    """Show a live spinner while a blocking operation runs.

    If ``status_fn`` is given, it must be a zero-arg callable returning a short
    string describing the current sub-stage; it will be rendered after the
    elapsed time.
    """
    stop = threading.Event()
    chars = "|/-\\"
    t0 = time.time()
    last_len = [0]

    def _spin():
        i = 0
        while not stop.is_set():
            elapsed = time.time() - t0
            extra = ""
            if status_fn is not None:
                try:
                    s = status_fn()
                    if s:
                        extra = f"  [{s}]"
                except Exception:
                    pass
            line = f"  {chars[i % len(chars)]} {msg} ... {elapsed:6.1f}s{extra}"
            pad = " " * max(0, last_len[0] - len(line))
            _spinner_write(f"\r{line}{pad}")
            last_len[0] = len(line)
            i += 1
            stop.wait(0.1)

    th = threading.Thread(target=_spin, daemon=True)
    th.start()
    try:
        yield
    finally:
        stop.set()
        th.join()
        elapsed = time.time() - t0
        line = f"  + {msg} ... done ({elapsed:.1f}s)"
        pad = " " * max(0, last_len[0] - len(line))
        _spinner_write(f"\r{line}{pad}\n")


@contextlib.contextmanager
def _suppress_stdout_stderr():
    """Redirect C-level + Python-level stdout/stderr to /dev/null."""
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            yield
    finally:
        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        os.close(devnull_fd)


# Known onnx2tf / TFLite-converter stage markers (substring -> friendly label)
_ONNX2TF_STAGES = [
    ("Model conversion started",     "loading ONNX graph"),
    ("Model loaded",                 "graph loaded"),
    ("Model optimizing started",     "optimizing graph"),
    ("Model optimizing complete",    "graph optimized"),
    ("saved_model output started",   "writing SavedModel"),
    ("saved_model output complete",  "SavedModel written"),
    ("Float32 tflite output started",          "converting FP32 tflite"),
    ("Float32 tflite output complete",         "FP32 tflite done"),
    ("Float16 tflite output started",          "converting FP16 tflite"),
    ("Float16 tflite output complete",         "FP16 tflite done"),
    ("Dynamic Range Quantization tflite output started",  "dynamic-range quant"),
    ("Dynamic Range Quantization tflite output complete", "dynamic-range done"),
    ("INT8 Quantization tflite output started",           "INT8 quantization"),
    ("INT8 Quantization tflite output complete",          "INT8 quant done"),
    ("Full Integer Quantization tflite output started",   "full-integer quant"),
    ("Full Integer Quantization tflite output complete",  "full-integer done"),
    ("Integer Quantization with int16 activations",       "INT16 act quant"),
    ("fully_quantize: 0, inference_type: 6, input_inference_type: FLOAT32", "calibrating ranges"),
    ("fully_quantize: 0, inference_type: 6, input_inference_type: INT8",    "writing INT8 weights"),
]


@contextlib.contextmanager
def _capture_pipeline_stage():
    """Redirect fds 1 & 2 to a pipe, parse for onnx2tf stage markers.

    Yields a zero-arg callable that returns the latest detected stage label
    (or empty string before any marker is seen). Captured output is otherwise
    suppressed.
    """
    stage = {"label": "starting"}
    r_fd, w_fd = os.pipe()
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    stop = threading.Event()
    op_count = {"n": 0}

    def _reader():
        buf = b""
        while not stop.is_set():
            try:
                chunk = os.read(r_fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                try:
                    text = line.decode("utf-8", errors="replace")
                except Exception:
                    continue
                # Detect known stage markers
                for needle, label in _ONNX2TF_STAGES:
                    if needle in text:
                        stage["label"] = label
                        break
                else:
                    # Count per-op conversions as a coarse progress hint
                    if "INFO: onnx_op_name" in text or "onnx_op_type" in text:
                        op_count["n"] += 1
                        stage["label"] = f"converting op {op_count['n']}"

    th = threading.Thread(target=_reader, daemon=True)
    th.start()

    global _SPINNER_FD
    prev_spinner_fd = _SPINNER_FD
    try:
        _SPINNER_FD = saved_stdout_fd
        os.dup2(w_fd, 1)
        os.dup2(w_fd, 2)
        os.close(w_fd)
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            yield lambda: stage["label"]
    finally:
        # Restore real fds first so any final writes go to the terminal
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)
        _SPINNER_FD = prev_spinner_fd
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        stop.set()
        try:
            os.close(r_fd)
        except OSError:
            pass
        th.join(timeout=1.0)


# Configuration
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
PRETRAINED_DIR = BASE_DIR / "pretrained"
DATASET_DIR = BASE_DIR / "Datasets"

ONNX_PATH = PRETRAINED_DIR / "MobileFaceNet.onnx"
FP32_TFLITE = MODEL_DIR / "mobilefacenet_FP32.tflite"
INT8_TFLITE = MODEL_DIR / "mobilefacenet_INT8.tflite"

INPUT_SIZE = 112  # 112x112 face crop

# Upstream PyTorch source (auto-cloned, used only for the TorchScript .pt)
MFN_GIT_URL = "https://github.com/foamliu/MobileFaceNet.git"
MFN_LOCAL_REPO = BASE_DIR / "third_party" / "MobileFaceNet"
MFN_SCRIPTED_PT = MFN_LOCAL_REPO / "pretrained_model" / "mobilefacenet_scripted.pt"





# ──────────────────────────────────────────────────────────────────────────────
# Source repo management (clone -> use scripted .pt -> remove)
# ──────────────────────────────────────────────────────────────────────────────
def _ensure_mfn_repo() -> Path:
    if MFN_SCRIPTED_PT.is_file():
        print(f"  MobileFaceNet repo already present: {MFN_LOCAL_REPO}")
        return MFN_LOCAL_REPO

    MFN_LOCAL_REPO.parent.mkdir(parents=True, exist_ok=True)
    with spinner(f"Cloning foamliu/MobileFaceNet -> {MFN_LOCAL_REPO}"):
        subprocess.run(
            ["git", "clone", "--depth", "1", MFN_GIT_URL, str(MFN_LOCAL_REPO)],
            check=True, capture_output=True,
        )
    if not MFN_SCRIPTED_PT.is_file():
        raise FileNotFoundError(
            f"Expected scripted checkpoint not found after clone: {MFN_SCRIPTED_PT}"
        )
    return MFN_LOCAL_REPO


def _cleanup_mfn_repo() -> None:
    third_party = MFN_LOCAL_REPO.parent
    if MFN_LOCAL_REPO.is_dir():
        print(f"  Removing cloned MobileFaceNet repo: {MFN_LOCAL_REPO}")
        shutil.rmtree(MFN_LOCAL_REPO, ignore_errors=True)
    if third_party.is_dir() and not any(third_party.iterdir()):
        third_party.rmdir()


# ──────────────────────────────────────────────────────────────────────────────
# LFW calibration dataset
# ──────────────────────────────────────────────────────────────────────────────
def _list_face_images(folder: Path, need: int) -> List[Path]:
    out: List[Path] = []
    if not folder.is_dir():
        return out
    for root, _, files in os.walk(folder):
        for f in sorted(files):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                out.append(Path(root) / f)
                if len(out) >= need:
                    return out
    return out


    for child in folder.iterdir():
        if not child.is_dir():
            continue
        probe = child / f"{child.name}_0001.jpg"
        if probe.is_file():
            return True
    return False








# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing: MTCNN-align -> 112x112 RGB -> normalise to [-1, 1]
#
# MobileFaceNet was trained on MTCNN-aligned 112x112 inputs where the eye
# centres, nose tip, and mouth corners land at a fixed canonical template.
# Feeding un-aligned images collapses accuracy, so every face that touches
# the model (calibration, inference, validation) goes through this same
# function. See utils/align.py for the alignment implementation.
#
# Aligned crops are cached on disk under <src_root>-aligned/ so repeated
# calibration / validation runs reuse the same MTCNN output (~1 sec/image
# -> instant cache hit).
# ──────────────────────────────────────────────────────────────────────────────
from utils.align import align_face, align_face_cached  # noqa: E402


def preprocess_face(image_path: Path,
                    cache_root: Path | None = None,
                    src_root: Path | None = None) -> np.ndarray:
    """Return a (1, 112, 112, 3) float32 tensor in NHWC, range [-1, 1].

    If ``cache_root`` is provided, the MTCNN-aligned crop is cached under it
    (keyed by the path relative to ``src_root``) so subsequent calls are I/O
    bound. ``RuntimeError`` is raised if MTCNN cannot find a face.
    """
    if cache_root is not None:
        aligned = align_face_cached(image_path, cache_root, src_root)
    else:
        aligned = align_face(image_path)             # uint8 (112, 112, 3) RGB
    arr = aligned.astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return np.expand_dims(arr, axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# Build model: TorchScript .pt -> ONNX
# ──────────────────────────────────────────────────────────────────────────────
def build_onnx() -> Path:
    """Load the upstream TorchScript checkpoint and export to ONNX."""
    with spinner("Importing torch"):
        import torch

    _ensure_mfn_repo()

    with spinner("Loading TorchScript checkpoint"):
        model = torch.jit.load(str(MFN_SCRIPTED_PT), map_location="cpu")
        model.eval()

    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    with spinner(f"Exporting to ONNX -> {ONNX_PATH.name}"):
        with torch.no_grad(), _suppress_stdout_stderr():
            # dynamo=False forces the legacy TorchScript-based exporter.
            # The new torch.export-based default (torch >= 2.5) cannot trace
            # ScriptModules and raises "Exporting a ScriptModule is not
            # supported." Since foamliu/MobileFaceNet ships a .pt
            # produced by torch.jit.script, we must use the legacy path.
            torch.onnx.export(
                model, dummy, str(ONNX_PATH),
                input_names=["input0"],
                output_names=["output0"],
                opset_version=18,
            )
    print(f"  ONNX saved ({ONNX_PATH.stat().st_size/1e6:.2f} MB)")

    del model
    _cleanup_mfn_repo()
    return ONNX_PATH


# ──────────────────────────────────────────────────────────────────────────────
# TFLite conversion (via onnx2tf)
# ──────────────────────────────────────────────────────────────────────────────
def _ensure_onnx2tf_dummy_npy() -> None:
    """onnx2tf checks for a small calibration .npy on disk; create a stub."""
    stub = Path("calibration_image_sample_data_20x128x128x3_float32.npy")
    if not stub.is_file():
        np.save(stub,
                np.random.RandomState(0).rand(20, 128, 128, 3).astype(np.float32))


def convert_fp32(onnx_path: Path) -> Path:
    with spinner("Importing onnx2tf (loads TensorFlow)"):
        with _suppress_stdout_stderr():
            import onnx2tf

    saved_model_dir = MODEL_DIR / "_tf_saved_model_fp32"
    print("\nConverting ONNX -> TFLite FP32 via onnx2tf ...")
    _ensure_onnx2tf_dummy_npy()

    with _capture_pipeline_stage() as get_stage:
        with spinner("onnx2tf FP32 conversion", status_fn=get_stage):
            onnx2tf.convert(
                input_onnx_file_path=str(onnx_path),
                output_folder_path=str(saved_model_dir),
                copy_onnx_input_output_names_to_tflite=True,
                output_signaturedefs=True,
                non_verbose=False,
            )

    auto_tflite = saved_model_dir / "MobileFaceNet_float32.tflite"
    if not auto_tflite.is_file():
        # onnx2tf may use a different naming convention; pick the first float32 tflite
        candidates = sorted(saved_model_dir.glob("*float32*.tflite"))
        if not candidates:
            candidates = sorted(saved_model_dir.glob("*.tflite"))
        if not candidates:
            raise FileNotFoundError(
                f"onnx2tf produced no .tflite in {saved_model_dir}"
            )
        auto_tflite = candidates[0]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(auto_tflite, FP32_TFLITE)
    print(f"  Saved {FP32_TFLITE} ({FP32_TFLITE.stat().st_size/1e6:.2f} MB)")
    return FP32_TFLITE


def convert_int8(onnx_path: Path, calib_paths: List[Path],
                 cache_root: Path | None = None,
                 src_root: Path | None = None) -> Path:
    with spinner("Importing onnx2tf (loads TensorFlow)"):
        with _suppress_stdout_stderr():
            import onnx2tf

    # Warm up MTCNN once so the alignment loop doesn't pay the load cost
    # silently on its first iteration.
    with spinner("Loading MTCNN face detector"):
        with _suppress_stdout_stderr():
            from utils.align import get_mtcnn
            get_mtcnn()

    print(f"\nPreparing {len(calib_paths)} calibration faces "
          f"(MTCNN-aligned to 112x112) ...")
    if cache_root is not None:
        print(f"  Aligned-crop cache: {cache_root}")
    calib_npy = MODEL_DIR / "_calib_data.npy"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    imgs = []
    skipped = 0
    iterator = tqdm(calib_paths, desc="  Aligning") if _HAS_TQDM else calib_paths
    for p in iterator:
        try:
            imgs.append(preprocess_face(p, cache_root=cache_root,
                                        src_root=src_root)[0])
        except RuntimeError:
            skipped += 1
    if skipped:
        print(f"  ! Skipped {skipped} image(s) where MTCNN found no face.")
    if len(imgs) < 50:
        raise RuntimeError(
            f"Only {len(imgs)} calibration faces survived alignment; "
            f"need at least 50. Try a larger --calib-num or a different --calib-dir."
        )
    np.save(calib_npy, np.stack(imgs).astype(np.float32))
    print(f"  Calibration data saved: {calib_npy}  (N={len(imgs)})")

    _ensure_onnx2tf_dummy_npy()

    out_dir = MODEL_DIR / "_onnx2tf_int8_out"
    print("  Converting ONNX -> TFLite INT8 via onnx2tf ...")
    print("  (typically 1-5 min: load -> optimize -> SavedModel -> calibrate -> INT8 weights)")
    with _capture_pipeline_stage() as get_stage:
        with spinner("onnx2tf INT8 conversion", status_fn=get_stage):
            onnx2tf.convert(
                input_onnx_file_path=str(onnx_path),
                output_folder_path=str(out_dir),
                copy_onnx_input_output_names_to_tflite=True,
                output_signaturedefs=True,
                non_verbose=False,
                output_integer_quantized_tflite=True,
                quant_type="per-channel",
                custom_input_op_name_np_data_path=[
                    ["input0", str(calib_npy), [0, 0, 0], [1, 1, 1]],
                ],
            )

    model_stem = onnx_path.stem
    auto_int8 = out_dir / f"{model_stem}_full_integer_quant.tflite"
    if not auto_int8.is_file():
        candidates = sorted(out_dir.glob("*integer_quant*.tflite"))
        if not candidates:
            raise FileNotFoundError(
                f"onnx2tf produced no INT8 .tflite in {out_dir}"
            )
        auto_int8 = candidates[0]

    shutil.copy2(auto_int8, INT8_TFLITE)
    print(f"  Saved {INT8_TFLITE} ({INT8_TFLITE.stat().st_size/1e6:.2f} MB)")
    return INT8_TFLITE


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download MobileFaceNet and convert to TFLite FP32/INT8."
    )
    parser.add_argument(
        "--mode", choices=["fp32", "int8", "all", "FP32", "INT8", "ALL"],
        default="all",
        help="Export mode: fp32, int8, or all (default: all).",
    )
    parser.add_argument(
        "--calib-num", type=int, default=1000,
        help="Number of LFW faces to use for INT8 calibration (default: 1000).",
    )
    parser.add_argument(
        "--calib-dir", type=str, default=None,
        help="Path to a directory of face images for INT8 calibration (required "
             "for INT8 mode). Each image may contain a face anywhere in the frame; "
             "MTCNN will locate and align it to the canonical 112x112 template "
             "before quantization.",
    )
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    mode = args.mode.upper()

    print("\n" + "=" * 64)
    print("  MobileFaceNet  ->  TFLite")
    print("=" * 64)

    onnx_path = ONNX_PATH if ONNX_PATH.is_file() else build_onnx()
    if onnx_path is ONNX_PATH and ONNX_PATH.is_file():
        print(f"ONNX already exists: {ONNX_PATH}")

    calib_paths: List[Path] = []
    cache_root: Path | None = None
    src_root: Path | None = None
    if mode in ("INT8", "ALL"):
        if args.calib_dir:
            lfw_dir = Path(args.calib_dir).expanduser().resolve()
            if not lfw_dir.is_dir():
                raise FileNotFoundError(f"--calib-dir not found: {lfw_dir}")
        else:
            raise FileNotFoundError(
                "--calib-dir is required for INT8 mode. "
                "Provide a directory containing at least 50 face images."
            )
        # Pull a few extra images in case MTCNN fails on some of them.
        calib_paths = _list_face_images(lfw_dir, need=int(args.calib_num * 1.2))
        if len(calib_paths) < 50:
            raise FileNotFoundError(
                f"Not enough images in {lfw_dir} "
                f"(found {len(calib_paths)}, need at least 50)."
            )
        calib_paths = calib_paths[: int(args.calib_num * 1.2)]

        # Cache aligned crops next to the source dataset:
        #   .../lfw-deepfunneled/   ->   .../lfw-deepfunneled-aligned/
        src_root = lfw_dir
        cache_root = lfw_dir.parent / f"{lfw_dir.name}-aligned"
        print(f"\nUsing up to {len(calib_paths)} source images from {lfw_dir} "
              f"(target {args.calib_num} aligned faces)")

    if mode in ("FP32", "ALL"):
        convert_fp32(onnx_path)
    if mode in ("INT8", "ALL"):
        convert_int8(onnx_path, calib_paths,
                     cache_root=cache_root, src_root=src_root)

    print("\n" + "=" * 64)
    print("Done. Models saved in:", MODEL_DIR)
    print("=" * 64)


if __name__ == "__main__":
    main()