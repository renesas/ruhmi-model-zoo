# Copyright 2026 Renesas Electronics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Download RNNoise model artifacts for this project.

Pipeline:
    1. INT8: sparse-clone Arm-software/ML-zoo and copy rnnoise_INT8.tflite.

Usage:
        python download_model.py                  # fetch INT8 model (default)
    python download_model.py --force         # overwrite existing artifacts
"""
from __future__ import annotations

import argparse
import contextlib
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

MODEL_INT8 = MODEL_DIR / "rnnoise_INT8.tflite"

# Arm ML-zoo INT8 source
ARM_GIT_URL = "https://github.com/ARM-software/ML-zoo.git"
ARM_BRANCH = "master"
ARM_SUBDIR = "models/noise_suppression/RNNoise/tflite_int8"
ARM_SPARSE_ROOT = "models/noise_suppression/RNNoise"
MLZOO_LOCAL_REPO = BASE_DIR / "third_party" / "ML-zoo"
MLZOO_MODEL_SRC = MLZOO_LOCAL_REPO / ARM_SUBDIR / "rnnoise_INT8.tflite"

EXPECTED_INT8_SIZE = 113_472


@contextlib.contextmanager
def spinner(msg: str):
    """Show a lightweight spinner while a blocking operation runs."""
    stop = threading.Event()
    chars = "|/-\\"
    t0 = time.time()

    def _spin() -> None:
        i = 0
        while not stop.is_set():
            elapsed = time.time() - t0
            sys.stdout.write(
                f"\r  {chars[i % len(chars)]} {msg} ... {elapsed:5.1f}s")
            sys.stdout.flush()
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
        sys.stdout.write(f"\r  + {msg} ... done ({elapsed:.1f}s)\n")
        sys.stdout.flush()


def _which(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def _run_cmd(cmd: list[str], msg: str) -> None:
    with spinner(msg):
        subprocess.run(cmd, check=True)


def _verify_int8(path: Path) -> bool:
    """Return True if path matches the expected Arm INT8 model size."""
    if not path.exists():
        print(f"  [-] Not present: {path}")
        return False
    size = path.stat().st_size
    if size != EXPECTED_INT8_SIZE:
        print(
            f"  [!] Size mismatch: got {size} B, expected {EXPECTED_INT8_SIZE} B")
        if size < 1024:
            print(
                "      Looks like a Git-LFS pointer file. Did you skip 'git lfs install'?")
        return False
    return True
# -----------------------------------------------------------------------------
# INT8 flow
# -----------------------------------------------------------------------------


def _ensure_mlzoo_repo() -> Path:
    if MLZOO_MODEL_SRC.is_file():
        print(f"  ML-zoo subtree already present: {MLZOO_LOCAL_REPO}")
        return MLZOO_LOCAL_REPO

    if not _which("git"):
        sys.exit("git is required. Install it and re-run.")
    if not _which("git-lfs"):
        print(
            "  [!] git-lfs not on PATH. The .tflite may become an LFS pointer file.")

    MLZOO_LOCAL_REPO.parent.mkdir(parents=True, exist_ok=True)

    print(f"  Cloning {ARM_GIT_URL}")
    print(f"       -> {MLZOO_LOCAL_REPO}  (sparse + LFS)")
    _run_cmd(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            "--depth=1",
            "--branch",
            ARM_BRANCH,
            ARM_GIT_URL,
            str(MLZOO_LOCAL_REPO),
        ],
        "git clone (sparse)",
    )
    _run_cmd(
        ["git", "-C", str(MLZOO_LOCAL_REPO),
         "sparse-checkout", "init", "--cone"],
        "sparse-checkout init",
    )
    _run_cmd(
        ["git", "-C", str(MLZOO_LOCAL_REPO), "sparse-checkout",
         "set", ARM_SPARSE_ROOT],
        "sparse-checkout set",
    )
    _run_cmd(
        ["git", "-C", str(MLZOO_LOCAL_REPO), "checkout", ARM_BRANCH],
        "git checkout",
    )

    if not MLZOO_MODEL_SRC.is_file():
        raise FileNotFoundError(
            f"Expected model not found after clone: {MLZOO_MODEL_SRC}")
    return MLZOO_LOCAL_REPO


def _cleanup_mlzoo_repo() -> None:
    third_party = MLZOO_LOCAL_REPO.parent
    if MLZOO_LOCAL_REPO.is_dir():
        print(f"  Removing cloned ML-zoo repo: {MLZOO_LOCAL_REPO}")
        shutil.rmtree(MLZOO_LOCAL_REPO, ignore_errors=True)
    if third_party.is_dir() and not any(third_party.iterdir()):
        third_party.rmdir()


def _copy_model(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    print(f"  Copied  {src.name}")
    print(f"       -> {dest}")


def _prepare_int8(force: bool) -> int:
    if MODEL_INT8.exists() and not force and _verify_int8(MODEL_INT8):
        print("  INT8 model already present.")
        return 0

    print("[int8 1/3] Cloning upstream Arm ML-zoo...")
    try:
        _ensure_mlzoo_repo()
    except subprocess.CalledProcessError as e:
        print(f"  [!] git failed (exit {e.returncode}). Check network/proxy.")
        return 2

    print("[int8 2/3] Extracting INT8 .tflite model...")
    _copy_model(MLZOO_MODEL_SRC, MODEL_INT8)

    print("[int8 3/3] Cleaning up clone...")
    _cleanup_mlzoo_repo()

    print("[int8 verify] Checking extracted file...")
    if not _verify_int8(MODEL_INT8):
        print("  INT8 download finished but verification FAILED.")
        return 2
    print("  INT8 model ready.")
    return 0


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch the RNNoise INT8 model artifact.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download or regenerate even if valid artifacts already exist.",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("  RNNoise model download / build")
    print("=" * 72)
    print(f"  INT8 target   : {MODEL_INT8}")
    print(f"  INT8 source   : {ARM_GIT_URL}")
    print()

    status = 0

    s = _prepare_int8(force=args.force)
    if s != 0:
        status = s

    print()
    if status == 0:
        print("Done.")
    else:
        print("Completed with errors.")
    return status


if __name__ == "__main__":
    sys.exit(main())
