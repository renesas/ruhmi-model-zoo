# SPDX-License-Identifier: Apache-2.0
"""Download TinyWav2Letter model artifacts from ARM ML-zoo and generate FP32 TFLite.

Sources:
  - pruned_tiny_wav2letter (SavedModel format) — ARM ML-zoo (Apache-2.0)
  - tiny_wav2letter_pruned_int8.tflite (pruned INT8 model) — ARM ML-zoo (Apache-2.0)

Pipeline:
  1. Sparse-clone ARM ML-zoo repo (tiny_wav2letter paths only)
  2. Copy tiny_wav2letter_pruned_int8.tflite to model/
  3. Copy SavedModel directory pruned_tiny_wav2letter to model/
  4. Remove the cloned repo
  5. Convert SavedModel to FP32 TFLite (via tf.lite.TFLiteConverter.from_saved_model)

Usage:
    python download_model.py
    python download_model.py --mode fp32
    python download_model.py --mode int8
    python download_model.py --mode both
"""

import argparse
import os
import shutil
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
REPO_TMP = os.path.join(BASE_DIR, "_ml_zoo_tmp")

ML_ZOO_URL = "https://github.com/ARM-software/ML-zoo.git"
SPARSE_DIR = "models/speech_recognition/tiny_wav2letter/tflite_pruned_int8"
REPO_SAVED_MODEL = f"{SPARSE_DIR}/recreate_code/saved_models/pruned_tiny_wav2letter"
REPO_INT8 = f"{SPARSE_DIR}/tiny_wav2letter_pruned_int8.tflite"


def _run(cmd, **kwargs):
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def _is_lfs_pointer(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(64).startswith(b"version https://git-lfs")
    except Exception:
        return False


def ensure_artifacts(mode: str) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)

    dest_saved_model = os.path.join(MODEL_DIR, "pruned_tiny_wav2letter")
    dest_int8 = os.path.join(MODEL_DIR, "tiny_wav2letter_pruned_int8.tflite")

    required_entries = []
    if mode in ("fp32", "both"):
        required_entries.append(("pruned_tiny_wav2letter", dest_saved_model, REPO_SAVED_MODEL))
    if mode in ("int8", "both"):
        required_entries.append(("tiny_wav2letter_pruned_int8.tflite", dest_int8, REPO_INT8))

    missing_entries = [
        (name, dest, repo_rel)
        for (name, dest, repo_rel) in required_entries
        if not os.path.isdir(dest) and not os.path.isfile(dest)
    ]
    missing = [name for (name, _, _) in missing_entries]
    if not missing:
        print("\nAll required artifacts already present. Skipping ML-zoo clone.")
        return

    if os.path.isdir(REPO_TMP):
        print(f"  Removing stale temp dir: {REPO_TMP}")
        shutil.rmtree(REPO_TMP)

    print(f"\nMissing artifacts for mode={mode}: {', '.join(missing)}")
    print("Cloning ARM ML-zoo (sparse, depth 1) ...")
    _run([
        "git", "clone", "--depth", "1", "--filter=blob:none",
        "--sparse", ML_ZOO_URL, REPO_TMP
    ])
    _run(["git", "sparse-checkout", "set", SPARSE_DIR], cwd=REPO_TMP)

    lfs_missing_rel_paths = []
    for _, _, repo_rel in missing_entries:
        src = os.path.join(REPO_TMP, repo_rel)
        if os.path.isfile(src) and _is_lfs_pointer(src):
            lfs_missing_rel_paths.append(repo_rel)

    if lfs_missing_rel_paths:
        include_paths = ",".join(lfs_missing_rel_paths)
        print(f"  LFS pointers detected for missing files - pulling: {include_paths}")
        _run(["git", "lfs", "pull", "--include", include_paths], cwd=REPO_TMP)

    for name, dest, repo_rel in missing_entries:
        src = os.path.join(REPO_TMP, repo_rel)
        if not os.path.exists(dest):
            if os.path.isdir(src):
                shutil.copytree(src, dest)
                print(f"  Copied directory {dest}")
            else:
                shutil.copy2(src, dest)
                print(f"  Copied {dest} ({os.path.getsize(dest) / 1e6:.2f} MB)")
        else:
            print(f"  Reusing existing {dest}")

    print(f"\nRemoving cloned repo: {REPO_TMP}")
    shutil.rmtree(REPO_TMP, ignore_errors=True)


def convert_fp32() -> str:
    import tensorflow as tf

    saved_model_dir = os.path.join(MODEL_DIR, "pruned_tiny_wav2letter")
    fp32_out = os.path.join(MODEL_DIR, "tiny_wav2letter_pruned_fp32.tflite")

    if not os.path.isdir(saved_model_dir):
        sys.exit(f"ERROR: SavedModel directory not found: {saved_model_dir}")

    print("\nLoading SavedModel as Keras model ...")
    model = tf.keras.models.load_model(saved_model_dir)

    # Wrap with fixed input shape (1, 296, 39) for TFLite export
    static_in = tf.keras.layers.Input((296, 39), batch_size=1)
    static_model = tf.keras.models.Model(
        inputs=[static_in], outputs=[model.call(static_in)]
    )

    print("Converting to FP32 TFLite ...")
    converter = tf.lite.TFLiteConverter.from_keras_model(static_model)
    tflite_bytes = converter.convert()

    with open(fp32_out, "wb") as f:
        f.write(tflite_bytes)
    print(f"  FP32 TFLite saved: {fp32_out} ({len(tflite_bytes) / 1e6:.2f} MB)")
    return fp32_out


def main():
    parser = argparse.ArgumentParser(
        description="Download TinyWav2Letter artifacts and generate FP32 TFLite."
    )
    parser.add_argument(
        "--mode",
        choices=["fp32", "int8", "both"],
        default="both",
        help="Select outputs: fp32, int8, or both (default: both).",
    )
    args = parser.parse_args()

    ensure_artifacts(args.mode)

    if args.mode in ["fp32", "both"]:
        convert_fp32()
    elif args.mode == "int8":
        print("\nINT8 mode selected: using downloaded tiny_wav2letter_pruned_int8.tflite as-is.")

    print("\nDone. Files in model/:")
    for name in sorted(os.listdir(MODEL_DIR)):
        p = os.path.join(MODEL_DIR, name)
        if os.path.isdir(p):
            print(f"  {name}/ (SavedModel directory)")
        else:
            print(f"  {name}  ({os.path.getsize(p) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
