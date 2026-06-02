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
# The pre-trained Anomaly Detection model is from the MLCommons Tiny benchmark:
#   https://github.com/mlcommons/tiny
# Licensed under the Apache License, Version 2.0.
"""
Download the Anomaly Detection (AD) Dense Autoencoder Keras model from
MLCommons Tiny and convert it to TFLite (float32 and/or int8).

The .h5 Keras model is downloaded from:
  https://github.com/mlcommons/tiny/tree/master/benchmark/training/
  anomaly_detection/trained_models/

BatchNormalization layers are folded into the preceding Dense layers before
TFLite conversion (following the reference 02_convert.py script).

INT8 quantisation uses the DCASE 2020 Task 2 ToyCar *training* data as the
calibration dataset.  The script downloads the dataset from Zenodo if not
already present.

Outputs (in model/):
  ad01_FP32.tflite  – float32 model (BN folded)
  ad01_INT8.tflite  – fully int8 quantised model (int8 input/output)

Usage:
  python download_model.py                           # both fp32 + int8
  python download_model.py --mode fp32               # fp32 only
  python download_model.py --mode int8               # int8 only
  python download_model.py --mode int8 --calib-num 500
"""

import argparse
import glob
import os
import sys
import urllib.request
import zipfile

import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.python.ops import math_ops
except Exception:
    print("TensorFlow is required to run this script. "
          "Install it with: pip install tensorflow")
    raise

try:
    import librosa
except ImportError:
    print("librosa is required for feature extraction. "
          "Install it with: pip install librosa")
    raise

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "model")

FP32_OUT = os.path.join(OUTPUT_DIR, "ad01_FP32.tflite")
INT8_OUT = os.path.join(OUTPUT_DIR, "ad01_INT8.tflite")

# H5 model URL (GitHub raw)
H5_URL = (
    "https://github.com/mlcommons/tiny/raw/master/"
    "benchmark/training/anomaly_detection/trained_models/ad01.h5"
)
H5_PATH = os.path.join(OUTPUT_DIR, "ad01.h5")

# DCASE 2020 Task 2 ToyCar dataset (Zenodo)
ZENODO_TOYCAR_URL = (
    "https://zenodo.org/record/3678171/files/dev_data_ToyCar.zip"
)
DATASET_DIR = os.path.join(HERE, "Datasets")
TOYCAR_DIR = os.path.join(DATASET_DIR, "ToyCar")

# Feature extraction parameters (must match training / baseline.yaml)
N_MELS     = 128
FRAMES     = 5
N_FFT      = 1024
HOP_LENGTH = 512
POWER      = 2.0
INPUT_DIM  = N_MELS * FRAMES  # 640


# ──────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────────────────────────────────────
def _download_with_progress(url: str, dest: str) -> None:
    """Download *url* → *dest* with a live progress bar."""
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


def download_file(url: str, dest: str) -> str:
    """Download a file if not already present."""
    if os.path.isfile(dest):
        size_kb = os.path.getsize(dest) / 1024
        print(f"  ✅ Already exists: {dest} ({size_kb:.1f} KB) — skipping.")
        return dest
    _download_with_progress(url, dest)
    size_kb = os.path.getsize(dest) / 1024
    print(f"  ✅ Saved to {dest} ({size_kb:.1f} KB)")
    return dest


# ──────────────────────────────────────────────────────────────────────────────
# Dataset download (DCASE 2020 ToyCar from Zenodo)
# ──────────────────────────────────────────────────────────────────────────────
def ensure_dataset(dataset_dir: str = None) -> str:
    """Ensure the DCASE 2020 ToyCar training dataset is available.

    Downloads from Zenodo if not already present.

    Returns
    -------
    str : path to ToyCar directory (containing train/ and test/ sub-dirs).
    """
    toycar_dir = os.path.join(dataset_dir or DATASET_DIR, "ToyCar")
    train_dir = os.path.join(toycar_dir, "train")

    if os.path.isdir(train_dir):
        n_wavs = len(glob.glob(os.path.join(train_dir, "*.wav")))
        if n_wavs > 0:
            print(f"  ✅ ToyCar training data found: {train_dir} ({n_wavs} WAV files)")
            return toycar_dir

    print("  ⚠️  DCASE 2020 ToyCar dataset not found. Downloading from Zenodo...")
    os.makedirs(dataset_dir or DATASET_DIR, exist_ok=True)

    zip_path = os.path.join(dataset_dir or DATASET_DIR, "dev_data_ToyCar.zip")
    _download_with_progress(ZENODO_TOYCAR_URL, zip_path)

    print("  Extracting archive...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dataset_dir or DATASET_DIR)

    # Clean up zip
    os.remove(zip_path)

    n_wavs = len(glob.glob(os.path.join(train_dir, "*.wav")))
    print(f"  ✅ Extracted ToyCar dataset: {toycar_dir} ({n_wavs} training WAV files)")
    return toycar_dir


# ──────────────────────────────────────────────────────────────────────────────
# Feature extraction (matches training pipeline in common.py)
# ──────────────────────────────────────────────────────────────────────────────
def file_to_vector_array(file_path,
                         n_mels=N_MELS,
                         frames=FRAMES,
                         n_fft=N_FFT,
                         hop_length=HOP_LENGTH,
                         power=POWER):
    """Extract log-mel spectrogram features from a WAV file.

    Returns
    -------
    np.ndarray, shape (num_vectors, n_mels * frames)
        Concatenated sliding-window feature vectors.
    """
    y, sr = librosa.load(file_path, sr=None, mono=True)

    # Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=power,
    )

    # Log transform
    log_mel_spectrogram = 20.0 / power * np.log10(
        np.maximum(mel_spectrogram, sys.float_info.epsilon)
    )

    # Central crop: take frames 50..250
    if log_mel_spectrogram.shape[1] > 250:
        log_mel_spectrogram = log_mel_spectrogram[:, 50:250]

    # Sliding window concatenation
    dims = n_mels * frames
    n_frames = log_mel_spectrogram.shape[1]
    vector_array_size = n_frames - frames + 1
    if vector_array_size < 1:
        return np.empty((0, dims), dtype=np.float32)

    vector_array = np.zeros((vector_array_size, dims), dtype=np.float32)
    for t in range(vector_array_size):
        vector_array[t, :] = log_mel_spectrogram[:, t:t + frames].T.flatten()

    return vector_array


# ──────────────────────────────────────────────────────────────────────────────
# Model loading and BN folding
# ──────────────────────────────────────────────────────────────────────────────
def load_and_fold_model(h5_path: str) -> tf.keras.Model:
    """Load the .h5 Keras model and fold BatchNormalization into Dense layers.

    This matches the reference 02_convert.py approach from the MLCommons Tiny
    benchmark.  After folding, all BN layers are removed, resulting in a
    simpler model that is more efficient for TFLite conversion.
    """
    model = keras.models.load_model(h5_path)
    print(f"  Original model loaded: {h5_path}")
    model.summary()

    print("\n  Folding BatchNormalization layers into Dense layers...")
    h = model.input
    skip = False
    for i in range(len(model.layers)):
        if skip:
            skip = False
            continue
        if isinstance(model.layers[i], keras.layers.Dense):
            if (i < len(model.layers) - 1 and
                    isinstance(model.layers[i + 1], keras.layers.BatchNormalization)):
                kernel, bias = model.layers[i].get_weights()
                gamma, beta, moving_mean, moving_variance = (
                    model.layers[i + 1].get_weights()
                )

                folded_kernel_multiplier = gamma * math_ops.rsqrt(
                    moving_variance + model.layers[i + 1].epsilon
                )
                folded_kernel = math_ops.mul(
                    folded_kernel_multiplier, kernel, name="folded_kernel"
                )
                folded_bias = math_ops.subtract(
                    beta,
                    moving_mean * folded_kernel_multiplier,
                    name="folded_bias",
                )

                model.layers[i].set_weights([folded_kernel, folded_bias])
                skip = True

        h = model.layers[i](h)

    folded_model = keras.Model(inputs=model.input, outputs=h)
    print("  ✅ BN folding complete")
    folded_model.summary()
    return folded_model


# ──────────────────────────────────────────────────────────────────────────────
# TFLite conversion
# ──────────────────────────────────────────────────────────────────────────────
def convert_fp32(model: tf.keras.Model, output_path: str) -> str:
    """Convert Keras model to FP32 TFLite."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    size_kb = os.path.getsize(output_path) / 1024
    print(f"  ✅ FP32 TFLite saved: {output_path} ({size_kb:.1f} KB)")
    return output_path


def convert_int8(model: tf.keras.Model, output_path: str,
                 calib_data: np.ndarray) -> str:
    """Convert Keras model to fully quantised INT8 TFLite.

    Parameters
    ----------
    model : tf.keras.Model
        The Keras model (with BN already folded).
    output_path : str
        Destination path for the .tflite file.
    calib_data : np.ndarray, shape (N, 640)
        Calibration feature vectors.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    def representative_dataset_gen():
        samples = calib_data[::5]
        iterator = tqdm(samples, desc="  INT8 calibration", unit="sample") if _HAS_TQDM else samples
        for sample in iterator:
            yield [np.expand_dims(sample.astype(np.float32), axis=0)]

    converter.representative_dataset = representative_dataset_gen
    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    size_kb = os.path.getsize(output_path) / 1024
    print(f"  ✅ INT8 TFLite saved: {output_path} ({size_kb:.1f} KB)")
    return output_path


def verify_tflite(path: str, tag: str) -> None:
    """Load a TFLite model and print IO details."""
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    print(f"  [{tag}] input  → shape={list(inp['shape'])}, "
          f"dtype={inp['dtype'].__name__}")
    print(f"  [{tag}] output → shape={list(out['shape'])}, "
          f"dtype={out['dtype'].__name__}")

    if inp["dtype"] != np.float32:
        s, z = inp["quantization"]
        print(f"  [{tag}] input  quant: scale={s:.6f}, zero_point={z}")
    if out["dtype"] != np.float32:
        s, z = out["quantization"]
        print(f"  [{tag}] output quant: scale={s:.6f}, zero_point={z}")

    print(f"  [{tag}] ✓ verification passed")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main(mode: str = "all", calib_num: int = 0):
    """
    Download the AD .h5 model and convert to TFLite.

    Parameters
    ----------
    mode : str
        "fp32" → FP32 TFLite only.
        "int8" → INT8 TFLite only.
        "all"  → Both FP32 + INT8 (default).
    calib_num : int
        Max calibration samples for INT8 (0 = use all available).
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Step 1: Download .h5 model ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 1: Downloading AD Autoencoder Keras (.h5) model")
    print("=" * 60)
    download_file(H5_URL, H5_PATH)

    # ── Step 2: Load and fold BN ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Loading model and folding BatchNormalization")
    print("=" * 60)
    model = load_and_fold_model(H5_PATH)

    # ── Step 3: Convert to FP32 TFLite ───────────────────────────────────
    if mode in ("fp32", "both", "all"):
        print("\n" + "=" * 60)
        print("STEP 3: Converting to FP32 TFLite")
        print("=" * 60)
        convert_fp32(model, FP32_OUT)
        verify_tflite(FP32_OUT, "FP32")

    # ── Step 4: Convert to INT8 TFLite ───────────────────────────────────
    if mode in ("int8", "both", "all"):
        step = "4" if mode in ("both", "all") else "3"
        print("\n" + "=" * 60)
        print(f"STEP {step}: Converting to INT8 TFLite (with calibration)")
        print("=" * 60)

        # Ensure the DCASE ToyCar dataset is available
        print("  Ensuring calibration dataset (ToyCar training data)...")
        toycar_dir = ensure_dataset(DATASET_DIR)
        train_dir = os.path.join(toycar_dir, "train")

        # Build calibration feature vectors
        train_files = sorted(glob.glob(os.path.join(train_dir, "*.wav")))
        if not train_files:
            print("  ❌ No training WAV files found for calibration!")
            sys.exit(1)

        if calib_num > 0:
            # Sub-sample for faster calibration
            step = max(1, len(train_files) // calib_num)
            train_files = train_files[::step][:calib_num]

        print(f"  Extracting features from {len(train_files)} training files...")
        all_vectors = []
        iterator = (tqdm(train_files, desc="  Feature extraction", unit="wav")
                    if _HAS_TQDM else train_files)
        for wav_path in iterator:
            vectors = file_to_vector_array(wav_path)
            all_vectors.append(vectors)

        calib_data = np.concatenate(all_vectors, axis=0)
        print(f"  Calibration data shape: {calib_data.shape}")

        convert_int8(model, INT8_OUT, calib_data)
        verify_tflite(INT8_OUT, "INT8")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("All done!")
    print("=" * 60)
    if mode in ("fp32", "both", "all") and os.path.isfile(FP32_OUT):
        print(f"  FP32 : {FP32_OUT}")
    if mode in ("int8", "both", "all") and os.path.isfile(INT8_OUT):
        print(f"  INT8 : {INT8_OUT}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download AD .h5 model and convert to TFLite."
    )
    parser.add_argument(
        "--mode",
        choices=["fp32", "int8", "both", "FP32", "INT8", "BOTH"],
        default="both",
        help="Conversion mode: fp32, int8, or all (default).",
    )
    parser.add_argument(
        "--calib-num",
        type=int,
        default=0,
        help="Max calibration files for INT8 (0 = use all). Default: 0.",
    )
    args = parser.parse_args()
    main(args.mode, args.calib_num)
