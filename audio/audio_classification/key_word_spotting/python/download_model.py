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
# The pre-trained KWS model is from the MLCommons Tiny benchmark:
#   https://github.com/mlcommons/tiny
# Licensed under the Apache License, Version 2.0.
"""
Download the KWS DS-CNN SavedModel from MLCommons Tiny and convert it to
TFLite (float32 and/or int8).

The SavedModel is downloaded from:
  https://github.com/mlcommons/tiny/tree/master/benchmark/training/
  keyword_spotting/trained_models/kws_ref_model

Int8 quantization uses WAV files from the Google Speech Commands v0.02
validation split as the calibration dataset.  The script will automatically
download and extract the dataset if the calibration directory is not found.

Outputs (in model/):
  kws_ref_model_FP32.tflite  – float32 model
  kws_ref_model_INT8.tflite  – fully int8 quantized model (input/output int8)

Usage:
  python convert_model.py                           # both fp32 + int8
  python convert_model.py --mode fp32               # fp32 only
  python convert_model.py --mode int8               # int8 only
  python convert_model.py --mode int8 --calib-num 500
  python convert_model.py --calib-dir Datasets/speech_commands_val
"""

import argparse
import glob
import os
import sys
import urllib.request
import wave

import numpy as np

try:
    import tensorflow as tf
except Exception:
    print("TensorFlow is required to run this script. "
          "Install it with: pip install tensorflow")
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

FP32_OUT = os.path.join(OUTPUT_DIR, "kws_ref_model_FP32.tflite")
INT8_OUT = os.path.join(OUTPUT_DIR, "kws_ref_model_INT8.tflite")

SAVED_MODEL_DIR = os.path.join(OUTPUT_DIR, "kws_ref_model")

# GitHub raw URLs for the SavedModel files
_BASE_RAW = (
    "https://github.com/mlcommons/tiny/raw/master/"
    "benchmark/training/keyword_spotting/trained_models/kws_ref_model"
)
SAVED_MODEL_FILES = {
    "saved_model.pb":                         f"{_BASE_RAW}/saved_model.pb",
    "variables/variables.data-00000-of-00001": f"{_BASE_RAW}/variables/variables.data-00000-of-00001",
    "variables/variables.index":               f"{_BASE_RAW}/variables/variables.index",
}

# Calibration directory — validation WAV files (same pattern as other models)
CALIB_DIR = os.path.join("Datasets", "speech_commands_val")

# 12-class keyword labels – order must match the training label map
# (Google Speech Commands v0.02 via tensorflow_datasets)
KWS_LABELS = [
    "down", "go", "left", "no", "off", "on",
    "right", "stop", "up", "yes", "silence", "unknown",
]

# Audio / feature extraction settings (must match training)
SAMPLE_RATE            = 16000
DESIRED_SAMPLES        = 16000      # 1 second @ 16 kHz
WINDOW_SIZE_SAMPLES    = 480        # 30 ms
WINDOW_STRIDE_SAMPLES  = 320        # 20 ms
NUM_MEL_BINS           = 40
DCT_COEFFICIENT_COUNT  = 10
LOWER_EDGE_HZ          = 20.0
UPPER_EDGE_HZ          = 4000.0


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


def list_wavs(folder: str, need: int) -> list:
    """Return up to *need* .wav file paths from *folder* (recursive)."""
    if not os.path.isdir(folder):
        return []
    out = []
    for root, _, files in os.walk(folder):
        for f in sorted(files):
            if f.lower().endswith(".wav"):
                out.append(os.path.join(root, f))
                if len(out) >= need:
                    return out
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 1. Download SavedModel
# ══════════════════════════════════════════════════════════════════════════════
def download_saved_model(dest_dir: str) -> str:
    """Download the KWS DS-CNN SavedModel from GitHub if not already present.

    Returns the path to the SavedModel directory.
    """
    pb_path = os.path.join(dest_dir, "saved_model.pb")
    if os.path.isfile(pb_path):
        print(f"  ✅ SavedModel already exists: {dest_dir} — skipping download.")
        return dest_dir

    print(f"  Downloading SavedModel to {dest_dir} ...")
    for rel_path, url in SAVED_MODEL_FILES.items():
        dest = os.path.join(dest_dir, rel_path)
        _download_with_progress(url, dest)

    print(f"  ✅ SavedModel downloaded: {dest_dir}")
    return dest_dir


# ══════════════════════════════════════════════════════════════════════════════
# 2. MFCC feature extraction (matches training pipeline)
# ══════════════════════════════════════════════════════════════════════════════
def extract_mfcc(audio: np.ndarray) -> np.ndarray:
    """Extract MFCC features from a float32 audio waveform.

    Parameters
    ----------
    audio : np.ndarray, shape (16000,), float32 normalised to [-1, 1]

    Returns
    -------
    np.ndarray, shape (49, 10, 1), float32
    """
    stfts = tf.signal.stft(
        audio,
        frame_length=WINDOW_SIZE_SAMPLES,
        frame_step=WINDOW_STRIDE_SAMPLES,
        window_fn=tf.signal.hann_window,
    )
    spectrograms = tf.abs(stfts)
    num_spectrogram_bins = stfts.shape[-1]

    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        NUM_MEL_BINS,
        num_spectrogram_bins,
        SAMPLE_RATE,
        LOWER_EDGE_HZ,
        UPPER_EDGE_HZ,
    )
    mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)
    mfccs = mfccs[..., :DCT_COEFFICIENT_COUNT]
    mfccs = tf.reshape(mfccs, [mfccs.shape[0], DCT_COEFFICIENT_COUNT, 1])
    return mfccs.numpy()


def preprocess_wav(wav_path: str) -> np.ndarray:
    """Load a WAV file and return MFCC features of shape (49, 10, 1).

    Replicates the training preprocessing:
      audio_float = audio.astype(float32)
      audio_float = audio_float / max(audio_float)
      audio_float = pad_or_truncate(audio_float, 16000)
      mfcc = extract_mfcc(audio_float)
    """
    raw = tf.io.read_file(wav_path)
    audio, sr = tf.audio.decode_wav(raw, desired_channels=1)
    audio = tf.squeeze(audio, axis=-1).numpy()  # shape (N,)

    # Normalise: divide by max (matching training code)
    max_val = np.max(audio)
    if max_val > 0:
        audio = audio / max_val
    elif max_val < 0:
        audio = audio / (-max_val)

    # Pad or truncate to 1 second
    if len(audio) < DESIRED_SAMPLES:
        audio = np.pad(audio, (0, DESIRED_SAMPLES - len(audio)))
    else:
        audio = audio[:DESIRED_SAMPLES]

    return extract_mfcc(audio)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Ensure calibration dataset (download Speech Commands if needed)
# ══════════════════════════════════════════════════════════════════════════════
def _extract_wavs_from_tfrecords(data_dir: str, output_dir: str) -> str:
    """Extract WAV files from the Speech Commands validation tfrecords
    into *output_dir* organised by keyword sub-folder.

    Returns the path to the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Find validation tfrecord shards
    pattern = os.path.join(data_dir, "speech_commands",
                           "*", "speech_commands-validation.tfrecord-*")
    shard_files = sorted(glob.glob(pattern))
    if not shard_files:
        raise FileNotFoundError(
            f"No validation tfrecord shards found matching: {pattern}"
        )

    print(f"  Found {len(shard_files)} validation shard(s)")

    raw_ds = tf.data.TFRecordDataset(shard_files)
    count = 0
    label_counts = {i: 0 for i in range(12)}

    items = raw_ds
    if _HAS_TQDM:
        items = tqdm(items, desc="  extracting WAVs", unit="sample")

    for raw_record in items:
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())

        label_idx = example.features.feature['label'].int64_list.value[0]
        audio_int16 = np.array(
            example.features.feature['audio'].int64_list.value,
            dtype=np.int16,
        )
        label_name = KWS_LABELS[label_idx] if label_idx < len(KWS_LABELS) \
            else f"class_{label_idx}"

        # Create sub-folder per keyword
        keyword_dir = os.path.join(output_dir, label_name)
        os.makedirs(keyword_dir, exist_ok=True)

        # Write WAV file
        wav_path = os.path.join(keyword_dir, f"{label_name}_{count:05d}.wav")
        with wave.open(wav_path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_int16.tobytes())

        label_counts[label_idx] = label_counts.get(label_idx, 0) + 1
        count += 1

    print(f"  ✅ Extracted {count} WAV files to {output_dir}")
    for i, name in enumerate(KWS_LABELS):
        print(f"     {name:>10s}: {label_counts.get(i, 0)}")
    return output_dir


def ensure_dataset(calib_dir: str) -> str:
    """Return a valid calibration directory with WAV files, downloading
    the Speech Commands v0.02 validation set if necessary.

    Priority:
      1. calib_dir already has WAV files  → use it as-is.
      2. Speech Commands tfrecords exist   → extract WAVs.
      3. Nothing exists                    → download via tfds, then extract.
    """
    # 1. Dataset already present
    if os.path.isdir(calib_dir) and list_wavs(calib_dir, need=1):
        n_wavs = len(list_wavs(calib_dir, need=99999))
        print(f"  ✅ Dataset found: {calib_dir} ({n_wavs} WAV files)")
        return calib_dir

    print(f"  ⚠️  Dataset not found at {calib_dir}")

    # Determine tfds data directory (check common locations)
    tfds_data_dir = os.path.join(os.getenv("HOME", "/tmp"), "data")

    # 2. Check if tfrecords already exist
    pattern = os.path.join(tfds_data_dir, "speech_commands",
                           "*", "speech_commands-validation.tfrecord-*")
    shard_files = sorted(glob.glob(pattern))

    if not shard_files:
        # 3. Download via tensorflow_datasets
        print(f"\n  Downloading Google Speech Commands v0.02 dataset ...")
        print(f"  (This may take a few minutes on the first run.)")
        print(f"  Data directory: {tfds_data_dir}")

        try:
            import tensorflow_datasets as tfds
        except ImportError:
            print("\n  ERROR: tensorflow_datasets is required to download "
                  "the Speech Commands dataset.")
            print("  Install it with: pip install tensorflow-datasets")
            print("  Or provide a directory of WAV files with --calib-dir.")
            sys.exit(1)

        tfds.load(
            "speech_commands",
            split="validation",
            data_dir=tfds_data_dir,
            download=True,
        )
        print(f"  ✅ Dataset downloaded to {tfds_data_dir}")

        # Re-check for tfrecord shards
        shard_files = sorted(glob.glob(pattern))
        if not shard_files:
            raise FileNotFoundError(
                f"Failed to find validation tfrecords after download. "
                f"Pattern: {pattern}"
            )

    # Extract WAVs from tfrecords into the calibration directory
    return _extract_wavs_from_tfrecords(tfds_data_dir, calib_dir)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Build representative dataset for int8 calibration
# ══════════════════════════════════════════════════════════════════════════════
def build_representative_dataset(calib_dir: str, num_samples: int):
    """Return a generator function that yields float32 MFCC features
    of shape (1, 49, 10, 1) preprocessed on-the-fly from WAV files.

    Balanced sampling: picks an equal number of WAVs from each keyword
    sub-folder (like VWW's person/notperson balanced sampling).
    """
    # Collect WAV files per sub-folder for balanced sampling
    subdirs = sorted([
        d for d in os.listdir(calib_dir)
        if os.path.isdir(os.path.join(calib_dir, d))
    ])

    if subdirs:
        # Balanced sampling across keyword sub-folders
        per_class = max(1, num_samples // len(subdirs))
        selected = []
        for subdir in subdirs:
            subdir_path = os.path.join(calib_dir, subdir)
            wavs = sorted([
                os.path.join(subdir_path, f)
                for f in os.listdir(subdir_path)
                if f.lower().endswith(".wav")
            ])
            # Evenly space the selection across the class
            step = max(1, len(wavs) // per_class)
            selected.extend(wavs[::step][:per_class])

        selected = selected[:num_samples]
    else:
        # Flat directory — just list WAVs
        selected = list_wavs(calib_dir, need=num_samples)

    if not selected:
        raise FileNotFoundError(
            f"No WAV files found in {calib_dir}"
        )

    print(f"  Calibration: {len(selected)} WAV files from {calib_dir}")
    if subdirs:
        print(f"    ({per_class} per class × {len(subdirs)} classes, "
              f"balanced sampling)")

    def gen():
        items = selected
        if _HAS_TQDM:
            items = tqdm(items, desc="  calibrating", unit="sample")
        for path in items:
            mfcc = preprocess_wav(path).astype(np.float32)  # (49, 10, 1)
            yield [mfcc[np.newaxis, ...]]  # shape (1, 49, 10, 1)

    return gen


# ══════════════════════════════════════════════════════════════════════════════
# 5. Convert to TFLite
# ══════════════════════════════════════════════════════════════════════════════
def _get_converter(saved_model_dir: str):
    """Return a TFLite converter, falling back to the v1 API if v2 fails."""
    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        return converter
    except (AttributeError, Exception) as e:
        print(f"  ⚠️  v2 converter failed ({e}), using tf.compat.v1 fallback")
        converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model(
            saved_model_dir)
        return converter


def convert_float32(saved_model_dir: str, output_path: str):
    """SavedModel → float32 TFLite."""
    converter = _get_converter(saved_model_dir)
    buf = converter.convert()
    with open(output_path, "wb") as f:
        f.write(buf)
    print(f"  ✓ float32 : {output_path}  ({len(buf)/1024:.1f} KB)")
    return buf


def convert_int8(saved_model_dir: str, output_path: str, rep_dataset_gen):
    """SavedModel → fully int8 quantized TFLite (input/output also int8)."""
    converter = _get_converter(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    buf = converter.convert()
    with open(output_path, "wb") as f:
        f.write(buf)
    print(f"  ✓ int8    : {output_path}  ({len(buf)/1024:.1f} KB)")
    return buf


# ══════════════════════════════════════════════════════════════════════════════
# 6. Verify
# ══════════════════════════════════════════════════════════════════════════════
def verify_tflite(path: str, tag: str) -> None:
    """Load a TFLite model and print IO details + run dummy inference."""
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

    # Sanity check shapes
    expected_input_shape = [1, 49, 10, 1]
    expected_output_shape = [1, 12]
    assert list(inp["shape"]) == expected_input_shape, \
        f"Unexpected input shape: {list(inp['shape'])} " \
        f"(expected {expected_input_shape})"
    assert list(out["shape"]) == expected_output_shape, \
        f"Unexpected output shape: {list(out['shape'])} " \
        f"(expected {expected_output_shape})"

    # Dummy inference
    dummy_input = np.zeros(expected_input_shape, dtype=inp["dtype"])
    interp.set_tensor(inp["index"], dummy_input)
    interp.invoke()
    output = interp.get_tensor(out["index"])

    if inp["dtype"] == np.float32:
        probs = output[0]
        predicted = int(np.argmax(probs))
        print(f"  [{tag}] dummy inference → class {predicted} "
              f"({KWS_LABELS[predicted]}), conf={probs[predicted]:.4f}")
    else:
        out_scale, out_zp = out["quantization"]
        probs_f = (output[0].astype(np.float32) - out_zp) * out_scale
        predicted = int(np.argmax(probs_f))
        print(f"  [{tag}] dummy inference → class {predicted} "
              f"({KWS_LABELS[predicted]}), conf={probs_f[predicted]:.4f}")

    print(f"  [{tag}] ✓ verification passed")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main(mode: str = "all", calib_num: int = 1000,
         calib_dir: str = CALIB_DIR):
    """
    Download the KWS SavedModel and convert to TFLite.

    Parameters
    ----------
    mode : str
        "fp32" → FP32 TFLite only.
        "int8" → INT8 TFLite only.
        "all"  → Both FP32 + INT8.
    calib_num : int
        Number of calibration samples for INT8 quantization.
    calib_dir : str
        Path to directory with validation WAV files.
        Auto-downloaded from Speech Commands v0.02 if not found.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Step 1: Download SavedModel ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 1: Downloading KWS SavedModel")
    print("=" * 60)
    sm_dir = download_saved_model(SAVED_MODEL_DIR)

    # ── Step 2: Verify SavedModel loads correctly ──────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Verifying SavedModel")
    print("=" * 60)
    try:
        loaded = tf.saved_model.load(sm_dir)
        sigs = list(loaded.signatures.keys())
        print(f"  SavedModel loaded: {sm_dir}")
        print(f"  Signatures       : {sigs}")
        if "serving_default" in loaded.signatures:
            sig = loaded.signatures["serving_default"]
            inp_info = {
                k: v.shape.as_list()
                for k, v in sig.structured_input_signature[1].items()
            }
            out_info = {
                k: v.shape.as_list()
                for k, v in sig.structured_outputs.items()
            }
            print(f"  Input  shapes    : {inp_info}")
            print(f"  Output shapes    : {out_info}")
    except Exception as e:
        print(f"  ⚠️  SavedModel verification skipped (non-fatal): {e}")
        print(f"  Proceeding with conversion — TFLiteConverter handles this separately.")

    # ── Step 3: Ensure calibration dataset (INT8 only) ───────────────────
    if mode in ("int8", "both", "all"):
        print("\n" + "=" * 60)
        print("STEP 3: Ensuring calibration dataset")
        print("=" * 60)
        calib_dir = ensure_dataset(calib_dir)

    # ── Step 4: Convert to FP32 TFLite ───────────────────────────────────
    if mode in ("fp32", "both", "all"):
        step_num = 3 if mode == "fp32" else 4
        print("\n" + "=" * 60)
        print(f"STEP {step_num}: Converting to FP32 TFLite")
        print("=" * 60)
        convert_float32(sm_dir, FP32_OUT)
        verify_tflite(FP32_OUT, "FP32")

    # ── Step 5: Convert to INT8 TFLite ───────────────────────────────────
    if mode in ("int8", "both", "all"):
        step_num = 4 if mode == "int8" else 5
        print("\n" + "=" * 60)
        print(f"STEP {step_num}: Converting to INT8 TFLite")
        print("=" * 60)
        rep_gen = build_representative_dataset(calib_dir, calib_num)
        convert_int8(sm_dir, INT8_OUT, rep_gen)
        verify_tflite(INT8_OUT, "INT8")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("All done!")
    print("=" * 60)
    if mode in ("fp32", "both", "all"):
        print(f"  FP32 : {FP32_OUT}")
    if mode in ("int8", "both", "all"):
        print(f"  INT8 : {INT8_OUT}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download KWS DS-CNN SavedModel from MLCommons Tiny "
                    "& convert to TFLite (FP32 / INT8)."
    )
    parser.add_argument(
        "--mode",
        choices=["fp32", "int8", "both", "FP32", "INT8", "BOTH"],
        default="both",
        help="Export mode: fp32 (FP32 only), int8 (INT8 only), all (both). "
             "Default: all.",
    )
    parser.add_argument(
        "--calib-num",
        type=int,
        default=1000,
        help="Number of calibration samples for INT8 quantization "
             "(default: 1000).",
    )
    parser.add_argument(
        "--calib-dir",
        type=str,
        default=CALIB_DIR,
        help=f"Path to directory with validation WAV files "
             f"(default: {CALIB_DIR}). Auto-downloaded from "
             f"Speech Commands v0.02 if not found.",
    )
    args = parser.parse_args()
    main(args.mode, args.calib_num, args.calib_dir)
