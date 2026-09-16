# SPDX-License-Identifier: Apache-2.0
"""Download and convert split-input YAMNet TFLite models.

This script builds the 96x64 log-mel patch YAMNet graph used for 
Quantization and exports it to TFLite in FP32 and INT16 formats.

  - FP32 split model  : Model/yamnet_FP32.tflite
  - INT16 split model : Model/yamnet_INT16.tflite
  - AudioSet class map: Model/yamnet_class_map.csv
  - YAMNet weights    : utils/yamnet.h5

The INT16 model uses the experimental TFLite 16x8 path:
  activations = int16, weights = int8, bias = int64, I/O = int16.

Usage:
    python download_model.py
    python download_model.py --mode fp32
    python download_model.py --mode int16
    python download_model.py --mode int16 --audio-dir <path/to/16k_wavs> --calib-count 3000
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import threading
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, List

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import h5py
import numpy as np
import tensorflow as tf

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


BASE_DIR = Path(__file__).resolve().parent
WORK_DIR = BASE_DIR / "utils"
MODEL_DIR = BASE_DIR / "model"

FP32_TFLITE = MODEL_DIR / "yamnet_FP32.tflite"
INT16_TFLITE = MODEL_DIR / "yamnet_INT16.tflite"
CLASS_MAP_CSV = MODEL_DIR / "yamnet_class_map.csv"

GOOGLE_RAW = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet"
WEIGHTS_URL = "https://storage.googleapis.com/audioset/yamnet.h5"
CLASS_MAP_URL = f"{GOOGLE_RAW}/yamnet_class_map.csv"

SAMPLE_RATE = 16000
PATCH_SHAPE = (96, 64)


@contextmanager
def spinner(message: str):
    if not sys.stdout.isatty():
        print(message)
        yield
        return

    stop = threading.Event()

    def run() -> None:
        frames = "|/-\\"
        idx = 0
        while not stop.is_set():
            sys.stdout.write(f"\r{message} {frames[idx % len(frames)]}")
            sys.stdout.flush()
            idx += 1
            time.sleep(0.1)
        sys.stdout.write(f"\r{message} done\n")
        sys.stdout.flush()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()


def download_if_missing(url: str, dest: Path, label: str) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        print(f"  Reusing {label}: {dest}")
        return dest
    print(f"  Downloading {label} -> {dest}")
    if tqdm is None:
        urllib.request.urlretrieve(url, dest)
    else:
        with tqdm(total=None, unit="B", unit_scale=True, desc=f"  {label}") as bar:
            last = 0

            def hook(blocks: int, block_size: int, total: int) -> None:
                nonlocal last
                if total > 0:
                    bar.total = total
                downloaded = blocks * block_size
                bar.update(max(0, downloaded - last))
                last = downloaded

            urllib.request.urlretrieve(url, dest, reporthook=hook)
    return dest


def ensure_yamnet_assets() -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    download_if_missing(f"{GOOGLE_RAW}/params.py", WORK_DIR / "params.py", "params.py")
    download_if_missing(f"{GOOGLE_RAW}/yamnet.py", WORK_DIR / "yamnet.py", "yamnet.py")
    download_if_missing(f"{GOOGLE_RAW}/features.py", WORK_DIR / "features.py", "features.py")
    download_if_missing(WEIGHTS_URL, WORK_DIR / "yamnet.h5", "yamnet.h5")
    download_if_missing(CLASS_MAP_URL, CLASS_MAP_CSV, "yamnet_class_map.csv")
    if str(WORK_DIR) not in sys.path:
        sys.path.insert(0, str(WORK_DIR))


def build_split_model():
    import params as params_mod
    import yamnet as yamnet_mod

    params_obj = params_mod.Params(sample_rate=SAMPLE_RATE, patch_hop_seconds=0.48)
    mel_input = tf.keras.Input(
        batch_size=1,
        shape=(params_obj.patch_frames, params_obj.patch_bands),
        dtype=tf.float32,
        name="mel_patch",
    )
    _predictions, embeddings = yamnet_mod.yamnet(mel_input, params_obj)
    classifier_input = tf.keras.layers.Reshape((1, 1, int(embeddings.shape[-1])))(embeddings)
    classifier = tf.keras.layers.Conv2D(
        filters=params_obj.num_classes,
        kernel_size=(1, 1),
        use_bias=True,
        name="classifier_conv",
    )(classifier_input)
    classifier = tf.keras.layers.Activation(
        activation=params_obj.classifier_activation,
        name="classifier_activation",
    )(classifier)
    predictions = tf.keras.layers.Reshape(
        (params_obj.num_classes,),
        name="classifier_output",
    )(classifier)
    model = tf.keras.Model(inputs=mel_input, outputs=predictions, name="yamnet_mel_patch")
    model.load_weights(str(WORK_DIR / "yamnet.h5"), by_name=True, skip_mismatch=True)

    with h5py.File(WORK_DIR / "yamnet.h5", "r") as hf:
        kernel = hf["logits/logits/kernel:0"][()]
        bias = hf["logits/logits/bias:0"][()]
    classifier_layer = model.get_layer("classifier_conv")
    classifier_layer.set_weights([kernel.reshape(1, 1, *kernel.shape), bias])
    print("  Loaded logits into equivalent 1x1 classifier Conv2D")
    return model, params_obj


def log_mel_patches_from_wav(path: str, params_mod, features_mod) -> np.ndarray | None:
    try:
        wav_bin = tf.io.read_file(path)
        wav, sr = tf.audio.decode_wav(wav_bin, desired_channels=1)
    except Exception:
        return None
    samples = tf.squeeze(wav, axis=-1).numpy()
    if sr.numpy() != SAMPLE_RATE:
        return None
    params_obj = params_mod.Params(sample_rate=SAMPLE_RATE, patch_hop_seconds=0.48)
    _spec, patches = features_mod.waveform_to_log_mel_spectrogram_patches(samples, params_obj)
    patches = patches.numpy()
    if patches.shape[0] == 0:
        return None
    return patches.astype(np.float32)


def list_calibration_wavs(folder: Path) -> List[str]:
    if not folder.is_dir():
        return []
    return sorted(glob.glob(str(folder / "*.wav")))


def make_representative_dataset(audio_dir: Path | None, count: int):
    import features as features_mod
    import params as params_mod

    patches: List[np.ndarray] = []
    wavs = list_calibration_wavs(audio_dir) if audio_dir else []
    if wavs:
        rng = np.random.default_rng(0)
        wavs_for_calib = wavs.copy()
        rng.shuffle(wavs_for_calib)
        total_patches_seen = 0
        scanned_wavs = 0
        patch_bar = tqdm(total=count, desc="  Real calib patches", unit="patch") if tqdm is not None else None
        for wav_path in wavs_for_calib:
            scanned_wavs += 1
            file_patches = log_mel_patches_from_wav(wav_path, params_mod, features_mod)
            if file_patches is None:
                continue
            total_patches_seen += file_patches.shape[0]

            # Add only what we still need; stop scanning WAVs once target is reached.
            remaining = count - len(patches)
            if remaining <= 0:
                break
            if file_patches.shape[0] <= remaining:
                patches.extend([patch.astype(np.float32) for patch in file_patches])
                if patch_bar is not None:
                    patch_bar.update(file_patches.shape[0])
            else:
                pick = rng.choice(file_patches.shape[0], size=remaining, replace=False)
                patches.extend([file_patches[i].astype(np.float32) for i in pick])
                if patch_bar is not None:
                    patch_bar.update(remaining)
                break

        if patch_bar is not None:
            patch_bar.close()

        print(
            f"  Calibration: {len(patches)} real patches from {total_patches_seen} seen patches "
            f"across {scanned_wavs} scanned WAVs"
        )

    rng = np.random.default_rng(0)
    synth_needed = count - len(patches)
    if synth_needed > 0:
        print(f"WARNING: No real calibration patches found; backfilling {synth_needed} synthetic patches")
        synth_iter = range(synth_needed)
        if tqdm is not None:
            synth_iter = tqdm(synth_iter, desc="  Synth patches", unit="patch")
        for _ in synth_iter:
            patch = rng.standard_normal(PATCH_SHAPE, dtype=np.float32) * 3.0 - 5.0
            patches.append(patch.astype(np.float32))
    print(f"  Calibration total: {len(patches)} patches ({count} target)")

    def gen() -> Iterable[list[np.ndarray]]:
        selected = patches[:count]
        iterator = tqdm(selected, desc="  Calib -> converter", unit="patch") if tqdm is not None else selected
        for patch in iterator:
            yield [patch[np.newaxis, ...]]

    return gen


def convert_fp32(model: tf.keras.Model, dest: Path) -> Path:
    print(f"\n[FP32] Converting -> {dest}")
    with spinner("  Running FP32 TFLite conversion"):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite = converter.convert()
    dest.write_bytes(tflite)
    print(f"  Wrote {len(tflite)} bytes")
    return dest


def validate_int16_tflm_compatibility(model_content: bytes) -> None:
    interpreter = tf.lite.Interpreter(
        model_content=model_content,
        experimental_preserve_all_tensors=True,
    )
    interpreter.allocate_tensors()
    tensors = {detail["index"]: detail for detail in interpreter.get_tensor_details()}

    for node_index, op in enumerate(interpreter._get_ops_details()):
        if op["op_name"] != "FULLY_CONNECTED":
            continue

        input_tensor = tensors[int(op["inputs"][0])]
        weights_tensor = tensors[int(op["inputs"][1])]
        weight_scales = weights_tensor["quantization_parameters"]["scales"]
        if input_tensor["dtype"] == np.int16 and len(weight_scales) != 1:
            raise RuntimeError(
                f"TFLM-incompatible FULLY_CONNECTED at node {node_index}: "
                f"INT16 activations require per-tensor INT8 weights for the "
                f"CMSIS-NN s16 kernel, but converter emitted {len(weight_scales)} scales"
            )

    print("  TFLM compatibility: INT16 FullyConnected weights are per-tensor")


def convert_int16(model: tf.keras.Model, dest: Path, audio_dir: Path | None, calib_count: int) -> Path:
    print(f"\n[INT16] Converting -> {dest}")
    print(f"  Representative dataset: {calib_count} log-mel patches from {audio_dir}")
    with spinner("  Running INT16 TFLite conversion"):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = make_representative_dataset(audio_dir, calib_count)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
        ]
        converter.inference_input_type = tf.int16
        converter.inference_output_type = tf.int16
        tflite = converter.convert()
    validate_int16_tflm_compatibility(tflite)
    dest.write_bytes(tflite)
    print(f"  Wrote {len(tflite)} bytes")
    return dest


def validate_class_map() -> None:
    with open(CLASS_MAP_CSV, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != 521:
        raise RuntimeError(f"Expected 521 class-map rows, got {len(rows)}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and convert standalone split-input YAMNet models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", choices=["fp32", "int16", "both"], default="both")
    parser.add_argument("--audio-dir", default=None,
                        help="Directory of 16 kHz mono WAV files for INT16 calibration. "
                             "If omitted, synthetic patches are used (not recommended for accuracy).")
    parser.add_argument("--calib-count", type=int, default=3000,
                        help="Number of representative log-mel patches for INT16 calibration")
    args = parser.parse_args()

    print("[1/3] Ensuring YAMNet assets...")
    ensure_yamnet_assets()
    with spinner("  Validating class map"):
        validate_class_map()

    print("[2/3] Building split-input Keras model...")
    with spinner("  Building split-input Keras model"):
        model, _params_obj = build_split_model()

    print("[3/3] Exporting requested TFLite models...")
    audio_dir = Path(args.audio_dir) if args.audio_dir else None

    if args.mode in ("fp32", "both"):
        convert_fp32(model, FP32_TFLITE)
    if args.mode in ("int16", "both"):
        convert_int16(model, INT16_TFLITE, audio_dir, args.calib_count)

    print("\nDone.")
    print(f"  FP32 : {FP32_TFLITE}")
    print(f"  INT16: {INT16_TFLITE}")
    print(f"  Map  : {CLASS_MAP_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
