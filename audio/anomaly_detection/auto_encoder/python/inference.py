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
Run inference with the Anomaly Detection (AD) Dense Autoencoder TFLite
model (float32 or int8).

The model is a fully-connected autoencoder that reconstructs log-mel
spectrogram features.  The anomaly score for a WAV file is the mean
reconstruction error (MSE) across all sliding-window feature vectors.

Feature extraction:
  - librosa mel spectrogram: n_mels=128, n_fft=1024, hop_length=512, power=2.0
  - log-mel energy: 20/power * log10(mel + eps)
  - central crop [:, 50:250]  →  200 frames
  - sliding window of 5 frames → vectors of dim 128 × 5 = 640
  - input shape per vector: (1, 640)

Example:
  python inference.py --audio sample.wav
  python inference.py --model model/ad01_FP32.tflite --audio sample.wav
  python inference.py --model model/ad01_INT8.tflite --audio sample.wav --verbose
"""
import argparse
import os
import sys

import numpy as np

try:
    import tensorflow as tf
except Exception:
    print("TensorFlow is required. Install with: pip install tensorflow")
    raise

try:
    import librosa
except ImportError:
    print("librosa is required. Install with: pip install librosa")
    raise


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "model", "ad01_FP32.tflite")

# Feature extraction parameters (must match training)
N_MELS     = 128
FRAMES     = 5
N_FFT      = 1024
HOP_LENGTH = 512
POWER      = 2.0
INPUT_DIM  = N_MELS * FRAMES  # 640


# ──────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────────────────────────────────────
def file_to_vector_array(file_path: str,
                         n_mels: int = N_MELS,
                         frames: int = FRAMES,
                         n_fft: int = N_FFT,
                         hop_length: int = HOP_LENGTH,
                         power: float = POWER) -> np.ndarray:
    """Extract log-mel spectrogram features from a WAV file.

    Parameters
    ----------
    file_path : str
        Path to a WAV file.

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
    n_time = log_mel_spectrogram.shape[1]
    vector_array_size = n_time - frames + 1
    if vector_array_size < 1:
        return np.empty((0, dims), dtype=np.float32)

    vector_array = np.zeros((vector_array_size, dims), dtype=np.float32)
    for t in range(vector_array_size):
        vector_array[t, :] = log_mel_spectrogram[:, t:t + frames].T.flatten()

    return vector_array


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────
def load_interpreter(model_path: str):
    """Load TFLite model and return interpreter + input/output details."""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    return interpreter, input_details, output_details


def predict(interpreter, input_details, output_details, data: np.ndarray):
    """Run autoencoder inference on feature vectors.

    Parameters
    ----------
    data : np.ndarray, shape (N, 640)
        Input feature vectors.

    Returns
    -------
    np.ndarray, shape (N, 640)
        Reconstructed feature vectors.
    """
    is_int8 = input_details["dtype"] == np.int8

    output_data = np.empty_like(data, dtype=np.float32)

    for i in range(data.shape[0]):
        sample = data[i:i + 1, :].astype(np.float32)

        if is_int8:
            in_scale, in_zp = input_details["quantization"]
            sample_q = np.clip(
                np.round(sample / in_scale) + in_zp, -128, 127
            ).astype(np.int8)
            interpreter.set_tensor(input_details["index"], sample_q)
        else:
            interpreter.set_tensor(input_details["index"], sample)

        interpreter.invoke()
        raw_out = interpreter.get_tensor(output_details["index"])

        if output_details["dtype"] == np.int8:
            out_scale, out_zp = output_details["quantization"]
            output_data[i:i + 1, :] = (
                raw_out.astype(np.float32) - out_zp
            ) * out_scale
        else:
            output_data[i:i + 1, :] = raw_out

    return output_data


def compute_anomaly_score(data: np.ndarray, reconstructed: np.ndarray) -> float:
    """Compute anomaly score as the mean MSE across all vectors.

    Parameters
    ----------
    data : np.ndarray, shape (N, 640)
        Original feature vectors.
    reconstructed : np.ndarray, shape (N, 640)
        Reconstructed feature vectors.

    Returns
    -------
    float
        Anomaly score (mean of per-vector MSEs).
    """
    errors = np.mean(np.square(data - reconstructed), axis=1)
    return float(np.mean(errors))


def run_inference(model_path: str, audio_path: str, verbose: bool = False):
    """Run anomaly detection inference on a single WAV file.

    Parameters
    ----------
    model_path : str
        Path to TFLite model (.tflite).
    audio_path : str
        Path to WAV file.
    verbose : bool
        Print detailed feature statistics.

    Returns
    -------
    float
        Anomaly score (mean reconstruction MSE).
    """
    # ── Load and preprocess audio ────────────────────────────────────────
    print(f"Loading audio: {audio_path}")
    data = file_to_vector_array(audio_path)

    if data.shape[0] == 0:
        print("  ⚠️  Audio file too short — no feature vectors extracted.")
        return float("nan")

    print(f"  Feature vectors: {data.shape[0]} × {data.shape[1]}")

    if verbose:
        print(f"  Feature stats: min={data.min():.4f}, max={data.max():.4f}, "
              f"mean={data.mean():.4f}, std={data.std():.4f}")

    # ── Load model ───────────────────────────────────────────────────────
    print(f"Loading model: {model_path}")
    interpreter, input_details, output_details = load_interpreter(model_path)

    is_int8 = input_details["dtype"] == np.int8
    tag = "INT8" if is_int8 else "FP32"

    if verbose:
        print(f"  [{tag}] input  → shape={list(input_details['shape'])}, "
              f"dtype={input_details['dtype'].__name__}")
        print(f"  [{tag}] output → shape={list(output_details['shape'])}, "
              f"dtype={output_details['dtype'].__name__}")
        if is_int8:
            in_s, in_z = input_details["quantization"]
            out_s, out_z = output_details["quantization"]
            print(f"  [{tag}] input  quant: scale={in_s:.6f}, zero_point={in_z}")
            print(f"  [{tag}] output quant: scale={out_s:.6f}, zero_point={out_z}")

    # ── Run autoencoder inference ────────────────────────────────────────
    print("Running inference...")
    reconstructed = predict(interpreter, input_details, output_details, data)

    # ── Compute anomaly score ────────────────────────────────────────────
    anomaly_score = compute_anomaly_score(data, reconstructed)

    per_vector_errors = np.mean(np.square(data - reconstructed), axis=1)

    # ── Derive ground truth from filename ────────────────────────────────
    fname = os.path.basename(audio_path).lower()
    if fname.startswith("anomaly"):
        ground_truth = "ANOMALY"
    elif fname.startswith("normal"):
        ground_truth = "NORMAL"
    else:
        ground_truth = "UNKNOWN"

    # ── Predicted label ──────────────────────────────────────────────────
    threshold = 10.7  # default threshold — tune per machine ID
    predicted = "ANOMALY" if anomaly_score >= threshold else "NORMAL"
    correct = (predicted == ground_truth) if ground_truth != "UNKNOWN" else None

    print(f"\n{'=' * 50}")
    print(f"  ANOMALY DETECTION RESULT")
    print(f"{'=' * 50}")
    print(f"  Audio file     : {os.path.basename(audio_path)}")
    print(f"  Model          : {os.path.basename(model_path)} ({tag})")
    print(f"  Feature vectors: {data.shape[0]}")
    print(f"  Anomaly score  : {anomaly_score:.6f}")
    print(f"  Per-vector MSE : min={per_vector_errors.min():.6f}, "
          f"max={per_vector_errors.max():.6f}")
    print(f"  Threshold      : {threshold:.2f}")
    print(f"  Predicted      : {predicted}")
    print(f"  Ground truth   : {ground_truth}")
    if correct is not None:
        print(f"  Result         : {'✅ CORRECT' if correct else '❌ WRONG'}")
    print(f"{'=' * 50}")

    return anomaly_score


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Anomaly Detection inference on a single WAV file."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Path to TFLite model. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to WAV file for anomaly detection.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed model IO and feature statistics.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print(f"Error: Model not found: {args.model}")
        print("Run download_pre_model.py first to download the pre-trained model.")
        sys.exit(1)

    if not os.path.isfile(args.audio):
        print(f"Error: Audio file not found: {args.audio}")
        sys.exit(1)

    run_inference(args.model, args.audio, args.verbose)


if __name__ == "__main__":
    main()
