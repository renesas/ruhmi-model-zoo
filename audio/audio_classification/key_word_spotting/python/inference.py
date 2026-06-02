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
Run inference with the Keyword Spotting (KWS) DS-CNN TFLite model
(float32 or int8).

The model expects 10-coefficient MFCC features extracted from 1-second,
16 kHz mono audio.  Input shape: (1, 49, 10, 1).

Example:
  python inference.py --model model/kws_ref_model_FP32.tflite --audio sample.wav
  python inference.py --model model/kws_ref_model_INT8.tflite --audio sample.wav
  python inference.py --model model/kws_ref_model_FP32.tflite --audio sample.wav --verbose
"""
import argparse
import os
import sys

import numpy as np
import tensorflow as tf

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model", "kws_ref_model_FP32.tflite")

# 12-class keyword labels – order must match the training label map
# (Google Speech Commands v0.02 via tensorflow_datasets)
KWS_LABELS = [
    "down", "go", "left", "no", "off", "on",
    "right", "stop", "up", "yes", "silence", "unknown",
]

# Audio / feature extraction settings (must match training)
SAMPLE_RATE    = 16000      # 16 kHz
CLIP_DURATION  = 1.0        # 1 second
DESIRED_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION)  # 16000

WINDOW_SIZE_MS  = 30.0
WINDOW_STRIDE_MS = 20.0
WINDOW_SIZE_SAMPLES  = int(SAMPLE_RATE * WINDOW_SIZE_MS / 1000)    # 480
WINDOW_STRIDE_SAMPLES = int(SAMPLE_RATE * WINDOW_STRIDE_MS / 1000) # 320

NUM_MEL_BINS = 40
DCT_COEFFICIENT_COUNT = 10   # number of MFCC coefficients
LOWER_EDGE_HZ = 20.0
UPPER_EDGE_HZ = 4000.0

# Expected model I/O shapes
EXPECTED_INPUT_SHAPE  = [1, 49, 10, 1]
EXPECTED_OUTPUT_SHAPE = [1, 12]


# ──────────────────────────────────────────────────────────────────────────────
# Audio loading
# ──────────────────────────────────────────────────────────────────────────────
def load_wav(wav_path: str) -> np.ndarray:
    """Load a WAV file and return float32 samples normalised to [-1, 1].

    The audio is resampled to 16 kHz mono and zero-padded or truncated
    to exactly 1 second (16 000 samples).
    """
    raw = tf.io.read_file(wav_path)
    audio, sr = tf.audio.decode_wav(raw, desired_channels=1)
    audio = tf.squeeze(audio, axis=-1)  # (samples,)

    # Resample if needed (simple case: just verify rate)
    sr_val = sr.numpy()
    if sr_val != SAMPLE_RATE:
        raise ValueError(
            f"Expected {SAMPLE_RATE} Hz audio, got {sr_val} Hz. "
            "Please resample before running inference."
        )

    # Pad or truncate to desired length
    audio = audio.numpy().astype(np.float32)
    if len(audio) < DESIRED_SAMPLES:
        audio = np.pad(audio, (0, DESIRED_SAMPLES - len(audio)), mode="constant")
    else:
        audio = audio[:DESIRED_SAMPLES]

    # Normalise to [-1, 1]
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    return audio


# ──────────────────────────────────────────────────────────────────────────────
# MFCC feature extraction (matches training pipeline in get_dataset.py)
# ──────────────────────────────────────────────────────────────────────────────
def extract_mfcc(audio: np.ndarray) -> np.ndarray:
    """Extract MFCC features from a 1-second float32 audio waveform.

    Parameters
    ----------
    audio : np.ndarray, shape (16000,), float32 in [-1, 1]

    Returns
    -------
    np.ndarray, shape (49, 10, 1), float32
        49 time frames × 10 MFCC coefficients × 1 channel.
    """
    # STFT
    stfts = tf.signal.stft(
        audio,
        frame_length=WINDOW_SIZE_SAMPLES,
        frame_step=WINDOW_STRIDE_SAMPLES,
        fft_length=None,
        window_fn=tf.signal.hann_window,
    )
    spectrograms = tf.abs(stfts)
    num_spectrogram_bins = stfts.shape[-1]

    # Mel filterbank
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        NUM_MEL_BINS,
        num_spectrogram_bins,
        SAMPLE_RATE,
        LOWER_EDGE_HZ,
        UPPER_EDGE_HZ,
    )
    mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)

    # Log-mel
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

    # MFCCs — take first DCT_COEFFICIENT_COUNT
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)
    mfccs = mfccs[..., :DCT_COEFFICIENT_COUNT]

    # Reshape to (49, 10, 1)
    mfccs = tf.reshape(mfccs, [mfccs.shape[0], DCT_COEFFICIENT_COUNT, 1])

    return mfccs.numpy()


# ──────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ──────────────────────────────────────────────────────────────────────────────
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum(axis=-1, keepdims=True)


def load_interpreter(model_path: str):
    """Load TFLite model and return interpreter + input/output details."""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    return interpreter, input_details, output_details


def run_inference(model_path: str, audio_path: str, labels=None, verbose=False):
    """Run KWS inference on a single WAV file.

    Parameters
    ----------
    model_path : str
        Path to TFLite model (.tflite).
    audio_path : str
        Path to 16 kHz mono WAV file.
    labels : list of str, optional
        Class labels.  Defaults to KWS_LABELS.
    verbose : bool
        Print model IO details and feature stats.

    Returns
    -------
    np.ndarray
        Softmax probability vector of shape (12,).
    """
    labels = labels or KWS_LABELS

    # ── Load and preprocess audio ────────────────────────────────────────
    print(f"Loading audio: {audio_path}")
    audio = load_wav(audio_path)
    mfcc = extract_mfcc(audio)  # shape (49, 10, 1)

    if verbose:
        print(f"  Audio samples : {len(audio)}, range [{audio.min():.4f}, {audio.max():.4f}]")
        print(f"  MFCC shape    : {mfcc.shape}")
        print(f"  MFCC range    : [{mfcc.min():.4f}, {mfcc.max():.4f}]")

    # ── Load model ───────────────────────────────────────────────────────
    interpreter, input_details, output_details = load_interpreter(model_path)

    input_dtype = input_details["dtype"]
    q_scale_in, q_zp_in = input_details.get("quantization", (0.0, 0))

    if verbose:
        print(f"  Input  : shape={list(input_details['shape'])}, "
              f"dtype={input_dtype.__name__}, quant=({q_scale_in}, {q_zp_in})")
        print(f"  Output : shape={list(output_details['shape'])}, "
              f"dtype={output_details['dtype'].__name__}")

    # ── Prepare input tensor ─────────────────────────────────────────────
    if input_dtype == np.float32:
        input_tensor = mfcc.astype(np.float32)
    elif input_dtype in (np.int8, np.uint8):
        if q_scale_in == 0:
            raise RuntimeError("Model input quantization scale is 0; cannot quantize.")
        q = np.round(mfcc / q_scale_in + q_zp_in).astype(input_dtype)
        if input_dtype == np.int8:
            q = np.clip(q, -128, 127)
        else:
            q = np.clip(q, 0, 255)
        input_tensor = q
    else:
        raise RuntimeError(f"Unsupported input dtype: {input_dtype}")

    # Add batch dimension: (49, 10, 1) → (1, 49, 10, 1)
    input_tensor = np.expand_dims(input_tensor, axis=0)

    # ── Run inference ────────────────────────────────────────────────────
    interpreter.set_tensor(input_details["index"], input_tensor)
    interpreter.invoke()
    raw_output = interpreter.get_tensor(output_details["index"])

    # ── Dequantize output if needed ──────────────────────────────────────
    out_dtype = output_details["dtype"]
    q_scale_out, q_zp_out = output_details.get("quantization", (0.0, 0))

    if out_dtype == np.float32:
        scores = raw_output.astype(np.float32).squeeze()
    else:
        scores = (raw_output.astype(np.float32) - q_zp_out) * q_scale_out
        scores = scores.squeeze()

    # The model's final layer is softmax, so the output is already a
    # probability distribution.  Only apply softmax when the outputs
    # don't look like probabilities (e.g. logits from a quantised model
    # where dequantization may shift the range).
    if np.all(scores >= 0) and np.isclose(scores.sum(), 1.0, atol=0.01):
        probs = scores
    else:
        probs = softmax(scores)

    # ── Print results ────────────────────────────────────────────────────
    print("\nResults:")
    for i, p in enumerate(probs):
        label = labels[i] if i < len(labels) else str(i)
        bar = "█" * int(p * 40)
        print(f"  {label:10s}: {p:.4f}  {bar}")

    predicted_idx = int(np.argmax(probs))
    predicted_label = labels[predicted_idx] if predicted_idx < len(labels) else str(predicted_idx)
    print(f"\nPredicted: {predicted_label} (index {predicted_idx}, confidence {probs[predicted_idx]:.4f})")

    return probs


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Run KWS DS-CNN inference on a WAV file."
    )
    parser.add_argument(
        "--model", "-m",
        default=MODEL_PATH,
        help="Path to TFLite model (.tflite)",
    )
    parser.add_argument(
        "--audio", "-a",
        required=True,
        help="Path to 16 kHz mono WAV file (1 second)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show model IO details and feature stats",
    )
    args = parser.parse_args()

    run_inference(args.model, args.audio, verbose=args.verbose)


if __name__ == "__main__":
    main()
