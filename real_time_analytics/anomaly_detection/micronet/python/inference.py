#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0

"""ARM MicroNet anomaly detection TFLite inference pipeline.

This script runs a single-model flow for the DCASE 2020 Task 2 "slider" set:

1) Load a WAV clip at 16 kHz mono.
2) Compute a 64-bin log-mel spectrogram with training-time parameters.
3) Slide 64-frame windows over the log-mel matrix (stride 20 frames).
4) Mean-pool every window from 64x64 down to 32x32.
5) Quantize each patch using the model's input scale/zero-point.
6) Run the INT8 TFLite model and dequantize the 8 output logits.
7) Score each window with score = -logit[machine_class] and average.
8) Emit an ANOMALY / normal verdict against a fixed threshold.

The geometric transforms (windowing, pooling) and quantization steps are kept
explicit so preprocessing math, TFLite I/O contracts, and score aggregation
can be audited independently. Runtime behavior matches the pinned ML-zoo
reference commit 7c32b097f7d94aae2cd0b98a8ed5a3ba81e66b18.

Reference:
    https://github.com/Arm-Examples/ML-zoo/tree/7c32b097f7d94aae2cd0b98a8ed5a3ba81e66b18/models/anomaly_detection

Usage:
    python inference.py --model model/ad_medium_int8.tflite --wav clip.wav --machine-id 0
    python inference.py --model model/ad_medium_int8.tflite --wav normal_id_02_00000000.wav
"""

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import librosa
import numpy as np

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite


# --------------------------------------------------------------------------- #
# Model / signal constants (must match the training-time recipe).             #
# --------------------------------------------------------------------------- #

# Audio front-end parameters.
SAMPLE_RATE: int = 16000            # WAV resample target, Hz.
N_FFT: int = 1024                   # FFT size for STFT.
FRAME_LEN: int = 1024               # STFT window length in samples.
FRAME_STRIDE: int = 512             # STFT hop length in samples.
N_MELS_RAW: int = 64                # Mel bins produced by librosa filterbank.
MEL_FMIN: int = 0                   # Mel filter lower frequency bound, Hz.
MEL_FMAX: int = 8000                # Mel filter upper frequency bound, Hz.
TRAINING_MEAN: float = -30.0        # ARM usecase.cmake mean shift for log-mel.

# Model input is 32 mel bins x 32 time frames; features are built as
# 64 mel bins x 64 time frames then mean-pooled 2x2.
PATCH_RAW_FRAMES: int = 64          # Time frames per outer window.
PATCH_MODEL_DIM: int = 32           # Spatial size after 2x2 mean pool.
OUTER_STRIDE_FRAMES: int = 20       # Outer stride between windows (frames).

# Empirical: normal clips score around -7.5, anomalous around -6.0 on the
# slider dev-test. -6.8 sits roughly at the boundary.
DEFAULT_THRESHOLD: float = -7.2

# ML-zoo labelmappings.txt (slider dev): id0, id2, id4, id6 -> outputs 0..3.
MACHINE_ID_TO_OUTPUT = {0: 0, 2: 1, 4: 2, 6: 3}


# --------------------------------------------------------------------------- #
# Custom exceptions (reserved for stricter argument / range validation).      #
# --------------------------------------------------------------------------- #

class InvalidMachineIdError(Exception):
    """Raised when the requested machine ID is not in the trained set."""
    pass


class ClipTooShortError(Exception):
    """Raised when a WAV clip has fewer than PATCH_RAW_FRAMES mel frames."""
    pass


# --------------------------------------------------------------------------- #
# Data containers.                                                            #
# --------------------------------------------------------------------------- #

@dataclass
class ClipScore:
    """End-to-end scoring result for a single WAV clip.

    Attributes:
        mean_score: Clip-level anomaly score (mean over window scores).
        window_scores: Per-window anomaly scores (higher = more anomalous).
        num_windows: Number of sliding windows scored on the clip.
    """
    mean_score: float
    window_scores: List[float]
    num_windows: int


# --------------------------------------------------------------------------- #
# Audio + feature front-end.                                                  #
# --------------------------------------------------------------------------- #

def load_audio(path: Union[str, Path]) -> np.ndarray:
    """Load WAV as 16 kHz mono float32 in the range [-1, 1].

    Args:
        path: Path to a WAV file readable by librosa.

    Returns:
        1-D np.float32 array of PCM samples, shape (N,).
    """
    audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    return audio.astype(np.float32)


def compute_log_mel(audio: np.ndarray) -> np.ndarray:
    """Return a (64, T) log-mel matrix consumed by the sliding window loop.

    Steps:
        1) Power mel spectrogram (Slaney norm, Hann window, no centering).
        2) 10 * log10 with an eps floor to avoid log(0).
        3) Shift by -TRAINING_MEAN so the value range matches training.

    Args:
        audio: 1-D float32 PCM samples returned by :func:`load_audio`.

    Returns:
        (64, T) float32 log-mel matrix. T depends on clip length; for a
        10-second clip and center=False, T is 311.
    """
    # Mel power spectrogram matching the training-time front-end exactly.
    spec = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=FRAME_STRIDE,
        win_length=FRAME_LEN,
        window="hann",
        center=False,
        n_mels=N_MELS_RAW,
        fmin=MEL_FMIN,
        fmax=MEL_FMAX,
        htk=False,
        norm="slaney",
        power=2.0,
    )
    # Clamp to a tiny positive value before log to keep the result finite.
    log_mel = 10.0 * np.log10(np.maximum(spec, np.finfo(np.float32).tiny))
    # Subtract training mean; TRAINING_MEAN is negative so this shifts up.
    log_mel -= TRAINING_MEAN
    return log_mel.astype(np.float32)


def mean_pool_2x2(patch: np.ndarray) -> np.ndarray:
    """Down-sample a (64, 64) patch to (32, 32) using 2x2 block averaging.

    ARM's C use-case does every-other picking here instead of averaging,
    which drops AUC by ~15 points. Mean-pool matches the training pipeline.

    Args:
        patch: (64, 64) float32 log-mel window.

    Returns:
        (32, 32) float32 array with pooled values.
    """
    return (
        patch
        .reshape(PATCH_MODEL_DIM, 2, PATCH_MODEL_DIM, 2)
        .mean(axis=(1, 3))
        .astype(np.float32)
    )


# --------------------------------------------------------------------------- #
# TFLite I/O helpers.                                                         #
# --------------------------------------------------------------------------- #

def quantize_input(patch_f32: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """Quantize a float32 patch to int8 using TFLite tensor quant parameters.

    Args:
        patch_f32: (32, 32) float32 input patch.
        scale: TFLite input tensor scale.
        zero_point: TFLite input tensor zero-point.

    Returns:
        (32, 32) int8 array clipped to the int8 numeric range.
    """
    q = np.round(patch_f32 / scale + zero_point)
    return np.clip(q, -128, 127).astype(np.int8)


def dequantize_output(out_q: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """Dequantize an int8 output tensor to float32 logits.

    Args:
        out_q: Raw output tensor as returned by the TFLite interpreter.
        scale: TFLite output tensor scale.
        zero_point: TFLite output tensor zero-point.

    Returns:
        float32 logits array with the same shape as ``out_q``.
    """
    return scale * (out_q.astype(np.float32) - zero_point)


# --------------------------------------------------------------------------- #
# Filename helpers.                                                           #
# --------------------------------------------------------------------------- #

# Match the ML-zoo slider naming convention ``..._id_XX_...``.
_MACHINE_ID_PATTERN = re.compile(r"_id_(\d{2})_")


def infer_machine_id_from_name(path: Union[str, Path]) -> Optional[int]:
    """Extract the DCASE slider machine ID from a WAV filename.

    Args:
        path: Path or filename following ``..._id_XX_...`` naming.

    Returns:
        Integer machine ID (e.g. 0, 2, 4, 6) or None if the pattern is absent.
    """
    # Match against the basename only so a directory like ``.../_id_99_.../``
    # cannot poison the parsed machine ID.
    match = _MACHINE_ID_PATTERN.search(Path(path).name)
    if match is None:
        return None
    return int(match.group(1))


# --------------------------------------------------------------------------- #
# End-to-end scoring.                                                         #
# --------------------------------------------------------------------------- #

def score_wav(
    interpreter: tflite.Interpreter,
    wav_path: Union[str, Path],
    machine_id: int,
    verbose: bool = False,
) -> ClipScore:
    """Compute the clip-level anomaly score for a WAV file.

    Pipeline per clip:
        1) Load audio (16 kHz mono float32).
        2) Compute (64, T) log-mel with training parameters.
        3) Slide 64-frame windows with stride 20.
        4) 2x2 mean-pool each window to (32, 32).
        5) Quantize, run TFLite inference, dequantize output logits.
        6) Compute window score = -logit[target_index_for_machine_id].
        7) Return the mean of window scores (higher is more anomalous).

    Args:
        interpreter: Allocated TFLite interpreter for ad_medium_int8.tflite.
        wav_path: Path to the WAV file to score.
        machine_id: DCASE slider machine ID (0, 2, 4, or 6).
        verbose: If True, print per-window diagnostics (logit + top-3 classes).

    Returns:
        :class:`ClipScore` with mean score, per-window scores and window count.
    """
    if machine_id not in MACHINE_ID_TO_OUTPUT:
        raise InvalidMachineIdError(
            f"Unknown machine id {machine_id}; expected one of "
            f"{list(MACHINE_ID_TO_OUTPUT)}"
        )
    # Map the DCASE slider machine ID to the corresponding output class index.
    out_idx = MACHINE_ID_TO_OUTPUT[machine_id]

    # Read tensor metadata once; quantization parameters are per-tensor.
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    in_scale, in_zp = in_det["quantization"]
    out_scale, out_zp = out_det["quantization"]

    # Front-end: audio -> log-mel (64, T).
    audio = load_audio(wav_path)
    log_mel = compute_log_mel(audio)
    total_frames = log_mel.shape[1]
    if total_frames < PATCH_RAW_FRAMES:
        raise ClipTooShortError(
            f"Clip too short: {total_frames} mel frames, "
            f"need >= {PATCH_RAW_FRAMES}"
        )

    # Enumerate outer sliding-window start indices. The clip-length check
    # above already guarantees ``starts`` is non-empty (at least ``[0]``).
    starts = list(
        range(0, total_frames - PATCH_RAW_FRAMES + 1, OUTER_STRIDE_FRAMES)
    )

    window_scores: List[float] = []
    for i, start in enumerate(starts):
        # (64, 64) log-mel window -> (32, 32) pooled patch -> quantized tensor.
        patch64 = log_mel[:, start:start + PATCH_RAW_FRAMES]
        patch32 = mean_pool_2x2(patch64)
        tensor_q = quantize_input(patch32, in_scale, in_zp).reshape(
            1, PATCH_MODEL_DIM, PATCH_MODEL_DIM, 1
        )

        # Single-window forward pass.
        interpreter.set_tensor(in_det["index"], tensor_q)
        interpreter.invoke()
        out_q = interpreter.get_tensor(out_det["index"])[0]

        # Dequantize the 8 output logits and score the target machine class.
        # The score is the negative target-class logit, so larger values mean
        # a more anomalous window.
        logits = dequantize_output(out_q, out_scale, out_zp)
        window_score = -float(logits[out_idx])
        window_scores.append(window_score)

        if verbose:
            # Rank all 8 outputs by dequantized value for a quick sanity view.
            top3 = np.argsort(-logits)[:3].tolist()
            print(
                f"  window {i + 1:2d}/{len(starts)}  "
                f"logit[{out_idx}]={logits[out_idx]:+.5f}  "
                f"score={window_score:+.4f}  top3_classes={top3}"
            )

    mean_score = float(np.mean(window_scores))
    return ClipScore(
        mean_score=mean_score,
        window_scores=window_scores,
        num_windows=len(starts),
    )


# --------------------------------------------------------------------------- #
# CLI plumbing.                                                               #
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    """Parse command-line options for model path, WAV path and thresholds."""
    parser = argparse.ArgumentParser(
        description="MicroNet anomaly detection inference on a single WAV clip."
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to ad_medium_int8.tflite",
    )
    parser.add_argument(
        "--wav", required=True,
        help="Path to input WAV",
    )
    parser.add_argument(
        "--machine-id", type=int, default=None,
        help="DCASE slider machine ID: 0, 2, 4, or 6. "
             "If omitted, auto-detected from filename (..._id_XX_...).",
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=(
            "Anomaly threshold on the mean -logit score "
            f"(default {DEFAULT_THRESHOLD})"
        ),
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Only print final score + verdict",
    )
    return parser.parse_args()


def _resolve_machine_id(cli_id: Optional[int], wav_path: str) -> int:
    """Return the effective machine ID, deriving from filename if unset."""
    if cli_id is not None:
        return cli_id
    inferred = infer_machine_id_from_name(wav_path)
    if inferred is None:
        print(
            "error: could not infer --machine-id from filename; "
            "pass it explicitly.",
            file=sys.stderr,
        )
        sys.exit(2)
    return inferred


def _print_header(args: argparse.Namespace, machine_id: int) -> None:
    """Emit the pre-inference banner in non-quiet mode."""
    print(f"Model      : {args.model}")
    print(f"WAV        : {args.wav}")
    print(
        f"Machine ID : {machine_id}  ->  "
        f"output index {MACHINE_ID_TO_OUTPUT[machine_id]}"
    )


def _print_result(
    result: ClipScore, threshold: float, verdict: str, quiet: bool
) -> None:
    """Format the final scoring output, honoring --quiet."""
    if quiet:
        print(f"{result.mean_score:+.4f}  {verdict}")
        return
    print()
    print(f"Windows scored : {result.num_windows}")
    print(f"Mean score     : {result.mean_score:+.4f}")
    print(f"Threshold      : {threshold:+.4f}")
    print(f"Verdict        : {verdict}")


def main() -> None:
    """Entry point: parse args, load model, score one WAV, print verdict."""
    args = parse_args()

    # Fail fast with a clear message if the user passed a bad path.
    if not Path(args.model).is_file():
        print(f"error: model not found: {args.model}", file=sys.stderr)
        sys.exit(2)
    if not Path(args.wav).is_file():
        print(f"error: wav not found: {args.wav}", file=sys.stderr)
        sys.exit(2)

    machine_id = _resolve_machine_id(args.machine_id, args.wav)

    # Instantiate the TFLite interpreter (uses tflite_runtime when available).
    interpreter = tflite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()

    if not args.quiet:
        _print_header(args, machine_id)

    try:
        result = score_wav(
            interpreter,
            args.wav,
            machine_id,
            verbose=not args.quiet,
        )
    except InvalidMachineIdError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
    except ClipTooShortError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)

    # A higher clip score means a stronger anomaly signal; compare it to the
    # configured threshold to decide whether the clip is anomalous.
    verdict = "ANOMALY" if result.mean_score > args.threshold else "normal"
    _print_result(result, args.threshold, verdict, args.quiet)


if __name__ == "__main__":
    main()
