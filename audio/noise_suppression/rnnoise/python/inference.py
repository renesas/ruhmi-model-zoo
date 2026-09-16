# Copyright 2026 Renesas Electronics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Single-WAV denoising with the RNNoise INT8 TFLite model.

Pipeline:
  1. Load the noisy WAV (resampled to 48 kHz if needed).
  2. Slice into 480-sample frames (10 ms each).
  3. For each frame:
       a. ``RNNoisePreProcess.process_frame()`` -> 42 DSP features.
       b. If non-silent: feed features + 3 previous GRU states to TFLite.
          The model returns 22 band gains, the VAD probability and 3 next states.
       c. ``RNNoisePreProcess.post_process()`` -> 480 denoised samples.
       d. Silent frames are replaced with 480 zeros.
  4. Concatenate frames; write int16 PCM at 48 kHz.

If ``--reference`` is supplied, wide-band PESQ is also printed
(noisy-vs-clean and denoised-vs-clean).

Usage:
    python inference.py --input  sample_audio/p257_036_noisy.wav
    python inference.py --input  sample_audio/p257_036_noisy.wav \\
                        --output sample_audio/p257_036_denoised.wav
    python inference.py --input     sample_audio/p257_036_noisy.wav \\
                        --reference sample_audio/p257_036_clean.wav   # + PESQ-wb
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Silence TensorFlow / absl C++ chatter before importing anything heavy
# ──────────────────────────────────────────────────────────────────────────────
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "3")
import warnings
warnings.filterwarnings("ignore")


@contextlib.contextmanager
def _silence_stderr():
    """Redirect fd 2 to /dev/null while a C-extension is loaded."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(2)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull)


import numpy as np
import soundfile as sf

try:
    import librosa
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False

try:
    import pesq as _pesq_mod
    _HAS_PESQ = True
except ImportError:
    _HAS_PESQ = False

with _silence_stderr():
    try:
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        try:
            import absl.logging
            absl.logging.set_verbosity(absl.logging.ERROR)
        except ImportError:
            pass
    except ImportError as e:
        sys.exit(f"tensorflow is required for TFLite inference: {e}")


BASE_DIR = Path(__file__).resolve().parent
MODEL_INT8 = BASE_DIR / "model" / "rnnoise_INT8.tflite"
sys.path.insert(0, str(BASE_DIR))

from utils import RNNoisePreProcess  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# RNNoise constants  (must match utils.rnnoise_preprocessing)
# ──────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 48_000
FRAME_SIZE = 480                # 10 ms @ 48 kHz
PESQ_SR = 16_000                # PESQ wb requires 16 kHz inputs

# GRU hidden-state sizes (from the Arm definition.yaml input nodes)
VAD_GRU_SIZE = 24
NOISE_GRU_SIZE = 48
DENOISE_GRU_SIZE = 96


# ──────────────────────────────────────────────────────────────────────────────
# TFLite helpers
# ──────────────────────────────────────────────────────────────────────────────
def _load_wav(path: Path, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Return a 1-D float32 array at ``target_sr``."""
    data, sr = sf.read(str(path), always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)        # downmix to mono
    data = data.astype(np.float32)

    if sr != target_sr:
        if not _HAS_LIBROSA:
            sys.exit(f"{path} is {sr} Hz; install librosa to auto-resample to {target_sr}.")
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)

    # RNNoise's reference C code operates on int16-scaled floats
    # ([-32768, +32767]). soundfile gives us [-1, 1] floats, so undo it.
    if np.max(np.abs(data)) <= 1.5:
        data = data * 32768.0

    return data


def _save_wav(path: Path, samples: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    """Write a 1-D array as int16 PCM."""
    samples = np.clip(samples, -32768, 32767).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, sr, subtype="PCM_16")


def _build_io_index(interpreter: "tf.lite.Interpreter") -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """Build name->details maps for inputs and outputs.

    Names emitted by the published Arm ``.tflite`` are::

        Inputs : main_input_int8, vad_gru_prev_state_int8,
                 noise_gru_prev_state_int8, denoise_gru_prev_state_int8
        Outputs: Identity_int8  (denoise GRU next state),
                 Identity_1_int8 (22 gains),
                 Identity_2_int8 (noise GRU next state),
                 Identity_3_int8 (VAD GRU next state),
                 Identity_4_int8 (VAD probability).

    Names may be renamed by some toolchains, so we also key by shape as a
    fallback (every shape is unique in this model).
    """
    ins  = {d["name"]: d for d in interpreter.get_input_details()}
    outs = {d["name"]: d for d in interpreter.get_output_details()}
    return ins, outs


def _find_by_shape(details: Dict[str, dict], shape) -> dict:
    target = tuple(shape)
    for d in details.values():
        if tuple(d["shape"]) == target:
            return d
    raise KeyError(f"No tensor with shape {target} found among {[tuple(d['shape']) for d in details.values()]}")


def _get(details: Dict[str, dict], *candidate_names: str, shape=None) -> dict:
    """Pick a tensor by name first, then by shape if names don't match."""
    for n in candidate_names:
        if n in details:
            return details[n]
    if shape is not None:
        return _find_by_shape(details, shape)
    raise KeyError(f"None of {candidate_names} present and no shape fallback given")


def _quantise(data: np.ndarray, detail: dict) -> np.ndarray:
    """Convert a float32 tensor to whatever dtype the TFLite input expects."""
    if detail["dtype"] == np.int8:
        scale, zero = detail["quantization"]
        q = data / scale + zero
        return np.clip(q, -128, 127).astype(np.int8)
    return data.astype(detail["dtype"])


def _dequantise(data: np.ndarray, detail: dict) -> np.ndarray:
    if detail["dtype"] == np.int8:
        scale, zero = detail["quantization"]
        return scale * (data.astype(np.float32) - zero)
    return data.astype(np.float32)


def _resolve_model_path(model_arg: str | None) -> Path:
    """Resolve model path from explicit argument or default INT8 model path."""
    if model_arg:
        return Path(model_arg).resolve()
    return MODEL_INT8.resolve()


# ──────────────────────────────────────────────────────────────────────────────
# Frame-by-frame denoise loop
# ──────────────────────────────────────────────────────────────────────────────
def denoise(noisy: np.ndarray, model_path: Path, *, verbose: bool = False) -> np.ndarray:
    """Run RNNoise over a 48 kHz noisy waveform and return the denoised waveform."""
    with _silence_stderr():
        interp = tf.lite.Interpreter(model_path=str(model_path))
        interp.allocate_tensors()
    ins, outs = _build_io_index(interp)

    feat_in     = _get(ins,  "main_input_int8",             shape=(1, 1, 42))
    vad_in      = _get(ins,  "vad_gru_prev_state_int8",     shape=(1, VAD_GRU_SIZE))
    noise_in    = _get(ins,  "noise_gru_prev_state_int8",   shape=(1, NOISE_GRU_SIZE))
    denoise_in  = _get(ins,  "denoise_gru_prev_state_int8", shape=(1, DENOISE_GRU_SIZE))

    denoise_state_out = _get(outs, "Identity_int8",   shape=(1, 1, DENOISE_GRU_SIZE))
    gains_out         = _get(outs, "Identity_1_int8", shape=(1, 1, 22))
    noise_state_out   = _get(outs, "Identity_2_int8", shape=(1, 1, NOISE_GRU_SIZE))
    vad_state_out     = _get(outs, "Identity_3_int8", shape=(1, 1, VAD_GRU_SIZE))
    vad_prob_out      = _get(outs, "Identity_4_int8", shape=(1, 1, 1))

    # Initial GRU hidden states are zero
    vad_state     = np.zeros((1, VAD_GRU_SIZE),     dtype=np.float32)
    noise_state   = np.zeros((1, NOISE_GRU_SIZE),   dtype=np.float32)
    denoise_state = np.zeros((1, DENOISE_GRU_SIZE), dtype=np.float32)

    pre = RNNoisePreProcess(training=False)

    n_frames = len(noisy) // FRAME_SIZE
    out_frames: List[np.ndarray] = []
    n_silent = 0
    n_voiced = 0
    t0 = time.perf_counter()

    for i in range(n_frames):
        window = noisy[i * FRAME_SIZE : (i + 1) * FRAME_SIZE]
        silence, features, X, P, Ex, Ep, Exp = pre.process_frame(window)

        if silence:
            n_silent += 1
            out_frames.append(np.zeros(FRAME_SIZE, dtype=np.int16))
            continue

        n_voiced += 1
        feats = np.expand_dims(features, (0, 1)).astype(np.float32)   # (1,1,42)

        interp.set_tensor(feat_in["index"],    _quantise(feats,         feat_in))
        interp.set_tensor(vad_in["index"],     _quantise(vad_state,     vad_in))
        interp.set_tensor(noise_in["index"],   _quantise(noise_state,   noise_in))
        interp.set_tensor(denoise_in["index"], _quantise(denoise_state, denoise_in))
        interp.invoke()

        gains          = _dequantise(interp.get_tensor(gains_out["index"]),         gains_out)
        new_denoise_st = _dequantise(interp.get_tensor(denoise_state_out["index"]), denoise_state_out)
        new_noise_st   = _dequantise(interp.get_tensor(noise_state_out["index"]),   noise_state_out)
        new_vad_st     = _dequantise(interp.get_tensor(vad_state_out["index"]),     vad_state_out)
        # vad_prob is informational; surfaced only in --verbose
        # vad_prob = _dequantise(interp.get_tensor(vad_prob_out["index"]), vad_prob_out)

        # Hidden states come out as (1, 1, N); we need (1, N) on the way back.
        denoise_state = np.squeeze(new_denoise_st, axis=1)
        noise_state   = np.squeeze(new_noise_st,   axis=1)
        vad_state     = np.squeeze(new_vad_st,     axis=1)

        # Feed the 22 gains through the post-processing filterbank.
        denoised = pre.post_process(silence, np.squeeze(gains), X, P, Ex, Ep, Exp)
        denoised = np.rint(denoised).astype(np.int16)
        out_frames.append(denoised)

    dt = time.perf_counter() - t0
    if verbose:
        rtf = (n_frames * FRAME_SIZE / SAMPLE_RATE) / max(dt, 1e-9)
        print(f"  Frames     : {n_frames}  ({n_voiced} voiced, {n_silent} silent)")
        print(f"  Wall time  : {dt*1000:.1f} ms  ({rtf:.1f}x realtime)")

    if not out_frames:
        return np.zeros(0, dtype=np.int16)
    return np.concatenate(out_frames, axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# Optional PESQ scoring
# ──────────────────────────────────────────────────────────────────────────────
def _pesq_wb(clean: np.ndarray, deg: np.ndarray) -> float:
    """Wide-band PESQ at 16 kHz. ``clean`` and ``deg`` are float32 in [-1, 1]."""
    if not _HAS_PESQ:
        return float("nan")
    return float(_pesq_mod.pesq(ref=clean, deg=deg, fs=PESQ_SR, mode="wb"))


def _resample_for_pesq(samples_48k: np.ndarray) -> np.ndarray:
    """Take a float array at 48 kHz (any scale) and produce float32 at 16 kHz in [-1, 1]."""
    x = samples_48k.astype(np.float32)
    if np.max(np.abs(x)) > 1.5:
        x = x / 32768.0
    if not _HAS_LIBROSA:
        sys.exit("librosa is required for PESQ resampling (16 kHz).")
    return librosa.resample(x, orig_sr=SAMPLE_RATE, target_sr=PESQ_SR)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser(
        description="Denoise a single WAV with the RNNoise INT8 TFLite model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",     required=True,                            help="Noisy input WAV.")
    p.add_argument("--output",    default=None,                              help="Output WAV (default: <input_stem>_denoised.wav).")
    p.add_argument("--reference", default=None,                              help="Clean reference WAV - enables PESQ-wb scoring.")
    p.add_argument("--model",     default=None,                              help="Explicit TFLite model path.")
    p.add_argument("--verbose",   action="store_true",                       help="Print per-stage timings and counts.")
    args = p.parse_args()

    in_path = Path(args.input).resolve()
    if not in_path.exists():
        sys.exit(f"input WAV not found: {in_path}")

    model_path = _resolve_model_path(args.model)
    if not model_path.exists():
        sys.exit(
            "model not found: "
            f"{model_path}\n"
            "  Run download_model.py to fetch the INT8 artifact, "
            "or pass --model with an explicit file."
        )

    out_path = (Path(args.output).resolve() if args.output
                else in_path.with_name(in_path.stem + "_denoised.wav"))

    print("=" * 72)
    print("  RNNoise TFLite  -  single-WAV denoise")
    print("=" * 72)
    print(f"  Input   : {in_path}")
    print(f"  Output  : {out_path}")
    print(f"  Model   : {model_path.name}  ({model_path.stat().st_size} B)")
    if args.reference:
        print(f"  Ref     : {args.reference}")
    print()

    noisy = _load_wav(in_path)
    print(f"  Loaded {len(noisy)} samples = {len(noisy)/SAMPLE_RATE:.2f} s @ {SAMPLE_RATE} Hz")

    denoised = denoise(noisy, model_path, verbose=args.verbose)
    _save_wav(out_path, denoised)
    print(f"  Wrote  {len(denoised)} samples -> {out_path.name}")

    if args.reference:
        if not _HAS_PESQ:
            print()
            print("  [!] pesq library not installed; skipping PESQ score.")
            print("      pip install pesq")
        else:
            ref_path = Path(args.reference).resolve()
            if not ref_path.exists():
                sys.exit(f"reference WAV not found: {ref_path}")
            clean   = _load_wav(ref_path)
            # Trim everything to the same length (multiple of FRAME_SIZE)
            n = min(len(clean), len(noisy), len(denoised))
            clean_16k    = _resample_for_pesq(clean[:n])
            noisy_16k    = _resample_for_pesq(noisy[:n])
            denoised_16k = _resample_for_pesq(denoised[:n])

            pesq_noisy    = _pesq_wb(clean_16k, noisy_16k)
            pesq_denoised = _pesq_wb(clean_16k, denoised_16k)
            print()
            print("  PESQ-wb (wide-band, 16 kHz):")
            print(f"    noisy    vs clean : {pesq_noisy:.3f}")
            print(f"    denoised vs clean : {pesq_denoised:.3f}   "
                  f"(delta = {pesq_denoised - pesq_noisy:+.3f})")
    print()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
