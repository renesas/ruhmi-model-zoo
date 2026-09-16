# Copyright (C) 2021 Arm Limited or its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run TinyWav2Letter TFLite inference on a single audio file.

Pipeline:
  1. Load audio at 16 kHz mono (librosa)
  2. Extract MFCC13 + delta + delta-delta -> (T, 39) features
  3. Sliding-window TFLite inference (window=296, context=98)
  4. Greedy CTC decode; optional LER/WER vs --transcript

Usage:
    python inference.py --model model/tiny_wav2letter_pruned_fp32.tflite --wav /path/to/your_audio.wav
    python inference.py --model model/tiny_wav2letter_pruned_int8.tflite --wav /path/to/your_audio.wav --transcript "decrease the heating in the kitchen"
"""

import argparse
import multiprocessing

import librosa
import numpy as np
import tensorflow as tf
from jiwer import wer


ALPHABET = "abcdefghijklmnopqrstuvwxyz' @"
BLANK_INDEX = len(ALPHABET) - 1  # '@' maps to CTC blank (index 28)
alphabet_dict = {c: ind for (ind, c) in enumerate(ALPHABET)}
index_dict = {ind: c for (ind, c) in enumerate(ALPHABET)}


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _normalize(values):
    return (values - np.mean(values)) / np.std(values)


def extract_mfcc(audio_file):
    audio_data, sample_rate = librosa.load(audio_file, sr=16000)
    mfcc = librosa.feature.mfcc(
        y=audio_data, sr=sample_rate, n_mfcc=13, n_fft=512, hop_length=160
    )
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    mfcc = np.concatenate(
        (_normalize(mfcc), _normalize(mfcc_delta), _normalize(mfcc_delta2)), axis=0
    )
    mfcc_out = mfcc.T.astype(np.float32)       # (T, 39)
    return np.expand_dims(mfcc_out, 0)          # (1, T, 39)


# ---------------------------------------------------------------------------
# CTC decode helpers (mirrors the notebook)
# ---------------------------------------------------------------------------

def _ctc_preparation(label_tensor, y_predict):
    if len(y_predict.shape) == 4:
        y_predict = tf.squeeze(y_predict, axis=1)
    y_predict = tf.transpose(y_predict, (1, 0, 2))
    sequence_lengths = label_tensor[:, 0]
    labels = label_tensor[:, 1:]
    idx = tf.where(tf.not_equal(labels, BLANK_INDEX))
    sparse_labels = tf.SparseTensor(
        idx, tf.gather_nd(labels, idx), tf.shape(labels, out_type=tf.int64)
    )
    return sparse_labels, sequence_lengths, y_predict


def _ints_to_string(ints):
    return "".join(
        index_dict[np.array(i).item()] for i in ints if np.array(i).item() != BLANK_INDEX
    )


def ctc_decode(logits, label_tensor=None):
    """
    Greedy CTC decode via tf.nn.ctc_greedy_decoder.

    Args:
        logits:       numpy array (1, 1, T, 29) — dequantized model output.
        label_tensor: optional (1, 1+N) int32 array ([seq_len, *transcript_ints])
                      for LER/WER computation.

    Returns:
        transcript (str), LER (float or None), WER (float or None)
    """
    if len(logits.shape) == 4:
        logits_t = tf.squeeze(logits, axis=1)          # (1, T, 29)
    logits_t = tf.transpose(logits_t, (1, 0, 2))       # (T, 1, 29)

    seq_len = tf.constant([logits_t.shape[0]], dtype=tf.int32)
    decoded, _ = tf.nn.ctc_greedy_decoder(logits_t, seq_len, merge_repeated=True)

    transcript = _ints_to_string(decoded[0].values)

    ler_val = None
    wer_val = None
    if label_tensor is not None:
        sparse_labels, sequence_lengths, lp = _ctc_preparation(label_tensor, logits)
        decoded2, _ = tf.nn.ctc_greedy_decoder(
            lp, tf.cast(sequence_lengths, tf.int32), merge_repeated=True
        )
        ler_val = float(tf.reduce_mean(
            tf.edit_distance(
                tf.cast(decoded2[0], tf.int32), tf.cast(sparse_labels, tf.int32)
            ).numpy()
        ))
        pred_str = _ints_to_string(decoded2[0].values)
        true_str = _ints_to_string(sparse_labels.values)
        wer_val = wer(pred_str, true_str)

    return transcript, ler_val, wer_val


# ---------------------------------------------------------------------------
# Sliding-window TFLite inference
# ---------------------------------------------------------------------------

def run_inference(tflite_path, data):
    """
    Run sliding-window inference on MFCC data.

    Args:
        tflite_path: path to tiny_wav2letter_int8.tflite
        data:        numpy array (1, T, 39) float32

    Returns:
        logits numpy array (1, 1, T_out, 29) float32
    """
    interpreter = tf.lite.Interpreter(
        model_path=tflite_path, num_threads=multiprocessing.cpu_count()
    )
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_dtype = input_details["dtype"]
    output_dtype = output_details["dtype"]

    input_scale, input_zero_point = (
        input_details["quantization"] if input_dtype != np.float32 else (1, 0)
    )
    output_scale, output_zero_point = (
        output_details["quantization"] if output_dtype != np.float32 else (1, 0)
    )
    
    print(f"\nInput quantization: scale={input_scale}, zero_point={input_zero_point}")
    print(f"Output quantization: scale={output_scale}, zero_point={output_zero_point}\n")

    # Quantize and pad input
    data = data / input_scale + input_zero_point
    if input_dtype is not np.float32:
        data = np.round(data)

    window = input_details["shape"][1]   # 296
    while data.shape[1] < window:
        data = np.append(data, data[:, -2:-1, :], axis=1)
    if data.shape[1] % 2 == 1:
        data = np.concatenate(
            [data, np.zeros((1, 1, data.shape[2]), dtype=input_dtype)], axis=1
        )

    context = 24 + 2 * (7 * 3 + 16)   # = 98
    inner = window - 2 * context        # = 100
    data_end = data.shape[1]
    data_pos = 0
    outputs = []

    while data_pos < data_end:
        if data_pos == 0:
            start, end = 0, window
            y_start, y_end = 0, (window - context) // 2
            data_pos = end - context
        elif data_pos + inner + context >= data_end:
            shift = (data_pos + inner + context) - data_end
            start = data_pos - context - shift
            end = start + window
            assert start >= 0
            y_start = (shift + context) // 2
            y_end = window // 2
            data_pos = data_end
        else:
            start = data_pos - context
            end = start + window
            y_start = context // 2
            y_end = y_start + inner // 2
            data_pos = end - context

        interpreter.set_tensor(
            input_details["index"], tf.cast(data[:, start:end, :], input_dtype)
        )
        interpreter.invoke()
        chunk = interpreter.get_tensor(output_details["index"])[:, :, y_start:y_end, :]
        chunk = output_scale * (chunk.astype(np.float32) - output_zero_point)
        outputs.append(chunk)

    return np.concatenate(outputs, axis=2)   # (1, 1, T_out, 29)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Tiny Wav2Letter TFLite INT8 inference")
    parser.add_argument("--model", required=True, help="Path to tiny_wav2letter_int8.tflite")
    parser.add_argument("--wav", required=True, help="Path to input audio file (.wav/.flac)")
    parser.add_argument(
        "--transcript", default=None,
        help="Optional ground-truth transcript (uppercase) for LER/WER evaluation"
    )
    args = parser.parse_args()

    label_tensor = None
    if args.transcript is not None:
        transcript_ints = [alphabet_dict[c] for c in args.transcript.lower()]
        seq_length = 0  # filled after MFCC extraction below
        _transcript_ints = transcript_ints  # saved for later

    mfcc = extract_mfcc(args.wav)   # (1, T, 39)

    if args.transcript is not None:
        seq_length = mfcc.shape[1] // 2
        label_tensor = np.expand_dims(
            np.concatenate([[seq_length], _transcript_ints]).astype(np.int32), 0
        )

    logits = run_inference(args.model, mfcc)
    transcript, ler_val, wer_val = ctc_decode(logits, label_tensor)

    print(f"Transcription : {transcript}")
    if ler_val is not None:
        print(f"LER           : {ler_val:.4f}")
        print(f"WER           : {wer_val:.4f}")


if __name__ == "__main__":
    main()