# Copyright 2026 Renesas Electronics Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Run inference with the Visual Wake Words TFLite model (FP32 or INT8).

Usage:
  python inference.py --image person.jpg
  python inference.py --image person.jpg --model model/vww_96_INT8.tflite
  python inference.py --image person.jpg --top-k 2
"""
import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
from PIL import Image
import tensorflow as tf

# ──────────────────────────────────────────────────────────────────────────────
# Defaults  (relative to this script's directory)
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "model", "vww_96_FP32.tflite")
LABELS_PATH  = os.path.join(BASE_DIR, "utils", "labels.txt")
SAMPLE_IMAGE = os.path.join(BASE_DIR, "sample_images", "COCO_val2014_000000000042.jpg")

INPUT_SIZE   = 96    # model expects 96×96 RGB
TOP_K        = 2     # only 2 classes: notperson / person


# ──────────────────────────────────────────────────────────────────────────────
# Label loading
# ──────────────────────────────────────────────────────────────────────────────
def load_labels(labels_path: str) -> List[str]:
    """Load class names from a text file (one label per line)."""
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]
    return labels


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────────────────────────────────────
def preprocess(image_path: str) -> np.ndarray:
    """
    Load image and return float32 array normalised to [0, 1].
    Shape: (1, 96, 96, 3)
    """
    img = Image.open(image_path).convert("RGB").resize(
        (INPUT_SIZE, INPUT_SIZE), Image.BILINEAR
    )
    arr = np.asarray(img, dtype=np.float32) / 255.0   # [0, 1]
    return np.expand_dims(arr, axis=0)                 # (1, 96, 96, 3)


# ──────────────────────────────────────────────────────────────────────────────
# Post-processing
# ──────────────────────────────────────────────────────────────────────────────
def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def postprocess(raw: np.ndarray,
                out_details: dict,
                labels: List[str],
                top_k: int) -> List[Tuple[int, str, float]]:
    """
    Dequantize (if INT8), apply softmax, return top-k results.
    """
    scores = raw.squeeze().astype(np.float32)

    # Dequantize INT8 output
    if out_details["dtype"] != np.float32:
        scale, zp = out_details["quantization"]
        scores = (scores - zp) * scale

    # VWW output is raw logits — apply softmax
    probs = softmax(scores)

    top_indices = np.argsort(probs)[::-1][:top_k]
    return [
        (int(i), labels[i] if i < len(labels) else f"class_{i}", float(probs[i]))
        for i in top_indices
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────
def run_inference(model_path: str,
                  image_path: str,
                  labels: List[str],
                  top_k: int) -> List[Tuple[int, str, float]]:
    interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    inp_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]

    img = preprocess(image_path)   # float32 (1, 96, 96, 3)

    # Quantize input for INT8 models
    if inp_det["dtype"] == np.float32:
        tensor = img
    else:
        scale, zp = inp_det["quantization"]
        if scale == 0:
            sys.exit("[ERROR] Input quantization scale is 0.")
        tensor = np.clip(
            np.round(img / scale + zp), -128, 127
        ).astype(inp_det["dtype"])

    interp.set_tensor(inp_det["index"], tensor)
    interp.invoke()
    raw = interp.get_tensor(out_det["index"])

    return postprocess(raw, out_det, labels, top_k)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Single-image TFLite inference for Visual Wake Words (person / notperson)."
    )
    parser.add_argument(
        "image", nargs="?", default=None,
        help="Path to input image (positional).",
    )
    parser.add_argument(
        "--image", dest="image_flag", default=None, metavar="PATH",
        help="Path to input image (alternative to positional).",
    )
    parser.add_argument(
        "--model", default=MODEL_PATH,
        help=f"Path to TFLite model (default: {MODEL_PATH}).",
    )
    parser.add_argument(
        "--labels", default=LABELS_PATH,
        help=f"Path to labels.txt (default: {LABELS_PATH}).",
    )
    parser.add_argument(
        "--top-k", type=int, default=TOP_K,
        help=f"Number of top predictions to show (default: {TOP_K}).",
    )
    args = parser.parse_args()

    # Resolve image path (--image flag takes priority over positional)
    image_path = args.image_flag or args.image or SAMPLE_IMAGE

    if not os.path.isfile(args.model):
        sys.exit(f"[ERROR] Model not found: {args.model}")
    if not os.path.isfile(image_path):
        sys.exit(f"[ERROR] Image not found: {image_path}")
    if not os.path.isfile(args.labels):
        sys.exit(f"[ERROR] Labels file not found: {args.labels}")

    labels = load_labels(args.labels)

    print(f"\nModel   : {args.model}")
    print(f"Image   : {image_path}")
    print(f"Input   : ({INPUT_SIZE}, {INPUT_SIZE}, 3)  normalised to [0, 1]")
    print(f"Classes : {labels}\n")

    results = run_inference(args.model, image_path, labels, args.top_k)

    print("─" * 45)
    print(f"{'Rank':<6} {'Class':>6}   {'Label':<15} {'Probability':>12}")
    print("─" * 45)
    for rank, (cls_idx, label, prob) in enumerate(results, start=1):
        print(f"  {rank:<4} {cls_idx:>6}   {label:<15} {prob:>12.6f}")
    print("─" * 45)

    # Binary decision
    top_label = results[0][1]
    top_prob  = results[0][2]
    print(f"\nDecision : {top_label.upper()}  ({top_prob*100:.1f}% confidence)")


if __name__ == "__main__":
    main()
