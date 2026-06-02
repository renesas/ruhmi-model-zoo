#!/usr/bin/env python3
"""
inference.py — Single-image TFLite FP32 inference for ShuffleNetV2 x0.5.

Usage:
  python inference.py <image_path>
  python inference.py --model Model/shufflenet_v2_x0_5_FP32.tflite sample_images/coco_cat.jpg
"""

import os
import sys
import argparse
from typing import List, Tuple

import numpy as np
from PIL import Image
import tensorflow as tf

# ──────────────────────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "Model", "shufflenet_v2_x0_5_FP32.tflite")
LABELS_PATH  = os.path.join(BASE_DIR, "utils", "imagenet_labels.txt")
SAMPLE_IMAGE = os.path.join(BASE_DIR, "sample_images", "coco_cat.jpg")

TOP_K        = 5
INPUT_HEIGHT = 224
INPUT_WIDTH  = 224

# ImageNet normalization (same as torchvision standard)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# 1. PRE-PROCESSING
# ──────────────────────────────────────────────────────────────────────────────
def preprocess(image_path: str,
               height: int = INPUT_HEIGHT,
               width: int  = INPUT_WIDTH) -> np.ndarray:
    """
    Standard ImageNet preprocessing for ShuffleNetV2:
      1. Resize shortest edge to 256.
      2. Center-crop to 224×224.
      3. Normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225].

    Returns
    -------
    np.ndarray  shape (1, H, W, 3)  dtype float32
    """
    img = Image.open(image_path).convert("RGB")

    # Resize shortest edge to 256
    w, h = img.size
    scale = 256.0 / min(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img = img.resize((new_w, new_h), resample=Image.BILINEAR)

    # Center crop to 224×224
    left = (new_w - width) // 2
    top  = (new_h - height) // 2
    img = img.crop((left, top, left + width, top + height))

    # To float, normalize
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD

    return np.expand_dims(arr, axis=0)  # (1, H, W, 3)


# ──────────────────────────────────────────────────────────────────────────────
# 2. LABEL LOADING
# ──────────────────────────────────────────────────────────────────────────────
def load_labels(labels_path: str) -> List[str]:
    """Load ImageNet 1000 class names from imagenet_labels.txt (one label per line)."""
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]
    assert len(labels) == 1000, f"Expected 1000 labels, got {len(labels)}"
    return labels


# ──────────────────────────────────────────────────────────────────────────────
# 3. POST-PROCESSING
# ──────────────────────────────────────────────────────────────────────────────
def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp     = np.exp(shifted)
    return exp / exp.sum()


def postprocess(raw_output: np.ndarray,
                labels: List[str],
                top_k: int = TOP_K) -> List[Tuple[int, str, float]]:
    """
    Convert raw model output to top-K predictions.

    ShuffleNetV2 from torchvision outputs raw logits (no softmax),
    so we always apply softmax here.
    """
    probs = np.squeeze(raw_output)  # (1000,)

    # Check if already probabilities (unlikely for this model)
    already_probs = (
        float(probs.min()) >= 0.0 and
        np.isclose(float(probs.sum()), 1.0, atol=1e-3)
    )
    if not already_probs:
        probs = softmax(probs)

    top_indices = np.argsort(probs)[::-1][:top_k]

    results: List[Tuple[int, str, float]] = []
    for idx in top_indices:
        idx   = int(idx)
        label = labels[idx] if idx < len(labels) else f"class_{idx}"
        prob  = float(probs[idx])
        results.append((idx, label, prob))
    return results


# ──────────────────────────────────────────────────────────────────────────────
# 4. TFLite RUNNER
# ──────────────────────────────────────────────────────────────────────────────
def run_tflite_inference(model_path: str,
                         image_path: str,
                         labels: List[str],
                         top_k: int = TOP_K) -> List[Tuple[int, str, float]]:
    """Load TFLite model, run single-image inference, return top-K results."""
    # BUILTIN_WITHOUT_DEFAULT_DELEGATES skips XNNPACK, which fails on
    # ShuffleNetV2's INT8 Reshape→Transpose→Reshape (channel-shuffle) pattern.
    from tensorflow.lite.python.interpreter import OpResolverType
    interpreter = tf.lite.Interpreter(
        model_path=model_path,
        experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    )
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_tensor = preprocess(image_path, INPUT_HEIGHT, INPUT_WIDTH)

    input_dtype = input_details[0]["dtype"]
    if input_dtype == np.int8:
        # INT8 model — quantise input
        scale, zp = input_details[0]["quantization"]
        input_tensor = np.clip(
            np.round(input_tensor / scale) + zp, -128, 127
        ).astype(np.int8)

    interpreter.set_tensor(input_details[0]["index"], input_tensor)
    interpreter.invoke()

    raw_output = interpreter.get_tensor(output_details[0]["index"])

    # Dequantise INT8 output if needed
    output_dtype = output_details[0]["dtype"]
    if output_dtype == np.int8:
        scale, zp = output_details[0]["quantization"]
        raw_output = (raw_output.astype(np.float32) - zp) * scale

    return postprocess(raw_output, labels, top_k)


# ──────────────────────────────────────────────────────────────────────────────
# 5. MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Single-image TFLite inference for ShuffleNetV2 x0.5."
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
        help=f"Path to imagenet_labels.txt (default: {LABELS_PATH}).",
    )
    parser.add_argument(
        "--top-k", type=int, default=TOP_K,
        help=f"Number of top predictions to show (default: {TOP_K}).",
    )
    args = parser.parse_args()

    # Resolve image path (--image flag takes priority over positional)
    image_path = args.image_flag or args.image or SAMPLE_IMAGE

    # Validate
    if not os.path.isfile(args.model):
        sys.exit(f"[ERROR] Model not found: {args.model}\n"
                 "        Run download_model.py first.")
    if not os.path.isfile(image_path):
        sys.exit(f"[ERROR] Image not found: {image_path}")
    if not os.path.isfile(args.labels):
        sys.exit(f"[ERROR] Labels not found: {args.labels}")

    labels = load_labels(args.labels)
    print(f"Loaded {len(labels)} labels from {args.labels}")

    print(f"\nModel   : {args.model}")
    print(f"Image   : {image_path}")
    print(f"Input   : ({INPUT_HEIGHT}, {INPUT_WIDTH}, 3)  "
          f"mean={list(IMAGENET_MEAN)}, std={list(IMAGENET_STD)}")
    print(f"Top-K   : {args.top_k}")
    print()

    results = run_tflite_inference(
        model_path=args.model,
        image_path=image_path,
        labels=labels,
        top_k=args.top_k,
    )

    print("─" * 60)
    print(f"{'Rank':<6} {'Class':>6}   {'Label':<35} {'Probability':>12}")
    print("─" * 60)
    for rank, (class_idx, label, prob) in enumerate(results, start=1):
        print(f"  {rank:<4} {class_idx:>6}   {label:<35} {prob:>12.6f}")
    print("─" * 60)


if __name__ == "__main__":
    main()
