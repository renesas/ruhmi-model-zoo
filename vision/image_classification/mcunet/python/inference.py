# SPDX-License-Identifier: BSD-3-Clause
import os
import sys
import argparse
from typing import List, Tuple

# Suppress TensorFlow and oneDNN info/warning logs before importing tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
from PIL import Image
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

# ──────────────────────────────────────────────────────────────────────────────
# Defaults  (paths are relative to the project root, i.e. where this script lives)
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "model", "mcunet_in0_FP32.tflite")
LABELS_PATH = os.path.join(BASE_DIR, "utils", "imagenet_labels.txt")

# Inference settings
TOP_K        = 5
INPUT_HEIGHT = 48
INPUT_WIDTH  = 48


# ──────────────────────────────────────────────────────────────────────────────
# 1.  PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────
def preprocess(image_path: str,
               height: int = INPUT_HEIGHT,
               width: int  = INPUT_WIDTH) -> np.ndarray:
    """
    Load an image and convert to a model-ready tensor using center-crop.

    Steps (matches MCUNet-in0 expected pre-processing):
      1. Open with Pillow and force RGB.
      2. Resize shortest edge to height * 256/224 (preserve aspect ratio).
      3. Center crop to (width, height).
      4. Cast to float32, normalize to [-1.0, 1.0].
      5. Add batch dimension → shape (1, height, width, 3).

    Returns
    -------
    np.ndarray  shape (1, H, W, 3)  dtype float32
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    # Resize shortest edge
    target_short = int(round(height * 256 / 224))
    ratio = target_short / min(w, h)
    new_w = int(round(w * ratio))
    new_h = int(round(h * ratio))
    img = img.resize((new_w, new_h), resample=Image.BILINEAR)

    # Center crop
    left = (new_w - width)  // 2
    top  = (new_h - height) // 2
    img  = img.crop((left, top, left + width, top + height))

    arr = np.array(img, dtype=np.float32)
    arr = arr / 127.5 - 1.0
    arr = np.expand_dims(arr, axis=0)
    return arr


# ──────────────────────────────────────────────────────────────────────────────
# 2.  LABEL LOADING
# ──────────────────────────────────────────────────────────────────────────────
def load_labels(labels_path: str) -> List[str]:
    """Load ImageNet label names from a plain text file (one label per line)."""
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]
    assert len(labels) == 1000, f"Expected 1000 classes, got {len(labels)}"
    return labels


# ──────────────────────────────────────────────────────────────────────────────
# 3.  POST-PROCESSING
# ──────────────────────────────────────────────────────────────────────────────
def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = logits - np.max(logits)
    exp     = np.exp(shifted)
    return exp / exp.sum()


def postprocess(raw_output: np.ndarray,
                labels: List[str],
                top_k: int = TOP_K) -> List[Tuple[int, str, float]]:
    """
    Convert raw model output to human-readable top-K predictions.

    Handles both FP32 and INT8 output.
    Applies softmax only when output is not already a probability vector.
    """
    probs = np.squeeze(raw_output)   # shape (1000,)

    if not (float(probs.min()) >= 0.0 and np.isclose(float(probs.sum()), 1.0, atol=1e-3)):
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
# 4.  TFLite RUNNER
# ──────────────────────────────────────────────────────────────────────────────
def run_tflite_inference(model_path:  str,
                         image_path:  str,
                         labels:      List[str],
                         top_k:       int = TOP_K) -> List[Tuple[int, str, float]]:
    """Load a TFLite model (FP32 or INT8) and run single-image inference."""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_tensor = preprocess(image_path, INPUT_HEIGHT, INPUT_WIDTH)

    # Handle both FP32 and INT8 input types
    if input_details[0]["dtype"] == np.int8:
        scale     = input_details[0]["quantization"][0]
        zero_point = input_details[0]["quantization"][1]
        input_tensor = np.clip(
            np.round(input_tensor / scale + zero_point), -128, 127
        ).astype(np.int8)

    interpreter.set_tensor(input_details[0]["index"], input_tensor)
    interpreter.invoke()

    raw_output = interpreter.get_tensor(output_details[0]["index"])

    # Dequantize INT8 output
    if output_details[0]["dtype"] == np.int8:
        scale      = output_details[0]["quantization"][0]
        zero_point = output_details[0]["quantization"][1]
        raw_output = (raw_output.astype(np.float32) - zero_point) * scale

    results = postprocess(raw_output, labels, top_k)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# 5.  MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Single-image TFLite inference for MCUNet-in0 (FP32 or INT8).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python inference.py path/to/image.jpg\n"
            "  python inference.py path/to/image.jpg --top-k 3\n"
            "  python inference.py path/to/image.jpg --model model/mcunet_in0_INT8.tflite\n"
        ),
    )
    parser.add_argument(
        "image", nargs="?", default=None,
        help="Path to input image (JPEG/PNG).",
    )
    parser.add_argument(
        "--image", dest="image_flag", default=None, metavar="PATH",
        help="Path to input image (JPEG/PNG). Alternative to positional argument.",
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

    image_path = args.image_flag or args.image

    if image_path is None:
        print("┌─────────────────────────────────────────────────────────┐")
        print("│              MCUNet-in0 — TFLite Inference              │")
        print("├─────────────────────────────────────────────────────────┤")
        print("│  [ERROR] Missing required argument: image               │")
        print("│          Please provide a path to an input image.       │")
        print("├─────────────────────────────────────────────────────────┤")
        print("│  Usage:  python inference.py <image> [options]          │")
        print("│                                                         │")
        print("│  Arguments:                                             │")
        print("│    image            Path to input image (JPEG/PNG)      │")
        print("│                                                         │")
        print("│  Options:                                               │")
        print("│    --model PATH     Path to TFLite model                │")
        print("│    --labels PATH    Path to imagenet_labels.txt         │")
        print("│    --top-k N        Number of top predictions (def: 5)  │")
        print("│    -h, --help       Show full help message              │")
        print("│                                                         │")
        print("│  Examples:                                              │")
        print("│    python inference.py image.jpg                        │")
        print("│    python inference.py image.jpg --top-k 3              │")
        print("└─────────────────────────────────────────────────────────┘")
        sys.exit(1)

    errors = []
    if not os.path.isfile(args.model):
        errors.append(f"  Model not found      : {args.model}")
    if not os.path.isfile(image_path):
        errors.append(f"  Image not found      : {image_path}")
    if not os.path.isfile(args.labels):
        errors.append(f"  Labels file not found: {args.labels}")
    if errors:
        print("[ERROR] One or more required files are missing:")
        for e in errors:
            print(e)
        sys.exit(1)

    labels  = load_labels(args.labels)
    results = run_tflite_inference(args.model, image_path, labels, args.top_k)

    model_tag = os.path.basename(args.model)
    print("=" * 50)
    print(f"Image : {os.path.basename(image_path)}")
    print(f"Model : {model_tag}")
    print("-" * 50)
    for rank, (class_id, label, prob) in enumerate(results, start=1):
        print(f"  #{rank:>2}  [{class_id:>4}]  {label:<30}  {prob:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
