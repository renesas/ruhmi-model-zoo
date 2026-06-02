import os
import sys
import argparse
from typing import List, Tuple

# Suppress TensorFlow and oneDNN info/warning logs before importing tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import cv2
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

# ──────────────────────────────────────────────────────────────────────────────
# Defaults  (paths are relative to this script's directory: python/)
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "model", "squeezenet1_1_INT8.tflite")
LABELS_PATH  = os.path.join(BASE_DIR, "utils", "imagenet_labels.txt")

# Inference settings
TOP_K        = 5
INPUT_HEIGHT = 224
INPUT_WIDTH  = 224

# ImageNet normalisation (same as torchvision SqueezeNet)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# 1.  PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────
def preprocess(image_path: str,
               height: int = INPUT_HEIGHT,
               width: int  = INPUT_WIDTH) -> np.ndarray:
    """
    Load an image file and convert it to a model-ready tensor.

    Steps (mirrors torchvision SqueezeNet pre-processing):
      1. Read with OpenCV and convert BGR → RGB.
      2. Resize to (width, height).
      3. Cast to float32, scale to [0, 1].
      4. Normalize: (pixel - mean) / std  using ImageNet statistics.
      5. Add batch dimension → shape (1, height, width, 3).

    Returns
    -------
    np.ndarray  shape (1, H, W, 3)  dtype float32
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (width, height))
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return np.expand_dims(img, axis=0)   # (1, H, W, 3)


# ──────────────────────────────────────────────────────────────────────────────
# 2.  LABEL LOADING
# ──────────────────────────────────────────────────────────────────────────────
def load_labels(labels_path: str) -> List[str]:
    """
    Load ImageNet label names from a plain text file (one label per line).
    Returns a list of label strings where labels[i] → class index i.
    """
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
                labels:     List[str],
                top_k:      int = TOP_K) -> List[Tuple[int, str, float]]:
    """
    Convert raw model output to human-readable top-K predictions.

    Returns
    -------
    List of (class_index, label_string, probability)
    sorted by descending probability.
    """
    probs = np.squeeze(raw_output)

    # Apply softmax if output doesn't look like probabilities
    already_probabilities = (
        float(probs.min()) >= 0.0 and
        np.isclose(float(probs.sum()), 1.0, atol=1e-3)
    )
    if not already_probabilities:
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
    """
    Load the TFLite FP32 model and run a single-image inference.

    Returns
    -------
    List of (class_index, label, probability) — top-k results.
    """
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_tensor = preprocess(image_path, INPUT_HEIGHT, INPUT_WIDTH)

    assert input_details[0]["dtype"] == np.float32, (
        f"Expected float32 input, got {input_details[0]['dtype']}. "
        "Use the INT8 inference script for quantized models."
    )

    interpreter.set_tensor(input_details[0]["index"], input_tensor)
    interpreter.invoke()

    raw_output = interpreter.get_tensor(output_details[0]["index"])

    return postprocess(raw_output, labels, top_k)


# ──────────────────────────────────────────────────────────────────────────────
# 5.  MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Single-image TFLite FP32 inference for SqueezeNet1.1.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python inference.py path/to/image.jpg\n"
            "  python inference.py path/to/image.jpg --top-k 3\n"
            "  python inference.py path/to/image.jpg --model ../../Models/squeezenet1_1_FP32.tflite\n"
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
        help=f"Path to FP32 TFLite model (default: {MODEL_PATH}).",
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

    # ── Resolve image path (--image flag takes priority over positional) ─────
    image_path = args.image_flag or args.image

    # ── Check image argument first ───────────────────────────────────────────
    if image_path is None:
        print("┌─────────────────────────────────────────────────────────┐")
        print("│             SqueezeNet1.1 — TFLite FP32 Inference       │")
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
        print("│    --model PATH     Path to FP32 TFLite model           │")
        print("│    --labels PATH    Path to imagenet_labels.txt         │")
        print("│    --top-k N        Number of top predictions (def: 5)  │")
        print("│    -h, --help       Show full help message              │")
        print("│                                                         │")
        print("│  Examples:                                              │")
        print("│    python inference.py image.jpg                        │")
        print("│    python inference.py image.jpg --top-k 3              │")
        print("└─────────────────────────────────────────────────────────┘")
        sys.exit(1)

    # ── Validate paths ───────────────────────────────────────────────────────
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

    if args.top_k < 1 or args.top_k > 1000:
        print(f"[ERROR] --top-k must be between 1 and 1000, got {args.top_k}")
        sys.exit(1)

    # ── Load labels ──────────────────────────────────────────────────────────
    labels = load_labels(args.labels)
    print(f"Loaded {len(labels)} labels from {args.labels}")

    # ── Run inference ────────────────────────────────────────────────────────
    print(f"\nModel   : {args.model}")
    print(f"Image   : {image_path}")
    print(f"Input   : ({INPUT_HEIGHT}, {INPUT_WIDTH}, 3)  ImageNet mean/std")
    print(f"Top-K   : {args.top_k}")
    print()

    try:
        results = run_tflite_inference(
            model_path = args.model,
            image_path = image_path,
            labels     = labels,
            top_k      = args.top_k,
        )
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")
        sys.exit(1)

    # ── Print results ────────────────────────────────────────────────────────
    print("─" * 60)
    print(f"{'Rank':<6} {'Class':>6}   {'Label':<35} {'Probability':>12}")
    print("─" * 60)
    for rank, (class_idx, label, prob) in enumerate(results, start=1):
        print(f"  {rank:<4} {class_idx:>6}   {label:<35} {prob:>12.6f}")
    print("─" * 60)


if __name__ == "__main__":
    main()
