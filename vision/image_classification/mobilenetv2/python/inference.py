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
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "model", "mobilenet_v2_FP32.tflite")
LABELS_PATH  = os.path.join(BASE_DIR, "utils", "imagenet_labels.txt")

# inference settings
TOP_K        = 5   # how many top predictions to show
INPUT_HEIGHT = 224
INPUT_WIDTH  = 224


def preprocess(image_path: str,
               height: int = INPUT_HEIGHT,
               width: int  = INPUT_WIDTH) -> np.ndarray:
    """
    Load an image file and convert it to a model-ready tensor.

    Returns
    -------
    np.ndarray  shape (1, H, W, 3)  dtype float32
    """
    # Step 1: load & convert color space
    img = Image.open(image_path).convert("RGB")

    # Step 2: resize  (PIL uses (width, height) ordering)
    img = img.resize((width, height), resample=Image.BILINEAR)

    # Step 3: to float32 numpy array  — shape (H, W, 3), values [0, 255]
    arr = np.array(img, dtype=np.float32)

    # Step 4: normalize to [-1, 1]
    arr = arr / 127.5 - 1.0

    # Step 5: add batch dimension → (1, H, W, 3)
    arr = np.expand_dims(arr, axis=0)

    return arr


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
# 3.  POST-PROCESSING  (manual — no keras.applications.imagenet_utils.decode_predictions)
# ──────────────────────────────────────────────────────────────────────────────
def softmax(logits: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax.
    Shifts by max before exponentiation to avoid overflow.
    """
    shifted = logits - np.max(logits)
    exp     = np.exp(shifted)
    return exp / exp.sum()


def postprocess(raw_output: np.ndarray,
                labels:     List[str],
                top_k:      int = TOP_K) -> List[Tuple[int, str, float]]:
    """
    Convert raw model output to human-readable top-K predictions.

    Steps:
      1. Squeeze to 1-D (remove batch dimension).
      2. Apply softmax only when necessary  (model already has softmax in its
         graph because classifier_activation='softmax' was set at build time,
         so values will already sum to ≈1.0 in the [0,1] range).
      3. argsort descending → take top-k indices.
      4. Map each index to its label string and probability.

    Parameters
    ----------
    raw_output : np.ndarray  shape (1, 1000) or (1000,)
    labels     : List[str]   length 1000
    top_k      : int         number of top predictions to return

    Returns
    -------
    List of (class_index: int, label: str, probability: float)
    sorted by descending probability.
    """
    # Step 1: squeeze to 1-D
    probs = np.squeeze(raw_output)   # shape (1000,)

    # Step 2: apply softmax only if values don't already look like probabilities.
    #         Heuristic: if any value is negative OR they don't sum to ~1 then
    #         the model returned raw logits → apply softmax.
    already_probabilities = (
        float(probs.min()) >= 0.0 and
        np.isclose(float(probs.sum()), 1.0, atol=1e-3)
    )
    if not already_probabilities:
        probs = softmax(probs)

    # Step 3: argsort descending, keep top-k
    top_indices = np.argsort(probs)[::-1][:top_k]   # shape (top_k,)

    # Step 4: build result list
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
    # ── Load & allocate interpreter ──────────────────────────────────────────
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # ── Pre-process ──────────────────────────────────────────────────────────
    input_tensor = preprocess(image_path, INPUT_HEIGHT, INPUT_WIDTH)

    # Sanity-check: this FP32 model expects float32 input
    assert input_details[0]["dtype"] == np.float32, (
        f"Expected float32 input, got {input_details[0]['dtype']}. "
        "Use the INT8 inference script for quantized models."
    )

    # ── Feed input & run ─────────────────────────────────────────────────────
    interpreter.set_tensor(input_details[0]["index"], input_tensor)
    interpreter.invoke()

    # ── Read output ──────────────────────────────────────────────────────────
    raw_output = interpreter.get_tensor(output_details[0]["index"])  # (1, 1000)

    # ── Post-process ─────────────────────────────────────────────────────────
    results = postprocess(raw_output, labels, top_k)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# 5.  MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Single-image TFLite FP32 inference for MobileNetV2.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python inference.py path/to/image.jpg\n"
            "  python inference.py path/to/image.jpg --top-k 3\n"
            "  python inference.py path/to/image.jpg --model Model/mobilenet_v2_FP32.tflite\n"
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
        print("│              MobileNetV2 — TFLite FP32 Inference        │")
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
    print(f"Input   : ({INPUT_HEIGHT}, {INPUT_WIDTH}, 3)  normalized to [-1, 1]")
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
