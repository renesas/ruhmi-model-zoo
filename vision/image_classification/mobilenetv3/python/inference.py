"""
Single-image TFLite inference for MobileNetV3-Small (minimalistic) 192×192.

Supports both FP32 and INT8 models — auto-detects the input dtype.

IMPORTANT – preprocessing:
  MobileNetV3 includes a Rescaling(1/127.5, offset=-1) layer inside the
  model graph, so the model expects raw pixel values in [0, 255] (float32).

Usage:
    python inference.py path/to/image.jpg
    python inference.py path/to/image.jpg --top-k 3
    python inference.py --image path/to/image.jpg --model model/mobilenet_v3_small_192_INT8.tflite
"""
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
# Defaults
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "model", "mobilenet_v3_small_192_FP32.tflite")
LABELS_PATH  = os.path.join(BASE_DIR, "utils", "imagenet_labels.txt")

TOP_K        = 5


# ──────────────────────────────────────────────────────────────────────────────
# 1.  PRE-PROCESSING
# ──────────────────────────────────────────────────────────────────────────────
def preprocess(image_path: str,
               height: int,
               width: int) -> np.ndarray:
    """
    Load an image and convert it to a model-ready tensor.

    MobileNetV3 expects float32 pixels in [0, 255].
    The model's built-in Rescaling layer normalises internally.

    Returns
    -------
    np.ndarray  shape (1, H, W, 3)  dtype float32  range [0, 255]
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((width, height), resample=Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)       # [0, 255]
    arr = np.expand_dims(arr, axis=0)            # (1, H, W, 3)
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
# 3.  POST-PROCESSING
# ──────────────────────────────────────────────────────────────────────────────
def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = logits - np.max(logits)
    exp     = np.exp(shifted)
    return exp / exp.sum()


def postprocess(raw_output: np.ndarray,
                labels:     List[str],
                top_k:      int = TOP_K,
                skip_softmax: bool = False) -> List[Tuple[int, str, float]]:
    """
    Convert raw model output to human-readable top-K predictions.

    When skip_softmax=True the output is used as-is (INT8 models have
    softmax baked in; quantisation rounding may make the sum ≠ 1.0).
    """
    probs = np.squeeze(raw_output)

    if not skip_softmax:
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
# 4.  TFLite RUNNER  (auto-detects FP32 vs INT8)
# ──────────────────────────────────────────────────────────────────────────────
def run_tflite_inference(model_path:  str,
                         image_path:  str,
                         labels:      List[str],
                         top_k:       int = TOP_K) -> List[Tuple[int, str, float]]:
    """
    Load a TFLite model and run single-image inference.
    Automatically handles FP32 and INT8 input/output tensors.
    Input size is auto-detected from the model's input tensor shape.
    """
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # ── Auto-detect input size from model ────────────────────
    input_shape = input_details[0]["shape"]  # (1, H, W, 3)
    height, width = int(input_shape[1]), int(input_shape[2])

    # ── Pre-process (always start with [0, 255] float32) ─────
    input_tensor = preprocess(image_path, height, width)

    # ── Quantise input if model expects int8 ─────────────────
    is_int8_input = (input_details[0]["dtype"] == np.int8)
    if is_int8_input:
        scale, zero_point = input_details[0]["quantization"]
        input_tensor = np.clip(
            np.round(input_tensor / scale) + zero_point, -128, 127
        ).astype(np.int8)

    # ── Feed & invoke ────────────────────────────────────────
    interpreter.set_tensor(input_details[0]["index"], input_tensor)
    interpreter.invoke()

    # ── Read & dequantise output ─────────────────────────────
    raw_output = interpreter.get_tensor(output_details[0]["index"])
    is_int8_output = (output_details[0]["dtype"] == np.int8)

    if is_int8_output:
        o_scale, o_zp = output_details[0]["quantization"]
        raw_output = (raw_output.astype(np.float32) - o_zp) * o_scale

    # ── Post-process ─────────────────────────────────────────
    # For INT8 models the softmax is baked into the graph; the dequantised
    # output is already a probability distribution.
    return postprocess(raw_output, labels, top_k, skip_softmax=is_int8_output)


# ──────────────────────────────────────────────────────────────────────────────
# 5.  MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Single-image TFLite inference for MobileNetV3-Small (minimalistic) 192×192.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python inference.py path/to/image.jpg\n"
            "  python inference.py path/to/image.jpg --top-k 3\n"
            "  python inference.py path/to/image.jpg --model model/mobilenet_v3_small_192_INT8.tflite\n"
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

    # ── Resolve image path (--image flag takes priority over positional) ─────
    image_path = args.image_flag or args.image

    # ── Check image argument first ───────────────────────────────────────────
    if image_path is None:
        print("┌────────────────────────────────────────────────────────┐")
        print("│         MobileNetV3-Small — TFLite Inference           │")
        print("├────────────────────────────────────────────────────────┤")
        print("│  [ERROR] Missing required argument: image              │")
        print("│          Please provide a path to an input image.      │")
        print("├────────────────────────────────────────────────────────┤")
        print("│  Usage:  python inference.py <image> [options]         │")
        print("│                                                        │")
        print("│  Arguments:                                            │")
        print("│    image            Path to input image (JPEG/PNG)     │")
        print("│                                                        │")
        print("│  Options:                                              │")
        print("│    --model PATH     Path to TFLite model               │")
        print("│    --labels PATH    Path to imagenet_labels.txt        │")
        print("│    --top-k N        Number of top predictions (def: 5) │")
        print("│    -h, --help       Show full help message             │")
        print("│                                                        │")
        print("│  Examples:                                             │")
        print("│    python inference.py image.jpg                       │")
        print("│    python inference.py image.jpg --top-k 3             │")
        print("└────────────────────────────────────────────────────────┘")
        sys.exit(1)

    # ── Validate paths ───────────────────────────────────────
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

    # ── Load labels ──────────────────────────────────────────
    labels = load_labels(args.labels)
    print(f"Loaded {len(labels)} labels from {args.labels}")

    # ── Detect model type and input size ────────────────────
    interp_tmp = tf.lite.Interpreter(model_path=args.model)
    interp_tmp.allocate_tensors()
    inp_details = interp_tmp.get_input_details()[0]
    inp_dtype = inp_details["dtype"]
    input_shape = inp_details["shape"]  # (1, H, W, 3)
    input_size = int(input_shape[1])
    del interp_tmp
    model_type = "INT8" if inp_dtype == np.int8 else "FP32"

    # ── Run inference ────────────────────────────────────────
    print(f"\nModel   : {args.model}  ({model_type})")
    print(f"Image   : {image_path}")
    print(f"Input   : ({input_size}, {input_size}, 3)  pixels in [0, 255] (auto-detected)")
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

    # ── Print results ────────────────────────────────────────
    print("─" * 60)
    print(f"{'Rank':<6} {'Class':>6}   {'Label':<35} {'Probability':>12}")
    print("─" * 60)
    for rank, (class_idx, label, prob) in enumerate(results, start=1):
        print(f"  {rank:<4} {class_idx:>6}   {label:<35} {prob:>12.6f}")
    print("─" * 60)


if __name__ == "__main__":
    main()
