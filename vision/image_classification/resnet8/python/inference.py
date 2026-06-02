"""
Single-image inference for ResNet8 CIFAR-10 TFLite model.

Usage:
    python inference.py sample_images/airplane.png
    python inference.py --model model/Resnet_INT8.tflite sample_images/airplane.png
    python inference.py --image sample_images/airplane.png --top-k 3
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image

# ---------- Paths (relative to this script) ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "Resnet_fp32.tflite")
TOP_K = 5

# ---------- CIFAR-10 class labels ----------
CIFAR10_LABELS = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def load_interpreter(model_path: str):
    """Load TFLite model and return interpreter + input/output details."""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    return interpreter, input_details, output_details


def preprocess_image(image_path: str, input_details: dict) -> np.ndarray:
    """
    Load and preprocess a PNG/JPEG image for inference.
    - Resizes to 32x32 RGB
    - Normalises to [0, 1] float32
    - Quantises to int8 if the model expects it
    - Returns array of shape [1, 32, 32, 3]
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize((32, 32), Image.BILINEAR)
    arr = np.array(img).astype(np.float32)  # [32, 32, 3]

    if input_details['dtype'] == np.int8:
        scale, zero_point = input_details['quantization']
        arr = (arr / scale + zero_point).astype(np.int8)

    return np.expand_dims(arr, axis=0)  # [1, 32, 32, 3]


def load_npy(npy_path: str) -> np.ndarray:
    """Load a pre-processed numpy array saved as .npy."""
    return np.load(npy_path)


def run_inference(interpreter, input_details: dict, output_details: dict,
                  input_data: np.ndarray) -> np.ndarray:
    """Run inference and return dequantised output logits."""
    interpreter.set_tensor(input_details['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details['index'])[0]  # shape [num_classes]

    # Dequantise if model output is int8
    if output_details['dtype'] == np.int8:
        scale, zero_point = output_details['quantization']
        output = (output.astype(np.float32) - zero_point) * scale

    return output


def main():
    parser = argparse.ArgumentParser(
        description="ResNet8 CIFAR-10 — Single-image TFLite inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "image", nargs="?", default=None,
        help="Path to input image (JPEG/PNG) or .npy file.",
    )
    parser.add_argument(
        "--image", dest="image_flag", default=None, metavar="PATH",
        help="Path to input image (JPEG/PNG). Alternative to positional argument.",
    )
    parser.add_argument(
        "--model", default=MODEL_PATH,
        help=f"Path to TFLite model (default: model/Resnet_fp32.tflite).",
    )
    parser.add_argument(
        "--top-k", type=int, default=TOP_K,
        help=f"Number of top predictions to show (default: {TOP_K}).",
    )
    args = parser.parse_args()

    # Resolve image path (positional or --image flag)
    image_path = args.image_flag or args.image
    if image_path is None:
        parser.error("Please provide an image path (positional or --image).")

    if not os.path.isfile(image_path):
        parser.error(f"File not found: {image_path}")

    # ---- Load model ----
    interpreter, input_details, output_details = load_interpreter(args.model)

    # ---- Load input ----
    ext = os.path.splitext(image_path)[1].lower()
    if ext == '.npy':
        input_data = load_npy(image_path)
    else:
        input_data = preprocess_image(image_path, input_details)

    # ---- Inference ----
    output = run_inference(interpreter, input_details, output_details, input_data)

    # ---- Top-K results ----
    top_k = min(args.top_k, len(CIFAR10_LABELS))
    top_indices = np.argsort(output)[::-1][:top_k]

    print("=" * 60)
    print(f"{'Rank':<6}{'Class':<8}{'Label':<20}{'Score'}")
    print("-" * 60)
    for rank, idx in enumerate(top_indices, 1):
        label = CIFAR10_LABELS[idx]
        score = float(output[idx])
        print(f"{rank:<6}{idx:<8}{label:<20}{score:.6f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
