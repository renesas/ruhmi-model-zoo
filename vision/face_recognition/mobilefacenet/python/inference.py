# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2026 Renesas Electronics Corporation. All rights reserved.
"""MobileFaceNet TFLite inference — face recognition and 1:1 verification.

The model produces a 128-D embedding for a 112x112 MTCNN-aligned RGB face.
Two faces are compared by cosine similarity of their L2-normalised embeddings.

Modes:
  Embedding only   : python inference.py --image sample_images/George_W_Bush_0014.jpg
  1:1 verification : python inference.py --image sample_images/George_W_Bush_0014.jpg \\
                                          --reference sample_images/George_W_Bush_0145.jpg

The default model is model/mobilefacenet_INT8.tflite.
Use --model to point at the FP32 variant.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "3")


@contextlib.contextmanager
def _silence_stderr():
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(2)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull)


import cv2
import numpy as np

try:
    with _silence_stderr():
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        try:
            import absl.logging
            absl.logging.set_verbosity(absl.logging.ERROR)
        except ImportError:
            pass
except ImportError as e:
    sys.exit(f"tensorflow is required for TFLite inference: {e}")


BASE_DIR      = Path(__file__).resolve().parent
DEFAULT_MODEL = BASE_DIR / "model" / "mobilefacenet_INT8.tflite"
EMBEDDING_DIM = 128
MATCH_THRESHOLD = 0.28

sys.path.insert(0, str(BASE_DIR))
from utils.align import align_face_bgr, get_mtcnn  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────────────────────────────────────
def preprocess_face(image_bgr: np.ndarray, input_details: dict) -> np.ndarray:
    aligned_rgb = align_face_bgr(image_bgr)          # uint8 (112, 112, 3)
    face = aligned_rgb.astype(np.float32) / 255.0
    face = (face - 0.5) / 0.5                        # -> [-1, 1]

    shape = input_details["shape"]
    if len(shape) == 4 and shape[1] == 3:            # NCHW layout
        face = face.transpose(2, 0, 1)
    face = np.expand_dims(face, axis=0)

    if input_details["dtype"] == np.int8:
        scale, zero_point = input_details["quantization"]
        face = np.clip(np.round(face / scale) + zero_point,
                       -128, 127).astype(np.int8)
    return face


# ──────────────────────────────────────────────────────────────────────────────
# Recognizer
# ──────────────────────────────────────────────────────────────────────────────
class FaceRecognizer:
    def __init__(self, model_path: os.PathLike):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Model not found: {model_path}")
        with _silence_stderr():
            self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
            self.interpreter.allocate_tensors()
        self.input_details  = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]
        print(f"  Model  : {model_path.name}")
        print(f"  Input  : {self.input_details['shape']} "
              f"{self.input_details['dtype'].__name__}")
        print(f"  Output : {self.output_details['shape']} "
              f"{self.output_details['dtype'].__name__}")
        with _silence_stderr():
            get_mtcnn()

    def get_embedding(self, face_bgr: np.ndarray) -> np.ndarray:
        inp = preprocess_face(face_bgr, self.input_details)
        self.interpreter.set_tensor(self.input_details["index"], inp)
        self.interpreter.invoke()
        raw = self.interpreter.get_tensor(self.output_details["index"])

        if self.output_details["dtype"] == np.int8:
            scale, zero_point = self.output_details["quantization"]
            emb = (raw.astype(np.float32) - zero_point) * scale
        else:
            emb = raw.astype(np.float32)

        emb = emb.flatten()
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb


# ──────────────────────────────────────────────────────────────────────────────
# Similarity
# ──────────────────────────────────────────────────────────────────────────────
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="MobileFaceNet TFLite inference")
    parser.add_argument("--image",     required=True, help="Input face image.")
    parser.add_argument("--reference", help="Reference image for 1:1 verification.")
    parser.add_argument(
        "--model", default=str(DEFAULT_MODEL),
        help=f"TFLite model path (default: {DEFAULT_MODEL.name}).",
    )
    parser.add_argument(
        "--threshold", type=float, default=MATCH_THRESHOLD,
        help=f"Cosine-similarity match threshold (default: {MATCH_THRESHOLD}).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print the first 10 embedding values.",
    )
    args = parser.parse_args()

    recognizer = FaceRecognizer(args.model)

    image = cv2.imread(args.image)
    if image is None:
        sys.exit(f"Cannot read image: {args.image}")

    t0 = time.time()
    try:
        embedding = recognizer.get_embedding(image)
    except RuntimeError as e:
        sys.exit(f"Face alignment failed for --image: {e}")
    dt_ms = (time.time() - t0) * 1000.0

    print(f"\nEmbedding extracted in {dt_ms:.1f} ms "
          f"(shape={embedding.shape}, ||x||={np.linalg.norm(embedding):.4f})")

    if args.verbose:
        print(f"  embedding[:10] = {embedding[:10]}")

    if args.reference:
        ref_image = cv2.imread(args.reference)
        if ref_image is None:
            sys.exit(f"Cannot read reference: {args.reference}")
        try:
            ref_emb = recognizer.get_embedding(ref_image)
        except RuntimeError as e:
            sys.exit(f"Face alignment failed for --reference: {e}")
        score   = cosine_similarity(embedding, ref_emb)
        verdict = "MATCH" if score >= args.threshold else "NO MATCH"
        print(f"\n1:1 verification : {verdict}")
        print(f"  cosine similarity : {score:.4f}")
        print(f"  threshold         : {args.threshold}")


if __name__ == "__main__":
    main()
