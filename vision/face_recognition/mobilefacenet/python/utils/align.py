# SPDX-License-Identifier: Apache-2.0
"""MTCNN face alignment for MobileFaceNet.

This module wraps the MTCNN detector from ``facenet-pytorch`` and warps the
detected face into the canonical 5-point template used by ArcFace / MobileFaceNet
training datasets. The warped output is a 112x112 uint8 RGB image with eye
centres, nose tip and mouth corners landing at fixed pixel coordinates.

Public API
----------
align_face(path) -> np.ndarray
    Read an image from disk, detect+align the largest face, return RGB uint8.
align_face_bgr(img_bgr) -> np.ndarray
    Same but takes an OpenCV BGR ndarray.
get_mtcnn() -> facenet_pytorch.MTCNN
    Singleton accessor; weights (~2 MB) are downloaded on first use.

The MTCNN instance is created lazily so importing this module is cheap and so
``torch`` is only loaded once per process.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

# Canonical 5-point landmark template for a 112x112 aligned face
# (left eye, right eye, nose tip, left mouth corner, right mouth corner).
# These coordinates are the ArcFace / InsightFace standard and are what
# MobileFaceNet was trained against.
ARCFACE_REF_PTS = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

OUTPUT_SIZE = 112

_MTCNN = None  # singleton


def get_mtcnn():
    """Return a process-wide singleton MTCNN detector (CPU)."""
    global _MTCNN
    if _MTCNN is None:
        # Silence the TF/torch startup banner that some envs emit.
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        from facenet_pytorch import MTCNN  # local import: heavy
        _MTCNN = MTCNN(
            image_size=OUTPUT_SIZE,
            margin=0,
            keep_all=False,        # only return the most prominent face
            select_largest=True,
            post_process=False,    # we do our own normalisation
            device="cpu",
        )
    return _MTCNN


def _warp_to_template(img_rgb: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Similarity-warp ``img_rgb`` so its 5 landmarks match ARCFACE_REF_PTS."""
    src = landmarks.astype(np.float32)
    M, _ = cv2.estimateAffinePartial2D(
        src, ARCFACE_REF_PTS, method=cv2.LMEDS
    )
    if M is None:
        raise RuntimeError("Could not estimate similarity transform from landmarks.")
    aligned = cv2.warpAffine(
        img_rgb, M, (OUTPUT_SIZE, OUTPUT_SIZE),
        borderValue=0.0,
    )
    return aligned


def align_face_rgb(img_rgb: np.ndarray) -> np.ndarray:
    """Detect+align the largest face in an RGB uint8 image.

    Returns a (112, 112, 3) uint8 RGB array. Raises ``RuntimeError`` if no
    face is found.
    """
    if img_rgb.dtype != np.uint8:
        img_rgb = img_rgb.astype(np.uint8)
    pil = Image.fromarray(img_rgb)
    mtcnn = get_mtcnn()
    # MTCNN.detect returns (boxes, probs, landmarks) when landmarks=True.
    boxes, probs, landmarks = mtcnn.detect(pil, landmarks=True)
    if boxes is None or len(boxes) == 0 or landmarks is None:
        raise RuntimeError("MTCNN found no face in image.")
    return _warp_to_template(img_rgb, landmarks[0])


def align_face_bgr(img_bgr: np.ndarray) -> np.ndarray:
    """Same as :func:`align_face_rgb` but accepts an OpenCV BGR image."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return align_face_rgb(rgb)


def align_face(image_path: os.PathLike) -> np.ndarray:
    """Read ``image_path`` (any format Pillow supports) and align its face."""
    image_path = Path(image_path)
    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    return align_face_rgb(arr)


def align_face_cached(
    src_path: os.PathLike,
    cache_root: os.PathLike,
    src_root: Optional[os.PathLike] = None,
) -> np.ndarray:
    """Read+align ``src_path``, persisting the result under ``cache_root``.

    The cache mirrors the source directory layout. If ``src_root`` is provided
    the relative path of ``src_path`` inside ``src_root`` is used as the cache
    key; otherwise the source filename alone is used.

    On subsequent calls the cached JPEG is loaded directly (no MTCNN run).
    """
    src_path = Path(src_path)
    cache_root = Path(cache_root)
    if src_root is not None:
        try:
            rel = src_path.relative_to(Path(src_root))
        except ValueError:
            rel = Path(src_path.name)
    else:
        rel = Path(src_path.name)
    cache_path = cache_root / rel.with_suffix(".jpg")

    if cache_path.is_file():
        img = Image.open(cache_path).convert("RGB")
        return np.asarray(img, dtype=np.uint8)

    aligned = align_face(src_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(aligned).save(cache_path, quality=95)
    return aligned
