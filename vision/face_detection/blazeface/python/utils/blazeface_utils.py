# Copyright 2026 Renesas Electronics Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# BlazeFace model from Google MediaPipe, PyTorch port by Matthijs Hollemans:
#   https://github.com/hollance/BlazeFace-PyTorch
# Licensed under the MIT License.
"""
BlazeFace Utilities — Shared Post-Processing & Helpers
=======================================================
Common code shared by both ONNX/ and TFLite/ sub-pipelines:

  - Anchor-based box decoding
  - Sigmoid score thresholding
  - BlazeFace-style weighted NMS
  - Image preprocessing
  - Visualization (draw boxes + keypoints)
  - WIDER FACE ground-truth parsing & evaluation helpers
  - Calibration image loading
"""

import glob
import os
import sys
import time
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
INPUT_SIZE = 128

# Post-processing parameters (from BlazeFace paper / MediaPipe)
SCORE_CLIPPING_THRESH  = 100.0
MIN_SCORE_THRESH       = 0.5
MIN_SUPPRESSION_THRESH = 0.3   # IoU threshold for weighted NMS
NUM_COORDS    = 16   # 4 bbox + 6 keypoints × 2
NUM_KEYPOINTS = 6

# Anchor decoding scales
X_SCALE = 128.0
Y_SCALE = 128.0
W_SCALE = 128.0
H_SCALE = 128.0

# Keypoint names
KEYPOINT_NAMES = [
    "right_eye", "left_eye", "nose",
    "mouth", "right_ear", "left_ear",
]

# Visualization colours (BGR)
BOX_COLOR      = (0, 255, 0)     # green
KEYPOINT_COLOR = (0, 0, 255)     # red
TEXT_COLOR     = (255, 255, 255)  # white

# WIDER FACE evaluation IoU threshold
IOU_THRESH = 0.5

# Source directory for PyTorch weights / anchors
BLAZEFACE_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    ))),
    "playground", "BlazeFace-PyTorch",
)

# WIDER FACE download URLs (HuggingFace mirrors)
WIDER_VAL_URL  = "https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_val.zip"
WIDER_ANNO_URL = "https://huggingface.co/datasets/wider_face/resolve/main/data/wider_face_split.zip"

DEFAULT_CALIB_NUM = 100


# ══════════════════════════════════════════════════════════════
#  POST-PROCESSING
# ══════════════════════════════════════════════════════════════

def decode_boxes(raw_boxes: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Decode raw regressor output into normalized box + keypoint coords.

    Parameters
    ----------
    raw_boxes : (N, 16)  —  raw regressor predictions per anchor
    anchors   : (N, 4)   —  [x_center, y_center, w, h] (normalized)

    Returns
    -------
    boxes : (N, 16) — [ymin, xmin, ymax, xmax, kp0_x, kp0_y, ..., kp5_x, kp5_y]
        All values in [0, 1] (normalized coordinates).
    """
    boxes = np.zeros_like(raw_boxes)

    x_center = raw_boxes[:, 0] / X_SCALE * anchors[:, 2] + anchors[:, 0]
    y_center = raw_boxes[:, 1] / Y_SCALE * anchors[:, 3] + anchors[:, 1]
    w        = raw_boxes[:, 2] / W_SCALE * anchors[:, 2]
    h        = raw_boxes[:, 3] / H_SCALE * anchors[:, 3]

    boxes[:, 0] = y_center - h / 2.0  # ymin
    boxes[:, 1] = x_center - w / 2.0  # xmin
    boxes[:, 2] = y_center + h / 2.0  # ymax
    boxes[:, 3] = x_center + w / 2.0  # xmax

    for k in range(NUM_KEYPOINTS):
        offset = 4 + k * 2
        boxes[:, offset]     = raw_boxes[:, offset]     / X_SCALE * anchors[:, 2] + anchors[:, 0]
        boxes[:, offset + 1] = raw_boxes[:, offset + 1] / Y_SCALE * anchors[:, 3] + anchors[:, 1]

    return boxes


def tensors_to_detections(raw_boxes: np.ndarray,
                          raw_scores: np.ndarray,
                          anchors: np.ndarray,
                          score_clipping_thresh: float = SCORE_CLIPPING_THRESH,
                          min_score_thresh: float = MIN_SCORE_THRESH
                          ) -> np.ndarray:
    """Convert raw model outputs to filtered detections.

    Parameters
    ----------
    raw_boxes  : (N, 16)  regressor output
    raw_scores : (N, 1)   classifier output
    anchors    : (N, 4)

    Returns
    -------
    detections : (M, 17) — [ymin, xmin, ymax, xmax, 6×(x,y) keypoints, score]
    """
    detection_boxes = decode_boxes(raw_boxes, anchors)

    scores = raw_scores.flatten()
    scores = np.clip(scores, -score_clipping_thresh, score_clipping_thresh)
    scores = 1.0 / (1.0 + np.exp(-scores))  # sigmoid

    mask = scores >= min_score_thresh
    boxes = detection_boxes[mask]
    scores = scores[mask]

    if len(boxes) == 0:
        return np.empty((0, 17))

    return np.concatenate([boxes, scores[:, np.newaxis]], axis=1)


def _iou(box_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """IoU of box_a (4,) vs boxes_b (M, 4) in [ymin, xmin, ymax, xmax]."""
    ymin = np.maximum(box_a[0], boxes_b[:, 0])
    xmin = np.maximum(box_a[1], boxes_b[:, 1])
    ymax = np.minimum(box_a[2], boxes_b[:, 2])
    xmax = np.minimum(box_a[3], boxes_b[:, 3])

    inter = np.maximum(0, ymax - ymin) * np.maximum(0, xmax - xmin)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    return inter / (area_a + area_b - inter + 1e-6)


def weighted_non_max_suppression(detections: np.ndarray,
                                 min_suppression_thresh: float = MIN_SUPPRESSION_THRESH
                                 ) -> np.ndarray:
    """BlazeFace-style weighted NMS.

    Overlapping detections are blended using score-weighted averaging
    rather than simply discarding lower-scoring boxes.

    Parameters
    ----------
    detections : (M, 17) — last column is score.

    Returns
    -------
    output : (K, 17)
    """
    if len(detections) == 0:
        return np.empty((0, 17))

    output = []
    remaining = np.argsort(-detections[:, 16])

    while len(remaining) > 0:
        detection = detections[remaining[0]].copy()
        first_box = detection[:4]
        other_boxes = detections[remaining, :4]

        ious = _iou(first_box, other_boxes)
        mask = ious > min_suppression_thresh
        overlapping = remaining[mask]
        remaining = remaining[~mask]

        if len(overlapping) > 1:
            coords = detections[overlapping, :16]
            scores = detections[overlapping, 16:17]
            total_score = scores.sum()
            weighted = (coords * scores).sum(axis=0) / total_score
            detection[:16] = weighted
            detection[16] = total_score / len(overlapping)

        output.append(detection)

    return np.array(output) if output else np.empty((0, 17))


# ══════════════════════════════════════════════════════════════
#  PREPROCESSING
# ══════════════════════════════════════════════════════════════

def preprocess_image(image: np.ndarray, input_size: int = INPUT_SIZE):
    """Resize BGR image to (input_size, input_size) RGB uint8.

    Returns
    -------
    img_resized : (H, W, 3) uint8 RGB
    scale_x, scale_y : float
    """
    orig_h, orig_w = image.shape[:2]
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (input_size, input_size),
                             interpolation=cv2.INTER_LINEAR)
    return img_resized, orig_w / input_size, orig_h / input_size


def normalize_image(img_uint8: np.ndarray) -> np.ndarray:
    """Normalize uint8 [0,255] → float32 [-1, 1]."""
    return img_uint8.astype(np.float32) / 127.5 - 1.0


# ══════════════════════════════════════════════════════════════
#  FULL DETECTION PIPELINE
# ══════════════════════════════════════════════════════════════

def detect_faces(model, image: np.ndarray,
                 score_thresh: float = MIN_SCORE_THRESH,
                 nms_thresh: float = MIN_SUPPRESSION_THRESH):
    """Run full face detection on a BGR image.

    ``model`` must have ``.predict(img_uint8)`` returning
    ``(raw_boxes, raw_scores)`` and ``.anchors``.

    Returns
    -------
    detections : (K, 17) — pixel coordinates
    latency_ms : float
    """
    img_resized, scale_x, scale_y = preprocess_image(image)

    t0 = time.perf_counter()
    raw_boxes, raw_scores = model.predict(img_resized)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    dets = tensors_to_detections(
        raw_boxes, raw_scores, model.anchors,
        min_score_thresh=score_thresh,
    )
    dets = weighted_non_max_suppression(dets, nms_thresh)

    if len(dets) == 0:
        return np.empty((0, 17)), latency_ms

    # Scale to original image pixels
    orig_h, orig_w = image.shape[:2]
    dets[:, 0] *= orig_h  # ymin
    dets[:, 1] *= orig_w  # xmin
    dets[:, 2] *= orig_h  # ymax
    dets[:, 3] *= orig_w  # xmax
    for k in range(NUM_KEYPOINTS):
        offset = 4 + k * 2
        dets[:, offset]     *= orig_w
        dets[:, offset + 1] *= orig_h

    return dets, latency_ms


# ══════════════════════════════════════════════════════════════
#  VISUALIZATION
# ══════════════════════════════════════════════════════════════

def draw_detections(image: np.ndarray, detections: np.ndarray) -> np.ndarray:
    """Draw bounding boxes, keypoints, and scores on a BGR image."""
    vis = image.copy()
    for det in detections:
        ymin, xmin, ymax, xmax = det[:4].astype(int)
        score = det[16]

        cv2.rectangle(vis, (xmin, ymin), (xmax, ymax), BOX_COLOR, 2)

        label = f"{score:.2f}"
        txt_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(vis, (xmin, ymin - txt_size[1] - 6),
                      (xmin + txt_size[0] + 4, ymin), BOX_COLOR, -1)
        cv2.putText(vis, label, (xmin + 2, ymin - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)

        for k in range(NUM_KEYPOINTS):
            kp_x = int(det[4 + k * 2])
            kp_y = int(det[4 + k * 2 + 1])
            cv2.circle(vis, (kp_x, kp_y), 2, KEYPOINT_COLOR, -1)

    return vis


# ══════════════════════════════════════════════════════════════
#  CALIBRATION IMAGE LOADING
# ══════════════════════════════════════════════════════════════

def _download(url: str, dest: str) -> None:
    """Download a file with progress."""
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    print(f"  Downloading {url}")
    print(f"           -> {dest}")
    try:
        from tqdm import tqdm

        class _Reporter:
            def __init__(self):
                self.pbar = None
            def __call__(self, block_num, block_size, total_size):
                if self.pbar is None:
                    self.pbar = tqdm(total=total_size, unit="B",
                                    unit_scale=True, unit_divisor=1024)
                self.pbar.update(block_size)

        urllib.request.urlretrieve(url, dest, _Reporter())
    except ImportError:
        urllib.request.urlretrieve(url, dest)
    print("  ✅ Download complete")


def ensure_calibration_images(dataset_dir: str,
                              num_images: int = DEFAULT_CALIB_NUM) -> str:
    """Ensure WIDER FACE val images are available for calibration.

    Returns path to image directory, or None if unavailable.
    """
    wider_dir = os.path.join(dataset_dir, "WIDER_val", "images")

    if os.path.isdir(wider_dir):
        imgs = glob.glob(os.path.join(wider_dir, "**", "*.jpg"), recursive=True)
        if len(imgs) >= num_images:
            print(f"  ✅ WIDER FACE val found: {len(imgs)} images")
            return wider_dir

    os.makedirs(dataset_dir, exist_ok=True)
    zip_path = os.path.join(dataset_dir, "WIDER_val.zip")
    try:
        if not os.path.isfile(zip_path):
            _download(WIDER_VAL_URL, zip_path)
        print("  Extracting WIDER_val.zip ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dataset_dir)
        print(f"  ✅ Extracted to {dataset_dir}")
        return wider_dir
    except Exception as e:
        print(f"  ⚠️  Could not download WIDER FACE: {e}")
        print("  Using synthetic calibration data instead.")
        return None


def load_calibration_images_numpy(calib_dir: str, num_images: int,
                                  input_size: int = INPUT_SIZE) -> list:
    """Load calibration images as preprocessed numpy arrays.

    Returns list of arrays, each shape (1, input_size, input_size, 3) float32
    in [-1, 1] range.
    """
    if calib_dir is None:
        print(f"  Generating {num_images} synthetic calibration images...")
        return [
            np.random.randint(0, 256, (1, input_size, input_size, 3)
                              ).astype(np.float32) / 127.5 - 1.0
            for _ in range(num_images)
        ]

    patterns = [
        os.path.join(calib_dir, "**", "*.jpg"),
        os.path.join(calib_dir, "**", "*.png"),
        os.path.join(calib_dir, "**", "*.jpeg"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))
    files = sorted(files)

    if len(files) == 0:
        print(f"  ⚠️  No images in {calib_dir}, using synthetic data")
        return [
            np.random.randint(0, 256, (1, input_size, input_size, 3)
                              ).astype(np.float32) / 127.5 - 1.0
            for _ in range(num_images)
        ]

    step = max(1, len(files) // num_images)
    selected = files[::step][:num_images]
    print(f"  Loading {len(selected)} calibration images from {calib_dir}")

    from tqdm import tqdm
    results = []
    for f in tqdm(selected, desc="  Loading calibration", unit="img", leave=False):
        img = cv2.imread(f)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (input_size, input_size))
        img = img.astype(np.float32) / 127.5 - 1.0
        results.append(img[np.newaxis, ...])

    return results


# ══════════════════════════════════════════════════════════════
#  WIDER FACE GROUND TRUTH & EVALUATION
# ══════════════════════════════════════════════════════════════

def download_wider_annotations(dataset_dir: str) -> str:
    """Download WIDER FACE annotations. Returns path to GT file."""
    anno_dir = os.path.join(dataset_dir, "wider_face_split")
    gt_file = os.path.join(anno_dir, "wider_face_val_bbx_gt.txt")

    if os.path.isfile(gt_file):
        return gt_file

    zip_path = os.path.join(dataset_dir, "wider_face_split.zip")
    os.makedirs(dataset_dir, exist_ok=True)

    if not os.path.isfile(zip_path):
        print(f"  Downloading WIDER FACE annotations...")
        urllib.request.urlretrieve(WIDER_ANNO_URL, zip_path)

    print(f"  Extracting annotations...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dataset_dir)

    if not os.path.isfile(gt_file):
        raise FileNotFoundError(f"Annotation file not found: {gt_file}")
    return gt_file


def parse_wider_face_gt(gt_file: str) -> dict:
    """Parse WIDER FACE ground truth.

    Returns {relative_path: [(x1, y1, w, h, blur, expr, illum, invalid, occ, pose), ...]}
    """
    annotations = {}
    with open(gt_file, "r") as f:
        while True:
            line = f.readline().strip()
            if not line:
                break
            img_path = line
            num_faces = int(f.readline().strip())
            boxes = []
            for _ in range(num_faces):
                parts = f.readline().strip().split()
                x1, y1, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                blur, expr, illum = int(parts[4]), int(parts[5]), int(parts[6])
                invalid, occ, pose = int(parts[7]), int(parts[8]), int(parts[9])
                boxes.append((x1, y1, w, h, blur, expr, illum, invalid, occ, pose))
            if num_faces == 0:
                f.readline()  # dummy line
            annotations[img_path] = boxes
    return annotations


def get_difficulty_label(box_info: tuple) -> str:
    """Classify GT box as Easy / Medium / Hard."""
    _, _, _, _, blur, _, _, invalid, occ, _ = box_info
    if invalid:
        return "invalid"
    if occ >= 2 or blur >= 2:
        return "hard"
    elif occ >= 1 or blur >= 1:
        return "medium"
    return "easy"


def filter_frontal_annotations(annotations: dict,
                                max_faces: int = 8,
                                min_face_px: int = 10) -> dict:
    """Return the subset of WIDER FACE annotations suitable for BlazeFace.

    BlazeFace is a *frontal-face* detector designed for near-frontal faces in
    selfie / video-call / portrait scenarios.  The full WIDER FACE val set
    contains crowd scenes (100 + tiny faces), heavy occlusion, and extreme
    atypical poses that are explicitly outside BlazeFace's design envelope.

    A WIDER FACE image is included in the frontal subset when ALL of the
    following hold:

    1. **All valid faces have typical pose** (``pose == 0`` in the GT).
       Atypical pose (profile, tilted > ~45°) makes the image unsuitable.
    2. **At most ``max_faces`` valid faces** (default 8).
       Crowd shots with dozens of tiny faces are excluded.
    3. **At least one easy-difficulty face** that is large enough
       (``min(w, h) >= min_face_px``, default 10 px).
       Images with only tiny/occluded faces are excluded.

    Parameters
    ----------
    annotations : dict
        Full output of ``parse_wider_face_gt()``.
    max_faces : int
        Maximum number of valid (non-invalid) faces allowed per image.
    min_face_px : int
        Minimum face size (min of w, h) to count as a valid detection target.

    Returns
    -------
    dict  — same format as input, restricted to frontal-suitable images.
    """
    frontal = {}
    for img_path, boxes in annotations.items():
        # Keep only valid, large-enough boxes
        valid = [b for b in boxes
                 if not b[7]                        # invalid flag == 0
                 and min(b[2], b[3]) >= min_face_px]  # w,h >= min_face_px

        if not valid:
            continue

        # Condition 1: all valid faces must be typical pose
        if any(b[9] != 0 for b in valid):           # pose == 0
            continue

        # Condition 2: not a crowd shot
        if len(valid) > max_faces:
            continue

        # Condition 3: at least one easy-difficulty face
        if not any(get_difficulty_label(b) == "easy" for b in valid):
            continue

        frontal[img_path] = boxes

    return frontal


def compute_iou_matrix(pred_boxes: np.ndarray,
                       gt_boxes: np.ndarray) -> np.ndarray:
    """IoU between pred_boxes (M, 4) and gt_boxes (N, 4) in [x1,y1,x2,y2]."""
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return np.empty((len(pred_boxes), len(gt_boxes)))

    px1, py1, px2, py2 = pred_boxes.T
    gx1, gy1, gx2, gy2 = gt_boxes.T

    inter_x1 = np.maximum(px1[:, None], gx1[None, :])
    inter_y1 = np.maximum(py1[:, None], gy1[None, :])
    inter_x2 = np.minimum(px2[:, None], gx2[None, :])
    inter_y2 = np.minimum(py2[:, None], gy2[None, :])

    inter = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
    area_p = np.maximum(0, px2 - px1) * np.maximum(0, py2 - py1)
    area_g = np.maximum(0, gx2 - gx1) * np.maximum(0, gy2 - gy1)

    union = area_p[:, None] + area_g[None, :] - inter
    return inter / (union + 1e-6)


def match_detections(pred_boxes: np.ndarray, pred_scores: np.ndarray,
                     gt_boxes: np.ndarray,
                     iou_thresh: float = IOU_THRESH):
    """Match predictions to GT. Returns (tp, fp) bool arrays."""
    if len(pred_boxes) == 0:
        return np.array([], dtype=bool), np.array([], dtype=bool)
    if len(gt_boxes) == 0:
        return np.zeros(len(pred_boxes), dtype=bool), np.ones(len(pred_boxes), dtype=bool)

    iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)
    order = np.argsort(-pred_scores)

    tp = np.zeros(len(pred_boxes), dtype=bool)
    fp = np.zeros(len(pred_boxes), dtype=bool)
    gt_matched = np.zeros(len(gt_boxes), dtype=bool)

    for i in order:
        best_gt = np.argmax(iou_matrix[i])
        if iou_matrix[i, best_gt] >= iou_thresh and not gt_matched[best_gt]:
            tp[i] = True
            gt_matched[best_gt] = True
        else:
            fp[i] = True

    return tp, fp


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Average Precision — VOC 2010+ area-under-curve (AUC) method.

    Appends sentinel values (recall=0,prec=1) and (recall=1,prec=0), then
    computes the area under the precision–recall curve using the
    monotone-decreasing envelope, identical to pycocotools / torchvision.
    This replaces the older 11-point interpolation which underestimates AP.
    """
    # Sentinel points
    rec  = np.concatenate(([0.0], recalls,    [1.0]))
    prec = np.concatenate(([1.0], precisions, [0.0]))

    # Make precision monotonically decreasing (right-to-left max)
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])

    # Sum area of rectangular strips where recall changes
    idx = np.where(rec[1:] != rec[:-1])[0]
    ap = np.sum((rec[idx + 1] - rec[idx]) * prec[idx + 1])
    return float(ap)


def compute_metrics(all_scores, all_tp, all_fp, total_gt) -> dict:
    """Compute precision, recall, F1, AP from accumulated results."""
    if not all_scores or total_gt == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "ap": 0.0,
                "num_gt": total_gt, "num_pred": len(all_scores),
                "tp_total": 0, "fp_total": 0, "fn_total": total_gt}

    scores = np.array(all_scores)
    tp = np.array(all_tp, dtype=bool)
    fp = np.array(all_fp, dtype=bool)

    order = np.argsort(-scores)
    tp, fp = tp[order], fp[order]

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recalls = cum_tp / float(total_gt)
    precisions = cum_tp / (cum_tp + cum_fp + 1e-10)

    tp_total = int(cum_tp[-1])
    fp_total = int(cum_fp[-1])
    fn_total = total_gt - tp_total

    prec = tp_total / (tp_total + fp_total + 1e-10)
    rec = tp_total / (total_gt + 1e-10)
    f1 = 2 * prec * rec / (prec + rec + 1e-10)
    ap = compute_ap(recalls, precisions)

    return {"precision": prec, "recall": rec, "f1": f1, "ap": ap,
            "num_gt": total_gt, "num_pred": len(all_scores),
            "tp_total": tp_total, "fp_total": fp_total, "fn_total": fn_total}


def evaluate_model_on_wider(model, annotations: dict, images_dir: str,
                            score_thresh: float = MIN_SCORE_THRESH,
                            nms_thresh: float = MIN_SUPPRESSION_THRESH,
                            limit: int = 0, model_name: str = "",
                            difficulty: str = "all") -> dict:
    """Evaluate any model wrapper on WIDER FACE val set.

    ``model`` must have ``.predict(img_uint8)`` and ``.anchors``.

    Parameters
    ----------
    difficulty : "all" | "easy" | "medium" | "hard"
        Filter GT boxes to a specific difficulty level for the *overall*
        metrics.  Per-difficulty breakdown is always computed.
        BlazeFace is designed for frontal faces; use "easy" for a fair
        comparison against the published benchmark numbers.
    """
    from tqdm import tqdm

    DIFFS = ["easy", "medium", "hard"]

    # Accumulators keyed by difficulty (plus "all")
    acc = {d: {"scores": [], "tp": [], "fp": [], "gt": 0}
           for d in DIFFS + ["all"]}

    num_images, errors = 0, 0

    img_paths = sorted(annotations.keys())
    if limit > 0:
        img_paths = img_paths[:limit]

    for img_rel in tqdm(img_paths, desc=f"  {model_name:15s}",
                        unit="img", leave=False):
        img_path = os.path.join(images_dir, img_rel)
        if not os.path.isfile(img_path):
            continue

        image = cv2.imread(img_path)
        if image is None:
            errors += 1
            continue

        num_images += 1

        # ── Ground truth: parse and label by difficulty ───────────────────
        gt_infos = annotations[img_rel]
        all_gt_boxes  = []   # list of [x1,y1,x2,y2]
        all_gt_diffs  = []   # "easy" | "medium" | "hard"

        for info in gt_infos:
            x1, y1, w, h = info[:4]
            diff = get_difficulty_label(info)
            if diff == "invalid" or w < 1 or h < 1:
                continue
            all_gt_boxes.append([x1, y1, x1 + w, y1 + h])
            all_gt_diffs.append(diff)

        gt_by_diff = {}
        for d in DIFFS:
            idxs = [i for i, dd in enumerate(all_gt_diffs) if dd == d]
            gt_by_diff[d] = (np.array([all_gt_boxes[i] for i in idxs])
                             if idxs else np.empty((0, 4)))
            acc[d]["gt"] += len(idxs)

        gt_all = (np.array(all_gt_boxes)
                  if all_gt_boxes else np.empty((0, 4)))
        acc["all"]["gt"] += len(all_gt_boxes)

        # ── Run model ─────────────────────────────────────────────────────
        try:
            img_resized, _, _ = preprocess_image(image)
            raw_boxes, raw_scores = model.predict(img_resized)
            dets = tensors_to_detections(raw_boxes, raw_scores, model.anchors,
                                         min_score_thresh=score_thresh)
            dets = weighted_non_max_suppression(dets, nms_thresh)
        except Exception:
            errors += 1
            continue

        if len(dets) == 0:
            continue

        # Scale to original pixels → [x1, y1, x2, y2]
        orig_h, orig_w = image.shape[:2]
        pred_boxes = np.stack([
            dets[:, 1] * orig_w,   # x1
            dets[:, 0] * orig_h,   # y1
            dets[:, 3] * orig_w,   # x2
            dets[:, 2] * orig_h,   # y2
        ], axis=1)
        pred_scores_arr = dets[:, 16]

        # ── Match against each GT set independently ───────────────────────
        # "all" — match against all valid GT boxes
        tp, fp = match_detections(pred_boxes, pred_scores_arr, gt_all)
        acc["all"]["scores"].extend(pred_scores_arr.tolist())
        acc["all"]["tp"].extend(tp.tolist())
        acc["all"]["fp"].extend(fp.tolist())

        # Per-difficulty — match predictions against only that subset's GT
        # so that TP/FP are correct for the difficulty-specific AP curve.
        for d in DIFFS:
            tp_d, fp_d = match_detections(pred_boxes, pred_scores_arr,
                                          gt_by_diff[d])
            acc[d]["scores"].extend(pred_scores_arr.tolist())
            acc[d]["tp"].extend(tp_d.tolist())
            acc[d]["fp"].extend(fp_d.tolist())

    # ── Compute metrics ───────────────────────────────────────────────────
    def _metrics(key):
        a = acc[key]
        return compute_metrics(a["scores"], a["tp"], a["fp"], a["gt"])

    # Overall results use the requested difficulty filter
    if difficulty in DIFFS:
        results = _metrics(difficulty)
        results["_eval_difficulty"] = difficulty
    else:
        results = _metrics("all")
        results["_eval_difficulty"] = "all"

    results["num_images"] = num_images
    results["errors"] = errors
    results["per_difficulty"] = {d: _metrics(d) for d in DIFFS
                                 if acc[d]["gt"] > 0}
    return results


def print_results(model_name: str, results: dict) -> None:
    """Pretty-print evaluation results."""
    diff_label = results.get("_eval_difficulty", "all").upper()
    print(f"\n  {model_name} Results  [overall GT filter: {diff_label}]:")
    print(f"    Images evaluated : {results['num_images']}")
    print(f"    GT faces         : {results['num_gt']}")
    print(f"    Predictions      : {results['num_pred']}")
    print(f"    TP / FP          : {results['tp_total']} / {results['fp_total']}")
    print(f"    Precision        : {results['precision'] * 100:.2f}%")
    print(f"    Recall           : {results['recall'] * 100:.2f}%")
    print(f"    F1               : {results['f1'] * 100:.2f}%")
    print(f"    AP @ IoU=0.5     : {results['ap'] * 100:.2f}%")
    if results.get("per_difficulty"):
        print(f"    Per-difficulty breakdown:")
        for diff in ["easy", "medium", "hard"]:
            if diff in results["per_difficulty"]:
                r = results["per_difficulty"][diff]
                print(f"      {diff:>6s}: P={r['precision'] * 100:.2f}%  "
                      f"R={r['recall'] * 100:.2f}%  F1={r['f1'] * 100:.2f}%  "
                      f"AP={r['ap'] * 100:.2f}%  GT={r['num_gt']}")


def print_comparison_table(all_results: dict) -> None:
    """Print a comparison table of all model results."""
    # Determine the difficulty label from the first result
    first = next(iter(all_results.values()))
    diff_label = first.get("_eval_difficulty", "all").upper()

    print(f"\n{'=' * 70}")
    print(f"  Comparison Summary  [overall GT filter: {diff_label}]")
    print(f"{'=' * 70}")
    print(f"  {'Model':15s} {'Precision':>10s} {'Recall':>10s} "
          f"{'F1':>10s} {'AP@0.5':>10s} {'#Imgs':>6s}")
    print(f"  {'─' * 15} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 6}")
    for key, res in all_results.items():
        print(f"  {key:15s} {res['precision'] * 100:9.2f}% {res['recall'] * 100:9.2f}% "
              f"{res['f1'] * 100:9.2f}% {res['ap'] * 100:9.2f}% {res['num_images']:6d}")
    print(f"{'=' * 70}")
