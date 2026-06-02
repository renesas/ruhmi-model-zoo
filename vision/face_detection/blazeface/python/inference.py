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
# BlazeFace model from Google MediaPipe, PINTO model zoo conversion:
#   https://github.com/PINTO0309/PINTO_model_zoo/tree/main/030_BlazeFace
# Licensed under the Apache License, Version 2.0.
"""
PINTOBlazeFace — TFLite Inference (Front Face)
===============================================
Runs BlazeFace front-face inference using the 4-output split TFLite models
produced by download_model.py (128×128 input).

Model output layout (4 separate tensors):
    boxes_s8   (1, 512, 16)  — stride-8  box regressors
    scores_s8  (1, 512,  1)  — stride-8  face scores
    scores_s16 (1, 384,  1)  — stride-16 face scores
    boxes_s16  (1, 384, 16)  — stride-16 box regressors

FP32 model: float32 I/O
INT8 model: int8 I/O, each output dequantized with its own scale/zp

Usage:
    python3 inference.py --image path/to/image.jpg
    python3 inference.py --image path/to/image.jpg --model model/blazeface_front_int8.tflite
    python3 inference.py --image path/to/image.jpg --thresh 0.6 --verbose
"""

import argparse
import os
import sys
import time
from typing import List, Tuple

import cv2
import numpy as np
import tensorflow as tf

# ──────────────────────────────────────────────────────────────────────────────
# Paths & constants
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_FP32  = os.path.join(BASE_DIR, "model", "blazeface_front_fp32.tflite")
MODEL_INT8  = os.path.join(BASE_DIR, "model", "blazeface_front_int8.tflite")
ANCHORS_PATH = os.path.join(BASE_DIR, "model", "anchors.npy")
OUTPUT_DIR   = os.path.join(BASE_DIR, "output")

INPUT_SIZE  = 128   # front model: 128×128

# Detection thresholds (mirror MediaPipe defaults)
MIN_SCORE_THRESH      = 0.5
MIN_SUPPRESSION_THRESH = 0.3

# Anchor decode scale (same for both axes, both models)
_XY_SCALE = 128.0
_WH_SCALE = 128.0
NUM_KEYPOINTS = 6


# ──────────────────────────────────────────────────────────────────────────────
# 1.  PRE-PROCESSING
# ──────────────────────────────────────────────────────────────────────────────
def preprocess(image_bgr: np.ndarray,
               size: int = INPUT_SIZE) -> np.ndarray:
    """Resize a BGR uint8 image and normalise to float32 [0, 1].

    Parameters
    ----------
    image_bgr : H×W×3 uint8 BGR image (any resolution).
    size      : Target square side in pixels (default: 128).

    Returns
    -------
    np.ndarray  shape (1, size, size, 3)  dtype float32, values in [0, 1].
    """
    img = cv2.resize(image_bgr, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img /= 255.0
    return img[np.newaxis, ...]   # (1, 128, 128, 3)


def quantize_input(img_fp32: np.ndarray,
                   scale: float,
                   zero_point: int) -> np.ndarray:
    """Quantize a float32 NHWC tensor to int8 for INT8 model input.

    Parameters
    ----------
    img_fp32   : (1, H, W, 3) float32 in [0, 1].
    scale      : Input quantization scale from the TFLite model.
    zero_point : Input zero-point from the TFLite model.

    Returns
    -------
    np.ndarray  same shape, dtype int8.
    """
    q = img_fp32 / scale + zero_point
    return np.clip(np.round(q), -128, 127).astype(np.int8)


# ──────────────────────────────────────────────────────────────────────────────
# 2.  MODEL LOADING
# ──────────────────────────────────────────────────────────────────────────────
def load_model(model_path: str) -> tf.lite.Interpreter:
    """Load a TFLite model and allocate tensors.

    Parameters
    ----------
    model_path : Path to the .tflite file.

    Returns
    -------
    tf.lite.Interpreter with tensors already allocated.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    return interp


def load_anchors(anchors_path: str) -> np.ndarray:
    """Load the (896, 4) BlazeFace front-face anchor array.

    Parameters
    ----------
    anchors_path : Path to anchors.npy produced by download_model.py.

    Returns
    -------
    np.ndarray  shape (896, 4) float32  [cx, cy, 1, 1].
    """
    anchors = np.load(anchors_path).astype(np.float32)
    assert anchors.shape == (896, 4), \
        f"Expected anchors (896,4), got {anchors.shape}"
    return anchors


# ──────────────────────────────────────────────────────────────────────────────
# 3.  INFERENCE (raw outputs)
# ──────────────────────────────────────────────────────────────────────────────
def run_interpreter(interp: tf.lite.Interpreter,
                    img_fp32: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Run one forward pass and return raw boxes + scores as float32.

    Handles both FP32 and INT8 models transparently:
      - FP32: input fed as-is, outputs read as float32.
      - INT8: input quantized to int8, outputs dequantized via each tensor's
              own scale/zero_point before being returned.

    Parameters
    ----------
    interp   : Allocated tf.lite.Interpreter.
    img_fp32 : (1, 128, 128, 3) float32 image in [0, 1].

    Returns
    -------
    raw_boxes  : (896, 16) float32 — concatenated stride-8 + stride-16 boxes.
    raw_scores : (896,  1) float32 — concatenated stride-8 + stride-16 scores.
    """
    inp_detail  = interp.get_input_details()[0]
    out_details = interp.get_output_details()
    is_int8     = inp_detail["dtype"] == np.int8

    # ── Feed input ────────────────────────────────────────────────────────────
    if is_int8:
        qp = inp_detail["quantization_parameters"]
        inp = quantize_input(img_fp32,
                             float(qp["scales"][0]),
                             int(qp["zero_points"][0]))
    else:
        inp = img_fp32

    interp.set_tensor(inp_detail["index"], inp)
    interp.invoke()

    # ── Read & dequantize outputs ─────────────────────────────────────────────
    def _fetch(od: dict) -> np.ndarray:
        t = interp.get_tensor(od["index"])
        if is_int8:
            qp = od["quantization_parameters"]
            t = (t.astype(np.float32) - float(qp["zero_points"][0])) \
                * float(qp["scales"][0])
        return t.astype(np.float32)

    # Group outputs: shape[-1]==1 → scores, shape[-1]==16 → boxes
    # Sort each group descending by anchor count (512 first, then 384)
    score_outs = sorted([o for o in out_details if o["shape"][-1] == 1],
                        key=lambda o: o["shape"][1], reverse=True)
    box_outs   = sorted([o for o in out_details if o["shape"][-1] == 16],
                        key=lambda o: o["shape"][1], reverse=True)

    scores_s8  = _fetch(score_outs[0]).reshape(512,  1)
    scores_s16 = _fetch(score_outs[1]).reshape(384,  1)
    boxes_s8   = _fetch(box_outs[0]).reshape(512, 16)
    boxes_s16  = _fetch(box_outs[1]).reshape(384, 16)

    raw_scores = np.concatenate([scores_s8,  scores_s16], axis=0)  # (896, 1)
    raw_boxes  = np.concatenate([boxes_s8,   boxes_s16],  axis=0)  # (896, 16)
    return raw_boxes, raw_scores


# ──────────────────────────────────────────────────────────────────────────────
# 4.  POST-PROCESSING
# ──────────────────────────────────────────────────────────────────────────────
def decode_boxes(raw_boxes: np.ndarray,
                 anchors: np.ndarray) -> np.ndarray:
    """Decode raw regressor output into normalised [ymin, xmin, ymax, xmax, kps…].

    Parameters
    ----------
    raw_boxes : (N, 16) raw box regressors from the model.
    anchors   : (N, 4)  [cx, cy, w, h]  (w=h=1 for BlazeFace).

    Returns
    -------
    boxes : (N, 16) normalised [0, 1] coords.
    """
    boxes = np.zeros_like(raw_boxes)

    cx = raw_boxes[:, 0] / _XY_SCALE * anchors[:, 2] + anchors[:, 0]
    cy = raw_boxes[:, 1] / _XY_SCALE * anchors[:, 3] + anchors[:, 1]
    w  = raw_boxes[:, 2] / _WH_SCALE * anchors[:, 2]
    h  = raw_boxes[:, 3] / _WH_SCALE * anchors[:, 3]

    boxes[:, 0] = cy - h / 2.0   # ymin
    boxes[:, 1] = cx - w / 2.0   # xmin
    boxes[:, 2] = cy + h / 2.0   # ymax
    boxes[:, 3] = cx + w / 2.0   # xmax

    for k in range(NUM_KEYPOINTS):
        off = 4 + k * 2
        boxes[:, off]     = raw_boxes[:, off]     / _XY_SCALE * anchors[:, 2] + anchors[:, 0]
        boxes[:, off + 1] = raw_boxes[:, off + 1] / _XY_SCALE * anchors[:, 3] + anchors[:, 1]

    return boxes


def tensors_to_detections(raw_boxes: np.ndarray,
                           raw_scores: np.ndarray,
                           anchors: np.ndarray,
                           score_thresh: float = MIN_SCORE_THRESH) -> np.ndarray:
    """Apply sigmoid, threshold, and decode boxes into (M, 17) detections.

    Parameters
    ----------
    raw_boxes    : (896, 16) float32.
    raw_scores   : (896,  1) float32.
    anchors      : (896,  4) float32.
    score_thresh : Minimum face score after sigmoid.

    Returns
    -------
    detections : (M, 17) float32 — [ymin, xmin, ymax, xmax, kp0x…kp5y, score].
                 Empty array (0, 17) if no faces pass the threshold.
    """
    scores = raw_scores.flatten()
    scores = np.clip(scores, -88.0, 88.0)     # avoid float32 overflow in exp
    scores = 1.0 / (1.0 + np.exp(-scores))    # sigmoid

    mask = scores >= score_thresh
    if mask.sum() == 0:
        return np.empty((0, 17), dtype=np.float32)

    decoded = decode_boxes(raw_boxes[mask], anchors[mask])
    return np.concatenate([decoded, scores[mask, np.newaxis]], axis=1)


def _iou(box_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """IoU between box_a (4,) and every row of boxes_b (M, 4).
    Boxes are in [ymin, xmin, ymax, xmax] format.
    """
    ymin = np.maximum(box_a[0], boxes_b[:, 0])
    xmin = np.maximum(box_a[1], boxes_b[:, 1])
    ymax = np.minimum(box_a[2], boxes_b[:, 2])
    xmax = np.minimum(box_a[3], boxes_b[:, 3])

    inter = np.maximum(0, ymax - ymin) * np.maximum(0, xmax - xmin)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    return inter / (area_a + area_b - inter + 1e-6)


def weighted_nms(detections: np.ndarray,
                 nms_thresh: float = MIN_SUPPRESSION_THRESH) -> np.ndarray:
    """BlazeFace weighted NMS — averages overlapping boxes by score weight.

    Parameters
    ----------
    detections : (M, 17) float32, score in column 16.
    nms_thresh : IoU threshold for grouping boxes (default: 0.3).

    Returns
    -------
    np.ndarray : (K, 17) float32, K ≤ M.
    """
    if len(detections) == 0:
        return np.empty((0, 17), dtype=np.float32)

    output    = []
    remaining = np.argsort(-detections[:, 16])

    while len(remaining) > 0:
        det     = detections[remaining[0]].copy()
        ious    = _iou(det[:4], detections[remaining, :4])
        overlap = remaining[ious > nms_thresh]
        remaining = remaining[ious <= nms_thresh]

        if len(overlap) > 1:
            coords      = detections[overlap, :16]
            scores      = detections[overlap, 16:17]
            total_score = scores.sum()
            det[:16]    = (coords * scores).sum(axis=0) / total_score
            det[16]     = total_score / len(overlap)

        output.append(det)

    return np.array(output, dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# 5.  HIGH-LEVEL RUNNER
# ──────────────────────────────────────────────────────────────────────────────
def detect_faces(interp: tf.lite.Interpreter,
                 anchors: np.ndarray,
                 image_bgr: np.ndarray,
                 score_thresh: float = MIN_SCORE_THRESH,
                 nms_thresh: float   = MIN_SUPPRESSION_THRESH) -> np.ndarray:
    """Full pipeline: preprocess → infer → postprocess → NMS.

    Parameters
    ----------
    interp      : Loaded and allocated tf.lite.Interpreter.
    anchors     : (896, 4) float32 anchor array.
    image_bgr   : H×W×3 uint8 BGR image (any resolution).
    score_thresh: Minimum face score threshold.
    nms_thresh  : IoU threshold for weighted NMS.

    Returns
    -------
    np.ndarray : (M, 17) float32 detections.
                 Each row: [ymin, xmin, ymax, xmax, kp0x…kp5y, score]
                 in normalised [0, 1] coordinates.
                 Returns shape (0, 17) when no faces are found.
    """
    img   = preprocess(image_bgr)
    boxes, scores = run_interpreter(interp, img)
    dets  = tensors_to_detections(boxes, scores, anchors, score_thresh)
    return weighted_nms(dets, nms_thresh)


# ──────────────────────────────────────────────────────────────────────────────
# 6.  VISUALISATION
# ──────────────────────────────────────────────────────────────────────────────
def draw_detections(image_bgr: np.ndarray,
                    detections: np.ndarray,
                    latency_ms: float,
                    is_int8: bool) -> np.ndarray:
    """Draw bounding boxes, keypoints and a HUD label onto a copy of the image.

    Parameters
    ----------
    image_bgr  : Original BGR image.
    detections : (M, 17) float32 from detect_faces().
    latency_ms : Inference latency in milliseconds (for HUD text).
    is_int8    : Whether the INT8 model was used (affects HUD label).

    Returns
    -------
    np.ndarray : Annotated BGR image (copy of input).
    """
    vis = image_bgr.copy()
    h, w = vis.shape[:2]

    for det in detections:
        ymin, xmin, ymax, xmax = det[:4]
        score = det[16]

        x1, y1 = int(xmin * w), int(ymin * h)
        x2, y2 = int(xmax * w), int(ymax * h)

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, f"{score:.2f}", (x1, max(y1 - 6, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)

        for k in range(NUM_KEYPOINTS):
            kx = int(det[4 + k * 2]     * w)
            ky = int(det[4 + k * 2 + 1] * h)
            cv2.circle(vis, (kx, ky), 3, (255, 0, 0), -1)

    dtype_tag = "INT8" if is_int8 else "FP32"
    label = f"BlazeFace PINTO {dtype_tag} | {len(detections)} face(s) | {latency_ms:.1f}ms"
    cv2.putText(vis, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 140, 255), 1, cv2.LINE_AA)

    return vis


# ──────────────────────────────────────────────────────────────────────────────
# 7.  MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="BlazeFace PINTO TFLite front-face inference on a single image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image",   required=True,
                        help="Path to input image (JPEG or PNG).")
    parser.add_argument("--model",   default=MODEL_FP32,
                        help="TFLite model path.")
    parser.add_argument("--anchors", default=ANCHORS_PATH,
                        help="anchors.npy path.")
    parser.add_argument("--thresh",  type=float, default=MIN_SCORE_THRESH,
                        help="Minimum face score threshold.")
    parser.add_argument("--nms",     type=float, default=MIN_SUPPRESSION_THRESH,
                        help="NMS IoU threshold.")
    parser.add_argument("--output",  default=None,
                        help="Save annotated image to this path.")
    parser.add_argument("--display", action="store_true",
                        help="Open a display window.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-detection details.")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        sys.exit(f"[ERROR] Image not found: {args.image}")
    if not os.path.isfile(args.model):
        sys.exit(
            f"[ERROR] Model not found: {args.model}\n"
            "Run  python3 download_model.py  first."
        )
    if not os.path.isfile(args.anchors):
        sys.exit(
            f"[ERROR] Anchors not found: {args.anchors}\n"
            "Run  python3 download_model.py  first."
        )

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"[INFO] Model  : {args.model}")
    interp  = load_model(args.model)
    anchors = load_anchors(args.anchors)
    is_int8 = interp.get_input_details()[0]["dtype"] == np.int8
    print(f"[INFO] Type   : {'INT8' if is_int8 else 'FP32'}")

    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"[ERROR] Could not read image: {args.image}")

    h, w = img.shape[:2]
    print(f"[INFO] Image  : {args.image}  ({w}×{h})")

    # ── Warm-up + timed run ───────────────────────────────────────────────────
    detect_faces(interp, anchors, img, args.thresh, args.nms)   # warm-up

    t0   = time.perf_counter()
    dets = detect_faces(interp, anchors, img, args.thresh, args.nms)
    ms   = (time.perf_counter() - t0) * 1e3

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"[INFO] Faces  : {len(dets)}  |  latency: {ms:.2f} ms")

    if args.verbose and len(dets):
        print(f"\n  {'#':<4} {'ymin':>6} {'xmin':>6} {'ymax':>6} {'xmax':>6} {'score':>7}")
        print("  " + "-" * 40)
        for i, d in enumerate(dets):
            print(f"  {i:<4} {d[0]:6.3f} {d[1]:6.3f} {d[2]:6.3f} {d[3]:6.3f} {d[16]:7.4f}")

    # ── Annotate & output ─────────────────────────────────────────────────────
    vis = draw_detections(img, dets, ms, is_int8)

    if args.output:
        cv2.imwrite(args.output, vis)
        print(f"[INFO] Saved  : {args.output}")

    if args.display:
        cv2.imshow("BlazeFace PINTO TFLite (Front)", vis)
        print("[INFO] Press any key to close ...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "output.jpg"), vis)
        print(f"[INFO] Display disabled, saved annotated image as {os.path.join(OUTPUT_DIR, 'output.jpg')}")


if __name__ == "__main__":
    main()
