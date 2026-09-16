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
# NanoDet-Plus model is from the NanoDet repository:
#   https://github.com/RangiLyu/nanodet
# Licensed under the Apache License, Version 2.0.
"""
NanoDet-Plus-m ONNX / TFLite Inference
========================================
Run object detection on a single image using the NanoDet-Plus-m model.
Supports both ONNX (.onnx) and TFLite (.tflite) model formats.

The model input size is auto-detected from the model file (default 320x320).
Preprocessing uses warp resize (direct resize, no letterbox) and
mean/std normalization (BGR).

Usage:
    python inference.py --image sample.jpg
    python inference.py --image sample.jpg --model model/nanodet-plus-m_320.onnx
    python inference.py --image sample.jpg --model model/nanodet-plus-m_320_FP32.tflite
    python inference.py --image sample.jpg --model model/nanodet-plus-m_320_INT8.tflite
    python inference.py --image sample.jpg --score 0.35 --nms 0.5
    python inference.py --image sample.jpg --output result.jpg --verbose
"""

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "nanodet-plus-m_320.onnx"

# NanoDet-Plus-m strides (4 FPN levels)
STRIDES = (8, 16, 32, 64)

# GFL reg_max
REG_MAX = 7

# Normalization (BGR order)
MEAN = np.array([103.53, 116.28, 123.675], dtype=np.float32)
STD = np.array([57.375, 57.12, 58.395], dtype=np.float32)

NUM_CLASSES = 80

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


# ──────────────────────────────────────────────────────────────
# Grid / center prior generation
# ──────────────────────────────────────────────────────────────
def generate_center_priors(input_h, input_w, strides=STRIDES):
    """Generate center priors (x, y, stride) for all FPN levels.

    Returns (centers, strides_arr) each of shape (N, 1) or (N, 2).
    centers: (N, 2) — [x, y] pixel coordinates of each anchor
    strides_arr: (N, 1) — stride value for each anchor
    """
    all_centers = []
    all_strides = []
    for s in strides:
        feat_h, feat_w = input_h // s, input_w // s
        yy, xx = np.meshgrid(
            np.arange(feat_h).astype(np.float32) * s,
            np.arange(feat_w).astype(np.float32) * s,
            indexing="ij",
        )
        grid = np.stack([xx.ravel(), yy.ravel()], axis=1)  # (H*W, 2)
        all_centers.append(grid)
        all_strides.append(np.full((feat_h * feat_w, 1), s, dtype=np.float32))
    return np.concatenate(all_centers, axis=0), np.concatenate(all_strides, axis=0)


# ──────────────────────────────────────────────────────────────
# Pre-processing (warp resize + normalize)
# ──────────────────────────────────────────────────────────────
def warp_resize(img, target_w, target_h):
    """Resize image directly to (target_w, target_h) — no letterbox.

    Returns (resized_img, scale_w, scale_h) where scale_w = target_w/orig_w.
    """
    ih, iw = img.shape[:2]
    resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    scale_w = target_w / iw
    scale_h = target_h / ih
    return resized, scale_w, scale_h


def preprocess_nchw(img, target_w, target_h):
    """Full preprocessing for ONNX (NCHW): warp resize + normalize + HWC→CHW.

    Returns (blob, scale_w, scale_h) where blob is (1, 3, H, W) float32.
    """
    resized, scale_w, scale_h = warp_resize(img, target_w, target_h)
    blob = resized.astype(np.float32)
    blob = (blob - MEAN) / STD
    blob = blob.transpose(2, 0, 1)[np.newaxis, ...]  # (1, 3, H, W)
    return blob, scale_w, scale_h


def preprocess_nhwc(img, target_w, target_h):
    """Full preprocessing for TFLite (NHWC): warp resize + normalize.

    Returns (blob, scale_w, scale_h) where blob is (1, H, W, 3) float32.
    """
    resized, scale_w, scale_h = warp_resize(img, target_w, target_h)
    blob = resized.astype(np.float32)
    blob = (blob - MEAN) / STD
    blob = blob[np.newaxis, ...]  # (1, H, W, 3)
    return blob, scale_w, scale_h


# ──────────────────────────────────────────────────────────────
# Decoding & post-processing
# ──────────────────────────────────────────────────────────────
def _softmax(x, axis=-1):
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def distribution_project(reg_preds, reg_max=REG_MAX):
    """Convert GFL distribution predictions to distances.

    reg_preds: (N, 4*(reg_max+1)) → (N, 4) distances
    """
    n = reg_preds.shape[0]
    reg_max_p1 = reg_max + 1
    reg = reg_preds.reshape(n, 4, reg_max_p1)  # (N, 4, 8)
    reg = _softmax(reg, axis=2)  # softmax over bins
    project = np.arange(reg_max_p1, dtype=np.float32)  # [0, 1, ..., 7]
    distances = (reg * project).sum(axis=2)  # (N, 4) — [left, top, right, bottom]
    return distances


def decode_nanodet_output(raw_output, centers, strides_arr, reg_max=REG_MAX):
    """Decode raw NanoDet-Plus output.

    raw_output: (1, N, 112) where first 80 = sigmoided cls, last 32 = raw reg
    Returns: (N, 7) — [x1, y1, x2, y2, score, cls_conf, cls_id]
    """
    preds = raw_output[0]  # (N, 112)
    num_classes = preds.shape[1] - 4 * (reg_max + 1)

    cls_scores = preds[:, :num_classes]  # already sigmoided from ONNX export
    reg_preds = preds[:, num_classes:]   # raw reg predictions

    # Decode boxes
    distances = distribution_project(reg_preds, reg_max)  # (N, 4)
    distances = distances * strides_arr  # scale by stride

    # distance2bbox
    x1 = centers[:, 0] - distances[:, 0]
    y1 = centers[:, 1] - distances[:, 1]
    x2 = centers[:, 0] + distances[:, 2]
    y2 = centers[:, 1] + distances[:, 3]

    # Best class per anchor
    cls_id = cls_scores.argmax(axis=1)
    cls_conf = cls_scores[np.arange(len(cls_id)), cls_id]

    # NanoDet-Plus has no separate objectness — score == cls_conf. Column 4
    # (score) and column 5 (cls_conf) are intentionally duplicated so the
    # detection tensor layout stays [x1, y1, x2, y2, score, cls_conf, cls_id],
    # matching the schema used by the other detectors in this repo.
    return np.stack(
        [x1, y1, x2, y2, cls_conf, cls_conf, cls_id.astype(np.float32)], axis=1
    )


def nms_boxes(detections, iou_thresh=0.6):
    """Standard NMS on detections array (N, 7)."""
    if len(detections) == 0:
        return detections

    x1, y1, x2, y2 = detections[:, 0], detections[:, 1], detections[:, 2], detections[:, 3]
    scores = detections[:, 4]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        idx = order[0]
        keep.append(idx)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[idx], x1[order[1:]])
        yy1 = np.maximum(y1[idx], y1[order[1:]])
        xx2 = np.minimum(x2[idx], x2[order[1:]])
        yy2 = np.minimum(y2[idx], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[idx] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_thresh]

    return detections[keep]


def postprocess(detections, scale_w, scale_h, score_thresh=0.35, nms_thresh=0.6,
                max_det=100):
    """Filter by score, rescale to original image coords, apply per-class NMS."""
    mask = detections[:, 4] > score_thresh
    dets = detections[mask]

    if len(dets) == 0:
        return np.empty((0, 7))

    # Rescale boxes from model input coords to original image coords
    dets[:, 0] = dets[:, 0] / scale_w
    dets[:, 1] = dets[:, 1] / scale_h
    dets[:, 2] = dets[:, 2] / scale_w
    dets[:, 3] = dets[:, 3] / scale_h

    unique_cls = np.unique(dets[:, 6].astype(int))
    final = []
    for c in unique_cls:
        cls_mask = dets[:, 6].astype(int) == c
        cls_dets = nms_boxes(dets[cls_mask], nms_thresh)
        final.append(cls_dets)

    if not final:
        return np.empty((0, 7))

    result = np.concatenate(final, axis=0)

    # Keep top max_det
    if len(result) > max_det:
        order = result[:, 4].argsort()[::-1][:max_det]
        result = result[order]

    return result


# ──────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────
def draw_results(image, results, class_names):
    """Overlay bounding boxes and labels onto the image."""
    palette = np.random.RandomState(42).randint(0, 256, size=(len(class_names), 3)).tolist()

    for det in results:
        x1, y1, x2, y2, score, _, cls_id = det
        cls_id = int(cls_id)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        color = tuple(palette[cls_id % len(palette)])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label_text = "{} {:.2f}".format(class_names[cls_id], score)
        txt_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(
            image, (x1, y1 - txt_size[1] - 4), (x1 + txt_size[0], y1), color, -1
        )
        cv2.putText(
            image, label_text, (x1, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )
    return image


# ──────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────
def load_model(model_path):
    """Load ONNX or TFLite model.

    Returns (session_or_interpreter, input_name, input_h, input_w, model_type).
    model_type is 'onnx' or 'tflite'.
    """
    model_path = str(model_path)
    if model_path.endswith(".tflite"):
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            import tensorflow.lite as tflite

        interpreter = tflite.Interpreter(model_path=model_path)
        try:
            interpreter.allocate_tensors()
        except RuntimeError:
            import tensorflow as tf
            interpreter = tf.lite.Interpreter(
                model_path=model_path, num_threads=1,
                experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
            )
            interpreter.allocate_tensors()
        inp = interpreter.get_input_details()[0]
        input_shape = inp["shape"]
        # NHWC: (1, H, W, 3)
        if len(input_shape) == 4 and input_shape[3] == 3:
            input_h, input_w = int(input_shape[1]), int(input_shape[2])
        else:
            input_h, input_w = int(input_shape[2]), int(input_shape[3])
        return interpreter, None, input_h, input_w, "tflite"
    else:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        session = ort.InferenceSession(model_path, providers=providers)
        inp = session.get_inputs()[0]
        # ONNX shape is (N, C, H, W); dims may be str/None for dynamic axes.
        shape = inp.shape
        try:
            input_h = int(shape[2])
            input_w = int(shape[3])
        except (TypeError, ValueError):
            input_h, input_w = 320, 320
        return session, inp.name, input_h, input_w, "onnx"


# ──────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────
def run_inference(session, input_name, input_h, input_w, image_path,
                  score_thresh=0.35, nms_thresh=0.6, verbose=False,
                  model_type="onnx"):
    """Run detection on a single image. Supports ONNX and TFLite models."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError("Cannot read image: {}".format(image_path))

    centers, strides_arr = generate_center_priors(input_h, input_w)

    t0 = time.perf_counter()

    if model_type == "tflite":
        interpreter = session
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        input_dtype = input_details["dtype"]

        # NHWC preprocessing
        blob_fp32, scale_w, scale_h = preprocess_nhwc(img, input_w, input_h)

        if input_dtype == np.int8:
            input_quant = input_details.get("quantization_parameters", {})
            input_scale = input_quant.get("scales", np.array([1.0]))[0]
            input_zp = input_quant.get("zero_points", np.array([0]))[0]
            blob_input = np.clip(
                np.round(blob_fp32 / input_scale + input_zp), -128, 127
            ).astype(np.int8)
        elif input_dtype == np.uint8:
            input_quant = input_details.get("quantization_parameters", {})
            input_scale = input_quant.get("scales", np.array([1.0]))[0]
            input_zp = input_quant.get("zero_points", np.array([0]))[0]
            blob_input = np.clip(
                np.round(blob_fp32 / input_scale + input_zp), 0, 255
            ).astype(np.uint8)
        else:
            blob_input = blob_fp32

        if verbose:
            print("  Input shape : {}".format(blob_input.shape))
            print("  Input dtype : {}".format(blob_input.dtype))

        interpreter.set_tensor(input_details["index"], blob_input)
        interpreter.invoke()
        raw = interpreter.get_tensor(output_details["index"])

        # Dequantize output if needed
        if raw.dtype != np.float32:
            output_quant = output_details.get("quantization_parameters", {})
            output_scale = output_quant.get("scales", np.array([1.0]))[0]
            output_zp = output_quant.get("zero_points", np.array([0]))[0]
            raw = (raw.astype(np.float32) - output_zp) * output_scale

        if raw.ndim == 2:
            raw = raw[np.newaxis, ...]

    else:
        # ONNX path: NCHW
        blob, scale_w, scale_h = preprocess_nchw(img, input_w, input_h)

        if verbose:
            print("  Input shape : {}".format(blob.shape))

        raw = session.run(None, {input_name: blob})[0]

    latency_ms = (time.perf_counter() - t0) * 1000

    dets = decode_nanodet_output(raw, centers, strides_arr)
    results = postprocess(dets, scale_w, scale_h, score_thresh, nms_thresh)

    return results, img, latency_ms


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="NanoDet-Plus-m Inference -- single image object detection"
    )
    ap.add_argument(
        "-m", "--model", default=str(MODEL_PATH),
        help="Path to ONNX or TFLite model (default: model/nanodet-plus-m_320.onnx).",
    )
    ap.add_argument(
        "-i", "--image", required=True,
        help="Input image path.",
    )
    ap.add_argument(
        "-o", "--output", default="outputs",
        help="Output image path (default: outputs/).",
    )
    ap.add_argument(
        "--score", type=float, default=0.35,
        help="Confidence threshold (default: 0.35).",
    )
    ap.add_argument(
        "--nms", type=float, default=0.6,
        help="NMS IoU threshold (default: 0.6).",
    )
    ap.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show model I/O details.",
    )
    args = ap.parse_args()

    session, input_name, input_h, input_w, model_type = load_model(args.model)
    print("Model   : {}".format(args.model))
    print("Type    : {}".format(model_type.upper()))
    print("Input   : {} x {} (auto-detected)".format(input_w, input_h))

    if args.verbose:
        if model_type == "onnx":
            out = session.get_outputs()[0]
            print("Output  : name={}, shape={}, dtype={}".format(out.name, out.shape, out.type))
        else:
            out_details = session.get_output_details()[0]
            print("Output  : shape={}, dtype={}".format(
                list(out_details["shape"]), out_details["dtype"].__name__))

    results, img, latency_ms = run_inference(
        session, input_name, input_h, input_w,
        args.image, args.score, args.nms, args.verbose, model_type,
    )

    print("Latency : {:.1f} ms".format(latency_ms))
    print("Detected: {} object(s)".format(len(results)))
    for r in results:
        cname = COCO_CLASSES[int(r[6])]
        print("  -> {:15s}  score={:.3f}  box=[{:.0f},{:.0f},{:.0f},{:.0f}]".format(
            cname, r[4], r[0], r[1], r[2], r[3]
        ))

    out_path = Path(args.output)
    if out_path.is_dir() or not out_path.suffix:
        out_path.mkdir(parents=True, exist_ok=True)
        out_path = out_path / "{}_result.jpg".format(Path(args.image).stem)
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    vis = draw_results(img.copy(), results, COCO_CLASSES)
    cv2.imwrite(str(out_path), vis)
    print("Saved   : {}".format(out_path))


if __name__ == "__main__":
    main()
