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
# YOLOX-Tiny model is from the Megvii YOLOX repository:
#   https://github.com/Megvii-BaseDetection/YOLOX
# Licensed under the Apache License, Version 2.0.
"""
YOLOX-Tiny ONNX / TFLite Inference
====================================
Run object detection on a single image using the YOLOX-Tiny model.
Supports both ONNX (.onnx) and TFLite (.tflite) model formats.

The model input size is auto-detected from the model file (default 416x416).
Preprocessing uses letterbox resize with gray (114) padding and 0-255 float
range (YOLOX convention -- no /255 normalization).

Usage:
    python inference.py --image sample.jpg
    python inference.py --image sample.jpg --model model/yolox_tiny.onnx
    python inference.py --image sample.jpg --model model/yolox_tiny_FP32.tflite
    python inference.py --image sample.jpg --model model/yolox_tiny_INT8.tflite
    python inference.py --image sample.jpg --score 0.25 --nms 0.5
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

MODEL_PATH   = BASE_DIR / "model" / "yolox_tiny.onnx"
PAD_VALUE    = 114
STRIDES      = (8, 16, 32)

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
# Grid generation (anchor-free decoding)
# ──────────────────────────────────────────────────────────────
def generate_yolox_grid(input_h, input_w, strides=STRIDES):
    """Build grid offsets and stride tensors for all YOLOX FPN levels."""
    all_grids, all_strides = [], []
    for s in strides:
        feat_h, feat_w = input_h // s, input_w // s
        yy, xx = np.meshgrid(np.arange(feat_h), np.arange(feat_w), indexing="ij")
        grid = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
        all_grids.append(grid)
        all_strides.append(np.full((feat_h * feat_w, 1), s, dtype=np.float32))
    return np.concatenate(all_grids, axis=0), np.concatenate(all_strides, axis=0)


# ──────────────────────────────────────────────────────────────
# Pre-processing
# ──────────────────────────────────────────────────────────────
def letterbox_resize(img, target_w, target_h):
    """Resize preserving aspect ratio, pad with gray (114)."""
    ih, iw = img.shape[:2]
    scale = min(target_w / iw, target_h / ih)
    new_w, new_h = int(iw * scale), int(ih * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    padded = np.full((target_h, target_w, 3), PAD_VALUE, dtype=np.uint8)
    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2
    padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
    return padded, scale, (pad_w, pad_h)


# ──────────────────────────────────────────────────────────────
# Decoding & post-processing
# ──────────────────────────────────────────────────────────────
def decode_yolox_output(raw_output, grids, expanded_strides):
    """Decode raw YOLOX output into [x1, y1, x2, y2, score, cls_conf, cls_id]."""
    preds = raw_output[0].copy()

    preds[:, :2] = (preds[:, :2] + grids) * expanded_strides
    preds[:, 2:4] = np.exp(preds[:, 2:4]) * expanded_strides

    cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    obj_conf = preds[:, 4:5]
    cls_scores = preds[:, 5:]
    cls_id = cls_scores.argmax(axis=1)
    cls_conf = cls_scores[np.arange(len(cls_id)), cls_id]
    scores = obj_conf.squeeze() * cls_conf

    return np.stack(
        [x1, y1, x2, y2, scores, cls_conf, cls_id.astype(np.float32)], axis=1
    )


def nms_boxes(detections, iou_thresh=0.45):
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


def postprocess(detections, scale, pad, score_thresh=0.3, nms_thresh=0.45):
    """Filter by score, undo letterbox, apply per-class NMS."""
    mask = detections[:, 4] > score_thresh
    dets = detections[mask]

    if len(dets) == 0:
        return np.empty((0, 7))

    dets[:, 0] = (dets[:, 0] - pad[0]) / scale
    dets[:, 1] = (dets[:, 1] - pad[1]) / scale
    dets[:, 2] = (dets[:, 2] - pad[0]) / scale
    dets[:, 3] = (dets[:, 3] - pad[1]) / scale

    unique_cls = np.unique(dets[:, 6].astype(int))
    final = []
    for c in unique_cls:
        cls_mask = dets[:, 6].astype(int) == c
        cls_dets = nms_boxes(dets[cls_mask], nms_thresh)
        final.append(cls_dets)

    return np.concatenate(final, axis=0) if final else np.empty((0, 7))


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
# Inference
# ──────────────────────────────────────────────────────────────
def load_model(model_path):
    """Load ONNX or TFLite model. Returns (session_or_interpreter, input_name, input_h, input_w, model_type).

    model_type is 'onnx' or 'tflite'.
    """
    model_path = str(model_path)
    if model_path.endswith(".tflite"):
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            import tensorflow.lite as tflite

        interpreter = tflite.Interpreter(model_path=model_path)
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
        input_h, input_w = int(inp.shape[2]), int(inp.shape[3])
        return session, inp.name, input_h, input_w, "onnx"


def run_inference(session, input_name, input_h, input_w, image_path,
                  score_thresh=0.3, nms_thresh=0.45, verbose=False,
                  model_type="onnx"):
    """Run detection on a single image. Supports ONNX and TFLite models."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError("Cannot read image: {}".format(image_path))

    padded, scale, pad = letterbox_resize(img, input_w, input_h)

    t0 = time.perf_counter()

    if model_type == "tflite":
        interpreter = session
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        input_dtype = input_details["dtype"]

        # NHWC blob
        blob_fp32 = padded.astype(np.float32)  # HWC, 0-255

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

        blob_input = blob_input[np.newaxis, ...]  # (1, H, W, 3)

        if verbose:
            print("  Input shape : {}".format(blob_input.shape))
            print("  Input dtype : {}".format(blob_input.dtype))
            print("  Scale       : {:.4f}".format(scale))
            print("  Padding     : {}".format(pad))

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
        # ONNX path
        blob = padded.astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...]

        if verbose:
            print("  Input shape : {}".format(blob.shape))
            print("  Scale       : {:.4f}".format(scale))
            print("  Padding     : {}".format(pad))

        raw = session.run(None, {input_name: blob})[0]

    latency_ms = (time.perf_counter() - t0) * 1000

    grids, strides = generate_yolox_grid(input_h, input_w)
    dets = decode_yolox_output(raw, grids, strides)
    results = postprocess(dets, scale, pad, score_thresh, nms_thresh)

    return results, img, latency_ms


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="YOLOX-Tiny ONNX Inference -- single image object detection"
    )
    ap.add_argument(
        "-m", "--model", default=str(MODEL_PATH),
        help="Path to ONNX model (default: model/yolox_tiny.onnx).",
    )
    ap.add_argument(
        "-i", "--image", required=True,
        help="Input image path.",
    )
    ap.add_argument(
        "-o", "--output", default=None,
        help="Save annotated image to this path.",
    )
    ap.add_argument(
        "--score", type=float, default=0.3,
        help="Confidence threshold (default: 0.3).",
    )
    ap.add_argument(
        "--nms", type=float, default=0.45,
        help="NMS IoU threshold (default: 0.45).",
    )
    ap.add_argument(
        "--display", action="store_true",
        help="Open a display window.",
    )
    ap.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print per-detection details and model I/O info.",
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

    if args.verbose and len(results):
        print("\n  {:<4} {:<15} {:>7} {:>6} {:>6} {:>6} {:>6}".format(
            "#", "class", "score", "x1", "y1", "x2", "y2"))
        print("  " + "-" * 60)
        for i, r in enumerate(results):
            cname = COCO_CLASSES[int(r[6])]
            print("  {:<4} {:<15} {:7.4f} {:6.0f} {:6.0f} {:6.0f} {:6.0f}".format(
                i, cname, r[4], r[0], r[1], r[2], r[3]))

    vis = draw_results(img.copy(), results, COCO_CLASSES)

    if args.output:
        out_path = Path(args.output)
        if out_path.is_dir() or not out_path.suffix:
            out_path.mkdir(parents=True, exist_ok=True)
            out_path = out_path / "{}_result.jpg".format(Path(args.image).stem)
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), vis)
        print("Saved   : {}".format(out_path))

    if args.display:
        cv2.imshow("YOLOX-Tiny Detection", vis)
        print("[INFO] Press any key to close ...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        output_dir = BASE_DIR / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        default_out = output_dir / "{}_result.jpg".format(Path(args.image).stem)
        cv2.imwrite(str(default_out), vis)
        print("Saved   : {}".format(default_out))


if __name__ == "__main__":
    main()
