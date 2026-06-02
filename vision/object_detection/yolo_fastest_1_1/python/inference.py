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
# Yolo-Fastest is from https://github.com/dog-qiuqiu/Yolo-Fastest
# Licensed under the MIT License.
"""
Run object detection on a single image using a Yolo-Fastest TFLite model.

Supports both FP32 and INT8 TFLite models.  The script auto-detects the
input dtype from the model metadata and applies the correct pre-processing:

  FP32 model : float32 NHWC input in [0, 1]
  INT8 model : int8 NHWC input, quantized via scale / zero_point from model

TFLite models use NHWC layout internally.  The TFLite FP32 model was
converted from the ONNX FP32 model (which is NCHW), so the TFLite outputs
are also in NHWC order: (1, grid_h, grid_w, 255).  The decode_head()
function here expects this NHWC layout.

Conversion:
    python download_model.py            # generates FP32 + INT8 TFLite

Example:
    python inference.py --image sample.jpg
    python inference.py --image sample.jpg --variant xl
    python inference.py --image sample.jpg --model model/yolo_fastest_1.1.tflite
    python inference.py --image sample.jpg --model model/yolo_fastest_1.1_xl.tflite
    python inference.py --image sample.jpg --model model/yolo_fastest_1.1_int8.tflite
    python inference.py --image sample.jpg --score 0.3 --nms 0.45
    python inference.py --image sample.jpg --output result.jpg --display
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
HERE          = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR    = os.path.join(HERE, "model")

_VARIANT_MODELS = {
    "base": os.path.join(_MODEL_DIR, "yolo_fastest_1.1.tflite"),
    "xl":   os.path.join(_MODEL_DIR, "yolo_fastest_1.1_xl.tflite"),
}
DEFAULT_MODEL = _VARIANT_MODELS["base"]
OUTPUT_DIR    = os.path.join(HERE, "output")

INPUT_W  = 320
INPUT_H  = 320
NUM_CLS  = 80

# Anchors from yolo-fastest-1.1.cfg (w, h) in input-image pixels.
# mask 0,1,2 → head1 (stride 16, 20×20 grid) — small objects
# mask 3,4,5 → head0 (stride 32, 10×10 grid) — large objects
ALL_ANCHORS  = [
    (12, 18), (37, 49), (52, 132),       # mask 0,1,2
    (115, 73), (119, 199), (242, 238),   # mask 3,4,5
]
HEAD_ANCHORS = [ALL_ANCHORS[3:6], ALL_ANCHORS[0:3]]
STRIDES      = [32, 16]

CONF_THRESH  = 0.25   # default confidence threshold
IOU_THRESH   = 0.45   # default NMS IoU threshold

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

_PALETTE = [
    (255, 56, 56),   (255, 157, 151), (255, 112, 31),  (255, 178, 29),
    (207, 210, 49),  (72,  249, 10),  (146, 204, 23),  (61,  219, 134),
    (26,  147, 52),  (0,   212, 187), (44,  153, 168),  (0,  194, 255),
    (52,  69,  147), (100, 45,  130), (127, 69,  193), (191, 33,  119),
]


# ──────────────────────────────────────────────────────────────────────────────
# 1.  TFLite INTERPRETER HELPER
# ──────────────────────────────────────────────────────────────────────────────
def load_interpreter(model_path: str):
    """Load a TFLite model and return (interpreter, input_detail, output_details).

    Parameters
    ----------
    model_path : str  Path to .tflite file.

    Returns
    -------
    interp          : tf.lite.Interpreter  Allocated interpreter.
    inp_detail      : dict  get_input_details()[0]
    out_details     : list  get_output_details()  (may be 1 or 2 tensors)
    is_int8         : bool  True when input dtype is int8.
    input_scale     : float  Quantization scale  (0.0 for FP32).
    input_zp        : int    Quantization zero-point  (0 for FP32).
    """
    try:
        import tensorflow as tf
        interp = tf.lite.Interpreter(model_path=model_path)
    except ImportError:
        # Fall back to tflite_runtime if full TF is not installed
        try:
            import tflite_runtime.interpreter as tflite
            interp = tflite.Interpreter(model_path=model_path)
        except ImportError:
            sys.exit(
                "[ERROR] Neither tensorflow nor tflite_runtime is installed.\n"
                "        pip install tensorflow   OR   pip install tflite-runtime"
            )

    interp.allocate_tensors()
    inp_detail  = interp.get_input_details()[0]
    out_details = interp.get_output_details()

    is_int8 = (inp_detail["dtype"] == np.int8)
    if is_int8:
        input_scale, input_zp = inp_detail["quantization"]
        input_zp = int(input_zp)
    else:
        input_scale, input_zp = 0.0, 0

    return interp, inp_detail, out_details, is_int8, input_scale, input_zp


# ──────────────────────────────────────────────────────────────────────────────
# 2.  PRE-PROCESSING
# ──────────────────────────────────────────────────────────────────────────────
def letterbox(img, new_w, new_h, pad_color=(114, 114, 114)):
    """Resize *img* preserving aspect ratio and pad to (new_h, new_w).

    Parameters
    ----------
    img       : np.ndarray  BGR uint8 image.
    new_w     : int         Target width in pixels.
    new_h     : int         Target height in pixels.
    pad_color : tuple       BGR fill colour for padding area (default grey).

    Returns
    -------
    padded : np.ndarray  (new_h, new_w, 3) uint8
    scale  : float       Resize scale applied to the original image.
    pad    : tuple       (pad_left, pad_top) pixels of padding added.
    """
    h, w     = img.shape[:2]
    scale    = min(new_w / w, new_h / h)
    rw, rh   = int(round(w * scale)), int(round(h * scale))
    resized  = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_LINEAR)
    pad_top  = (new_h - rh) // 2
    pad_left = (new_w - rw) // 2
    padded   = np.full((new_h, new_w, 3), pad_color, dtype=np.uint8)
    padded[pad_top:pad_top + rh, pad_left:pad_left + rw] = resized
    return padded, scale, (pad_left, pad_top)


def preprocess_fp32(img_bgr):
    """Letterbox + normalize to float32 NHWC [0, 1].

    Parameters
    ----------
    img_bgr : np.ndarray  BGR uint8 image, any size.

    Returns
    -------
    blob  : np.ndarray  (1, INPUT_H, INPUT_W, 3) float32  NHWC
    scale : float       Letterbox scale factor.
    pad   : tuple       (pad_left, pad_top) padding in pixels.
    """
    padded, scale, pad = letterbox(img_bgr, INPUT_W, INPUT_H)
    rgb  = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    blob = rgb.astype(np.float32) / 255.0
    return blob[np.newaxis, ...], scale, pad   # NHWC


def preprocess_int8(img_bgr, input_scale: float, input_zp: int):
    """Letterbox + quantize to int8 NHWC.

    Steps:
      1. Letterbox-resize to (INPUT_H, INPUT_W).
      2. Convert BGR → RGB and normalize to [0, 1].
      3. Quantize: q = round(pixel / input_scale) + input_zero_point.
      4. Clip to [-128, 127] and cast to int8.

    Parameters
    ----------
    img_bgr     : np.ndarray  BGR uint8 image.
    input_scale : float       Quantization scale from TFLite model metadata.
    input_zp    : int         Quantization zero-point from TFLite model metadata.

    Returns
    -------
    blob  : np.ndarray  (1, INPUT_H, INPUT_W, 3) int8  NHWC
    scale : float       Letterbox scale factor.
    pad   : tuple       (pad_left, pad_top) padding in pixels.
    """
    padded, scale, pad = letterbox(img_bgr, INPUT_W, INPUT_H)
    rgb  = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    arr  = rgb.astype(np.float32) / 255.0
    arr_q = np.round(arr / input_scale) + input_zp
    arr_q = np.clip(arr_q, -128, 127).astype(np.int8)
    return arr_q[np.newaxis, ...], scale, pad   # NHWC


# ──────────────────────────────────────────────────────────────────────────────
# 3.  YOLO HEAD DECODING
# ──────────────────────────────────────────────────────────────────────────────
def sigmoid(x):
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))


def decode_head_nhwc(raw, anchors, stride, conf_thresh):
    """Decode one raw YOLO output head from NHWC TFLite output layout.

    TFLite outputs are NHWC: shape (1, grid_h, grid_w, 255).
    We reshape / process accordingly, then produce the same boxes/scores/ids
    as the ONNX NCHW variant in ../inference.py.

    Parameters
    ----------
    raw         : np.ndarray  Shape (1, grid_h, grid_w, 255)  NHWC float32.
    anchors     : list        [(w, h), ...] anchor sizes for this head.
    stride      : int         Grid stride (32 for 10×10, 16 for 20×20).
    conf_thresh : float       Minimum objectness × class score to keep.

    Returns
    -------
    boxes     : np.ndarray  (N, 4) float32  [x1, y1, x2, y2] in input space.
    scores    : np.ndarray  (N,)   float32  confidence per detection.
    class_ids : np.ndarray  (N,)   int32    predicted class index.
    """
    n_anchors      = len(anchors)
    grid_h, grid_w = raw.shape[1], raw.shape[2]   # NHWC: 1 H W C

    # Reshape: (1, H, W, na*(5+nc)) → (1, H, W, na, 5+nc)
    pred = raw[0].reshape(grid_h, grid_w, n_anchors, 5 + NUM_CLS)
    # → (na, H, W, 5+nc)  to match ../inference.py layout after decode
    pred = pred.transpose(2, 0, 1, 3)             # (na, H, W, 85)

    grid_y, grid_x = np.mgrid[0:grid_h, 0:grid_w]
    tx  = sigmoid(pred[..., 0])
    ty  = sigmoid(pred[..., 1])
    tw  = pred[..., 2]
    th  = pred[..., 3]
    obj = sigmoid(pred[..., 4])

    boxes_l, scores_l, ids_l = [], [], []
    for a, (aw, ah) in enumerate(anchors):
        cx = (tx[a] + grid_x) * stride
        cy = (ty[a] + grid_y) * stride
        w  = np.exp(tw[a]) * aw
        h  = np.exp(th[a]) * ah

        cls_prob  = sigmoid(pred[a, ..., 5:])
        class_ids = np.argmax(cls_prob, axis=-1)
        class_scr = cls_prob[np.arange(grid_h)[:, None],
                              np.arange(grid_w)[None, :],
                              class_ids]
        conf = obj[a] * class_scr
        mask = conf >= conf_thresh
        if not mask.any():
            continue

        x1 = cx[mask] - w[mask] / 2
        y1 = cy[mask] - h[mask] / 2
        x2 = cx[mask] + w[mask] / 2
        y2 = cy[mask] + h[mask] / 2
        boxes_l.append(np.stack([x1, y1, x2, y2], axis=1))
        scores_l.append(conf[mask])
        ids_l.append(class_ids[mask])

    if not boxes_l:
        return (np.empty((0, 4), np.float32),
                np.empty((0,),   np.float32),
                np.empty((0,),   np.int32))
    return (np.concatenate(boxes_l),
            np.concatenate(scores_l),
            np.concatenate(ids_l))


def apply_nms(boxes, scores, class_ids, iou_thresh):
    """Apply per-class NMS using cv2.dnn.NMSBoxes.

    Parameters
    ----------
    boxes     : np.ndarray  (N, 4) float32  [x1, y1, x2, y2].
    scores    : np.ndarray  (N,)   float32  confidence per box.
    class_ids : np.ndarray  (N,)   int      class index per box.
    iou_thresh: float               IoU threshold for suppression.

    Returns
    -------
    keep : list of int  Indices of surviving boxes.
    """
    keep = []
    for cls in np.unique(class_ids):
        idx  = np.where(class_ids == cls)[0]
        xywh = [[float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                 for x1, y1, x2, y2 in boxes[idx]]
        nms_idx = cv2.dnn.NMSBoxes(xywh, scores[idx].tolist(), 0.0, iou_thresh)
        if len(nms_idx):
            keep.extend(idx[np.array(nms_idx).flatten()].tolist())
    return keep


def postprocess(outputs, conf_thresh, iou_thresh):
    """Decode both YOLO heads (NHWC) and apply NMS.

    Parameters
    ----------
    outputs     : list  Two raw output arrays from TFLite (NHWC layout).
    conf_thresh : float Confidence threshold.
    iou_thresh  : float NMS IoU threshold.

    Returns
    -------
    boxes     : np.ndarray  (N, 4) float32  [x1, y1, x2, y2] in input space.
    scores    : np.ndarray  (N,)   float32
    class_ids : np.ndarray  (N,)   int32
    """
    all_b, all_s, all_c = [], [], []
    for raw, anchors, stride in zip(outputs, HEAD_ANCHORS, STRIDES):
        b, s, c = decode_head_nhwc(raw, anchors, stride, conf_thresh)
        all_b.append(b); all_s.append(s); all_c.append(c)

    boxes     = np.concatenate(all_b)
    scores    = np.concatenate(all_s)
    class_ids = np.concatenate(all_c)
    if not len(boxes):
        return boxes, scores, class_ids

    keep = apply_nms(boxes, scores, class_ids, iou_thresh)
    return boxes[keep], scores[keep], class_ids[keep]


def unscale_boxes(boxes, scale, pad, orig_w, orig_h):
    """Map boxes from letterboxed space back to original image coordinates.

    Parameters
    ----------
    boxes  : np.ndarray  (N, 4) float32  [x1, y1, x2, y2] in 320×320 space.
    scale  : float        Letterbox scale factor.
    pad    : tuple        (pad_left, pad_top) padding in pixels.
    orig_w : int          Original image width.
    orig_h : int          Original image height.

    Returns
    -------
    np.ndarray  (N, 4) float32  clipped to [0, orig_w/orig_h].
    """
    pad_left, pad_top = pad
    out = boxes.copy()
    out[:, [0, 2]] = np.clip((out[:, [0, 2]] - pad_left) / scale, 0, orig_w)
    out[:, [1, 3]] = np.clip((out[:, [1, 3]] - pad_top)  / scale, 0, orig_h)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 4.  VISUALISATION
# ──────────────────────────────────────────────────────────────────────────────
def draw_results(img, boxes, scores, class_ids):
    """Draw bounding boxes and labels on a copy of *img*.

    Parameters
    ----------
    img       : np.ndarray  BGR uint8 image.
    boxes     : np.ndarray  (N, 4) float32  [x1, y1, x2, y2].
    scores    : np.ndarray  (N,)   float32  confidence per box.
    class_ids : np.ndarray  (N,)   int      class index per box.

    Returns
    -------
    np.ndarray  Annotated BGR image.
    """
    out = img.copy()
    for box, score, cid in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = [int(v) for v in box]
        color = _PALETTE[int(cid) % len(_PALETTE)]
        name  = COCO_NAMES[int(cid)] if cid < len(COCO_NAMES) else str(cid)
        label = f"{name}: {score:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        (txt_w, txt_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - txt_h - 6), (x1 + txt_w, y1), color, -1)
        cv2.putText(out, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 5.  TFLite RUNNER
# ──────────────────────────────────────────────────────────────────────────────
def run_inference(model_path, image_path,
                  conf_thresh=CONF_THRESH, iou_thresh=IOU_THRESH,
                  output_path=None, show=True):
    """Load the TFLite model and run single-image detection.

    Automatically detects FP32 vs INT8 from the model's input dtype.

    Parameters
    ----------
    model_path  : str    Path to the TFLite model file.
    image_path  : str    Path to the input image.
    conf_thresh : float  Confidence threshold (default: CONF_THRESH).
    iou_thresh  : float  NMS IoU threshold (default: IOU_THRESH).
    output_path : str    If set, save annotated image to this path.
    show        : bool   Display result window if True.

    Returns
    -------
    boxes     : np.ndarray  (N, 4) float32  detections in original image space.
    scores    : np.ndarray  (N,)   float32
    class_ids : np.ndarray  (N,)   int32
    """
    interp, inp_detail, out_details, is_int8, input_scale, input_zp = \
        load_interpreter(model_path)

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        sys.exit(f"[ERROR] Could not read image: {image_path}")
    orig_h, orig_w = img_bgr.shape[:2]

    # ── Pre-process ───────────────────────────────────────────────────────────
    if is_int8:
        blob, scale, pad = preprocess_int8(img_bgr, input_scale, input_zp)
        mode_str = f"INT8 (scale={input_scale:.6f}, zp={input_zp})"
    else:
        blob, scale, pad = preprocess_fp32(img_bgr)
        mode_str = "FP32"

    # ── Run inference ─────────────────────────────────────────────────────────
    interp.set_tensor(inp_detail["index"], blob)
    t0 = time.perf_counter()
    interp.invoke()
    t1 = time.perf_counter()

    # ── Collect outputs ───────────────────────────────────────────────────────
    # TFLite preserves output order from conversion.
    # Both outputs are NHWC: (1, grid_h, grid_w, 255)
    # Larger grid = stride 16 (20×20), smaller = stride 32 (10×10)
    raw_outs = [interp.get_tensor(d["index"]) for d in out_details]

    # Dequantize INT8 outputs if necessary
    float_outs = []
    for raw, det in zip(raw_outs, out_details):
        if raw.dtype == np.int8:
            out_scale, out_zp = det["quantization"]
            raw_f = (raw.astype(np.float32) - float(out_zp)) * out_scale
            float_outs.append(raw_f)
        else:
            float_outs.append(raw.astype(np.float32))

    # Sort by spatial size: stride-32 head first (10×10), then stride-16 (20×20)
    float_outs.sort(key=lambda x: x.shape[1])   # ascending grid_h → [10, 20]

    boxes, scores, class_ids = postprocess(float_outs, conf_thresh, iou_thresh)
    if len(boxes):
        boxes = unscale_boxes(boxes, scale, pad, orig_w, orig_h)

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"Model   : {model_path}  [{mode_str}]")
    print(f"Image   : {image_path}  ({orig_w}×{orig_h})")
    print(f"Latency : {(t1 - t0) * 1000:.1f} ms")
    print(f"Objects : {len(boxes)}\n")

    print("─" * 60)
    print(f"  {'#':>3}  {'Class':<20}  {'Conf':>6}  "
          f"{'x1':>5}  {'y1':>5}  {'x2':>5}  {'y2':>5}")
    print("─" * 60)
    for i, (box, score, cid) in enumerate(zip(boxes, scores, class_ids)):
        name = COCO_NAMES[int(cid)] if cid < len(COCO_NAMES) else str(cid)
        x1, y1, x2, y2 = [int(v) for v in box]
        print(f"  {i + 1:>3}  {name:<20}  {score:>6.3f}  "
              f"{x1:>5}  {y1:>5}  {x2:>5}  {y2:>5}")
    print("─" * 60)

    annotated = draw_results(img_bgr, boxes, scores, class_ids)
    if output_path:
        cv2.imwrite(output_path, annotated)
        print(f"\nSaved   : {output_path}")
    else:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, "result.jpg")
        cv2.imwrite(output_path, annotated)
        print(f"\nSaved   : {output_path}")

    if show:
        cv2.imshow("Yolo-Fastest TFLite", annotated)
        print("[INFO] Press any key to close ...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return boxes, scores, class_ids


# ──────────────────────────────────────────────────────────────────────────────
# 6.  MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Single-image TFLite inference for Yolo-Fastest 1.1 (80-class COCO). "
            "Supports FP32 and INT8 TFLite models."
        )
    )
    parser.add_argument(
        "--variant", choices=list(_VARIANT_MODELS.keys()), default=None,
        help=(
            "Model variant shortcut: 'base' (yolo-fastest-1.1) or "
            "'xl' (yolo-fastest-1.1-xl). Ignored when --model is given explicitly."
        ),
    )
    parser.add_argument(
        "--model", "-m", default=None,
        help="Path to TFLite model file (overrides --variant).",
    )
    parser.add_argument(
        "--image", "-i", required=True,
        help="Path to input image.",
    )
    parser.add_argument(
        "--score", type=float, default=CONF_THRESH,
        help=f"Confidence threshold (default: {CONF_THRESH}).",
    )
    parser.add_argument(
        "--nms", type=float, default=IOU_THRESH,
        help=f"NMS IoU threshold (default: {IOU_THRESH}).",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Save annotated result image to this path.",
    )
    parser.add_argument(
        "--display", action="store_true",
        help="Open a display window.",
    )
    args = parser.parse_args()

    # Resolve model path: explicit --model > --variant > default (base)
    model_path = args.model or _VARIANT_MODELS.get(args.variant or "base", DEFAULT_MODEL)

    if not os.path.isfile(model_path):
        sys.exit(
            f"[ERROR] Model not found: {model_path}\n"
            "        Run: python download_model.py"
        )

    if not os.path.isfile(args.image):
        sys.exit(f"[ERROR] Image not found: {args.image}")

    run_inference(
        model_path  = model_path,
        image_path  = args.image,
        conf_thresh = args.score,
        iou_thresh  = args.nms,
        output_path = args.output,
        show        = args.display,
    )


if __name__ == "__main__":
    main()
