# Copyright 2026 Renesas Electronics Corporation
#
# SPDX-License-Identifier: Apache-2.0
"""
TFLite inference for PoseNet (MobileNetV1).

Runs a single-image PoseNet TFLite model (FP32 or full-INT8). FP32 vs INT8
is auto-detected from the interpreter's input tensor dtype.

Usage:
  python inference.py --image sample.jpg                                # default FP32 model
  python inference.py --image sample.jpg --model model/posenet_mbv1_050_257_INT8.tflite
  python inference.py --image sample.jpg --out annotated.jpg            # save overlay
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import posenet                                                  # noqa: E402
from utils.progress import spinner                              # noqa: E402

# ------------------------------------------------------------------------------
# Locations & defaults
# ------------------------------------------------------------------------------
MODEL_DIR     = SCRIPT_DIR / "model"
DEFAULT_MODEL = MODEL_DIR / "posenet_mbv1_050_257_FP32.tflite"

# Decoder defaults (PersonLab multi-pose decoder) - tuned on COCO val2017.
# Exposed as CLI flags so users can trade recall vs precision per deployment.
DEFAULT_MAX_POSE       = 10
DEFAULT_MIN_POSE_SCORE = 0.25
DEFAULT_MIN_PART_SCORE = 0.25


def _model_id_from_filename(tflite_path: Path) -> int:
    """Parse MobileNet width multiplier (x100) from filename.

    Convention: posenet_mbv1_<model_id>_<input_size>_<suffix>.tflite
    TFLite metadata does not carry this, so the filename is the source.
    """
    parts = tflite_path.stem.split("_")
    if len(parts) < 5 or parts[0] != "posenet" or parts[1] != "mbv1":
        raise ValueError(
            f"unexpected filename {tflite_path.name}; "
            f"expected posenet_mbv1_<id>_<size>_<suffix>.tflite")
    try:
        return int(parts[2])
    except ValueError as e:
        raise ValueError(f"cannot parse model_id from {tflite_path.name}") from e


def _dims_from_interpreter(it, tflite_name: str) -> tuple[int, int]:
    """Return ``(input_size, output_stride)`` from an already-loaded interpreter.

    input_size    <- input tensor shape [1, S, S, 3]
    output_stride <- derived from heatmap head: (input_size - 1) / (head_h - 1)
    """
    inp_shape = it.get_input_details()[0]["shape"]
    if len(inp_shape) != 4 or inp_shape[1] != inp_shape[2]:
        raise ValueError(
            f"unexpected input shape {list(inp_shape)} in {tflite_name}; "
            f"want [1, S, S, 3]")
    input_size = int(inp_shape[1])

    head_h = next((int(d["shape"][1]) for d in it.get_output_details()
                   if int(d["shape"][-1]) == 17), None)
    if head_h is None:
        raise RuntimeError(f"no heatmap head (17 ch) in {tflite_name}")
    return input_size, (input_size - 1) // (head_h - 1)


# ------------------------------------------------------------------------------
# Output-head mapping (TFLite can re-order outputs; map by channel count)
# ------------------------------------------------------------------------------
def _map_heads_nhwc(outs_by_name, swap_disp_heads=False):
    """
    PoseNet has 4 heads:
       heatmap          : 17 channels
       offset           : 34 channels
       displacement_fwd : 32 channels
       displacement_bwd : 32 channels
    NHWC tensors are returned; the caller converts to CHW.

    ``swap_disp_heads`` overrides the name-based heuristic.  Pass True when
    the TFLite runtime returns the two 32-channel heads in bwd-first order
    (e.g. TFLite INT8 export where names are positional: PartitionedCall:N).
    """
    by_c = {17: [], 34: [], 32: []}
    for name, arr in outs_by_name:
        if arr.ndim != 4:
            raise ValueError(f"Expected NHWC, got shape {arr.shape} for {name}")
        c = arr.shape[-1]
        if c not in by_c:
            raise ValueError(f"Unexpected channel count {c} for {name}")
        by_c[c].append((name, arr))

    if len(by_c[17]) != 1 or len(by_c[34]) != 1 or len(by_c[32]) != 2:
        raise ValueError(
            f"PoseNet head layout mismatch (got 17/{len(by_c[17])}, "
            f"34/{len(by_c[34])}, 32/{len(by_c[32])})")

    heatmap = by_c[17][0][1]
    offset  = by_c[34][0][1]

    if swap_disp_heads:
        # Explicit swap: caller determined that by_c[32][0] is bwd, [1] is fwd.
        disp_fwd, disp_bwd = by_c[32][1][1], by_c[32][0][1]
    else:
        # Name-based heuristic for backends with semantic tensor names.
        name0 = by_c[32][0][0].lower()
        name1 = by_c[32][1][0].lower()
        if "bwd" in name0 or "fwd" in name1:
            disp_fwd, disp_bwd = by_c[32][1][1], by_c[32][0][1]
        else:
            disp_fwd, disp_bwd = by_c[32][0][1], by_c[32][1][1]

    return heatmap, offset, disp_fwd, disp_bwd


# ------------------------------------------------------------------------------
# TFLite runners
# ------------------------------------------------------------------------------
def _load_tflite_interpreter(tflite_path: Path):
    import tensorflow as tf
    it = tf.lite.Interpreter(model_path=str(tflite_path), num_threads=4)
    it.allocate_tensors()
    return it


def run_tflite_fp32(interpreter, nhwc_f32):
    inp = interpreter.get_input_details()[0]
    outs = interpreter.get_output_details()
    interpreter.set_tensor(inp["index"], nhwc_f32.astype(inp["dtype"]))
    t0 = time.perf_counter()
    interpreter.invoke()
    dt = (time.perf_counter() - t0) * 1000
    raw = [(d["name"], interpreter.get_tensor(d["index"])) for d in outs]
    return raw, dt


def run_tflite_int8(interpreter, nhwc_f32):
    inp = interpreter.get_input_details()[0]
    outs = interpreter.get_output_details()

    s_in, z_in = inp["quantization"]
    if s_in == 0.0:
        x_q = nhwc_f32.astype(inp["dtype"])
    else:
        x_q = np.round(nhwc_f32 / s_in + z_in).clip(-128, 127).astype(inp["dtype"])
    interpreter.set_tensor(inp["index"], x_q)
    t0 = time.perf_counter()
    interpreter.invoke()
    dt = (time.perf_counter() - t0) * 1000

    raw = []
    for d in outs:
        q = interpreter.get_tensor(d["index"])
        s_out, z_out = d["quantization"]
        if s_out == 0.0:
            arr = q.astype(np.float32)
        else:
            arr = (q.astype(np.float32) - z_out) * s_out
        raw.append((d["name"], arr))
    return raw, dt


# ------------------------------------------------------------------------------
# End-to-end inference for a single image
# ------------------------------------------------------------------------------
def infer_image(src_bgr,
                runner_fn,
                input_size: int,
                output_stride: int,
                *,
                swap_disp_heads: bool = False,
                max_pose: int = DEFAULT_MAX_POSE,
                min_pose_score: float = DEFAULT_MIN_POSE_SCORE,
                min_part_score: float = DEFAULT_MIN_PART_SCORE):
    """Run end-to-end inference; returns ``(annotated_img, info_dict)``."""
    nhwc = posenet.preprocess_image(src_bgr, (input_size, input_size),
                                    layout="NHWC")
    raw_out, dt = runner_fn(nhwc)

    # Strip batch dim and NHWC -> CHW float tensor for the decoder.
    heatmap, offset, disp_fwd, disp_bwd = (
        torch.from_numpy(np.transpose(t[0], (2, 0, 1))).float()
        for t in _map_heads_nhwc(raw_out, swap_disp_heads=swap_disp_heads))

    ps, ks, kc = posenet.decode_multiple_poses(
        heatmap, offset, disp_fwd, disp_bwd,
        output_stride=output_stride,
        max_pose_detections=max_pose,
        min_pose_score=min_pose_score)

    # Rescale keypoints from model-input pixels back to source-image pixels.
    src_h, src_w = src_bgr.shape[:2]
    kc *= np.array([src_h, src_w]) / input_size

    annotated = posenet.draw_skel_and_kp(
        src_bgr, ps, ks, kc,
        min_pose_score=min_pose_score,
        min_part_score=min_part_score)

    return annotated, {
        "inference_ms": dt,
        "pose_scores": ps,
        "keypoint_scores": ks,
        "keypoint_coords": kc,
        "heatmap_range": (float(heatmap.min()), float(heatmap.max())),
    }


def print_poses(info, min_pose_score=DEFAULT_MIN_POSE_SCORE,
                min_part_score=DEFAULT_MIN_PART_SCORE):
    print(f"\n  Inference time : {info['inference_ms']:.1f} ms")
    print(f"  Heatmap range  : [{info['heatmap_range'][0]:.3f}, {info['heatmap_range'][1]:.3f}]")
    ps = info["pose_scores"]
    ks = info["keypoint_scores"]
    kc = info["keypoint_coords"]
    detected = 0
    for pi, s in enumerate(ps):
        if s < min_pose_score:
            continue
        detected += 1
        print(f"\n  Pose #{pi}  score={s:.3f}")
        for ki, (sc, c) in enumerate(zip(ks[pi], kc[pi])):
            if sc < min_part_score:
                continue
            print(f"    {posenet.PART_NAMES[ki]:<16s} score={sc:.3f}  (y,x)=({c[0]:7.1f},{c[1]:7.1f})")
    if detected == 0:
        print("\n  (no poses above threshold)")


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="PoseNet MobileNetV1 multi-person inference (TFLite)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--image", required=True,
                   help="path to a single input image")
    p.add_argument("--model", default=str(DEFAULT_MODEL),
                   help="path to a PoseNet TFLite model (FP32 or INT8)")
    p.add_argument("--out", help="save annotated image to this path")

    p.add_argument("--max-pose",       type=int,   default=DEFAULT_MAX_POSE,
                   help="maximum poses returned per image")
    p.add_argument("--min-pose-score", type=float, default=DEFAULT_MIN_POSE_SCORE,
                   help="discard whole pose if root-keypoint score < this")
    p.add_argument("--min-part-score", type=float, default=DEFAULT_MIN_PART_SCORE,
                   help="discard individual keypoint if score < this")

    args = p.parse_args()

    # Resolve model path and detect FP32 vs INT8 from the interpreter itself.
    tflite_path = Path(args.model)
    if not tflite_path.is_file():
        raise SystemExit(f"model not found: {tflite_path}")

    with spinner(f"Loading TFLite model ({tflite_path.name})"):
        interpreter = _load_tflite_interpreter(tflite_path)

    input_dtype = interpreter.get_input_details()[0]["dtype"]
    if input_dtype == np.int8:
        model_type = "INT8"
        runner = run_tflite_int8
    else:
        model_type = "FP32"
        runner = run_tflite_fp32

    model_id = _model_id_from_filename(tflite_path)
    input_size, output_stride = _dims_from_interpreter(interpreter, tflite_path.name)
    runner_fn = lambda x: runner(interpreter, x)

    img_path = Path(args.image)
    src = cv2.imread(str(img_path))
    if src is None:
        raise SystemExit(f"cannot read {img_path}")

    print(f"{'='*60}")
    print(f"  PoseNet MobileNetV1-{model_id:03d} @ {input_size}x{input_size}  "
          f"(output_stride={output_stride})")
    print(f"  Model : {tflite_path.name} ({model_type})")
    print(f"  Image : {img_path.name}")
    print(f"{'='*60}")

    # INT8 TFLite export returns the two 32-ch displacement heads in bwd-first
    # order (PartitionedCall:3 before PartitionedCall:2); swap to correct.
    swap_disp_heads = (model_type == "INT8")

    annotated, info = infer_image(
        src, runner_fn, input_size, output_stride,
        swap_disp_heads=swap_disp_heads,
        max_pose=args.max_pose,
        min_pose_score=args.min_pose_score,
        min_part_score=args.min_part_score)

    print_poses(info,
                min_pose_score=args.min_pose_score,
                min_part_score=args.min_part_score)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), annotated)
        print(f"\n  Annotated image -> {out_path}")


if __name__ == "__main__":
    main()
