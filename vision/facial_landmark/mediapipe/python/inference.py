#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""MTCNN + TFLite face landmark inference pipeline.

This script runs a two-stage flow:
1) Detect faces with MTCNN (or assume already-cropped face when appropriate).
2) Predict 468 face landmarks with a MediaPipe-style TFLite landmark model.

The implementation intentionally keeps the geometric transforms explicit so that
ROI handling, quantization, and projection math can be inspected and debugged.

Usage:
    # Single image (uses default model and output directory)
    python AI_docker/facial_landmark/media_pipe_final/inference.py \\
        --input AI_docker/facial_landmark/media_pipe_final/sample_images/000000008532.jpg


    # Custom model and output directory
    python AI_docker/facial_landmark/media_pipe_final/inference.py \\
        --input path/to/image_or_dir \\
        --model AI_docker/facial_landmark/media_pipe_final/model/face_landmark_INT8.tflite \\
        --output-dir AI_docker/facial_landmark/media_pipe_final/outputs

Arguments:
    --input       Path to a face image or directory of images (required).
    --model       Path to the TFLite landmark model
                  (default: <script_dir>/model/face_landmark_INT8.tflite).
    --output-dir  Directory where rendered mesh overlays are saved
                  (default: <script_dir>/outputs).
"""

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw
from PIL.Image import Image as PILImage, Resampling, Transform, Transpose

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite


MODEL_NAME = "face_landmark_INT8.tflite"
NUM_DIMS = 3
NUM_LANDMARKS = 468
DETECTION_THRESHOLD = 0.5
# Confidence threshold used by MTCNN proposals.
MTCNN_MIN_CONFIDENCE = 0.8
# ROI enlargement factor to include useful context around the face.
ROI_SCALE = 1.5
# Optional normalized ROI center offsets (relative to ROI size).
ROI_SHIFT_X = 0.0
ROI_SHIFT_Y = 0.0


class InvalidEnumError(Exception):
    """Reserved custom exception for enum validation in extended variants."""
    pass

class CoordinateRangeError(Exception):
    """Reserved custom exception for coordinate range checks."""
    pass

class ArgumentError(Exception):
    """Reserved custom exception for argument validation."""
    pass

@dataclass
class ImageTensor:
    """Container for tensorized image and metadata used for back-projection.

    Attributes:
        tensor_data: Float image data after preprocessing.
        padding: Normalized (left, top, right, bottom) padding ratios.
        original_size: Source image size as (width, height).
    """
    tensor_data: np.ndarray
    padding: Tuple[float, float, float, float]
    original_size: Tuple[int, int]


@dataclass
class Rect:
    """Rotated rectangle used for ROI extraction and landmark reprojection.

    Coordinates are either normalized [0, 1] or absolute pixel space,
    controlled by `normalized`.
    """
    x_center: float
    y_center: float
    width: float
    height: float
    rotation: float
    normalized: bool

    @property
    def size(self) -> Union[Tuple[float, float], Tuple[int, int]]:
        """Return (width, height) in native units; cast to int in pixel mode."""
        w, h = self.width, self.height
        return (w, h) if self.normalized else (int(w), int(h))

    def scaled(
        self, size: Tuple[float, float], normalize: bool = False
    ) -> "Rect":
        """Convert rectangle between normalized and pixel coordinate systems."""
        if self.normalized == normalize:
            return self
        sx, sy = size
        if normalize:
            sx, sy = 1 / sx, 1 / sy
        return Rect(self.x_center * sx, self.y_center * sy,
                    self.width * sx, self.height * sy,
                    self.rotation, normalized=False)

    def points(self) -> np.ndarray:
        """Return four rectangle corner points, applying rotation if present."""
        x, y = self.x_center, self.y_center
        w, h = self.width / 2, self.height / 2
        pts = [(x - w, y - h), (x + w, y - h), (x + w, y + h), (x - w, y + h)]
        if self.rotation == 0:
            return pts
        s, c = np.sin(self.rotation), np.cos(self.rotation)
        t = np.array(pts) - (x, y)
        r = np.array([[c, s], [-s, c]])
        return np.matmul(t, r) + (x, y)


@dataclass
class Landmark:
    """Single face landmark in normalized image coordinates with relative depth."""
    x: float
    y: float
    z: float


@dataclass
class FaceDetection:
    """Normalized face detection representation from MTCNN output parsing."""
    bbox: Tuple[float, float, float, float]
    score: float
    left_eye: Optional[Tuple[float, float]] = None
    right_eye: Optional[Tuple[float, float]] = None


def image_to_tensor(
    image: Union[PILImage, np.ndarray, str],
    roi: Optional[Rect] = None,
    output_size: Optional[Tuple[int, int]] = None,
    keep_aspect_ratio: bool = False,
    output_range: Tuple[float, float] = (0.0, 1.0),
    flip_horizontal: bool = False,
) -> ImageTensor:
    """Warp selected ROI to model input tensor while tracking projection metadata.

    Steps:
    1) Normalize input image type (path/array/PIL) to RGB PIL.
    2) Build source ROI corner points (with optional rotation).
    3) Perspective-warp ROI to output tensor frame.
    4) Optionally letterbox/pillarbox to preserve ROI aspect ratio.
    5) Convert to float range expected by model.
    """
    img = _normalize_image(image)
    image_size = img.size
    if roi is None:
        # Full-image ROI in normalized coordinates.
        roi = Rect(0.5, 0.5, 1.0, 1.0, rotation=0.0, normalized=True)
    roi = roi.scaled(image_size)
    if output_size is None:
        output_size = (int(roi.size[0]), int(roi.size[1]))
    width, height = (roi.size if keep_aspect_ratio else output_size)
    src_points = roi.points()
    dst_points = [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)]
    coeffs = _perspective_transform_coeff(src_points, dst_points)
    roi_image = img.transform(size=(width, height), method=Transform.PERSPECTIVE,
                              data=coeffs, resample=Resampling.BILINEAR)
    if img != image:
        img.close()
    pad_x, pad_y = 0.0, 0.0
    if keep_aspect_ratio:
        # Compute symmetric padding to keep ROI geometry undistorted.
        out_aspect = output_size[1] / output_size[0]
        roi_aspect = roi.height / roi.width
        new_width, new_height = int(roi.width), int(roi.height)
        if out_aspect > roi_aspect:
            new_height = int(roi.width * out_aspect)
            pad_y = (1 - roi_aspect / out_aspect) / 2
        else:
            new_width = int(roi.height / out_aspect)
            pad_x = (1 - out_aspect / roi_aspect) / 2
        if new_width != int(roi.width) or new_height != int(roi.height):
            pad_h, pad_v = int(pad_x * new_width), int(pad_y * new_height)
            roi_image = roi_image.transform(
                size=(new_width, new_height), method=Transform.EXTENT,
                data=(-pad_h, -pad_v, new_width - pad_h, new_height - pad_v))
        roi_image = roi_image.resize(output_size, resample=Resampling.BILINEAR)
    if flip_horizontal:
        roi_image = roi_image.transpose(method=Transpose.FLIP_LEFT_RIGHT)
    min_val, max_val = output_range
    # Scale uint8 [0,255] into model-specific numeric range.
    tensor_data = np.asarray(roi_image, dtype=np.float32)
    tensor_data *= (max_val - min_val) / 255
    tensor_data += min_val
    return ImageTensor(tensor_data,
                       padding=(pad_x, pad_y, pad_x, pad_y),
                       original_size=image_size)


def sigmoid(data: np.ndarray) -> np.ndarray:
    """Numerically simple sigmoid used for face-presence head output."""
    return 1 / (1 + np.exp(-data))


def project_landmarks(
    data: Union[Sequence[Landmark], np.ndarray],
    *,
    tensor_size: Tuple[int, int],
    image_size: Tuple[int, int],
    padding: Tuple[float, float, float, float],
    roi: Optional[Rect],
    flip_horizontal: bool = False,
) -> List[Landmark]:
    """Map raw model landmark coordinates back to the original image space.

    The landmark model predicts coordinates in tensor pixel space; this function
    reverses preprocessing (normalization, optional horizontal flip, padding,
    and rotated ROI placement) to recover image-normalized landmark positions.
    """
    if not isinstance(data, np.ndarray):
        points = np.array([(pt.x, pt.y, pt.z) for pt in data], dtype="float32")
    else:
        points = data.reshape(-1, 3)
    width, height = tensor_size
    # Convert tensor pixel coordinates to [0, 1]-normalized tensor frame.
    points /= (width, height, width)
    if flip_horizontal:
        points[:, 0] *= -1
        points[:, 0] += 1
    if any(padding):
        # Remove synthetic padding introduced by keep_aspect_ratio preprocessing.
        left, top, right, bottom = padding
        h_scale = 1 - (left + right)
        v_scale = 1 - (top + bottom)
        points -= (left, top, 0.0)
        points /= (h_scale, v_scale, h_scale)
    if roi is None:
        return [Landmark(x, y, z) for (x, y, z) in points]
    norm_roi = roi.scaled(image_size, normalize=True)
    # Rotate from ROI-local axes back into image axes.
    sin, cos = np.sin(roi.rotation), np.cos(roi.rotation)
    matrix = np.array([[cos, sin, 0.0], [-sin, cos, 0.0], [1.0, 1.0, 1.0]])
    points -= (0.5, 0.5, 0.0)
    rotated = np.matmul(points * (1, 1, 0), matrix)
    points *= (0, 0, 1)
    points += rotated
    points *= (norm_roi.width, norm_roi.height, norm_roi.width)
    points += (norm_roi.x_center, norm_roi.y_center, 0.0)
    return [Landmark(x, y, z) for (x, y, z) in points]


def _perspective_transform_coeff(
    src_points: np.ndarray,
    dst_points: np.ndarray,
) -> np.ndarray:
    """Solve 8-parameter perspective transform from destination to source points."""
    matrix = []
    for (x, y), (X, Y) in zip(dst_points, src_points):
        matrix.extend([
            [x, y, 1.0, 0.0, 0.0, 0.0, -X * x, -X * y],
            [0.0, 0.0, 0.0, x, y, 1.0, -Y * x, -Y * y],
        ])
    A = np.array(matrix, dtype=np.float32)
    B = np.array(src_points, dtype=np.float32).reshape(8)
    return np.linalg.solve(A, B)


def _normalize_image(image: Union[PILImage, np.ndarray, str]) -> PILImage:
    """Accept path/array/PIL input and return an RGB PIL image."""
    if isinstance(image, PILImage) and image.mode != "RGB":
        return image.convert(mode="RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image, mode="RGB")
    if not isinstance(image, PILImage):
        return Image.open(image)
    return image


def _mtcnn_detections(image: PILImage, min_confidence: float = 0.8) -> List[FaceDetection]:
    """Run MTCNN and convert detections to local FaceDetection objects."""
    try:
        from mtcnn import MTCNN
    except ImportError as exc:
        raise ImportError(
            "MTCNN is not installed. Install it with: pip install mtcnn"
        ) from exc

    detector = MTCNN()
    rgb = np.asarray(image)
    results = detector.detect_faces(rgb)

    detections: List[FaceDetection] = []
    for det in results:
        confidence = float(det.get("confidence", 0.0))
        if confidence < min_confidence:
            continue
        x, y, w, h = det["box"]
        x1, y1 = float(x), float(y)
        x2, y2 = float(x + w), float(y + h)
        kpts = det.get("keypoints", {})
        left_eye = tuple(kpts["left_eye"]) if "left_eye" in kpts else None
        right_eye = tuple(kpts["right_eye"]) if "right_eye" in kpts else None
        detections.append(
            FaceDetection(
                bbox=(x1, y1, x2, y2),
                score=confidence,
                left_eye=left_eye,
                right_eye=right_eye,
            )
        )
    return detections


def _detection_to_roi(
    detection: FaceDetection,
    image_size: Tuple[int, int],
    scale: float = 1.5,
    shift_x: float = 0.0,
    shift_y: float = 0.0,
) -> Rect:
    """Convert a face bounding box (+ optional eye keypoints) into landmark ROI.

    The ROI is square, enlarged by `scale`, and optionally rotated by eye angle
    so the landmark model receives an aligned crop.
    """
    img_w, img_h = image_size
    x1, y1, x2, y2 = detection.bbox
    # Clamp detector output to image bounds before ROI derivation.
    x1 = min(max(x1, 0.0), float(img_w - 1))
    y1 = min(max(y1, 0.0), float(img_h - 1))
    x2 = min(max(x2, 0.0), float(img_w - 1))
    y2 = min(max(y2, 0.0), float(img_h - 1))

    # Ensure proper corner ordering even if detector emits inverted boxes.
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    side = max(bw, bh) * scale
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0

    # Apply MediaPipe-style optional ROI center translation in ROI units.
    cx += shift_x * side
    cy += shift_y * side

    rotation = 0.0
    if detection.left_eye is not None and detection.right_eye is not None:
        # Rotate ROI so eyes become approximately horizontal.
        lx, ly = detection.left_eye
        rx, ry = detection.right_eye
        rotation = float(-np.arctan2(ly - ry, rx - lx))

    return Rect(
        x_center=cx / img_w,
        y_center=cy / img_h,
        width=side / img_w,
        height=side / img_h,
        rotation=rotation,
        normalized=True,
    )


class FaceLandmark:
    """TFLite wrapper for 468-point face landmark inference."""

    def __init__(self, model_path: Optional[str] = None) -> None:
        """Initialize interpreter and resolve model I/O tensor metadata."""
        if model_path is None:
            my_path = os.path.abspath(__file__)
            model_path = os.path.join(os.path.dirname(my_path), "model")
        if model_path.endswith(".tflite"):
            self.model_path = model_path
        else:
            self.model_path = os.path.join(model_path, MODEL_NAME)
        self.interpreter = tflite.Interpreter(model_path=self.model_path)
        self.input_details = self.interpreter.get_input_details()[0]
        self.input_index = self.input_details["index"]
        self.input_shape = self.input_details["shape"]
        self.output_details = self.interpreter.get_output_details()
        self.data_index = self.output_details[0]["index"]
        self.face_index = self.output_details[1]["index"]
        data_shape = self.output_details[0]["shape"]
        num_expected_elements = NUM_DIMS * NUM_LANDMARKS
        if data_shape[-1] < num_expected_elements:
            raise ValueError(
                f"incompatible model: {data_shape} < {num_expected_elements}"
            )
        self.interpreter.allocate_tensors()

    def _quantize_input(self, float_tensor: np.ndarray) -> np.ndarray:
        """Convert float input to model dtype using tensor quantization params."""
        dtype = self.input_details["dtype"]
        scale, zero_point = self.input_details["quantization"]
        if dtype == np.float32:
            return float_tensor.astype(np.float32)
        if scale == 0:
            return float_tensor.astype(dtype)
        q = float_tensor / scale + zero_point
        if dtype == np.int8:
            return np.clip(np.round(q), -128, 127).astype(np.int8)
        if dtype == np.uint8:
            return np.clip(np.round(q), 0, 255).astype(np.uint8)
        raise ValueError(f"Unsupported input dtype: {dtype}")

    def _dequantize_output(self, index: int) -> np.ndarray:
        """Read output tensor and dequantize integers back to float domain."""
        raw = self.interpreter.get_tensor(index)
        details = next(
            (od for od in self.output_details if od["index"] == index), None
        )
        if details is None:
            return raw.astype(np.float32)
        scale, zero_point = details["quantization"]
        if np.issubdtype(details["dtype"], np.integer) and scale != 0:
            return scale * (raw.astype(np.float32) - float(zero_point))
        return raw.astype(np.float32)

    def __call__(
        self,
        image: Union[PILImage, np.ndarray, str],
        roi: Optional[Rect] = None,
    ) -> List[Landmark]:
        """Run one forward pass and return projected landmarks.

        Returns an empty list when the model face-presence score is below
        DETECTION_THRESHOLD.
        """
        height, width = self.input_shape[1:3]
        image_data = image_to_tensor(
            image,
            roi,
            output_size=(width, height),
            keep_aspect_ratio=False,
            output_range=(0.0, 1.0),
        )
        input_data = self._quantize_input(image_data.tensor_data[np.newaxis])
        self.interpreter.set_tensor(self.input_index, input_data)
        self.interpreter.invoke()
        raw_data = self._dequantize_output(self.data_index)
        raw_face = self._dequantize_output(self.face_index)
        # Model outputs a face-presence logit; sigmoid converts it to probability.
        face_flag = sigmoid(raw_face).flatten()[-1]
        if face_flag < DETECTION_THRESHOLD:
            return []
        height, width = self.input_shape[1:3]
        return project_landmarks(
            raw_data,
            tensor_size=(width, height),
            image_size=image_data.original_size,
            padding=image_data.padding,
            roi=roi,
        )


FACE_LANDMARK_CONNECTIONS = [
    # Canonical MediaPipe face mesh edges for visualization.
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314),
    (314, 405), (405, 321), (321, 375), (375, 291), (61, 185), (185, 40),
    (40, 39), (39, 37), (37, 0), (0, 267), (267, 269),
    (269, 270), (270, 409), (409, 291), (78, 95), (95, 88), (88, 178),
    (178, 87), (87, 14), (14, 317), (317, 402), (402, 318), (318, 324),
    (324, 308), (78, 191), (191, 80), (80, 81), (81, 82), (82, 13), (13, 312),
    (312, 311), (311, 310), (310, 415), (415, 308),
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154),
    (154, 155), (155, 133), (33, 246), (246, 161), (161, 160), (160, 159),
    (159, 158), (158, 157), (157, 173), (173, 133),
    (46, 53), (53, 52), (52, 65), (65, 55), (70, 63), (63, 105), (105, 66),
    (66, 107),
    (263, 249), (249, 390), (390, 373), (373, 374), (374, 380), (380, 381),
    (381, 382), (382, 362), (263, 466), (466, 388), (388, 387), (387, 386),
    (386, 385), (385, 384), (384, 398), (398, 362),
    (276, 283), (283, 282), (282, 295), (295, 285), (300, 293), (293, 334),
    (334, 296), (296, 336),
    (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389),
    (389, 356), (356, 454), (454, 323), (323, 361), (361, 288), (288, 397),
    (397, 365), (365, 379), (379, 378), (378, 400), (400, 377), (377, 152),
    (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172),
    (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162),
    (162, 21), (21, 54), (54, 103), (103, 67), (67, 109), (109, 10),
]


def draw_face_mesh(image: PILImage, landmarks: Sequence[Landmark]) -> PILImage:
    """Draw mesh edges and per-point dots on a copy of the input image."""
    out = image.copy()
    if not landmarks:
        return out
    draw = ImageDraw.Draw(out)
    width, height = out.size
    points = [(lm.x * width, lm.y * height) for lm in landmarks]
    for i, j in FACE_LANDMARK_CONNECTIONS:
        if i < len(points) and j < len(points):
            draw.line([points[i], points[j]], fill=(0, 255, 0), width=1)
    for x, y in points:
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(255, 0, 0))
    return out


def draw_bbox(image: PILImage, bbox: Tuple[float, float, float, float], color=(255, 255, 0)) -> PILImage:
    """Draw one rectangle around a detected face."""
    out = image.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle(bbox, outline=color, width=2)
    return out


def list_input_images(input_path: Path) -> List[Path]:
    """Return a single image path or all supported images from a directory."""
    if input_path.is_file():
        return [input_path]
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [p for p in sorted(input_path.iterdir()) if p.suffix.lower() in exts]


def parse_args() -> argparse.Namespace:
    """Parse command-line options for model path, input path, and outputs."""
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Cropped-image facial landmark inference using original functions"
    )
    parser.add_argument("--input", required=True,
                        help="Path to cropped face image or directory")
    parser.add_argument("--model", default=str(base_dir / "model" / MODEL_NAME),
                        help="Path to face_landmark.tflite")
    parser.add_argument("--output-dir", default=str(base_dir / "outputs"),
                        help="Directory for rendered outputs")
    return parser.parse_args()


def main() -> None:
    """Entry point for batch/individual image inference and rendering."""
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    images = list_input_images(input_path)
    if not images:
        raise RuntimeError(f"No images found in: {input_path}")

    model = FaceLandmark(model_path=str(model_path))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in images:
        # Preserve original image and render overlays on a copy.
        image = Image.open(image_path).convert("RGB")
        out_img = image.copy()

        # Detect faces first and run inference on ROI of highest-confidence detection.
        detections = _mtcnn_detections(
            image, min_confidence=MTCNN_MIN_CONFIDENCE
        )

        if not detections:
            # Save unchanged image so output folder still mirrors processed inputs.
            out_path = output_dir / f"{image_path.stem}_mesh.png"
            out_img.save(out_path)
            print(f"[warn] {image_path.name}: no face detected -> {out_path}")
            continue

        # Process highest-confidence faces first; all detections are rendered.
        detections.sort(key=lambda d: d.score, reverse=True)
        for det in detections:
            roi = _detection_to_roi(
                det,
                image.size,
                scale=ROI_SCALE,
                shift_x=ROI_SHIFT_X,
                shift_y=ROI_SHIFT_Y,
            )
            landmarks = model(image, roi=roi)
            if landmarks:
                out_img = draw_face_mesh(out_img, landmarks)
            out_img = draw_bbox(out_img, det.bbox)

        out_path = output_dir / f"{image_path.stem}_mesh.png"
        out_img.save(out_path)
        print(
            f"[ok] {image_path.name}: mode=auto(full) faces={len(detections)} -> {out_path}"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
