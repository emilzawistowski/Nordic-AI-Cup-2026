from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from dtos import DroneFlybyPredictionDto
from utils import clip_bbox_to_frame, view_bbox_to_global


ROOT = Path(__file__).resolve().parent

OBJECTNESS_MODEL_PATH = (
    ROOT / "models" / "drone_objectness_yolo11s_v1.pt"
)

CLASSIFIER_MODEL_PATH = (
    ROOT / "models" / "drone_classifier_yolo11s_v1.pt"
)

_objectness_model = None
_classifier_model = None


def get_objectness_model():
    global _objectness_model

    if _objectness_model is None:
        _objectness_model = YOLO(
            str(OBJECTNESS_MODEL_PATH)
        )

    return _objectness_model


def get_classifier_model():
    global _classifier_model

    if _classifier_model is None:
        _classifier_model = YOLO(
            str(CLASSIFIER_MODEL_PATH)
        )

    return _classifier_model


def make_square_crop(image, box, padding_factor):
    x1, y1, x2, y2 = map(float, box)

    width = x2 - x1
    height = y2 - y1
    object_size = max(width, height)

    if object_size <= 0:
        return None

    side = max(
        24.0,
        object_size * (1.0 + 2.0 * padding_factor),
    )

    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0

    left = int(round(center_x - side / 2.0))
    top = int(round(center_y - side / 2.0))
    right = int(round(center_x + side / 2.0))
    bottom = int(round(center_y + side / 2.0))

    pad_left = max(0, -left)
    pad_top = max(0, -top)
    pad_right = max(0, right - image.shape[1])
    pad_bottom = max(0, bottom - image.shape[0])

    left = max(0, left)
    top = max(0, top)
    right = min(image.shape[1], right)
    bottom = min(image.shape[0], bottom)

    crop = image[top:bottom, left:right]

    if crop.size == 0:
        return None

    if pad_left or pad_top or pad_right or pad_bottom:
        crop = cv2.copyMakeBorder(
            crop,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_REFLECT_101,
        )

    return cv2.resize(
        crop,
        (224, 224),
        interpolation=cv2.INTER_CUBIC,
    )


def classify_boxes(image, boxes):
    classifier = get_classifier_model()

    crops = []
    crop_metadata = []

    padding_factors = (0.35, 0.90)

    for box_index, box in enumerate(boxes):
        for padding_factor in padding_factors:
            crop = make_square_crop(
                image,
                box,
                padding_factor,
            )

            if crop is None:
                continue

            crops.append(crop)
            crop_metadata.append(box_index)

    if not crops:
        return {}

    results = classifier.predict(
        source=crops,
        imgsz=224,
        batch=min(32, len(crops)),
        device="mps",
        verbose=False,
    )

    probability_sums = {}
    probability_counts = {}
    names = None

    for box_index, result in zip(
        crop_metadata,
        results,
    ):
        if result.probs is None:
            continue

        probabilities = (
            result.probs.data
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

        names = result.names

        if box_index not in probability_sums:
            probability_sums[box_index] = np.zeros_like(
                probabilities
            )
            probability_counts[box_index] = 0

        probability_sums[box_index] += probabilities
        probability_counts[box_index] += 1

    classifications = {}

    for box_index, probability_sum in probability_sums.items():
        mean_probabilities = (
            probability_sum
            / probability_counts[box_index]
        )

        class_id = int(np.argmax(mean_probabilities))
        confidence = float(mean_probabilities[class_id])
        class_name = names[class_id]

        classifications[box_index] = (
            class_name,
            confidence,
        )

    return classifications


def detect_objects(image, request):
    detector = get_objectness_model()

    results = detector.predict(
        source=image,
        imgsz=960,
        conf=0.025,
        iou=0.50,
        max_det=80,
        device="mps",
        verbose=False,
    )

    if not results:
        return []

    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        return []

    boxes = result.boxes.xyxy.detach().cpu().numpy()
    objectness_scores = (
        result.boxes.conf.detach().cpu().numpy()
    )

    classifications = classify_boxes(
        image,
        boxes,
    )

    height, width = image.shape[:2]
    annotations = []

    for box_index, (box, objectness) in enumerate(
        zip(boxes, objectness_scores)
    ):
        classification = classifications.get(box_index)

        if classification is None:
            continue

        object_id, class_confidence = classification

        x1, y1, x2, y2 = map(float, box)

        if x2 <= x1 or y2 <= y1:
            continue

        view_bbox = (
            x1 / width,
            y1 / height,
            x2 / width,
            y2 / height,
        )

        global_bbox = view_bbox_to_global(
            view_bbox,
            request.view.source_region_xyxy,
            request.original_width,
            request.original_height,
        )

        global_bbox = clip_bbox_to_frame(global_bbox)

        if global_bbox is None:
            continue

        combined_confidence = float(
            np.sqrt(
                max(0.0, float(objectness))
                * max(0.0, class_confidence)
            )
        )

        if combined_confidence < 0.04:
            continue

        annotations.append(
            DroneFlybyPredictionDto(
                object_id=object_id,
                bbox=list(global_bbox),
                confidence=min(
                    1.0,
                    combined_confidence,
                ),
            )
        )

    return annotations


def warmup_model():
    dummy_image = np.zeros(
        (540, 960, 3),
        dtype=np.uint8,
    )

    detector = get_objectness_model()

    detector.predict(
        source=dummy_image,
        imgsz=960,
        conf=0.025,
        iou=0.50,
        max_det=80,
        device="mps",
        verbose=False,
    )

    classifier = get_classifier_model()

    dummy_crops = [
        np.zeros((224, 224, 3), dtype=np.uint8),
        np.zeros((224, 224, 3), dtype=np.uint8),
    ]

    classifier.predict(
        source=dummy_crops,
        imgsz=224,
        batch=2,
        device="mps",
        verbose=False,
    )
