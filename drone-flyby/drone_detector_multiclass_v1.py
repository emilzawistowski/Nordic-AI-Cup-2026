from pathlib import Path

from ultralytics import YOLO

from dtos import DroneFlybyPredictionDto
from utils import clip_bbox_to_frame, view_bbox_to_global


MODEL_PATH = Path(__file__).parent / "models" / "drone_yolo_v1.pt"

_model = None


def get_model():
    global _model

    if _model is None:
        _model = YOLO(str(MODEL_PATH))

    return _model


def detect_objects(image, request):
    model = get_model()

    results = model.predict(
        source=image,
        imgsz=960,
        conf=0.05,
        iou=0.55,
        max_det=100,
        device="mps",
        verbose=False,
    )

    annotations = []

    if not results:
        return annotations

    result = results[0]
    height, width = image.shape[:2]

    if result.boxes is None:
        return annotations

    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy()

    for box, confidence, class_id in zip(
        boxes,
        confidences,
        class_ids,
    ):
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

        object_id = result.names[int(class_id)]

        annotations.append(
            DroneFlybyPredictionDto(
                object_id=object_id,
                bbox=list(global_bbox),
                confidence=float(confidence),
            )
        )

    return annotations


def warmup_model():
    import numpy as np

    model = get_model()
    dummy_image = np.zeros((540, 960, 3), dtype=np.uint8)

    model.predict(
        source=dummy_image,
        imgsz=960,
        conf=0.05,
        iou=0.55,
        max_det=100,
        device="mps",
        verbose=False,
    )
