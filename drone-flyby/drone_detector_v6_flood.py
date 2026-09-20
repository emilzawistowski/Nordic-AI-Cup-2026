"""V6 recall-tail flood detector (inference-only, no training).

V4 = drone_detector_v2 (models/drone_yolo_v2_augmented.pt, conf 0.05)
     -> TemporalTracker (drops confidence < 0.025).

This module reuses the exact same weights/device and additionally surfaces
the detections V4 discards (sub-0.05 boxes, plus alternative class labels on
the same boxes via multi-label NMS), emitted with confidence strictly BELOW
every confidence V4 can emit. Under standard COCO ranking this cannot lower
any class's AP; it can only add true positives at the tail.

Design (detect_two_tier):
  Pass 1: model.predict(imgsz=960, conf=0.001, iou=0.55, max_det=300,
          device="mps", verbose=False), single-label, class-aware NMS.
    primary  = raw conf >= 0.05  (must be IDENTICAL to
               drone_detector_v2.detect_objects output)
    extras_A = raw conf < 0.05
  Pass 2 (variant B only): same call with multi-label NMS enabled by
    temporarily patching the exact call site used by the detect predictor
    (ultralytics.utils.nms.non_max_suppression, called from
    ultralytics.models.yolo.detect.predict.DetectionPredictor.postprocess
    in installed ultralytics 8.4.155) with
    functools.partial(orig, multi_label=True) inside a context manager,
    restored afterwards.
    extras_B = every pass-2 detection not already in primary
               (same label and IoU > 0.55 with a primary box).

Extra confidence mapping (monotone, strictly below the tracker's 0.025 floor):
    extra_conf = 0.001 + 0.0189 * raw_conf   -> always in [0.001, 0.0199]
"""

import contextlib
import functools
import math

from dtos import DroneFlybyPredictionDto
from utils import clip_bbox_to_frame, view_bbox_to_global

# Reuse the exact model loader (same weights file, same lazy singleton).
from drone_detector_v2 import MODEL_PATH, get_model

PRIMARY_CONF_THRESHOLD = 0.05
FLOOD_CONF_THRESHOLD = 0.001
FLOOD_IOU = 0.55
FLOOD_MAX_DET = 300
FLOOD_DEVICE = "mps"
FLOOD_IMGSZ = 960

# Tracker floor in drone_tracker_v4.TemporalTracker.update: annotations with
# confidence < 0.025 are dropped, so 0.025 is the minimum confidence V4 emits.
_TRACKER_FLOOR = 0.025


def extra_confidence(raw_conf):
    """Map a raw sub-threshold confidence into the recall-tail band.

    Monotone in raw_conf; always in [0.001, 0.0199], strictly below the
    tracker's 0.025 floor (and therefore below every V4 annotation).
    """
    mapped = 0.001 + 0.0189 * float(raw_conf)
    if not math.isfinite(mapped):
        return 0.001
    return min(max(mapped, 0.001), 0.0199)


@contextlib.contextmanager
def _multilabel_nms():
    """Enable multi-label NMS for one predict call, then restore.

    Patches the exact NMS call site used by the detect predictor
    (ultralytics.utils.nms.non_max_suppression) with
    functools.partial(orig, multi_label=True).
    """
    import ultralytics.utils.nms as nms_module

    orig = nms_module.non_max_suppression
    nms_module.non_max_suppression = functools.partial(orig, multi_label=True)
    try:
        yield
    finally:
        nms_module.non_max_suppression = orig


def _raw_predict(image, multi_label=False):
    """Run one low-threshold predict; return (boxes, confidences, class_ids, names)."""
    model = get_model()
    if multi_label:
        with _multilabel_nms():
            results = model.predict(
                source=image,
                imgsz=FLOOD_IMGSZ,
                conf=FLOOD_CONF_THRESHOLD,
                iou=FLOOD_IOU,
                max_det=FLOOD_MAX_DET,
                device=FLOOD_DEVICE,
                verbose=False,
            )
    else:
        results = model.predict(
            source=image,
            imgsz=FLOOD_IMGSZ,
            conf=FLOOD_CONF_THRESHOLD,
            iou=FLOOD_IOU,
            max_det=FLOOD_MAX_DET,
            device=FLOOD_DEVICE,
            verbose=False,
        )
    if not results:
        return [], [], [], {}
    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return [], [], [], result.names
    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy()
    return boxes, confidences, class_ids, result.names


def _to_global(image, request, box):
    """Convert a view-pixel xyxy box to frame-global normalized coords.

    Returns None for zero-area/invalid/NaN boxes. Mirrors the conversion in
    drone_detector_v2 (view_bbox_to_global + clip_bbox_to_frame) with an
    extra finite-value guard so we never emit NaN.
    """
    x1, y1, x2, y2 = (float(c) for c in box)
    if not all(math.isfinite(c) for c in (x1, y1, x2, y2)):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    height, width = image.shape[:2]
    view_bbox = (x1 / width, y1 / height, x2 / width, y2 / height)
    global_bbox = view_bbox_to_global(
        view_bbox,
        request.view.source_region_xyxy,
        request.original_width,
        request.original_height,
    )
    global_bbox = clip_bbox_to_frame(global_bbox)
    if global_bbox is None:
        return None
    gx1, gy1, gx2, gy2 = (float(c) for c in global_bbox)
    if not all(math.isfinite(c) for c in (gx1, gy1, gx2, gy2)):
        return None
    if not (0 <= gx1 < gx2 <= 1 and 0 <= gy1 < gy2 <= 1):
        return None
    return (gx1, gy1, gx2, gy2)


def _iou(a, b):
    """IoU of two [x1, y1, x2, y2] boxes (any consistent scale)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def detect_two_tier(image, request, include_multilabel=False):
    """Split one low-threshold inference into (primary, extras).

    primary: raw conf >= 0.05, identical to drone_detector_v2.detect_objects.
    extras: sub-0.05 boxes (variant A) plus, when include_multilabel is True,
      every multi-label-NMS detection not already in primary (variant B),
      each with confidence mapped strictly below the tracker floor.
    Both lists contain DroneFlybyPredictionDto in frame-global normalized
    coordinates. Never emits NaN or zero-area boxes.
    """
    boxes, confidences, class_ids, names = _raw_predict(image, multi_label=False)

    primary = []
    extras = []
    primary_keys = []  # (label, global_bbox) for variant-B dedupe
    for box, raw_conf, class_id in zip(boxes, confidences, class_ids):
        raw_conf = float(raw_conf)
        if not math.isfinite(raw_conf):
            continue
        global_bbox = _to_global(image, request, box)
        if global_bbox is None:
            continue
        label = names[int(class_id)]
        if raw_conf >= PRIMARY_CONF_THRESHOLD:
            primary.append(
                DroneFlybyPredictionDto(
                    object_id=label,
                    bbox=list(global_bbox),
                    confidence=raw_conf,
                )
            )
            primary_keys.append((label, global_bbox))
        elif raw_conf >= FLOOD_CONF_THRESHOLD:
            extras.append(
                DroneFlybyPredictionDto(
                    object_id=label,
                    bbox=list(global_bbox),
                    confidence=extra_confidence(raw_conf),
                )
            )

    if include_multilabel:
        try:
            boxes_b, confs_b, ids_b, names_b = _raw_predict(image, multi_label=True)
        except Exception:
            boxes_b, confs_b, ids_b, names_b = [], [], [], {}
        for box, raw_conf, class_id in zip(boxes_b, confs_b, ids_b):
            raw_conf = float(raw_conf)
            if not math.isfinite(raw_conf) or raw_conf < FLOOD_CONF_THRESHOLD:
                continue
            global_bbox = _to_global(image, request, box)
            if global_bbox is None:
                continue
            label = names_b[int(class_id)]
            duplicate = any(
                plabel == label and _iou(global_bbox, pbox) > 0.55
                for plabel, pbox in primary_keys
            )
            if duplicate:
                continue
            extras.append(
                DroneFlybyPredictionDto(
                    object_id=label,
                    bbox=list(global_bbox),
                    confidence=extra_confidence(raw_conf),
                )
            )

    return primary, extras
