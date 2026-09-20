"""V6 recall-tail flood combiner (inference-only).

predict_flood(request, tracker, image):
  detect_two_tier -> feed ONLY primary into TemporalTracker.update
  (tracker memory never sees extras) -> tracker annotations (unchanged)
  + extras (current view only, not stored).

Extras dedupe: greedy by descending extra_conf, skip if same label and
IoU > 0.55 with any tracker annotation or any already kept extra.
Different-label duplicates on the same box are intentionally allowed
(class hedging).

Output sorted by confidence descending, capped at 500 total (tracker
annotations always kept first — automatic since every extra_conf is
strictly below the tracker floor).

Extras generation is wrapped in try/except: on any failure, fall back to
plain V4 output.
"""

import math
import os

from drone_detector_v6_flood import detect_two_tier

MAX_ANNOTATIONS = 500
EXTRAS_DEDUPE_IOU = 0.55


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _valid(annotation):
    try:
        x1, y1, x2, y2 = (float(c) for c in annotation.bbox)
        conf = float(annotation.confidence)
    except (TypeError, ValueError):
        return False
    return (
        all(math.isfinite(c) for c in (x1, y1, x2, y2, conf))
        and 0 <= x1 < x2 <= 1
        and 0 <= y1 < y2 <= 1
        and 0 <= conf <= 1
    )


def _variant_multilabel(include_multilabel=None):
    """Resolve variant: explicit flag wins, else V6_MULTILABEL env (default A)."""
    if include_multilabel is not None:
        return bool(include_multilabel)
    return os.environ.get("V6_MULTILABEL", "0").strip().lower() in ("1", "true", "yes", "b", "ab")


def predict_flood(request, tracker, image, include_multilabel=None):
    """V4 tracker annotations + recall-tail extras for one frame."""
    primary, extras = detect_two_tier(
        image, request, include_multilabel=_variant_multilabel(include_multilabel)
    )
    annotations = tracker.update(image, request, primary)
    try:
        kept = []
        ordered = sorted(extras, key=lambda d: float(d.confidence), reverse=True)
        for extra in ordered:
            if not _valid(extra):
                continue
            duplicate = False
            for existing in list(annotations) + kept:
                if existing.object_id != extra.object_id:
                    continue
                if _iou(list(existing.bbox), list(extra.bbox)) > EXTRAS_DEDUPE_IOU:
                    duplicate = True
                    break
            if duplicate:
                continue
            kept.append(extra)
        combined = list(annotations) + kept
        combined.sort(key=lambda d: float(d.confidence), reverse=True)
        return combined[:MAX_ANNOTATIONS]
    except Exception:
        # Any extras failure falls back to plain V4 output.
        return annotations
