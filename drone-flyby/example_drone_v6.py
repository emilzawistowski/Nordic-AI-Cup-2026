"""Attempt V6: V4 tracker + recall-tail flood extras, unchanged camera.

Same structure as example_v4_backup.py (Lock, TemporalTracker, decode_view,
requested_view=choose_next_view imported from example_v2_backup) but the
detector call goes through drone_flood_v6.predict_flood, which feeds ONLY
the V4-identical primary detections into the tracker and appends sub-floor
extras for the current view. Variant A (sub-0.05 boxes) vs A+B (plus
multi-label class hedging) is selected with the V6_MULTILABEL env var
("1" = A+B, default "0" = A); no code change or restart of the logic is
needed, but restart the server between measured runs for clean tracker state.

Exports predict and warmup_model.
"""
from threading import Lock

from dtos import DroneFlybyPredictResponseDto
from drone_detector_v2 import warmup_model as _warmup_v2
from drone_flood_v6 import predict_flood
from drone_tracker_v4 import TemporalTracker
from example_v2_backup import choose_next_view
from utils import decode_view

_tracker = TemporalTracker()
_lock = Lock()


def predict(request):
    with _lock:
        image = decode_view(request.view)
        annotations = predict_flood(request, _tracker, image)
        return DroneFlybyPredictResponseDto(
            request_id=request.request_id, frame=request.frame,
            annotations=annotations, requested_view=choose_next_view(request),
        )


def warmup_model():
    _warmup_v2()
    try:
        import numpy as np

        from drone_detector_v6_flood import get_model

        model = get_model()
        dummy_image = np.zeros((540, 960, 3), dtype=np.uint8)
        # Warm the low-threshold flood path (same weights/device as V2).
        model.predict(
            source=dummy_image,
            imgsz=960,
            conf=0.001,
            iou=0.55,
            max_det=300,
            device="mps",
            verbose=False,
        )
    except Exception:
        pass
