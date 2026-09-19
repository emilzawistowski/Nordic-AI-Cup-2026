"""Attempt 5: V2 detections + motion-compensated memory, unchanged camera."""
from threading import Lock

from dtos import DroneFlybyPredictResponseDto
from drone_detector_v2 import detect_objects, warmup_model
from drone_tracker_v4 import TemporalTracker
from example_v2_backup import choose_next_view
from utils import decode_view

_tracker = TemporalTracker()
_lock = Lock()


def predict(request):
    with _lock:
        image = decode_view(request.view)
        detections = detect_objects(image, request)
        annotations = _tracker.update(image, request, detections)
        return DroneFlybyPredictResponseDto(
            request_id=request.request_id, frame=request.frame,
            annotations=annotations, requested_view=choose_next_view(request),
        )
