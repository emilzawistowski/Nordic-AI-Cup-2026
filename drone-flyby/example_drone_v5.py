"""Attempt 6 candidate: unchanged V2 detector/V4 tracker, adaptive V5 camera."""
from contextlib import asynccontextmanager
from threading import Lock

from fastapi import FastAPI

from dtos import DroneFlybyPredictRequestDto, DroneFlybyPredictResponseDto
from drone_camera_v5 import AdaptiveCamera
from drone_detector_v2 import detect_objects, warmup_model
from drone_tracker_v4 import TemporalTracker
from utils import decode_view, validate_response

_tracker = TemporalTracker()
_camera = AdaptiveCamera()
_lock = Lock()


def predict(request):
    with _lock:
        image = decode_view(request.view)
        detections = detect_objects(image, request)
        annotations = _tracker.update(image, request, detections)
        state = _tracker.states[request.sequence_id]
        command = _camera.choose(request, state.tracks, _tracker.last_motion)
        return DroneFlybyPredictResponseDto(
            request_id=request.request_id, frame=request.frame,
            annotations=annotations, requested_view=command,
        )


@asynccontextmanager
async def lifespan(app):
    warmup_model()
    yield


app = FastAPI(lifespan=lifespan)


@app.get('/')
def health():
    return {'candidate': 'drone-v5'}


@app.post('/predict', response_model=DroneFlybyPredictResponseDto)
def endpoint(request: DroneFlybyPredictRequestDto):
    response = predict(request)
    validate_response(response)
    return response


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=9055)
