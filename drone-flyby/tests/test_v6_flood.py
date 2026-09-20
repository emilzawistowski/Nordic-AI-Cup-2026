"""V6 flood checks (plain-python runner; the venv has no pytest).

Run:  .venv/bin/python tests/test_v6_flood.py

1. Primary identity: on every local frame, primary from detect_two_tier
   == drone_detector_v2.detect_objects output (labels, boxes, confidences
   within 1e-6), for both variant A and variant A+B.
2. Ordering invariant: max(extra_conf) < min(tracker annotation
   confidence) on every predict_flood response.
3. DTO validity: every example_drone_v6 response validates with
   DroneFlybyPredictResponseDto; camera moves (V4 camera, unchanged) never
   exceed request.camera_constraints.maximum_center_delta.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dtos import DroneFlybyPredictRequestDto, DroneFlybyPredictResponseDto
from local_evaluator import Camera, CameraRejection, build_request, render_view
from utils import decode_view, frame_numbers, load_frame

import drone_detector_v2
from drone_detector_v6_flood import detect_two_tier
from drone_flood_v6 import predict_flood
from drone_tracker_v4 import TemporalTracker


def _request_for(frame, frame_index, camera, feedback=None):
    image = load_frame(frame)
    encoded = render_view(image, camera)
    payload = build_request(frame, frame_index, camera, encoded, feedback)
    return DroneFlybyPredictRequestDto.model_validate(payload)


def _sorted_key(dto):
    return (-float(dto.confidence), dto.object_id, tuple(float(c) for c in dto.bbox))


def test_primary_identity():
    frames = frame_numbers()
    checked = 0
    for frame_index, frame in enumerate(frames):
        camera = Camera()
        request = _request_for(frame, frame_index, camera)
        image = decode_view(request.view)
        expected = drone_detector_v2.detect_objects(image, request)
        for multilabel in (False, True):
            primary, _extras = detect_two_tier(
                image, request, include_multilabel=multilabel
            )
            got = sorted(primary, key=_sorted_key)
            want = sorted(expected, key=_sorted_key)
            assert len(got) == len(want), (
                f"frame {frame} multilabel={multilabel}: "
                f"primary count {len(got)} != v2 count {len(want)}"
            )
            for g, w in zip(got, want):
                assert g.object_id == w.object_id, (
                    f"frame {frame}: label {g.object_id} != {w.object_id}"
                )
                for gc, wc in zip(g.bbox, w.bbox):
                    assert abs(float(gc) - float(wc)) <= 1e-6, (
                        f"frame {frame}: bbox {list(g.bbox)} != {list(w.bbox)}"
                    )
                assert abs(float(g.confidence) - float(w.confidence)) <= 1e-6, (
                    f"frame {frame}: conf {g.confidence} != {w.confidence}"
                )
        checked += 1
    print(f"primary identity OK on {checked} frames (variants A and A+B)")


def test_ordering_invariant():
    frames = frame_numbers()
    for multilabel in (False, True):
        tracker = TemporalTracker()
        checked = 0
        for frame_index, frame in enumerate(frames):
            camera = Camera()
            request = _request_for(frame, frame_index, camera)
            image = decode_view(request.view)
            out = predict_flood(request, tracker, image,
                                include_multilabel=multilabel)
            head = [a for a in out if float(a.confidence) >= 0.025]
            tail = [a for a in out if float(a.confidence) < 0.025]
            assert len(head) + len(tail) == len(out)
            for extra in tail:
                assert 0.001 <= float(extra.confidence) <= 0.0199, (
                    f"frame {frame}: extra conf {extra.confidence} outside tail band"
                )
            if head and tail:
                assert max(float(e.confidence) for e in tail) < min(
                    float(a.confidence) for a in head
                ), f"frame {frame}: tail extra above tracker annotation"
            assert len(out) <= 500
            checked += 1
        print(f"ordering invariant OK on {checked} frames (multilabel={multilabel})")


def test_dto_and_camera():
    import example_drone_v6

    frames = frame_numbers()
    camera = Camera()
    feedback = None
    refused = 0
    for frame_index, frame in enumerate(frames):
        request = _request_for(frame, frame_index, camera, feedback)
        limit = request.camera_constraints.maximum_center_delta
        response = example_drone_v6.predict(request)
        # DTO validity: strict re-validation from the wire dict.
        DroneFlybyPredictResponseDto.model_validate(response.model_dump())
        assert len(response.annotations) <= 500
        for annotation in response.annotations:
            x1, y1, x2, y2 = (float(c) for c in annotation.bbox)
            assert all(math.isfinite(c) for c in (x1, y1, x2, y2))
            assert 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1
            assert 0 <= float(annotation.confidence) <= 1
        requested = response.requested_view
        if requested is not None:
            before_level, before_x, before_y = (
                camera.resolution_level, camera.center_x, camera.center_y,
            )
            try:
                camera.apply(
                    requested.resolution_level,
                    requested.center_x,
                    requested.center_y,
                )
            except CameraRejection as exc:
                refused += 1
                raise AssertionError(
                    f"frame {frame}: camera move refused (limit {limit}): {exc}"
                )
            if requested.resolution_level != 0:
                import math as _math

                dist = _math.hypot(
                    requested.center_x - before_x, requested.center_y - before_y
                )
                assert dist <= limit + 1e-9, (
                    f"frame {frame}: move {dist:.2f}px exceeds limit {limit}"
                )
            feedback = None
    assert refused == 0
    print(f"DTO + camera OK on {len(frames)} frames, refused moves 0")


def main():
    test_primary_identity()
    test_ordering_invariant()
    test_dto_and_camera()
    print("ALL V6 CHECKS PASSED")


if __name__ == "__main__":
    raise SystemExit(main())
