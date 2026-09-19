import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from dtos import (
    MAXIMUM_CENTER_DELTA_PIXELS,
    DroneFlybyPredictionDto,
    DroneFlybyPredictRequestDto,
    DroneFlybyPredictResponseDto,
    RequestedViewDto,
)
from utils import clip_bbox_to_frame, decode_view, view_bbox_to_global
from drone_detector_v2 import detect_objects

logger = logging.getLogger(__name__)


def predict(request: DroneFlybyPredictRequestDto) -> DroneFlybyPredictResponseDto:
    """Answer one frame: report detections and pick the next camera position."""
    if request.camera_command_feedback is not None:
        feedback = request.camera_command_feedback
        logger.warning(
            'Camera command from frame %s was ignored: %s',
            feedback.frame,
            feedback.reason,
        )

    logger.info(
        "camera frame=%s level=%s center=(%s,%s) region=%s",
        request.frame,
        request.view.resolution_level,
        request.view.center_x,
        request.view.center_y,
        request.view.source_region_xyxy,
    )

    image = decode_view(request.view)

    try:
        annotations = detect_objects(image, request)
    except Exception:
        logger.exception('Detector failed on frame %s', request.frame)
        annotations = []

    return DroneFlybyPredictResponseDto(
        request_id=request.request_id,
        frame=request.frame,
        annotations=annotations,
        requested_view=choose_next_view(request),
    )


_sweep_direction: Dict[str, int] = {}


def choose_next_view(
    request: DroneFlybyPredictRequestDto,
) -> Optional[RequestedViewDto]:
    constraints = request.camera_constraints
    current = request.view
    allowed = [level for level in constraints.allowed_resolution_levels if level > 0]
    if not allowed:
        return None

    target_level = min(max(allowed), current.resolution_level + 1)
    bounds = constraints.bounds_for_level(target_level)
    if bounds is None:
        return None

    if current.resolution_level == 0:
        centre_x = (bounds.minimum_center_x + bounds.maximum_center_x) // 2
        centre_y = (bounds.minimum_center_y + bounds.maximum_center_y) // 2
        return RequestedViewDto(
            resolution_level=target_level,
            center_x=int(centre_x),
            center_y=int(centre_y),
        )

    direction = _sweep_direction.setdefault(request.sequence_id, 1)

    limit = constraints.maximum_center_delta or MAXIMUM_CENTER_DELTA_PIXELS[
        current.resolution_level
    ]
    step = int(limit * 0.9)

    centre_x = current.center_x + direction * step
    if centre_x > bounds.maximum_center_x or centre_x < bounds.minimum_center_x:
        direction = -direction
        _sweep_direction[request.sequence_id] = direction
        centre_x = current.center_x + direction * step

    centre_y = current.center_y

    centre_x = int(min(max(centre_x, bounds.minimum_center_x), bounds.maximum_center_x))
    centre_y = int(min(max(centre_y, bounds.minimum_center_y), bounds.maximum_center_y))

    return RequestedViewDto(
        resolution_level=target_level,
        center_x=centre_x,
        center_y=centre_y,
    )
