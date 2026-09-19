"""Causal 2D acquisition with overlap-preserving, uncertainty-aware revisits.

The policy only sees request geometry and live V4 tracks. The coverage map is
registered with the same measured scene transform as the tracker; it contains
no labels, scene IDs, or prerecorded object locations.
"""
from collections import OrderedDict
from dataclasses import dataclass, field
import math

import cv2
import numpy as np

from dtos import RequestedViewDto


@dataclass
class CameraState:
    index: int = -1
    shape: tuple = ()
    coverage: np.ndarray = field(default_factory=lambda: np.zeros((24, 40), np.float32))
    direction: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0]))


class AdaptiveCamera:
    def __init__(self, max_sequences=8):
        self.states = OrderedDict()
        self.max_sequences = max_sequences

    def choose(self, request, tracks=(), motion=None):
        current, constraints = request.view, request.camera_constraints
        shape = (request.original_width, request.original_height)
        key, index = request.sequence_id, request.frame_index
        state = self.states.get(key)
        if state is None or index <= state.index or shape != state.shape:
            state = CameraState(shape=shape)
        self.states[key] = state
        self.states.move_to_end(key)
        while len(self.states) > self.max_sequences:
            self.states.popitem(last=False)
        gap = max(1, index - state.index)
        scale = np.array([40 / shape[0], 24 / shape[1]])
        if state.index >= 0 and motion is not None:
            matrix = motion[0].copy()
            matrix[:, :2] = np.diag(scale) @ matrix[:, :2] @ np.diag(1 / scale)
            matrix[:, 2] *= scale
            state.coverage = cv2.warpAffine(state.coverage, matrix, (40, 24))
        elif state.index >= 0:
            # Registration failed: reduce trust in previous coverage, just as
            # the tracker discards historical boxes on failed registration.
            state.coverage *= 0.25
        state.coverage *= 0.97 ** gap
        state.index = index
        xs = (np.arange(40) + 0.5) / scale[0]
        ys = (np.arange(24) + 0.5) / scale[1]
        xx, yy = np.meshgrid(xs, ys)
        x1, y1, x2, y2 = current.source_region_xyxy
        seen = (xx >= x1) & (xx < x2) & (yy >= y1) & (yy < y2)
        quality = (0.08, 0.35, 1.0)[current.resolution_level]
        state.coverage[seen] = np.maximum(state.coverage[seen], quality)

        allowed = constraints.allowed_resolution_levels
        level = min(2, current.resolution_level + 1)
        if level not in allowed:
            level = current.resolution_level
        bounds = constraints.bounds_for_level(level)
        if bounds is None or level not in allowed:
            return None
        center = np.array([current.center_x, current.center_y], dtype=float)
        minimum = np.array([bounds.minimum_center_x, bounds.minimum_center_y])
        maximum = np.array([bounds.maximum_center_x, bounds.maximum_center_y])
        limit = max(0.0, float(constraints.maximum_center_delta))
        if level != current.resolution_level:
            # Center-preserving zoom maximizes pixel correspondence across
            # scale changes. Never assume a level transition exempts motion.
            target = np.clip(center, minimum, maximum).astype(int)
            if np.linalg.norm(target - center) > limit:
                return None
            return RequestedViewDto(resolution_level=int(level), center_x=int(target[0]), center_y=int(target[1]))

        width, height = bounds.width, bounds.height
        candidates = [np.clip(center, minimum, maximum)]
        for angle in np.linspace(0, 2 * math.pi, 16, endpoint=False):
            delta = np.array([math.cos(angle) * width, math.sin(angle) * height])
            # Product of residual width and height is >=0.55, even diagonally.
            delta *= 0.32
            distance = np.linalg.norm(delta)
            if distance > limit * 0.88 and distance > 0:
                delta *= limit * 0.88 / distance
            candidates.append(np.clip(center + delta, minimum, maximum))

        best, best_score = None, -float('inf')
        for raw in candidates:
            target = np.rint(raw).astype(int)
            delta = target - center
            distance = float(np.linalg.norm(delta))
            if distance > limit:
                continue
            mask = (abs(xx - target[0]) < width / 2) & (abs(yy - target[1]) < height / 2)
            if not mask.any():
                continue
            novelty = float(np.mean(1.0 - state.coverage[mask]))
            demand = 0.0
            for track in tracks:
                box = track.box
                mid = (box[:2] + box[2:]) / 2
                if abs(mid[0] - target[0]) > width * 0.40 or abs(mid[1] - target[1]) > height * 0.40:
                    continue
                size = max(1.0, float(min(box[2:] - box[:2])))
                urgency = min(1.0, max((index - track.observed) / 18.0, track.uncertainty / (0.20 * size)))
                demand += float(track.confidence) * urgency
            continuity = float(np.dot(delta / max(distance, 1), state.direction))
            utility = novelty + 0.20 * min(demand, 2.0) + 0.025 * continuity
            if utility > best_score:
                best, best_score = target, utility
        if best is None:
            return None
        displacement = best - center
        if np.linalg.norm(displacement) > 1:
            state.direction = displacement / np.linalg.norm(displacement)
        return RequestedViewDto(resolution_level=int(level), center_x=int(best[0]), center_y=int(best[1]))
