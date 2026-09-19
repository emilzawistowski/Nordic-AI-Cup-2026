"""Causal, crop-aware track memory. Uses received pixels only, never scene labels."""
from collections import OrderedDict
from dataclasses import dataclass, field

import cv2
import numpy as np

from dtos import DroneFlybyPredictionDto


def iou(a, b):
    size = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))
    intersection = float(np.prod(size))
    union = float(np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]) - intersection)
    return intersection / max(union, 1e-9)


def warp_box(box, matrix):
    x1, y1, x2, y2 = box
    corners = np.array([[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]])
    points = corners @ matrix.T
    return np.concatenate([points.min(axis=0), points.max(axis=0)])


@dataclass
class Track:
    label: str
    box: np.ndarray
    confidence: float
    observed: int
    uncertainty: float = 0.0


@dataclass
class State:
    index: int = -1
    points: object = None
    descriptors: object = None
    tracks: list = field(default_factory=list)
    shape: tuple = ()


class TemporalTracker:
    """Bounded per-sequence state; conservative expiry when registration fails."""

    def __init__(self, max_age=18, max_sequences=8):
        self.max_age = max_age
        self.max_sequences = max_sequences
        self.states = OrderedDict()
        self.orb = cv2.ORB_create(nfeatures=1800, fastThreshold=12)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.last_motion = None

    def features(self, image, region):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        if descriptors is None:
            return np.empty((0, 2)), None
        points = np.array([k.pt for k in keypoints], dtype=np.float64)
        x1, y1, x2, y2 = region
        points *= [(x2 - x1) / image.shape[1], (y2 - y1) / image.shape[0]]
        points += [x1, y1]
        return points, descriptors

    def motion(self, previous, points, descriptors, gap):
        if previous.descriptors is None or descriptors is None or gap > 6:
            return None
        pairs = self.matcher.knnMatch(previous.descriptors, descriptors, k=2)
        matches = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.70 * p[1].distance]
        # One-to-one correspondences avoid repeated textures dominating RANSAC.
        unique = {}
        for match in sorted(matches, key=lambda m: m.distance):
            unique.setdefault(match.trainIdx, match)
        matches = list(unique.values())
        if len(matches) < 16:
            return None
        source = np.array([previous.points[m.queryIdx] for m in matches])
        target = np.array([points[m.trainIdx] for m in matches])
        matrix, inliers = cv2.estimateAffinePartial2D(
            source, target, method=cv2.RANSAC, ransacReprojThreshold=4.0,
            maxIters=1000, confidence=0.995,
        )
        if matrix is None or inliers is None or not np.isfinite(matrix).all():
            return None
        mask = inliers.ravel().astype(bool)
        if mask.sum() < 12 or mask.mean() < 0.45:
            return None
        if np.min(np.ptp(source[mask], axis=0)) < 80:
            return None
        scale = np.linalg.norm(matrix[:, 0])
        if not 0.90 ** gap <= scale <= 1.10 ** gap:
            return None
        residual = np.linalg.norm(source[mask] @ matrix[:, :2].T + matrix[:, 2] - target[mask], axis=1)
        error = float(np.quantile(residual, 0.9))
        if error > 5:
            return None
        return matrix, error

    def update(self, image, request, detections):
        key, index = request.sequence_id, request.frame_index
        shape = (request.original_width, request.original_height)
        state = self.states.get(key)
        if state is None or index <= state.index or state.shape != shape:
            state = State(shape=shape)
        self.states[key] = state
        self.states.move_to_end(key)
        while len(self.states) > self.max_sequences:
            self.states.popitem(last=False)

        points, descriptors = self.features(image, request.view.source_region_xyxy)
        motion = self.motion(state, points, descriptors, index - state.index)
        self.last_motion = motion
        tracks = []
        if motion is not None:
            matrix, error = motion
            for track in state.tracks:
                track.box = warp_box(track.box, matrix)
                track.uncertainty += max(1.0, error)
                size = track.box[2:] - track.box[:2]
                if index - track.observed <= self.max_age and track.uncertainty < 0.20 * min(size):
                    tracks.append(track)
        # No reliable transform: discard memory instead of emitting stale boxes.
        normalizer = np.array([*shape, *shape], dtype=float)
        fresh = sorted(detections, key=lambda d: d.confidence, reverse=True)
        observed = []
        for detection in fresh:
            box = np.array(detection.bbox) * normalizer
            if any(t.label == detection.object_id and iou(t.box, box) > 0.55 for t in observed):
                continue
            # Current pixels take precedence over any overlapping historical label.
            tracks = [t for t in tracks if iou(t.box, box) <= 0.40]
            observed.append(Track(detection.object_id, box, float(detection.confidence), index))

        region = np.array(request.view.source_region_xyxy, dtype=float)
        retained = []
        for track in tracks:
            # A missed object fully inside the current crop has contradictory evidence.
            inside = np.all(track.box[:2] >= region[:2]) and np.all(track.box[2:] <= region[2:])
            if inside and index - track.observed > 2:
                continue
            clipped = np.clip(track.box / normalizer, 0, 1)
            if np.any(clipped[2:] <= clipped[:2]):
                continue
            retained.append(track)

        state.tracks = (observed + retained)[:500]
        state.index, state.points, state.descriptors = index, points, descriptors
        annotations = []
        for track in state.tracks:
            box = np.clip(track.box / normalizer, 0, 1)
            if np.any(box[2:] <= box[:2]):
                continue
            confidence = track.confidence * 0.94 ** (index - track.observed)
            if confidence < 0.025:
                continue
            annotations.append(DroneFlybyPredictionDto(
                object_id=track.label, bbox=box.tolist(), confidence=float(confidence),
            ))
        return annotations
