import math

from dtos import RequestedViewDto


_sequence_state = {}


def _move_towards(current_x, current_y, target_x, target_y, limit):
    delta_x = target_x - current_x
    delta_y = target_y - current_y
    distance = math.hypot(delta_x, delta_y)

    if distance <= limit or distance == 0:
        return int(target_x), int(target_y), True

    scale = limit / distance

    next_x = current_x + delta_x * scale
    next_y = current_y + delta_y * scale

    return int(round(next_x)), int(round(next_y)), False


def _build_scan_path(bounds):
    x_min = bounds.minimum_center_x
    x_max = bounds.maximum_center_x
    y_min = bounds.minimum_center_y
    y_max = bounds.maximum_center_y

    x_positions = [
        x_min,
        x_min + (x_max - x_min) // 4,
        x_min + (x_max - x_min) // 2,
        x_min + 3 * (x_max - x_min) // 4,
        x_max,
    ]

    y_positions = [
        y_min,
        y_min + (y_max - y_min) // 2,
        y_max,
    ]

    path = []

    for row_index, center_y in enumerate(y_positions):
        row = [
            (int(center_x), int(center_y))
            for center_x in x_positions
        ]

        if row_index % 2 == 1:
            row.reverse()

        path.extend(row)

    return path


def choose_next_view_v2(request):
    constraints = request.camera_constraints
    current = request.view

    allowed = list(constraints.allowed_resolution_levels)

    if current.resolution_level == 0:
        if 1 not in allowed:
            return None

        bounds = constraints.bounds_for_level(1)

        if bounds is None:
            return None

        return RequestedViewDto(
            resolution_level=1,
            center_x=int(
                (bounds.minimum_center_x + bounds.maximum_center_x) // 2
            ),
            center_y=int(
                (bounds.minimum_center_y + bounds.maximum_center_y) // 2
            ),
        )

    if current.resolution_level == 1:
        if 2 not in allowed:
            return None

        bounds = constraints.bounds_for_level(2)

        if bounds is None:
            return None

        return RequestedViewDto(
            resolution_level=2,
            center_x=int(current.center_x),
            center_y=int(current.center_y),
        )

    if current.resolution_level != 2 or 2 not in allowed:
        return None

    bounds = constraints.bounds_for_level(2)

    if bounds is None:
        return None

    path = _build_scan_path(bounds)

    state = _sequence_state.setdefault(
        request.sequence_id,
        {"target_index": 0},
    )

    target_index = state["target_index"] % len(path)
    target_x, target_y = path[target_index]

    movement_limit = max(
        1.0,
        float(constraints.maximum_center_delta) * 0.92,
    )

    next_x, next_y, reached = _move_towards(
        current.center_x,
        current.center_y,
        target_x,
        target_y,
        movement_limit,
    )

    next_x = min(
        max(next_x, bounds.minimum_center_x),
        bounds.maximum_center_x,
    )
    next_y = min(
        max(next_y, bounds.minimum_center_y),
        bounds.maximum_center_y,
    )

    if reached:
        state["target_index"] = (target_index + 1) % len(path)

    return RequestedViewDto(
        resolution_level=2,
        center_x=int(next_x),
        center_y=int(next_y),
    )
