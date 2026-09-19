import math

from dtos import MAXIMUM_CENTER_DELTA_PIXELS, RequestedViewDto


_route_state = {}


def _clamp(value, minimum, maximum):
    return int(min(max(value, minimum), maximum))


def _make_targets(bounds):
    columns = 7
    rows = 4

    xs = [
        int(round(
            bounds.minimum_center_x
            + index
            * (bounds.maximum_center_x - bounds.minimum_center_x)
            / (columns - 1)
        ))
        for index in range(columns)
    ]

    ys = [
        int(round(
            bounds.minimum_center_y
            + index
            * (bounds.maximum_center_y - bounds.minimum_center_y)
            / (rows - 1)
        ))
        for index in range(rows)
    ]

    targets = []

    for row_index, center_y in enumerate(ys):
        row_xs = xs if row_index % 2 == 0 else list(reversed(xs))

        for center_x in row_xs:
            targets.append((center_x, center_y))

    return targets


def choose_next_view_safe(request):
    constraints = request.camera_constraints
    current = request.view
    sequence_id = request.sequence_id

    if request.frame_index == 0:
        _route_state[sequence_id] = 0

    if current.resolution_level == 0:
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
        bounds = constraints.bounds_for_level(2)

        if bounds is None:
            return None

        return RequestedViewDto(
            resolution_level=2,
            center_x=_clamp(
                current.center_x,
                bounds.minimum_center_x,
                bounds.maximum_center_x,
            ),
            center_y=_clamp(
                current.center_y,
                bounds.minimum_center_y,
                bounds.maximum_center_y,
            ),
        )

    bounds = constraints.bounds_for_level(2)

    if bounds is None:
        return None

    targets = _make_targets(bounds)
    target_index = _route_state.get(sequence_id, 0) % len(targets)
    target_x, target_y = targets[target_index]

    delta_x = target_x - current.center_x
    delta_y = target_y - current.center_y
    distance = math.hypot(delta_x, delta_y)

    if distance <= 8.0:
        target_index = (target_index + 1) % len(targets)
        _route_state[sequence_id] = target_index

        target_x, target_y = targets[target_index]
        delta_x = target_x - current.center_x
        delta_y = target_y - current.center_y
        distance = math.hypot(delta_x, delta_y)

    limit = float(
        constraints.maximum_center_delta
        or MAXIMUM_CENTER_DELTA_PIXELS[current.resolution_level]
    )

    safe_limit = max(1.0, limit * 0.88)

    if distance > safe_limit:
        scale = safe_limit / distance
        command_x = int(round(current.center_x + delta_x * scale))
        command_y = int(round(current.center_y + delta_y * scale))
    else:
        command_x = int(target_x)
        command_y = int(target_y)

    command_x = _clamp(
        command_x,
        bounds.minimum_center_x,
        bounds.maximum_center_x,
    )
    command_y = _clamp(
        command_y,
        bounds.minimum_center_y,
        bounds.maximum_center_y,
    )

    return RequestedViewDto(
        resolution_level=2,
        center_x=command_x,
        center_y=command_y,
    )
