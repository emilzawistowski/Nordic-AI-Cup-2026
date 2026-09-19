import math
import random
from src.utils.DTOs import ActionRequest

TAU = 2.0 * math.pi

# Module-level tracking dict for trajectory extrapolation (Mechanism 1)
# Key: agent_id (int) -> Value: list of relative positions [(x, y), ...] from previous tick
AGENT_PREDATOR_HISTORY: dict[int, list[tuple[float, float]]] = {}


def normalize_angle(angle: float) -> float:
    """Normalize an angle to [-pi, pi]."""
    return (angle + math.pi) % TAU - math.pi


def weighted_escape_direction(predators: list[dict]) -> float:
    """
    Calculate weighted escape direction away from all observed predators.
    Angles are relative to agent heading.
    """
    escape_x = 0.0
    escape_y = 0.0

    for predator in predators:
        distance = max(float(predator.get("distance", 1.0)), 1.0)
        predator_angle = float(predator.get("angle", 0.0))

        escape_angle = normalize_angle(predator_angle + math.pi)
        weight = 6000.0 / (distance * distance + 1.0)

        escape_x += math.cos(escape_angle) * weight
        escape_y += math.sin(escape_angle) * weight

    if abs(escape_x) < 1e-12 and abs(escape_y) < 1e-12:
        return math.pi

    return normalize_angle(math.atan2(escape_y, escape_x))


def nearest_observation(observations: list[dict], object_type: str):
    candidates = [
        o for o in observations
        if o.get("type") == object_type and "distance" in o and "angle" in o
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda o: float(o["distance"]))


def predict_predator_positions(agent_id: int, predators: list[dict]) -> list[dict]:
    """
    Mechanism 1: Predictive Predator Avoidance (trajectory extrapolation).
    Converts relative polar coords to relative Cartesian coords, estimates relative velocity
    from previous tick, and extrapolates 1.5 ticks ahead.
    Falls back to current position when no history exists for a predator (first sighting).
    """
    if not predators:
        AGENT_PREDATOR_HISTORY.pop(agent_id, None)
        return []

    # Convert current predator observations to relative Cartesian (x, y)
    curr_coords: list[tuple[float, float]] = []
    for p in predators:
        d = float(p["distance"])
        a = float(p["angle"])
        px = d * math.cos(a)
        py = d * math.sin(a)
        curr_coords.append((px, py))

    prev_coords = AGENT_PREDATOR_HISTORY.get(agent_id, [])

    predicted_predators: list[dict] = []

    for idx, (px, py) in enumerate(curr_coords):
        orig_p = predators[idx]
        best_match = None
        min_dist = float("inf")

        # Find closest predator from previous tick within a max movement threshold (40 units)
        for prev_x, prev_y in prev_coords:
            dist = math.hypot(px - prev_x, py - prev_y)
            if dist < min_dist and dist < 40.0:
                min_dist = dist
                best_match = (prev_x, prev_y)

        if best_match is not None:
            # Trajectory extrapolation: estimate velocity vector (dx, dy) between ticks
            dx = px - best_match[0]
            dy = py - best_match[1]
            # Extrapolate 1.5 ticks ahead
            pred_x = px + 1.5 * dx
            pred_y = py + 1.5 * dy

            pred_dist = max(math.hypot(pred_x, pred_y), 1.0)
            pred_angle = normalize_angle(math.atan2(pred_y, pred_x))
            predicted_predators.append({"distance": pred_dist, "angle": pred_angle})
        else:
            # First sighting fallback: use current observed position
            predicted_predators.append({
                "distance": float(orig_p["distance"]),
                "angle": float(orig_p["angle"])
            })

    # Save current positions for next tick comparison
    AGENT_PREDATOR_HISTORY[agent_id] = curr_coords
    return predicted_predators


def action_decision(observation_response: dict, rng: random.Random) -> ActionRequest:
    agent_id = int(observation_response["agent_id"])
    observations = observation_response.get("observations", [])

    energy = float(observation_response["energy"])
    age = float(observation_response["age"])
    speed = float(observation_response["speed"])
    sprint_speed = float(observation_response["sprint_speed"])
    hearing_radius = float(observation_response["hearing_radius"])
    max_energy = float(observation_response["max_energy"])

    predators = [
        o for o in observations
        if o.get("type") == "Predator" and "distance" in o and "angle" in o
    ]
    edges = [
        o for o in observations
        if o.get("type") == "Edge" and "coords" in o
    ]
    trees = [
        o for o in observations
        if o.get("type") == "Tree" and "distance" in o and "angle" in o
    ]

    nearest_fruit = nearest_observation(observations, "Fruit")
    nearest_tree = nearest_observation(observations, "Tree")

    energy_ratio = energy / max(max_energy, 1.0)

    move_distance = 0.0
    move_direction = 0.0
    turn_angle = 0.0
    spawn_agent = False

    # 1. Highest Priority: Predictive Predator Avoidance (Mechanism 1)
    if predators:
        # Trajectory extrapolation for visible predators
        predicted_predators = predict_predator_positions(agent_id, predators)
        move_direction = weighted_escape_direction(predicted_predators)

        # Wall-bounce check: deflect angle when near wall
        for edge_obs in edges:
            coords = edge_obs.get("coords")
            if coords and len(coords) == 2:
                (x1, y1), (x2, y2) = coords
                mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                edge_dist = math.hypot(mx, my)
                if edge_dist < 35.0:
                    edge_angle = math.atan2(my, mx)
                    diff = normalize_angle(move_direction - edge_angle)
                    if abs(diff) < math.pi / 2.0:
                        move_direction = normalize_angle(move_direction + math.copysign(math.pi / 2.5, diff))

        # Sprinting decision safety net: sprint whenever actual predator is within 55 units and energy > 20
        nearest_pred_dist = min(float(p["distance"]) for p in predators)
        if nearest_pred_dist <= 55.0 and energy > 20.0:
            move_distance = sprint_speed
        else:
            move_distance = speed

        nearest_pred = min(predators, key=lambda p: float(p["distance"]))
        pred_angle = float(nearest_pred["angle"])
        turn_angle = max(-0.15, min(0.15, pred_angle))

    # 2. Fruit Harvesting
    elif nearest_fruit is not None:
        fruit_dist = float(nearest_fruit["distance"])
        fruit_angle = normalize_angle(float(nearest_fruit["angle"]))

        move_direction = fruit_angle
        move_distance = min(speed, fruit_dist)
        turn_angle = max(-0.10, min(0.10, fruit_angle))

    # 3. Zone-Based Foraging Allocation (Mechanism 2)
    elif trees and (energy_ratio < 0.85 or age > 60.0):
        if len(trees) > 1:
            # Deterministically order trees by angle/distance and pick using agent_id modulo num_trees
            sorted_trees = sorted(trees, key=lambda t: (float(t["angle"]), float(t["distance"])))
            chosen_tree = sorted_trees[agent_id % len(sorted_trees)]
        else:
            chosen_tree = nearest_tree

        tree_dist = float(chosen_tree["distance"])
        tree_angle = normalize_angle(float(chosen_tree["angle"]))

        approach_offset = 0.35 if agent_id % 2 == 0 else -0.35
        move_direction = normalize_angle(tree_angle + approach_offset)

        if tree_dist > 35.0:
            move_distance = speed * 0.90
        elif tree_dist > 18.0:
            move_distance = speed * 0.50
        else:
            move_distance = speed * 0.30
            move_direction = normalize_angle(tree_angle + math.pi / 2.0)

        turn_angle = max(-0.08, min(0.08, tree_angle))

    # 4. Economical Wandering
    else:
        phase = agent_id * 1.61803398875
        slow_wave = math.sin(age * 0.055 + phase)
        faster_wave = math.sin(age * 0.017 + phase * 0.7)
        move_direction = 0.50 * slow_wave + 0.20 * faster_wave

        if energy_ratio < 0.35:
            move_distance = speed * 0.90
        elif energy_ratio < 0.75:
            move_distance = speed * 0.65
        else:
            move_distance = speed * 0.40

        turn_angle = 0.035 * math.sin(age * 0.08 + phase)

    # 5. Reproduction Strategy (Mechanism 3 Note)
    # Mechanism 3 (Time-Aware Reproduction Cutoff) is skipped here because
    # ObservationResponse / observation_response dict does not expose simulation time
    # or remaining time (sim_time is only in StepResponse).
    # Standard reproduction logic: Cost: 100. Spawn when energy >= 185.0 and age >= 10.0
    if not predators and energy >= 185.0 and age >= 10.0:
        spawn_agent = True
        move_distance = min(move_distance, speed * 0.25)

    return ActionRequest(
        agent_id=agent_id,
        move_distance=float(max(0.0, move_distance)),
        move_direction=float(normalize_angle(move_direction)),
        turn_angle=float(normalize_angle(turn_angle)),
        spawn_agent=spawn_agent,
    )
