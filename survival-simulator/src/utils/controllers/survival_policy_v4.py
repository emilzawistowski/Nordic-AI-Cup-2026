import math
import random
from src.utils.DTOs import ActionRequest

TAU = 2.0 * math.pi


def normalize_angle(angle: float) -> float:
    """Normalize an angle to [-pi, pi]."""
    return (angle + math.pi) % TAU - math.pi


def weighted_escape_direction(predators: list[dict]) -> float:
    """
    Calculate escape direction away from all observed predators.
    Angles are relative to current agent heading.
    """
    escape_x = 0.0
    escape_y = 0.0

    for predator in predators:
        distance = max(float(predator.get("distance", 1.0)), 1.0)
        predator_angle = float(predator.get("angle", 0.0))

        escape_angle = normalize_angle(predator_angle + math.pi)
        # Strongly weight nearby predators
        weight = 1000.0 / (distance * distance + 1.0)

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

    nearest_fruit = nearest_observation(observations, "Fruit")
    nearest_tree = nearest_observation(observations, "Tree")

    energy_ratio = energy / max(max_energy, 1.0)

    move_distance = 0.0
    move_direction = 0.0
    turn_angle = 0.0
    spawn_agent = False

    # 1. Highest Priority: Escape Predators
    if predators:
        nearest_pred_dist = min(float(p["distance"]) for p in predators)
        move_direction = weighted_escape_direction(predators)

        # Wall avoidance while escaping: check if escape direction points into a very close edge
        for edge_obs in edges:
            coords = edge_obs.get("coords")
            if coords and len(coords) == 2:
                (x1, y1), (x2, y2) = coords
                # Edge center relative to agent
                mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                edge_dist = math.hypot(mx, my)
                if edge_dist < 30.0:
                    edge_angle = math.atan2(my, mx)
                    # Adjust move_direction away from wall
                    diff = normalize_angle(move_direction - edge_angle)
                    if abs(diff) < math.pi / 2.0:
                        move_direction = normalize_angle(move_direction + math.copysign(math.pi / 3.0, diff))

        # Sprinting decision: sprint only if predator is within dangerous distance and energy is sufficient
        sprint_threshold = max(35.0, 2.5 * sprint_speed)
        if nearest_pred_dist <= sprint_threshold and energy > 25.0:
            move_distance = sprint_speed
        else:
            move_distance = speed

        # Look slightly toward predator to keep it in vision cone while running away
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

    # 3. Search near Trees for newly spawned fruits
    elif nearest_tree is not None and energy_ratio < 0.80:
        tree_dist = float(nearest_tree["distance"])
        tree_angle = normalize_angle(float(nearest_tree["angle"]))

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

    # 5. Smart Reproduction Strategy
    # Spawning cost is 100 energy. Child starts with 75 energy.
    # Spawn whenever energy >= 170 and no predators in sight!
    if not predators and energy >= 170.0 and age >= 15.0:
        spawn_agent = True
        move_distance = min(move_distance, speed * 0.30)

    return ActionRequest(
        agent_id=agent_id,
        move_distance=float(max(0.0, move_distance)),
        move_direction=float(normalize_angle(move_direction)),
        turn_angle=float(normalize_angle(turn_angle)),
        spawn_agent=spawn_agent,
    )
