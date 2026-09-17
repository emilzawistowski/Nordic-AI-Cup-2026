import math
import random

from src.utils.DTOs import ActionRequest


TAU = 2.0 * math.pi


def normalize_angle(angle: float) -> float:
    """Normalize an angle to [-pi, pi]."""
    return (angle + math.pi) % TAU - math.pi


def weighted_escape_direction(predators: list[dict]) -> float:
    """
    Calculate one escape direction from all observed predators.

    Nearby predators receive much more weight than distant predators.
    Observation angles are relative to the agent's current direction.
    """
    escape_x = 0.0
    escape_y = 0.0

    for predator in predators:
        distance = max(float(predator.get("distance", 1.0)), 1.0)
        predator_angle = float(predator.get("angle", 0.0))

        escape_angle = normalize_angle(predator_angle + math.pi)
        weight = 1.0 / (distance * distance)

        escape_x += math.cos(escape_angle) * weight
        escape_y += math.sin(escape_angle) * weight

    if abs(escape_x) < 1e-12 and abs(escape_y) < 1e-12:
        return math.pi

    return normalize_angle(math.atan2(escape_y, escape_x))


def nearest_observation(observations: list[dict], object_type: str):
    """Return the nearest observation of a requested type."""
    candidates = [
        observation
        for observation in observations
        if observation.get("type") == object_type
        and "distance" in observation
        and "angle" in observation
    ]

    if not candidates:
        return None

    return min(candidates, key=lambda observation: float(observation["distance"]))


def action_decision(
    observation_response: dict,
    rng: random.Random,
) -> ActionRequest:
    """
    Deterministic, energy-aware survival policy.

    Priority:
    1. Escape predators.
    2. Move towards the nearest fruit.
    3. Search near trees when energy is low.
    4. Explore deterministically and economically.
    """
    agent_id = int(observation_response["agent_id"])
    observations = observation_response.get("observations", [])

    energy = float(observation_response["energy"])
    age = float(observation_response["age"])
    speed = float(observation_response["speed"])
    sprint_speed = float(observation_response["sprint_speed"])
    hearing_radius = float(observation_response["hearing_radius"])
    max_energy = float(observation_response["max_energy"])

    predators = [
        observation
        for observation in observations
        if observation.get("type") == "Predator"
        and "distance" in observation
        and "angle" in observation
    ]

    nearest_fruit = nearest_observation(observations, "Fruit")
    nearest_tree = nearest_observation(observations, "Tree")

    energy_ratio = energy / max(max_energy, 1.0)

    move_distance = 0.0
    move_direction = 0.0
    turn_angle = 0.0
    spawn_agent = False

    # Highest priority: escape predators.
    if predators:
        nearest_predator_distance = min(
            float(predator["distance"]) for predator in predators
        )

        move_direction = weighted_escape_direction(predators)

        immediate_danger = max(
            25.0,
            2.5 * sprint_speed,
            0.30 * hearing_radius,
        )

        moderate_danger = max(
            55.0,
            5.0 * sprint_speed,
            0.70 * hearing_radius,
        )

        if nearest_predator_distance <= immediate_danger and energy_ratio > 0.22:
            move_distance = sprint_speed
        else:
            move_distance = speed

        # Look partly towards the predator while moving away from it.
        # This preserves tracking without paying for a full turn.
        nearest_predator = min(
            predators,
            key=lambda predator: float(predator["distance"]),
        )
        predator_angle = float(nearest_predator["angle"])

        max_turn = 0.10 if nearest_predator_distance <= moderate_danger else 0.05
        turn_angle = max(-max_turn, min(max_turn, predator_angle))

    # Second priority: collect energy.
    elif nearest_fruit is not None:
        fruit_distance = float(nearest_fruit["distance"])
        fruit_angle = normalize_angle(float(nearest_fruit["angle"]))

        move_direction = fruit_angle

        # Never overshoot a nearby fruit.
        move_distance = min(speed, max(0.0, fruit_distance))

        # Rotate slowly towards the target to keep it in the vision cone.
        turn_angle = max(-0.08, min(0.08, fruit_angle))

    # Third priority: search an area where fruit may appear.
    elif nearest_tree is not None and energy_ratio < 0.72:
        tree_distance = float(nearest_tree["distance"])
        tree_angle = normalize_angle(float(nearest_tree["angle"]))

        # Do not repeatedly collide with the centre of the tree.
        approach_offset = 0.30 if agent_id % 2 == 0 else -0.30
        move_direction = normalize_angle(tree_angle + approach_offset)

        if tree_distance > 35.0:
            move_distance = speed * 0.85
        elif tree_distance > 18.0:
            move_distance = speed * 0.45
        else:
            # Circle slowly near the tree instead of pushing into it.
            move_distance = speed * 0.25
            move_direction = normalize_angle(tree_angle + math.pi / 2.0)

        turn_angle = max(-0.06, min(0.06, tree_angle))

    # Last priority: economical deterministic exploration.
    else:
        phase = agent_id * 1.61803398875
        slow_wave = math.sin(age * 0.055 + phase)
        faster_wave = math.sin(age * 0.017 + phase * 0.7)

        move_direction = 0.55 * slow_wave + 0.25 * faster_wave

        # Hungry agents search more actively. Well-fed agents conserve energy.
        if energy_ratio < 0.35:
            move_distance = speed * 0.90
        elif energy_ratio < 0.75:
            move_distance = speed * 0.65
        else:
            move_distance = speed * 0.40

        # Small scanning motion. Large turns are unnecessarily expensive.
        turn_angle = 0.035 * math.sin(age * 0.08 + phase)

    return ActionRequest(
        agent_id=agent_id,
        move_distance=float(max(0.0, move_distance)),
        move_direction=float(normalize_angle(move_direction)),
        turn_angle=float(normalize_angle(turn_angle)),
        spawn_agent=spawn_agent,
    )
