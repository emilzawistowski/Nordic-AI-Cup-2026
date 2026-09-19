"""
survival_policy_v37.py — "Predator Awareness via rel_dir + Strategic Flee Direction"

Key NEW insight: each predator observation includes a 'rel_dir' field.
  rel_dir = (arctan2(agent.y - pred.y, agent.x - pred.x) - predator.direction)
  = how much the predator is FACING toward us (in predator's local frame).
  - rel_dir ≈ 0: predator is looking directly at us (dangerous)
  - |rel_dir| ≈ pi: predator is facing away from us (we are behind predator)

If predator is facing AWAY (|rel_dir| > pi/2), it is looking the other way and
may not have detected us. We can move gently away without sprinting.

If predator IS facing us (|rel_dir| < pi/2), it is actively pursuing; sprint hard.

This reduces unnecessary sprinting when predators happen to be facing away,
saving energy for when it truly matters.
"""

import math
import random
from src.utils.DTOs import ActionRequest

TAU = 2.0 * math.pi


def normalize_angle(angle: float) -> float:
    return (angle + math.pi) % TAU - math.pi


def weighted_escape_direction(predators: list) -> float:
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


def nearest_observation(observations: list, object_type: str):
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

    # 1. PREDATOR AVOIDANCE
    if predators:
        nearest_pred = min(predators, key=lambda p: float(p["distance"]))
        nearest_pred_dist = float(nearest_pred["distance"])
        nearest_pred_rel_dir = float(nearest_pred.get("rel_dir", 0.0))

        move_direction = weighted_escape_direction(predators)

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

        # Predator facing us? (rel_dir ≈ 0 means predator looks at us)
        predator_facing_us = abs(nearest_pred_rel_dir) < math.pi / 2.0
        # Within hearing range (predator can hear us regardless of direction)
        in_hearing_range = nearest_pred_dist < 90.0  # predator hearing_radius = 60, but buffer

        if nearest_pred_dist <= 55.0 and energy > 20.0:
            # Always sprint when very close
            move_distance = sprint_speed
        elif predator_facing_us and nearest_pred_dist <= 100.0 and energy > 20.0:
            # Sprint if predator is actively looking at us and within range
            move_distance = sprint_speed
        elif in_hearing_range:
            # Predator might hear us - move at full walk speed
            move_distance = speed
        else:
            # Predator is far and not looking - gentle movement
            move_distance = speed * 0.75

        pred_angle = float(nearest_pred["angle"])
        turn_angle = max(-0.15, min(0.15, pred_angle))

    # 2. FRUIT HARVESTING
    elif nearest_fruit is not None:
        fruit_dist = float(nearest_fruit["distance"])
        fruit_angle = normalize_angle(float(nearest_fruit["angle"]))

        move_direction = fruit_angle
        move_distance = min(speed, fruit_dist)
        turn_angle = max(-0.10, min(0.10, fruit_angle))

    # 3. TREE FORAGING
    elif nearest_tree is not None and (energy_ratio < 0.85 or age > 60.0):
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

    # 4. ECONOMICAL WANDERING
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

    # 5. REPRODUCTION
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
