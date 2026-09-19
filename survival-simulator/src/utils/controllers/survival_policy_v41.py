"""
survival_policy_v41.py — "Multi-Predator Zoning: Flee from Nearest, Check Rear"

When multiple predators are present, our current weighted escape vector is:
 SUM(escape_weight_i / dist_i^2). This averages all predators equally.

Problem: If 3 predators are all to the left and 1 predator is to the right (nearest),
the weighted vector pulls LEFT (toward the cluster of 3), which is WRONG.

Fix: ALWAYS prioritize fleeing from the nearest predator first. Use secondary predators
only to break ties between directions that are equally good from nearest predator.

Also: Only the NEAREST predator should determine sprint vs walk decision.
"""

import math
import random
from src.utils.DTOs import ActionRequest

TAU = 2.0 * math.pi


def normalize_angle(angle: float) -> float:
    return (angle + math.pi) % TAU - math.pi


def primary_escape_direction(predators: list, nearest_pred_angle: float) -> float:
    """
    Primary escape is directly away from nearest predator.
    Secondary predators add gentle adjustment to avoid them too.
    """
    primary_escape = normalize_angle(nearest_pred_angle + math.pi)

    if len(predators) == 1:
        return primary_escape

    # Adjustment from other predators (with much lower weight)
    adj_x = math.cos(primary_escape)
    adj_y = math.sin(primary_escape)

    nearest_dist = min(float(p["distance"]) for p in predators)

    for pred in predators:
        dist = max(float(pred.get("distance", 1.0)), 1.0)
        # Skip nearest predator (already handled by primary_escape)
        if abs(dist - nearest_dist) < 1.0:
            continue
        angle = float(pred.get("angle", 0.0))
        sec_escape = normalize_angle(angle + math.pi)
        # Secondary predators get small weight relative to nearest
        weight = 150.0 / (dist + 1.0)
        adj_x += math.cos(sec_escape) * weight * 0.15
        adj_y += math.sin(sec_escape) * weight * 0.15

    if abs(adj_x) < 1e-12 and abs(adj_y) < 1e-12:
        return primary_escape
    return normalize_angle(math.atan2(adj_y, adj_x))


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

    # 1. Predator Avoidance
    if predators:
        nearest_pred = min(predators, key=lambda p: float(p["distance"]))
        nearest_pred_dist = float(nearest_pred["distance"])
        nearest_pred_angle = float(nearest_pred["angle"])

        move_direction = primary_escape_direction(predators, nearest_pred_angle)

        # Wall avoidance
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

        if nearest_pred_dist <= 55.0 and energy > 20.0:
            move_distance = sprint_speed
        else:
            move_distance = speed

        turn_angle = max(-0.15, min(0.15, nearest_pred_angle))

    # 2. Fruit Harvesting
    elif nearest_fruit is not None:
        fruit_dist = float(nearest_fruit["distance"])
        fruit_angle = normalize_angle(float(nearest_fruit["angle"]))

        move_direction = fruit_angle
        move_distance = min(speed, fruit_dist)
        turn_angle = max(-0.10, min(0.10, fruit_angle))

    # 3. Tree Foraging
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

    # 5. Reproduction
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
