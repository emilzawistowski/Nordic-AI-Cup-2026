"""
survival_policy_v48.py — "v11 Champion Base + Predator Line-of-Sight Choke Escape"

Analysis:
In predator.py, a predator CHARGES at sprint speed (15) when:
  abs(agent_looking_dir) > pi/2 OR distance < hearing_radius * 1.5 (60 * 1.5 = 90u)

Where agent_looking_dir from predator frame is the agent's predator["angle"] observation!
When predator["angle"] ≈ 0, agent is in FRONT of predator (looking away or facing it).
When abs(predator["angle"]) > pi/2, agent is BEHIND predator (predator must PIVOT, speed drops to 10.6).

Hypothesis:
If an agent is in front of predator (angle < pi/2) at dist 55-90u, predator CHARGES.
If agent turns slightly so its velocity is NOT parallel to predator line of sight,
it forces predator to constantly re-evaluate line of sight and pivot instead of straight charge!

In v48:
- Apply a 15-degree (0.26 rad) tangential angle offset to escape direction when distance is 50-80u.
- Keep exact v11 values for everything else.
"""

import math
import random
from src.utils.DTOs import ActionRequest

TAU = 2.0 * math.pi


def normalize_angle(angle: float) -> float:
    return (angle + math.pi) % TAU - math.pi


def weighted_escape_direction(predators: list, agent_id: int) -> float:
    escape_x = 0.0
    escape_y = 0.0
    for predator in predators:
        distance = max(float(predator.get("distance", 1.0)), 1.0)
        predator_angle = float(predator.get("angle", 0.0))

        # Direct escape angle
        escape_angle = normalize_angle(predator_angle + math.pi)

        # Tangential offset (0.26 rad ≈ 15 deg) when 50-80u to break straight line charge
        if 50.0 <= distance <= 80.0:
            tangent_side = 0.26 if agent_id % 2 == 0 else -0.26
            escape_angle = normalize_angle(escape_angle + tangent_side)

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

    # 1. Predator Avoidance
    if predators:
        nearest_pred_dist = min(float(p["distance"]) for p in predators)
        move_direction = weighted_escape_direction(predators, agent_id)

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

        nearest_pred = min(predators, key=lambda p: float(p["distance"]))
        pred_angle = float(nearest_pred["angle"])
        turn_angle = max(-0.15, min(0.15, pred_angle))

    # 2. Fruit Harvesting (v11 exact)
    elif nearest_fruit is not None:
        fruit_dist = float(nearest_fruit["distance"])
        fruit_angle = normalize_angle(float(nearest_fruit["angle"]))

        move_direction = fruit_angle
        move_distance = min(speed, fruit_dist)
        turn_angle = max(-0.10, min(0.10, fruit_angle))

    # 3. Tree Foraging (v11 exact)
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

    # 4. Economical Wandering (v11 exact)
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

    # 5. Reproduction (v11 exact)
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
