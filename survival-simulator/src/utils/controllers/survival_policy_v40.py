"""
survival_policy_v40.py — "Lateral Dodge + Energy-Tiered Sprint"

Critical insight from predator.py:
- Predator CHARGES (speed 15) when abs(agent_looking_dir) > pi/2 (predator behind us)
- Predator PIVOTS (speed ~10.6 net) when abs(agent_looking_dir) < pi/2 (we face it)

So the worst case: predator is RIGHT BEHIND US. 
At distance d, predator closes at: 15 (predator) - 20 (agent sprint) = -5 (we pull away).
If we're walking (speed 10): 15 - 10 = +5 (predator closes!).

The KEY is: once predator is within 80 units, we must SPRINT or it will eventually catch us.
Sprint cost: speed * 0.05 + (dist-speed) * 0.5 per tick... actually from code:
  sprint cost = speed * 0.05 + (move_distance - speed) * 0.5
  = 20 * 0.05 + (20-10) * 0.5 = 1.0 + 5.0 = 6.0 per tick

Walk cost: 10 * 0.05 = 0.5 per tick.

Sprint is 12x more expensive! So we want to sprint as LITTLE as possible.

Strategy:
- At distance < 50: sprint at full speed (unavoidable)
- At distance 50-80: if predator "angle" tells us it's in cone ±pi/2, do lateral dodge
  The lateral direction: perpendicular to predator approach, weighted toward best open space.
- At distance > 80: walk away while foraging normally.
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

    # 1. Predator Avoidance with tiered strategy
    if predators:
        nearest_pred = min(predators, key=lambda p: float(p["distance"]))
        nearest_pred_dist = float(nearest_pred["distance"])
        nearest_pred_angle = float(nearest_pred["angle"])

        # Primary escape direction
        escape_dir = weighted_escape_direction(predators)

        # Check wall avoidance
        for edge_obs in edges:
            coords = edge_obs.get("coords")
            if coords and len(coords) == 2:
                (x1, y1), (x2, y2) = coords
                mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                edge_dist = math.hypot(mx, my)
                if edge_dist < 35.0:
                    edge_angle = math.atan2(my, mx)
                    diff = normalize_angle(escape_dir - edge_angle)
                    if abs(diff) < math.pi / 2.0:
                        escape_dir = normalize_angle(escape_dir + math.copysign(math.pi / 2.5, diff))

        if nearest_pred_dist <= 50.0 and energy > 20.0:
            # IMMEDIATE DANGER: full sprint directly away
            move_distance = sprint_speed
            move_direction = escape_dir
        elif nearest_pred_dist <= 80.0 and energy > 20.0:
            # MODERATE DANGER: lateral dodge - move perpendicular to predator
            # Lateral options: +pi/2 or -pi/2 from predator direction
            lateral_left = normalize_angle(nearest_pred_angle + math.pi + math.pi / 2.0)
            lateral_right = normalize_angle(nearest_pred_angle + math.pi - math.pi / 2.0)
            # Bias toward the one that keeps us away from center (walls already handled)
            # Use agent_id to deterministically pick a side (persists per agent)
            lateral_dir = lateral_left if agent_id % 2 == 0 else lateral_right
            # Blend lateral with escape: 60% lateral, 40% direct escape
            lx = 0.6 * math.cos(lateral_dir) + 0.4 * math.cos(escape_dir)
            ly = 0.6 * math.sin(lateral_dir) + 0.4 * math.sin(escape_dir)
            move_direction = normalize_angle(math.atan2(ly, lx))
            move_distance = sprint_speed  # still sprint in this range
        else:
            # FAR PREDATOR: walk away
            move_direction = escape_dir
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
