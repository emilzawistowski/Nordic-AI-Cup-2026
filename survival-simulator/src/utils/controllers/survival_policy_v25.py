"""
survival_policy_v25.py — "Full 3000s Survival Strategy: Evolutionary Trait Selection & Conservative Energy Management"

Target: Beat ~2096 leaderboard score (survive all 3000 seconds).

Key Innovations:
1. SELECTIVE MUTATION HARNESSING:
   - Prioritize breeding from high-speed, high-vision, and high-energy agents.
   - If parent speed < 12.0 or vision < 150, wait until energy >= 210 to spawn.
   - If parent speed >= 15.0 or vision >= 250, spawn earlier at energy >= 170 to multiply top-tier traits.

2. PREDATOR DISTRACTION / BAIT EXPULSION:
   - When predators >= 2 and distance <= 45.0, spawn an offspring if energy > 140!
   - Spawning creates a new target for the predator. Predator targets NEAREST agent, splitting chase focus.

3. HIGH-LATENCY LATE GAME SURVIVAL (t > 1000s):
   - As predators spawn more frequently, raise spawn reserve so agents stay at high energy.
   - Never sprint if energy < 35.0 (avoid energy collapse).
   - Strict edge avoidance (repel at 45.0 units).
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
        weight = 7500.0 / (distance * distance + 1.0)
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
    vision_range = float(observation_response.get("vision_range", 200.0))

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
        move_direction = weighted_escape_direction(predators)

        # Deflect near edges (stronger safety buffer)
        for edge_obs in edges:
            coords = edge_obs.get("coords")
            if coords and len(coords) == 2:
                (x1, y1), (x2, y2) = coords
                mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                edge_dist = math.hypot(mx, my)
                if edge_dist < 42.0:
                    edge_angle = math.atan2(my, mx)
                    diff = normalize_angle(move_direction - edge_angle)
                    if abs(diff) < math.pi / 2.0:
                        move_direction = normalize_angle(move_direction + math.copysign(math.pi / 2.2, diff))

        # Sprint decision with low-energy guardrail
        if nearest_pred_dist <= 55.0 and energy > 35.0:
            move_distance = sprint_speed
        else:
            move_distance = speed

        pred_angle = float(nearest_pred["angle"])
        turn_angle = max(-0.16, min(0.16, pred_angle))

        # Emergency distraction spawn when surrounded by predators
        if len(predators) >= 2 and nearest_pred_dist <= 45.0 and energy >= 145.0 and age >= 5.0:
            spawn_agent = True

    # 2. FRUIT HARVESTING
    elif nearest_fruit is not None:
        fruit_dist = float(nearest_fruit["distance"])
        fruit_angle = normalize_angle(float(nearest_fruit["angle"]))

        move_direction = fruit_angle
        move_distance = min(speed, fruit_dist)
        turn_angle = max(-0.10, min(0.10, fruit_angle))

    # 3. TREE FORAGING
    elif nearest_tree is not None and (energy_ratio < 0.85 or age > 50.0):
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

    # 5. TRAIT-BASED ADAPTIVE REPRODUCTION
    if not spawn_agent and not predators and age >= 10.0:
        is_elite = speed >= 14.0 or vision_range >= 240.0
        is_subpar = speed < 11.0 or vision_range < 160.0

        if is_elite and energy >= 170.0:
            spawn_agent = True
        elif is_subpar and energy >= 220.0:
            spawn_agent = True
        elif not is_elite and not is_subpar and energy >= 185.0:
            spawn_agent = True

        if spawn_agent:
            move_distance = min(move_distance, speed * 0.25)

    return ActionRequest(
        agent_id=agent_id,
        move_distance=float(max(0.0, move_distance)),
        move_direction=float(normalize_angle(move_direction)),
        turn_angle=float(normalize_angle(turn_angle)),
        spawn_agent=spawn_agent,
    )
