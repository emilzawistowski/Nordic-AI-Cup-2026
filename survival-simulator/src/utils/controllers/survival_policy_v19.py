"""
survival_policy_v19.py — "Face + Kite + Aggressive Spawn"

Same face-the-predator exploit as v18, but with:
- Lower spawn threshold (155 energy) for faster population growth early on
- More agents = predators split attention between them
- Biome-awareness: don't sprint in swamp (0.5x movement penalty wastes energy)
- Better sprint control: only sprint when truly necessary
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
    biome = observation_response.get("biome", "grassland")

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

    # Biome modifiers
    in_swamp = biome == "swamp"
    in_river = biome == "river"
    slow_biome = in_swamp or in_river

    move_distance = 0.0
    move_direction = 0.0
    turn_angle = 0.0
    spawn_agent = False

    # -----------------------------------------------------------------------
    # 1. PREDATOR AVOIDANCE — Face-and-Kite
    # -----------------------------------------------------------------------
    if predators:
        nearest_pred = min(predators, key=lambda p: float(p["distance"]))
        nearest_pred_dist = float(nearest_pred["distance"])
        nearest_pred_angle = float(nearest_pred["angle"])

        raw_escape = weighted_escape_direction(predators)

        # Facing predator? → predator pivots (slow mode)
        # Not facing? → predator charges → turn to face immediately!
        facing_predator = abs(nearest_pred_angle) < math.pi / 2

        if facing_predator:
            # In pivot mode — move perp to escape while maintaining facing
            perp_cw  = normalize_angle(nearest_pred_angle - math.pi / 2)
            perp_ccw = normalize_angle(nearest_pred_angle + math.pi / 2)
            dot_cw  = math.cos(perp_cw)  * math.cos(raw_escape) + math.sin(perp_cw)  * math.sin(raw_escape)
            dot_ccw = math.cos(perp_ccw) * math.cos(raw_escape) + math.sin(perp_ccw) * math.sin(raw_escape)
            move_direction = perp_cw if dot_cw > dot_ccw else perp_ccw
            # Maintain facing with gentle correction
            turn_angle = max(-0.12, min(0.12, nearest_pred_angle * 0.25))
        else:
            # Not facing → turn to face while still fleeing
            move_direction = raw_escape
            turn_angle = max(-0.30, min(0.30, nearest_pred_angle * 0.7))

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

        # Sprint: only when truly close and have energy
        # In slow biomes sprinting wastes extra energy for same effective movement
        if nearest_pred_dist <= 50.0 and energy > 30.0 and not slow_biome:
            move_distance = sprint_speed
        elif nearest_pred_dist <= 80.0 and energy > 50.0 and not slow_biome:
            move_distance = speed * 1.0
        else:
            move_distance = speed * 0.85

    # -----------------------------------------------------------------------
    # 2. FRUIT HARVESTING
    # -----------------------------------------------------------------------
    elif nearest_fruit is not None:
        fruit_dist = float(nearest_fruit["distance"])
        fruit_angle = normalize_angle(float(nearest_fruit["angle"]))
        move_direction = fruit_angle
        move_distance = min(speed, fruit_dist)
        turn_angle = max(-0.10, min(0.10, fruit_angle))

    # -----------------------------------------------------------------------
    # 3. TREE FORAGING (age-aware)
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # 4. WANDERING
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # 5. AGGRESSIVE EARLY REPRODUCTION
    # Spawn earlier (155u) to build population fast while predators are few.
    # Fewer predators early → each new agent has higher survival chance.
    # -----------------------------------------------------------------------
    if not predators and energy >= 155.0 and age >= 8.0:
        spawn_agent = True
        move_distance = min(move_distance, speed * 0.25)

    return ActionRequest(
        agent_id=agent_id,
        move_distance=float(max(0.0, move_distance)),
        move_direction=float(normalize_angle(move_direction)),
        turn_angle=float(normalize_angle(turn_angle)),
        spawn_agent=spawn_agent,
    )
