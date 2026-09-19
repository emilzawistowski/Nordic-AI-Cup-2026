"""
survival_policy_v42.py — "Potential Field Navigation (PFN)"

Instead of rule-based priority, use a unified potential field:
- Predators create REPULSIVE fields (push agent away)
- Fruit creates ATTRACTIVE fields (pull agent toward)
- Trees create weak ATTRACTIVE fields (guide toward fruit sources)
- Walls create REPULSIVE fields (prevent corners)

The agent moves in the direction of the net force vector.
This allows simultaneous handling of multiple goals and threats.

Key advantage: An agent near fruit AND near a distant predator will move
toward the fruit while slightly angling away from the predator —
rather than completely abandoning fruit for a distant threat.

Parameters tuned around v11's proven values.
"""

import math
import random
from src.utils.DTOs import ActionRequest

TAU = 2.0 * math.pi


def normalize_angle(angle: float) -> float:
    return (angle + math.pi) % TAU - math.pi


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

    edges = [
        o for o in observations
        if o.get("type") == "Edge" and "coords" in o
    ]

    energy_ratio = energy / max(max_energy, 1.0)

    # ─── Potential Field Accumulation ────────────────────────────────
    fx, fy = 0.0, 0.0

    predators = []
    for obs in observations:
        obs_type = obs.get("type")
        dist = max(float(obs.get("distance", 1.0)), 0.5)
        angle = float(obs.get("angle", 0.0))

        if obs_type == "Predator" and "distance" in obs and "angle" in obs:
            predators.append(obs)
            # Strong repulsion from predator
            # At dist=55: weight ≈ 6000/3025 ≈ 2.0; at dist=20: 6000/401 ≈ 15
            repulsion_weight = 6000.0 / (dist * dist + 1.0)
            escape_angle = normalize_angle(angle + math.pi)
            fx += math.cos(escape_angle) * repulsion_weight
            fy += math.sin(escape_angle) * repulsion_weight

        elif obs_type == "Fruit" and "distance" in obs and "angle" in obs:
            # Attraction toward fruit (scaled by energy need)
            hunger_scale = max(0.3, 1.0 - energy_ratio)
            fruit_energy = float(obs.get("energy", 40.0))  # default
            attr_weight = 80.0 * hunger_scale * (fruit_energy / 60.0) / max(dist * 0.5, 1.0)
            fx += math.cos(angle) * attr_weight
            fy += math.sin(angle) * attr_weight

        elif obs_type == "Tree" and "distance" in obs and "angle" in obs:
            # Weak attraction toward trees (fruit source)
            hunger_scale = max(0.0, 0.85 - energy_ratio)
            if hunger_scale > 0 or age > 60.0:
                tree_weight = 20.0 * (hunger_scale + (0.2 if age > 60.0 else 0.0)) / max(dist * 0.3, 1.0)
                approach_offset = 0.35 if agent_id % 2 == 0 else -0.35
                approach_angle = normalize_angle(angle + approach_offset)
                fx += math.cos(approach_angle) * tree_weight
                fy += math.sin(approach_angle) * tree_weight

    # Wall repulsion
    for edge_obs in edges:
        coords = edge_obs.get("coords")
        if coords and len(coords) == 2:
            (x1, y1), (x2, y2) = coords
            mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            edge_dist = max(math.hypot(mx, my), 1.0)
            if edge_dist < 60.0:
                edge_angle = math.atan2(my, mx)
                wall_weight = 3000.0 / (edge_dist * edge_dist + 1.0)
                repel_angle = normalize_angle(edge_angle + math.pi)
                fx += math.cos(repel_angle) * wall_weight
                fy += math.sin(repel_angle) * wall_weight

    # Wandering impulse when no strong signal
    force_magnitude = math.hypot(fx, fy)
    if force_magnitude < 5.0:
        phase = agent_id * 1.61803398875
        slow_wave = math.sin(age * 0.055 + phase)
        faster_wave = math.sin(age * 0.017 + phase * 0.7)
        wander_angle = 0.50 * slow_wave + 0.20 * faster_wave
        fx += math.cos(wander_angle) * 10.0
        fy += math.sin(wander_angle) * 10.0
        force_magnitude = math.hypot(fx, fy)

    move_direction = normalize_angle(math.atan2(fy, fx))

    # ─── Speed Decision ───────────────────────────────────────────────
    move_distance = 0.0
    if predators:
        nearest_pred_dist = min(float(p["distance"]) for p in predators)
        if nearest_pred_dist <= 55.0 and energy > 20.0:
            move_distance = sprint_speed
        else:
            move_distance = speed
    else:
        # No predators: scale speed by urgency
        urgency = min(1.0, force_magnitude / 30.0)
        if energy_ratio < 0.35:
            move_distance = speed * (0.5 + 0.4 * urgency)
        elif energy_ratio < 0.75:
            move_distance = speed * (0.35 + 0.3 * urgency)
        else:
            move_distance = speed * (0.25 + 0.15 * urgency)

    # Turn toward target direction
    nearest_pred = min(predators, key=lambda p: float(p["distance"])) if predators else None
    if nearest_pred:
        pred_angle = float(nearest_pred["angle"])
        turn_angle = max(-0.15, min(0.15, pred_angle))
    else:
        turn_angle = max(-0.08, min(0.08, move_direction))

    # ─── Reproduction ─────────────────────────────────────────────────
    spawn_agent = False
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
