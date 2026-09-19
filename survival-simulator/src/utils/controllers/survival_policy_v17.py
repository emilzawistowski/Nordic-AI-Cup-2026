import math
import random
from src.utils.DTOs import ActionRequest

TAU = 2.0 * math.pi


def normalize_angle(angle: float) -> float:
    """Normalize an angle to [-pi, pi]."""
    return (angle + math.pi) % TAU - math.pi


def weighted_escape_direction(predators: list[dict]) -> float:
    escape_x = 0.0
    escape_y = 0.0

    for predator in predators:
        distance = max(float(predator.get("distance", 1.0)), 1.0)
        predator_angle = float(predator.get("angle", 0.0))

        escape_angle = normalize_angle(predator_angle + math.pi)
        weight = 12000.0 / (distance * distance + 1.0)

        escape_x += math.cos(escape_angle) * weight
        escape_y += math.sin(escape_angle) * weight

    if abs(escape_x) < 1e-12 and abs(escape_y) < 1e-12:
        return math.pi

    return normalize_angle(math.atan2(escape_y, escape_x))


def get_tree_cluster_centroid(trees: list[dict]) -> tuple[float, float, int]:
    if not trees:
        return 0.0, 0.0, 0
    cx = 0.0
    cy = 0.0
    for tree in trees:
        d = float(tree["distance"])
        a = float(tree["angle"])
        cx += d * math.cos(a)
        cy += d * math.sin(a)
    n = len(trees)
    avg_x = cx / n
    avg_y = cy / n
    dist = math.hypot(avg_x, avg_y)
    angle = normalize_angle(math.atan2(avg_y, avg_x))
    return angle, dist, n


def get_best_fruit(fruits: list[dict]) -> dict:
    return max(
        fruits,
        key=lambda f: float(f.get("energy", 50.0)) / (float(f["distance"]) + 5.0)
    )


def action_decision(observation_response: dict, rng: random.Random) -> ActionRequest:
    agent_id = int(observation_response["agent_id"])
    observations = observation_response.get("observations", [])

    energy = float(observation_response["energy"])
    age = float(observation_response["age"])
    speed = float(observation_response["speed"])
    sprint_speed = float(observation_response["sprint_speed"])
    hearing_radius = float(observation_response["hearing_radius"])
    max_energy = float(observation_response["max_energy"])
    current_biome = str(observation_response.get("biome", "forest"))

    predators = [
        o for o in observations
        if o.get("type") == "Predator" and "distance" in o and "angle" in o
    ]
    fruits = [
        o for o in observations
        if o.get("type") == "Fruit" and "distance" in o and "angle" in o
    ]
    trees = [
        o for o in observations
        if o.get("type") == "Tree" and "distance" in o and "angle" in o
    ]
    edges = [
        o for o in observations
        if o.get("type") == "Edge" and "coords" in o
    ]

    energy_ratio = energy / max(max_energy, 1.0)
    in_bad_biome = current_biome in ["swamp", "river", "desert"]

    move_distance = 0.0
    move_direction = 0.0
    turn_angle = 0.0
    spawn_agent = False

    # 1. Highest Priority: Trait-Aware Predator Evasion & Pivot Lock
    if predators:
        nearest_pred_dist = min(float(p["distance"]) for p in predators)
        move_direction = weighted_escape_direction(predators)

        # Wall deflection math
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

        # Trait-Aware Speed Selection
        if speed >= 13.5:
            move_distance = speed
        elif nearest_pred_dist <= 60.0 and energy > 18.0:
            move_distance = sprint_speed
        else:
            move_distance = speed

        # Look TOWARDS nearest predator to force pivot lock
        nearest_p = min(predators, key=lambda p: float(p["distance"]))
        pred_angle = float(nearest_p["angle"])
        turn_angle = max(-0.20, min(0.20, pred_angle))

    # 2. Fruit Harvesting
    elif fruits:
        best_f = get_best_fruit(fruits)
        fruit_dist = float(best_f["distance"])
        fruit_angle = normalize_angle(float(best_f["angle"]))

        move_direction = fruit_angle
        move_distance = min(speed, fruit_dist)
        turn_angle = max(-0.12, min(0.12, fruit_angle))

    # 3. Tree Cluster Centroid & Age-Compensated Foraging
    elif trees and (energy_ratio < 0.88 or age > 50.0 or in_bad_biome):
        cluster_angle, cluster_dist, count = get_tree_cluster_centroid(trees)

        if count >= 2:
            move_direction = cluster_angle
            if cluster_dist > 30.0:
                move_distance = speed * (1.00 if in_bad_biome else 0.90)
            elif cluster_dist > 15.0:
                move_distance = speed * 0.50
            else:
                move_distance = speed * 0.30
                move_direction = normalize_angle(cluster_angle + math.pi / 2.0)
            turn_angle = max(-0.08, min(0.08, cluster_angle))
        else:
            best_tree = trees[0]
            tree_dist = float(best_tree["distance"])
            tree_angle = normalize_angle(float(best_tree["angle"]))
            approach_offset = 0.35 if agent_id % 2 == 0 else -0.35
            move_direction = normalize_angle(tree_angle + approach_offset)
            if tree_dist > 35.0:
                move_distance = speed * (1.00 if in_bad_biome else 0.90)
            elif tree_dist > 18.0:
                move_distance = speed * 0.50
            else:
                move_distance = speed * 0.30
                move_direction = normalize_angle(tree_angle + math.pi / 2.0)
            turn_angle = max(-0.08, min(0.08, tree_angle))

    # 4. Rapid 2-Second Radar Sweep Exploration
    else:
        phase = agent_id * 1.61803398875
        slow_wave = math.sin(age * 0.055 + phase)
        faster_wave = math.sin(age * 0.017 + phase * 0.7)
        move_direction = 0.50 * slow_wave + 0.20 * faster_wave

        if energy_ratio < 0.35 or in_bad_biome:
            move_distance = speed * 0.95
        elif energy_ratio < 0.75:
            move_distance = speed * 0.65
        else:
            move_distance = speed * 0.40

        # Rapid radar sweep (2-second cycle)
        turn_angle = 0.10 * math.sin(age * 0.35 + phase)

    # 5. Champion Reproduction Reserve Strategy
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
