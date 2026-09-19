import math
import random
from src.utils.DTOs import ActionRequest

TAU = 2.0 * math.pi


def normalize_angle(angle: float) -> float:
    """Normalize an angle to [-pi, pi]."""
    return (angle + math.pi) % TAU - math.pi


def weighted_escape_direction(predators: list[dict], other_agents: list[dict]) -> float:
    escape_x = 0.0
    escape_y = 0.0

    for predator in predators:
        distance = max(float(predator.get("distance", 1.0)), 1.0)
        predator_angle = float(predator.get("angle", 0.0))

        escape_angle = normalize_angle(predator_angle + math.pi)
        weight = 12000.0 / (distance * distance + 1.0)

        escape_x += math.cos(escape_angle) * weight
        escape_y += math.sin(escape_angle) * weight

    # Mild species dispersal
    for agent_obs in other_agents:
        distance = max(float(agent_obs.get("distance", 1.0)), 1.0)
        if distance < 90.0:
            a_angle = float(agent_obs.get("angle", 0.0))
            rep_angle = normalize_angle(a_angle + math.pi)
            weight = 300.0 / (distance * distance + 5.0)
            escape_x += math.cos(rep_angle) * weight
            escape_y += math.sin(rep_angle) * weight

    if abs(escape_x) < 1e-12 and abs(escape_y) < 1e-12:
        return math.pi

    return normalize_angle(math.atan2(escape_y, escape_x))


def get_wall_repulsion(edges: list[dict], safe_dist: float = 40.0) -> tuple[float, float]:
    rep_x = 0.0
    rep_y = 0.0

    for edge_obs in edges:
        coords = edge_obs.get("coords")
        if not coords or len(coords) != 2:
            continue
        (x1, y1), (x2, y2) = coords
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        dist = math.hypot(mx, my)
        if dist < safe_dist:
            angle = math.atan2(my, mx)
            rep_angle = normalize_angle(angle + math.pi)
            weight = 2500.0 / (dist * dist + 1.0)
            rep_x += math.cos(rep_angle) * weight
            rep_y += math.sin(rep_angle) * weight

    return rep_x, rep_y


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
    other_agents = [
        o for o in observations
        if o.get("type") == "Agent" and "distance" in o and "angle" in o
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

    move_distance = 0.0
    move_direction = 0.0
    turn_angle = 0.0
    spawn_agent = False

    # 1. Highest Priority: Predator Avoidance
    if predators:
        nearest_pred_dist = min(float(p["distance"]) for p in predators)
        move_direction = weighted_escape_direction(predators, other_agents)

        # Deflect from walls when fleeing
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

        # Sprint threshold: sprint if predator is within 60 units and energy > 20
        if nearest_pred_dist <= 60.0 and energy > 20.0:
            move_distance = sprint_speed
        else:
            move_distance = speed

        nearest_p = min(predators, key=lambda p: float(p["distance"]))
        pred_angle = float(nearest_p["angle"])
        turn_angle = max(-0.16, min(0.16, pred_angle))

    # 2. Fruit Harvesting (Value = distance)
    elif fruits:
        best_fruit = min(fruits, key=lambda f: float(f["distance"]))
        fruit_dist = float(best_fruit["distance"])
        fruit_angle = normalize_angle(float(best_fruit["angle"]))

        move_direction = fruit_angle
        move_distance = min(speed, fruit_dist)
        turn_angle = max(-0.12, min(0.12, fruit_angle))

    # 3. Tree Foraging
    elif trees and energy_ratio < 0.85:
        best_tree = min(trees, key=lambda t: float(t["distance"]))
        tree_dist = float(best_tree["distance"])
        tree_angle = normalize_angle(float(best_tree["angle"]))

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

    # 4. Exploration with Wall Avoidance
    else:
        wx, wy = get_wall_repulsion(edges, safe_dist=40.0)
        if abs(wx) > 1e-9 or abs(wy) > 1e-9:
            move_direction = normalize_angle(math.atan2(wy, wx))
            move_distance = speed * 0.70
            turn_angle = max(-0.12, min(0.12, move_direction))
        else:
            phase = agent_id * 1.61803398875
            slow_wave = math.sin(age * 0.055 + phase)
            faster_wave = math.sin(age * 0.017 + phase * 0.7)
            declump_dir = weighted_escape_direction([], other_agents) if other_agents else 0.0

            move_direction = 0.40 * slow_wave + 0.20 * faster_wave + 0.40 * declump_dir

            if energy_ratio < 0.35:
                move_distance = speed * 0.90
            elif energy_ratio < 0.75:
                move_distance = speed * 0.65
            else:
                move_distance = speed * 0.40

            turn_angle = 0.04 * math.sin(age * 0.15 + phase)

    # 5. Safe Reserve Reproduction Strategy
    # Cost: 100 energy.
    # Parent must retain at least 80 energy AND at least 45% max_energy post-spawning
    spawn_threshold = max(180.0, 0.45 * max_energy + 100.0)
    near_predators = [p for p in predators if float(p["distance"]) < 85.0]

    if not near_predators and energy >= spawn_threshold and age >= 10.0:
        spawn_agent = True
        move_distance = min(move_distance, speed * 0.25)

    return ActionRequest(
        agent_id=agent_id,
        move_distance=float(max(0.0, move_distance)),
        move_direction=float(normalize_angle(move_direction)),
        turn_angle=float(normalize_angle(turn_angle)),
        spawn_agent=spawn_agent,
    )
