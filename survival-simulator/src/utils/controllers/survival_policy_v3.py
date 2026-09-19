import math
import random
from src.utils.DTOs import ActionRequest

TAU = 2.0 * math.pi


def normalize_angle(angle: float) -> float:
    """Normalize an angle to [-pi, pi]."""
    return (angle + math.pi) % TAU - math.pi


def get_closest_point_on_segment(px: float, py: float, x1: float, y1: float, x2: float, y2: float):
    """Find the closest point on segment (x1,y1)-(x2,y2) to point (px,py)."""
    dx = x2 - x1
    dy = y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-12:
        return x1, y1
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / len_sq))
    return x1 + t * dx, y1 + t * dy


def compute_escape_vector(predators: list[dict], edges: list[dict], hearing_radius: float) -> tuple[float, float]:
    """
    Compute net escape vector combining predator repulsion and edge repulsion.
    Angles in observations are relative to agent's current heading.
    """
    escape_x = 0.0
    escape_y = 0.0

    # 1. Repulsion from predators
    for predator in predators:
        dist = max(float(predator.get("distance", 1.0)), 1.0)
        p_angle = float(predator.get("angle", 0.0))

        # Repulsion angle is opposite to predator angle
        rep_angle = normalize_angle(p_angle + math.pi)

        # Strongly weight close predators (inverse cubic/quadratic force)
        weight = 10000.0 / (dist * dist * dist + 10.0)
        escape_x += math.cos(rep_angle) * weight
        escape_y += math.sin(rep_angle) * weight

    # 2. Repulsion from visible edges (walls/obstacles)
    for edge_obs in edges:
        coords = edge_obs.get("coords")
        if not coords or len(coords) != 2:
            continue
        (x1, y1), (x2, y2) = coords
        cx, cy = get_closest_point_on_segment(0.0, 0.0, x1, y1, x2, y2)
        dist = math.hypot(cx, cy)
        if dist < 45.0:  # Only repulse when near edge
            edge_angle = math.atan2(cy, cx)
            rep_angle = normalize_angle(edge_angle + math.pi)
            weight = 5000.0 / (dist * dist + 5.0)
            escape_x += math.cos(rep_angle) * weight
            escape_y += math.sin(rep_angle) * weight

    if abs(escape_x) < 1e-9 and abs(escape_y) < 1e-9:
        return 0.0, 0.0

    escape_angle = normalize_angle(math.atan2(escape_y, escape_x))
    magnitude = math.hypot(escape_x, escape_y)
    return escape_angle, magnitude


def action_decision(observation_response: dict, rng: random.Random) -> ActionRequest:
    agent_id = int(observation_response["agent_id"])
    observations = observation_response.get("observations", [])

    energy = float(observation_response["energy"])
    age = float(observation_response["age"])
    speed = float(observation_response["speed"])
    sprint_speed = float(observation_response["sprint_speed"])
    hearing_radius = float(observation_response["hearing_radius"])
    max_energy = float(observation_response["max_energy"])

    # Categorize observations
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
    move_distance = 0.0
    move_direction = 0.0
    turn_angle = 0.0
    spawn_agent = False

    # Priority 1: Predator avoidance with vector field
    if predators:
        nearest_pred_dist = min(float(p["distance"]) for p in predators)
        escape_angle, magnitude = compute_escape_vector(predators, edges, hearing_radius)

        if magnitude > 1e-9:
            move_direction = escape_angle
        else:
            # Fallback away from nearest predator
            nearest_p = min(predators, key=lambda p: float(p["distance"]))
            move_direction = normalize_angle(float(nearest_p["angle"]) + math.pi)

        # Speed logic: sprint only under critical threat and sufficient energy
        critical_threat_dist = max(28.0, 1.8 * sprint_speed)
        if nearest_pred_dist <= critical_threat_dist and energy > 35.0:
            move_distance = sprint_speed
        else:
            move_distance = speed

        # Smooth turn towards escape direction
        turn_angle = max(-0.12, min(0.12, move_direction))

    # Priority 2: Fruit Collection
    elif fruits:
        # Pick best fruit (closest)
        best_fruit = min(fruits, key=lambda f: float(f["distance"]))
        fruit_dist = float(best_fruit["distance"])
        fruit_angle = normalize_angle(float(best_fruit["angle"]))

        move_direction = fruit_angle
        move_distance = min(speed, fruit_dist)
        turn_angle = max(-0.10, min(0.10, fruit_angle))

    # Priority 3: Tree Foraging (where fruits spawn)
    elif trees and energy_ratio < 0.85:
        best_tree = min(trees, key=lambda t: float(t["distance"]))
        tree_dist = float(best_tree["distance"])
        tree_angle = normalize_angle(float(best_tree["angle"]))

        if tree_dist > 30.0:
            move_direction = tree_angle
            move_distance = speed * 0.85
        elif tree_dist > 15.0:
            # Orbit around tree at radius ~20 to spot newly spawned fruit
            orbit_offset = 0.40 if agent_id % 2 == 0 else -0.40
            move_direction = normalize_angle(tree_angle + orbit_offset)
            move_distance = speed * 0.50
        else:
            # Slow orbit when very close
            move_direction = normalize_angle(tree_angle + math.pi / 2.0)
            move_distance = speed * 0.25

        turn_angle = max(-0.08, min(0.08, tree_angle))

    # Priority 4: Smart Exploration with Wall Avoidance
    else:
        # Check edge proximity for bouncing away from boundaries
        close_edges = []
        for e in edges:
            coords = e.get("coords")
            if coords and len(coords) == 2:
                (x1, y1), (x2, y2) = coords
                cx, cy = get_closest_point_on_segment(0.0, 0.0, x1, y1, x2, y2)
                d = math.hypot(cx, cy)
                if d < 35.0:
                    close_edges.append((cx, cy, d))

        if close_edges:
            # Avoid closest wall
            cx, cy, d = min(close_edges, key=lambda x: x[2])
            wall_angle = math.atan2(cy, cx)
            move_direction = normalize_angle(wall_angle + math.pi)
            move_distance = speed * 0.60
            turn_angle = max(-0.10, min(0.10, move_direction))
        else:
            # Sine wave exploration based on agent_id and age
            phase = agent_id * 2.399963229728653  # Golden angle ratio
            wave = math.sin(age * 0.04 + phase)
            move_direction = 0.45 * wave

            if energy_ratio < 0.30:
                move_distance = speed * 0.90
            elif energy_ratio < 0.70:
                move_distance = speed * 0.65
            else:
                move_distance = speed * 0.45

            turn_angle = 0.04 * math.sin(age * 0.07 + phase)

    # Controlled Reproduction Logic
    # 100 energy cost, child starts with 75 energy.
    # Spawn if energy > 220, age < 90, and no predators present.
    if not predators and energy > 220.0 and age < 90.0:
        # Avoid spawning every single tick by adding a small phase window
        spawn_window = (age + agent_id * 5.1) % 25.0
        if spawn_window < 0.20:
            spawn_agent = True
            move_distance = min(move_distance, speed * 0.20)

    return ActionRequest(
        agent_id=agent_id,
        move_distance=float(max(0.0, move_distance)),
        move_direction=float(normalize_angle(move_direction)),
        turn_angle=float(normalize_angle(turn_angle)),
        spawn_agent=spawn_agent,
    )
