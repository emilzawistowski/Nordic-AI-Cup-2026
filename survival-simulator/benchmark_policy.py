import argparse
import random
import statistics
import time

import pygame

from src.core import SimulationCore
from src.utils.controllers.dummy_agent_policy import action_decision


DEFAULT_SEEDS = [
    2204119010,
    1,
    42,
    137,
    2026,
    12345,
    8675309,
    31415926,
    271828182,
    3735928559,
]


def run_simulation(seed: int) -> dict:
    sim = SimulationCore(seed=seed)
    action_rng = random.Random(seed)
    actions = []

    started = time.perf_counter()

    while True:
        state = sim.step(actions)

        actions = []
        for agent, agent_state in zip(sim.env.agents, state["observations"]):
            action = action_decision(agent_state, action_rng)
            actions.append((agent.agent_id, action))

        if state["num_agents"] == 0 or sim.env.time > 3000:
            elapsed = time.perf_counter() - started
            return {
                "seed": seed,
                "score": float(state["score"]),
                "sim_time": float(sim.env.time),
                "agents": int(state["num_agents"]),
                "full_survival": sim.env.time > 3000,
                "wall_seconds": elapsed,
            }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=DEFAULT_SEEDS,
    )
    args = parser.parse_args()

    pygame.init()

    results = []

    for index, seed in enumerate(args.seeds, start=1):
        result = run_simulation(seed)
        results.append(result)

        status = "FULL" if result["full_survival"] else "DEAD"

        print(
            f"[{index:02d}/{len(args.seeds):02d}] "
            f"seed={result['seed']:10d} "
            f"score={result['score']:10.3f} "
            f"sim_time={result['sim_time']:8.1f} "
            f"status={status} "
            f"wall={result['wall_seconds']:7.2f}s",
            flush=True,
        )

    pygame.quit()

    scores = [result["score"] for result in results]
    survival_times = [result["sim_time"] for result in results]
    wall_times = [result["wall_seconds"] for result in results]
    full_runs = sum(result["full_survival"] for result in results)

    print()
    print("=== SUMMARY ===")
    print(f"Runs:              {len(results)}")
    print(f"Mean score:        {statistics.fmean(scores):.3f}")
    print(f"Median score:      {statistics.median(scores):.3f}")
    print(f"Minimum score:     {min(scores):.3f}")
    print(f"Maximum score:     {max(scores):.3f}")
    print(f"Score stdev:       {statistics.pstdev(scores):.3f}")
    print(f"Mean survival:     {statistics.fmean(survival_times):.1f}")
    print(f"Median survival:   {statistics.median(survival_times):.1f}")
    print(f"Full survivals:    {full_runs}/{len(results)}")
    print(f"Total wall time:   {sum(wall_times):.2f}s")


if __name__ == "__main__":
    main()
