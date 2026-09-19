# AUDIT STEP 1 — Files Inventory (2026-09-19)

## 1. HISTORY-survival.md
- Score formula: time survived + fruit.energy/1000. Survive first, eat second.
- Best online: 749.842 (v11). Local v11: 898.485 (10 seeds).
- Leader #1: 2096.489 — gap ~2.8x online, ~2.3x local.
- Log sparse: v0 baseline, v1=v11, v12-v50 summarized as "parameter tuning, clustering prevention" with no per-version scores.
- v51: 795.812 bench (vs 898.485 v11). Did not promote. Server stays on v11.
- v50 (untested): tree forage 0.85→0.70, age 60→90 to reduce clustering.
- Dead ends noted: aggressive spawn→starvation, strict conserve→predator deaths, tree clustering→group kills, complex wall-avoid→marginal.
- Next ideas: bench v50, flee-zone memory, sweep energy_ratio 0.5-0.9 / age 40-120, spawn energy >200.

## 2. WORKFLOW.md
- Rules: never touch simulation_server.py/local_evaluator.py; version new files; bench via `benchmark_any_policy.py --policy ...`; promote only if > old; commit each attempt; append HISTORY.
- Parallel work via worktrees, no direct merges.

## 3. v11 (active)
- Escape: weighted 6000/(d²+1), sprint if dist<=55 & energy>20 else walk.
- Fruit > tree > wander priority. Tree if energy_ratio<0.85 or age>60, offset ±0.35, orbit <18u.
- Wander: sine phase=id*1.618, speed 0.9/0.65/0.4 by energy.
- Spawn if no predator & energy>=185 & age>=10, slow to 0.25*speed.

## 4. v40 (failed)
- Same base + tiered flee: <=50 sprint direct, 50-80 sprint with 60% lateral +40% escape (side by id%2), >80 walk.
- Keeps v11 forage/spawn. Adds docstring cost analysis (sprint 6.0/tick vs walk 0.5).

## 5. v51 (failed, 795.812)
- Adds: (a) predictive escape via 1.5-tick extrapolation from consecutive-tick relative velocity, fallback current pos; (b) zone tree alloc: sort by (angle,dist), pick sorted[id%N]; (c) repro cutoff skipped (no sim_time in ObservationResponse).
- Sprint logic same as v11 (<=55). Rest identical.

## Gaps / Missing
- No files for v12-v39, v41-v50 contents; no per-attempt scores; no Plan E1-E6 file found in paths read. Need those for STEP 2 if available.
