# Attempt History — survival-simulator

## Scoring formula
Score = total simulation time survived + sum(fruit.energy / 1000)
Full survival (3000s) unlocks extra score from fruit collection.
Key: survive first, eat fruit second.

## Active policy: survival_policy_v11.py (imported by agent_server.py)
Latest experimental: survival_policy_v50.py (NOT yet benchmarked or deployed)

## Attempt log
| # | Date | Avg score (10 seeds) | Policy | What changed |
|---|------|----------------------|--------|--------------|
| 0 | 2026-09-17 | - | baseline | Original |
| 1-10 | 2026-09-18 | - | v3-v11 | Various rule iterations |
| ... | 2026-09-18/19 | - | v12-v50 | Parameter tuning, clustering prevention |
| CURRENT | - | ? | v11 | Active in server (v11 confirmed best so far) |

## v11 known behavior
- Predator avoidance: weighted escape, sprint threshold 55 units
- Fruit harvest on sight
- Tree forage if energy_ratio < 0.85 OR age > 60
- Sine-wave wander (spreads agents geographically)
- Spawn if energy >= 185, age >= 10, no predators

## v50 changes (untested)
- Tree forage threshold: energy_ratio < 0.70 (was 0.85) AND age > 90 (was 60)
- Goal: reduce tree clustering → fewer predator kills

## Dead ends (v3–v50)
- Aggressive early spawning → starvation
- Strict energy conservation → predator deaths
- Tree clustering → easy predator group kills
- Complex wall-avoidance → marginal benefit

## Next ideas (ranked by expected impact)
1. Benchmark v50 vs v11 on 10 seeds: python benchmark_any_policy.py --policy src.utils.controllers.survival_policy_v50
2. Flee-zone memory: remember last N predator positions, avoid those areas
3. Parametric sweep of energy_ratio threshold (0.5–0.9) and age threshold (40–120)
4. Consider spawning only when energy > 200 (more buffer before spawn cost)

## How to run benchmark
```bash
cd survival-simulator
python benchmark_any_policy.py --policy src.utils.controllers.survival_policy_v11
python benchmark_any_policy.py --policy src.utils.controllers.survival_policy_v50
```

---
*Append new attempts below this line*
