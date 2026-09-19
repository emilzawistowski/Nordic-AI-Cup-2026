# Attempt History — survival-simulator

## Scoring formula
Score = total simulation time survived + sum(fruit.energy / 1000)
Full survival (3000s) unlocks extra score from fruit collection.
Key: survive first, eat fruit second.

## Leaderboard Status
- **Current Submitted Best**: 749.842
- **Leaderboard #1 Score**: 2096.489

## Active policy: survival_policy_v11.py (imported by agent_server.py)
Latest experimental: survival_policy_v50.py (NOT yet benchmarked or deployed)

## Attempt log
| # | Date | Online/Bench Score | Policy | What changed |
|---|------|--------------------|--------|--------------|
| 0 | 2026-09-17 | - | baseline | Original |
| 1 | 2026-09-18 | 749.842 | v11 | Active in server (Current best online score) |
| ... | 2026-09-18/19 | - | v12-v50 | Parameter tuning, clustering prevention |

## v11 known behavior
- Predator avoidance: weighted escape, sprint threshold 55 units
- Fruit harvest on sight
- Tree forage if energy_ratio < 0.85 OR age > 60
- Sine-wave wander (spreads agents geographically)
- Spawn if energy >= 185, age >= 10, no predators

## v50 changes (untested)
- Tree forage threshold: energy_ratio < 0.70 (was 0.85) AND age > 90 (was 60)
- Goal: reduce tree clustering → reduce predator kills

## Dead ends (v3–v50)
- Aggressive early spawning → starvation
- Strict energy conservation → predator deaths
- Tree clustering → easy predator group kills
- Complex wall-avoidance → marginal benefit

## Next ideas (ranked by expected impact)
1. Benchmark v50 vs v11 on 10 seeds to see if average score > 749.842
2. Flee-zone memory: remember last N predator positions, avoid those areas
3. Parametric sweep of energy_ratio threshold (0.5–0.9) and age threshold (40–120)
4. Consider spawning only when energy > 200 (more buffer before spawn cost)

---
*Append new attempts below this line*
