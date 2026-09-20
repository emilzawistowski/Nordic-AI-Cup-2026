# v54 Fast-Breeder — FAILED

Date: 2026-09-20
Change vs v11: flight economy (walk if speed >= 15.5) + elite breeding (top 3% only)
Local mean (10 seeds): **783.485**
v11 baseline (same session): ~917 (fresh), 864 (earlier run)
Delta: **~-135 vs v11**

## Ratchet diagnostic
- Ratchet fired at t=600 in only 3/10 seeds
- Two colonies died before t=600
- Even in seeds where ratchet fired, colony died before t=1200
- All 9,650 fast-agent predator responses used walking-cost flight, as designed

## Conclusion
Fast-breeder hypothesis is FALSIFIED on these seeds.
Reason: mutation rate too slow to establish speed>=15.5 lineage before predators
scale (4-5 by t=400). Without enough fast agents, flight economy has no effect.

## Key mechanic discovery (preserve for future work)
From environment.py `update_entity_position`:
- Sprint is silently blocked when `energy < max_energy / 5` (not `energy > 20` as v11 assumes)
- Movement cost: 0.05/unit for distance <= speed, 0.05*speed + 0.5*(distance-speed) above
- Agent at speed=16 walking = 0.8/tick. Sprinting at 20 = ~5.6/tick (7x)
- Predator sprint = 15. Speed >= 15.5 agents outrun a charging predator by WALKING
These facts are correct, but not exploitable without a fast population.

## Combined audit history
| Version | Local mean | Change | Result |
|---|---|---|---|
| v11 | ~900 | baseline | PRODUCTION |
| v52 camp-in-place | 616.9 | idle instead of wander | FAILED (-28%) |
| v52 predator ctrl | 679.4 | short-horizon interception | FAILED (-25%) |
| v53 patch memory | 755.7 | tree abandonment after dry spell | FAILED (-16%) |
| v54 fast-breeder | 783.5 | flight economy + elite breeding | FAILED (-15%) |
| v51 predictive escape | 795.8 | 1.5-tick extrapolation | FAILED (-11%) |

## Final decision
**Deploy v11.** All attempted architectural changes lost to the baseline.
Remaining gap to leader (2096) requires RL or population-level coordination,
not single-agent heuristic changes.

## Reference files
- _v54_FAILED_fast_breeder.py — preserved for the sprint-gate discovery
- v52/v53 in codex worktree ~/.codex/worktrees/2940/ (not copied to main)
