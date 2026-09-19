# Survival Simulator — Experiment Plan E1-E6

## Baseline
- v11: 898.485 local, 749.842 online
- Leader: 2096.489

## Experiments (priority order)
- E2 (v55): Senescence-aware demography. HIGHEST priority. Aging economics dominate late-game.
- E4 (v55): Energy economy. HIGH priority. Camping near trees, avoid unnecessary sprinting.
- E1 (v52): Directed evolution. MEDIUM. Free traits (max_energy, vision) should be exploited.
- E3 (v54): Predator counter-play. LOW priority. Predators are rare early game.
- E5 (v56): Persistent map via dead reckoning. ONLY if time remains.
- E6: Combine winners, sweep 2-3 params on DEV, confirm on HOLDOUT.

## Promotion gate
- Beat v11's 898.485 on DEFAULT
- Paired CI > 0 on DEV and HOLDOUT
- Full survival count does not drop
- Latency acceptable
