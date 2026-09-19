# AUDIT STEP 2B — Pattern Analysis
1. Dominant: PREDATOR (12/40), then SPAWN (9), FORAGE (8), ENERGY (6), WANDER (3), OTHER (2). ~70% touch flee/spawn/forage balance.
2. Senescence: Yes. v45 directly (0.01*age drain, age-compensated reserve). Partial: v21 (tree age 60->50), v50 (age forage trigger), v20/v26 late-game thresholds.
3. Free traits: Yes. v20 (phase + trait exploit), v23 (speed-aware sprint to 65u), v25 (selective high-speed/vision breeding), v38 (sprint_speed evolution), v44/v49 (inheritance guard).
4. Most tweaked: Escape weight 6000->5k/8k/10k/12k + sprint trigger 55u->50/65u + spawn 185->155/165/190/210. Escape/sprint touched in ~18 versions.
5. Hypothesis: v11 sits at Pareto optimum (flee-if-<=55 + 185 spawn + 0.85/60 forage). Each variant optimizes one leg and breaks another: stronger flee burns energy->starvation; higher spawn reserve stalls population; conservative forage loses fruit bonus. No variant jointly solves predator scaling + senescence + dispersal, so all regress to mean.
