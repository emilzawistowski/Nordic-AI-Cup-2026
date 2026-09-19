# AUDIT STEP 4 — Experiment: Forest-Anchor + Senescent Decoy
## Name
Forest-Anchor + Senescent Decoy (new policy, not a v11 weight tweak).
## Core logic (replaces flee>fruit>tree>wander)
- Predator: young (age<120): sprint only if dist<=55, else hold position. Old (age>=120): decoy mode — sprint directly away from colony anchor to drag predator off, no energy guard, accept death.
- Fruit: harvest only if dist<=40u. Ignore distant fruit (avoids 0.5/tick chase for 20-60 energy that rots at age 100 anyway).
- Tree: anchor to nearest visible tree, prefer Forest/Swamp biome; orbit <18u at 0.3x or stop. Never migrate between trees.
- Idle: default is STOP (0.1/tick), not sine-wander. Move only to eat anchored fruit or flee. Spawn only if young (age<100) & energy>=185 & no predator within 55u; old never spawn.
## Why it beats v11
- Energy: idle 0.1 vs wander 0.5 saves 0.4/tick. Over 1000s = 400 energy = 4 spawns or 66 sprints. Funds late-game fleeing v11 cannot afford.
- Senescence: age 150 drain is 1.5/tick — old agents are already dead economically. Converting them to decoys buys young agents predator-free ticks instead of wasting 185-energy spawns on them (v45 failure fixed).
- Dispersal via anchors, not wander: static anchors on high-yield biomes (Forest 1.0 vs River 0.0) raise fruit/tick without movement cost.
## Risks
- Clustering: static anchors risk multi-kill; mitigate by one-anchor-per-agent (id%N) + decoy pull.
- Drought: bad anchor (Desert/River) starves; need biome fallback to nearest fruit<=40u.
- Fruit loss: ignoring far fruit cuts bonus if survival reaches 3000s.
## Success metric
- `benchmark_any_policy.py` 10 default seeds + 20-seed broad. Win if mean >v11 (898/10-seed, 822/20-seed) AND survival time up. Split deaths: starve vs predator vs age; fruit bonus separately.
