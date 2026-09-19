# Survival Simulator — Implementation Plan (post-audit)

## Baseline
v11: 898.485 local (10 seeds), 749.842 online. Leader: 2096.489.

## Audit findings (from AUDIT-STEP1..4)
- 70% of v12-v51 tweaked flee/spawn/forage (local perturbations of v11)
- v11 sits at Pareto optimum for its architecture
- Nobody changed priority order, nobody camped
- Idle 0.1/tick vs walk 0.5/tick: 5x saving over 3000s = ~1200 energy free
- Senescence: old agents need ROLE CHANGE, not higher spawn threshold

## Experiments (sequential, one change each)
### v52 = Camp-in-place
- Change: replace sine-wander with idle (move_distance=0)
- Everything else = v11
- Expected: save ~1200 energy/3000s
- Success: bench > 898.485 on 10 seeds

### v53 = Camp-at-tree (only if v52 wins)
- Add: anchor to nearest tree, orbit <18u, prefer Forest/Swamp
- Expected: fruit access without wander cost

### v54 = Senescent decoy (only if v53 wins)
- Old agents (age>120): sprint away from colony to pull predator off
- Young agents: conserve energy, don't help
- Expected: young survivor rate up

## Rules
- One change per experiment
- Bench each on 10 seeds before commit
- Do not promote unless > v11
- Git commit each attempt
