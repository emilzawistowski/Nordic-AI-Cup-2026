# AUDIT STEP 2A — v12-v51 Classification vs v11
| Version | Key change vs v11 (one line) | Category |
|---|---|---|
| v12 | Escape 6k->8k + value-weighted fruit energy/(d+5) | FORAGE |
| v13 | Escape 6k->10k + value-weighted fruit | FORAGE |
| v14 | Inverse-cubic escape 50k/(d^3+10) + best fruit | PREDATOR |
| v15 | Escape 10k + tree cluster centroid | FORAGE |
| v16 | Escape weight 6k->12k only | PREDATOR |
| v17 | Escape weight 6k->12k only (repeat) | PREDATOR |
| v18 | Face-the-predator to force pivot not charge | PREDATOR |
| v19 | Face+kite + spawn 185->155 + no swamp sprint | SPAWN |
| v20 | Phase repro 165 early / 195 late + trait exploit | SPAWN |
| v21 | Escape gain up + tree age trigger 60->50 | FORAGE |
| v22 | Dynamic escape scaling, sprint 55 + guardrail | PREDATOR |
| v23 | Speed-aware sprint to 65u, cutoff 25, golden tree offset | ENERGY |
| v24 | Wander turn smoothing 0.035->0.025 | WANDER |
| v25 | Selective breeding + conservative energy for 3000s | SPAWN |
| v26 | Late-game sprint 55->65 + tree reserve <50% | ENERGY |
| v27 | Wall buffer 35->32 only | OTHER |
| v28 | v7 hybrid: spawn 190/12, escape 5k, tree 0.82 | SPAWN |
| v29 | Spawn 210/15 + sprint only if energy>40 + wall 45 | SPAWN |
| v30 | Multi-agent repulsion <40u to disperse | WANDER |
| v31 | Fruit sprint if >70% & >80u + speed scaling | FORAGE |
| v32 | Wall 36.5 + sprint to 65u if 2+ predators | ENERGY |
| v33 | Flee turn bound 0.15->0.08 | PREDATOR |
| v34 | Per-agent wander frequencies (id%5) | WANDER |
| v35 | Speed-scaled move + flee limit 0.7x if ratio<0.20 | ENERGY |
| v36 | Ultra-conservative + hard sprint floor | ENERGY |
| v37 | rel_dir predator-facing awareness | PREDATOR |
| v38 | Aggressive breeding for sprint_speed evolution | SPAWN |
| v39 | Fruit sprint harvester + population engine | FORAGE |
| v40 | Lateral dodge 60/40 + tiered sprint 50/80u | PREDATOR |
| v41 | Flee nearest not average + rear check | PREDATOR |
| v42 | Unified potential-field navigation | OTHER |
| v43 | Dynamic sprint floor 40-65u by energy/count | ENERGY |
| v44 | Trait inheritance guard on spawn | SPAWN |
| v45 | Age-compensated spawn reserve (0.01*age drain) | SPAWN |
| v46 | High-yield tree cluster (age>=20) attraction | FORAGE |
| v47 | Density spawn guard + smooth edge | SPAWN |
| v48 | Line-of-sight choke + tangential escape | PREDATOR |
| v49 | Inheritance guard + 0.15 rad tangential offset | PREDATOR |
| v50 | Age-threshold forage trigger (reduce clustering) | FORAGE |
| v51 | 1.5-tick predictive escape + zone tree alloc | PREDATOR |
