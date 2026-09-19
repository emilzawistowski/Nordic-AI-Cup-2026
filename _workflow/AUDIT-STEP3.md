# AUDIT STEP 3 — Strategic Findings
## (a) Why priority order never changed
- Flee>fruit>tree>wander looks "obviously correct" — no one questions survival-first.
- Workflow rewards safe tweaks (promote only if >v11); reordering risks catastrophic bench, so workers tune weights instead.
- No energy model: without idle 0.1 vs walk 0.5 vs sprint 6 math, flee-first feels free. It is not (12x).
- Result: 40 variants polish the same decision tree, never test energy-first or camp-first.
## (b) Why v45 senescence fix failed
- v45 raised spawn reserve for old parents but kept old-agent behavior identical (same wander/flee/forage burn).
- Old drain is -0.01*age/tick: age 150 = 1.5/tick, 3x walk cost. A threshold delays death by ~50 ticks, does not prevent it.
- Fewer spawns = smaller population buffer when predators scale, so group wipes earlier. Paid cost, got no new capability.
- Needed role change (decoy/suicide/idle), not a higher gate.
## (c) Most under-exploited mechanic
- Movement economy + biome anchor. Nobody camps: all variants sine-wander at 0.4-0.9x speed (0.5/tick) vs idle 0.1/tick.
- Saving 0.4/tick x 3000s = ~1200 energy/agent = 12 spawns or 200 sprints, for free.
- Combined with biome (Forest 1.0 / Swamp 0.9 vs Desert 0.1 / River 0.0): anchoring on Forest trees + idling is untested. Free traits (vision/max_energy, no upkeep) second.
