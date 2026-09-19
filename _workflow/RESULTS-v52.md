# v52 Camp-in-place — FAILED

Date: 2026-09-19
Change vs v11: final else branch = idle instead of sine-wander

| Seed | v11 | v52 | Δ |
|---|---|---|---|
| 2204119010 | 942.5 | 468.6 | -474 |
| 1 | 978.5 | 769.7 | -209 |
| 42 | 885.2 | 782.2 | -103 |
| 137 | 842.1 | 883.4 | +41 |
| 2026 | 856.8 | 577.7 | -279 |
| 12345 | 697.0 | 358.9 | -338 |
| 8675309 | 955.8 | 525.5 | -430 |
| 31415926 | 788.3 | 637.7 | -151 |
| 271828182 | 790.1 | 470.6 | -319 |
| 3735928559 | 910.8 | 694.6 | -216 |
| **Mean** | **864.7** | **616.9** | **-247.8** |

## Interpretation
Camping kills. Wandering is not waste - it provides:
1. Fruit discovery (fruit only grows on trees)
2. Dispersion (avoids predator group kills)

## Lesson for next experiments
Conditional camping (only when near tree with fruit) may work.
Complete camping does not.

## Anomaly
Seed 42 in smoke test (--seeds 42): 868.8. In full run: 782.2.
Same seed, same policy. Benchmark has non-determinism or seed flag is interpreted differently.
