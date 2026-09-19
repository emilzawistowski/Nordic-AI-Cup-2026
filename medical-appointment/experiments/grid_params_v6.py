"""Step 1 — hardcoded constants in the v4 shrink path (attempt 8 audit).

Production shrink = medical_evidence_v4.calibrate_indexed_span
(inherited from medical_spans_v2, attempt 3):

| # | Constant | Value | Location | Tunable? |
|---|----------|-------|----------|----------|
| 1 | START_SHIFT (ds) | +0.3s | v4:29, spans_v2:32 | YES — grid {0.0,0.15,0.3,0.45,0.6} |
| 2 | END_TRIM (de) | 0.0s | v4:30, spans_v2:33 | YES — grid {0.0,0.25,0.5} (E1 said trim hurts; retest) |
| 3 | F-beta | 1.0 (balanced F1) | v4:43-53 | YES — grid {0.5,1.0,2.0}; beta<1 favors precision=shorter spans, directly attacks CONTAINS over-cover |
| 4 | MAX_RUN (cap on sub-run length) | uncapped in v4 (spans_v2: MAX_RUN=3 legacy) | v4:126-142 | YES — grid {1,2,3,inf}; gold ~= 2 sentences |
| 5 | Fallback trigger tau | best_score <= 0 | v4:152 | YES — grid {0.0, 0.15} |
| 6 | Shift on fallback | False (full range returned WITHOUT +0.3) | v4:153 | YES — grid {False, True}; note the inconsistency: fallback spans skip the start-bias correction |
| 7 | Tie-break on equal score | first-max-wins (earliest start, then shortest) | v4:146-150 | Fixed for coarse grid; revisit in refine if near-threshold |
| 8 | _STOP word list | 46 stopwords | v4:32-36 | NO — linguistic fixed list, not a numeric knob |
| 9 | Sentence split regex | .?! | v4:57 | NO — structural (attempt-6: granularity is load-bearing) |

Coarse grid: 5 x 3 x 3 x 4 x 2 x 2 = 720 combos, nested 5-fold CV.
Success bar: held-out tIoU delta >= +0.01 over v4 (0.5386 on cache).
"""

import logging
import re

logger = logging.getLogger(__name__)

# Re-export of the production constants for the gridsearch script.
from medical_evidence import normalize_tokens  # noqa: F401

BASELINE = {
    "ds": 0.3,
    "de": 0.0,
    "beta": 1.0,
    "max_run": float("inf"),
    "tau": 0.0,
    "shift_on_fallback": False,
}

COARSE_GRID = {
    "ds": [0.0, 0.15, 0.3, 0.45, 0.6],
    "de": [0.0, 0.25, 0.5],
    "beta": [0.5, 1.0, 2.0],
    "max_run": [1, 2, 3, float("inf")],
    "tau": [0.0, 0.15],
    "shift_on_fallback": [False, True],
}

REFINE_GRID = {
    # Filled after coarse results; narrow around the winner.
    "ds": [],
    "de": [],
    "beta": [],
    "max_run": [],
    "tau": [],
    "shift_on_fallback": [],
}
