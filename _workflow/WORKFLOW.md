# Nordic AI Cup 2026 — Master Workflow

## How This Works

You iterate in cycles:
1. **Browser AI** (Claude/Gemini/DeepSeek) — generates ideas and produces a ready-to-paste
   prompt for the coding AI. This is the main source of new ideas, not just a fallback.
2. **Coding AI** (Antigravity/Cursor) — receives the prompt, implements, tests, commits.
   Use this sparingly (token budget). Each session should have a single clear task.
3. Repeat.

---

## Files You Need to Know

```
Nordic-AI-Cup-2026/
├── _architecture.txt            ← Full snapshot: scores, files, models, dead ends
├── _workflow/
│   ├── WORKFLOW.md              ← THIS FILE
│   ├── HISTORY-medical.md       ← Attempt log for medical-appointment
│   ├── HISTORY-drone.md         ← Attempt log for drone-flyby
│   └── HISTORY-survival.md      ← Attempt log for survival-simulator
├── medical-appointment/
│   ├── example.py               ← ENTRY POINT
│   ├── api.py                   ← DO NOT TOUCH
│   └── dtos.py                  ← DO NOT TOUCH
├── drone-flyby/
│   ├── example.py               ← ENTRY POINT
│   ├── api.py                   ← DO NOT TOUCH
│   └── dtos.py                  ← DO NOT TOUCH
└── survival-simulator/
    ├── agent_server.py          ← ENTRY POINT (not example.py!)
    └── simulation_server.py     ← DO NOT TOUCH
```

---

## Rules Every Coding AI Must Follow

1. **Never modify**: `api.py`, `dtos.py`, `simulation_server.py`, `local_evaluator.py`
2. **Never delete** existing files — add new versioned files instead
3. **Before any change**: record the current local evaluator score
4. **After any change**: run local evaluator, compare scores
5. **Only promote** if new score > old score — revert otherwise
6. **Always commit** after each attempt: `git add -A && git commit -m "<usecase>: attempt <N>, score <X>"`
7. **Always append** to the HISTORY file for the use case being worked on
8. **Models**: keep all `.pt` files in `<usecase>/models/` with versioned names

---

## Local Evaluator Commands

```bash
# medical-appointment
cd medical-appointment && python api.py &
python local_evaluator.py

# drone-flyby
cd drone-flyby && python api.py &
python local_evaluator.py

# survival-simulator (benchmark mode — no server needed)
cd survival-simulator
python benchmark_any_policy.py --policy src.utils.controllers.survival_policy_vN
```

---

## Working in Parallel

When running two or more coding AIs on the same use case simultaneously,
each worker must have its own isolated copy of the repo to avoid HISTORY conflicts.

### Setup (do this once per parallel experiment)

```bash
# From the main repo root
git worktree add ../nordic-exp-drone-2 winning-solution
```

Or simply duplicate the folder manually. Each copy gets its own branch:

```bash
cd ../nordic-exp-drone-2
git checkout -b exp/drone-grounding-dino
```

### HISTORY in parallel work

Each experiment branch keeps its own `_workflow/HISTORY-<usecase>.md`.
When an experiment wins (beats the score on the main branch):

1. Copy the winning files back to main repo (`winning-solution` branch)
2. Manually append the winning attempt entry to the main `HISTORY-<usecase>.md`
3. Commit on `winning-solution`
4. Delete or archive the experiment worktree

### Rule: never merge directly

Do not `git merge` experiment branches into `winning-solution` — copy only the
files that changed (e.g. `example.py`, `medical_reasoner_v2.py`) and commit manually.
This keeps the main branch clean.

