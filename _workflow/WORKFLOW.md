# Nordic AI Cup 2026 — Master Workflow

## How This Works

You iterate in cycles. Each cycle:
1. **Browser AI** (Claude/Gemini/DeepSeek) = **strategist** — reads context, proposes idea
2. **Coding AI** (Antigravity/Cursor/etc.) = **builder** — receives a ready-to-execute prompt, makes changes, runs evaluator, commits
3. **Browser AI** = reads result, decides next step

You can run multiple coding AIs in parallel on different machines/worktrees.

---

## Files You Need to Know

```
Nordic-AI-Cup-2026/
├── _architecture.txt        ← Full snapshot: scores, files, models, dead ends
├── _workflow/
│   ├── WORKFLOW.md          ← THIS FILE (paste at top of every new chat)
│   ├── HISTORY-medical.md   ← Attempt log for medical-appointment
│   ├── HISTORY-drone.md     ← Attempt log for drone-flyby
│   └── HISTORY-survival.md  ← Attempt log for survival-simulator
├── medical-appointment/
│   ├── example.py           ← ENTRY POINT (the only file that gets submitted)
│   ├── api.py               ← DO NOT TOUCH
│   └── dtos.py              ← DO NOT TOUCH
├── drone-flyby/
│   ├── example.py           ← ENTRY POINT
│   ├── api.py               ← DO NOT TOUCH
│   └── dtos.py              ← DO NOT TOUCH
└── survival-simulator/
    ├── agent_server.py      ← ENTRY POINT (not example.py!)
    └── simulation_server.py ← DO NOT TOUCH
```

---

## Rules Every Coding AI Must Follow

1. **Never modify**: `api.py`, `dtos.py`, `simulation_server.py`, `local_evaluator.py`
2. **Never delete** existing files — add new versioned files instead (e.g. `medical_reasoner_v2.py`)
3. **Before any change**: note the current local evaluator score
4. **After any change**: run local evaluator, compare score
5. **Only promote** if new score > old score (no regressions allowed)
6. **Always commit** after each attempt: `git add -A && git commit -m "<usecase>: attempt <N>, score <X>"`
7. **Always append** to the HISTORY file for the use case you're working on
8. **Models** are referenced by relative path — keep all `.pt` files in `<usecase>/models/`

---

## Local Evaluator Commands

```bash
# medical-appointment
cd medical-appointment
python api.py &            # start server (port 9051)
python local_evaluator.py  # run evaluator

# drone-flyby
cd drone-flyby
python api.py &            # start server (port 9053)
python local_evaluator.py  # run evaluator

# survival-simulator
cd survival-simulator
python simulation_server.py &  # start sim server (port 9050)
python agent_server.py &       # start agent server (port 9052)
python local_evaluator.py      # run evaluator
```

---

## How to Work in Parallel (Multiple Machines / Worktrees)

Each parallel worker needs its OWN copy of the repo to avoid file conflicts:

```bash
# Create a parallel worktree for e.g. drone experiments
cd ~/Nordic-AI-Cup-2026
git worktree add ../drone-exp-1 winning-solution
```

Or just clone to a separate folder and work there.
**Rule**: only merge back to `winning-solution` branch if local score improves.

---

## Prompt Template for Browser AI (Claude/Gemini/DeepSeek)

Paste this at the start of every browser chat:

---
**[START OF CONTEXT — PASTE AT TOP OF EVERY NEW CHAT]**

I am competing in the Nordic AI Cup 2026 (https://github.com/amboltio/Nordic-AI-Cup-2026).
I need your help strategizing improvements to my solution.

**My current scores (local evaluator):**
- medical-appointment: 0.6394 (Accuracy 0.974, tIoU 0.416)
- drone-flyby: 0.0075 online (local mAP 0.090)
- survival-simulator: [run benchmark to get score]

**Architecture summary (paste _architecture.txt here)**
**Attempt history for this use case (paste HISTORY-<usecase>.md here)**

Your job: propose ONE concrete improvement I should try next.
Output format: a ready-to-paste prompt for a coding AI (Antigravity/Cursor).
The prompt must:
- State which files to modify (and which NOT to touch)
- Give exact code changes or clear algorithm description
- Tell the coding AI how to run the local evaluator and what threshold counts as success
- Include: "Append result to _workflow/HISTORY-<usecase>.md"

**[END OF CONTEXT]**

---

## Prompt Template for Coding AI (Antigravity/Cursor)

The browser AI will generate this for you. But every prompt MUST include:

```
FRAMEWORK RULES (always apply):
- Repo: ~/Nordic-AI-Cup-2026, branch: winning-solution
- DO NOT modify: api.py, dtos.py, simulation_server.py, local_evaluator.py
- DO NOT delete any files. Add versioned new files.
- Current best score: [X] — only keep changes that beat this
- After finishing: git add -A && git commit -m "<usecase>: attempt <N>, score <X>"
- Append to _workflow/HISTORY-<usecase>.md: date, attempt #, what changed, score
```

