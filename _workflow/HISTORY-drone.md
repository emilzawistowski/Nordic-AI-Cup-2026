# Attempt History — drone-flyby

## Scoring formula
mAP-based (COCO mAP@0.50 on hidden validation frames).
Score depends on: correct class labels + tight bounding boxes + camera steering.

## Production status — 2026-09-19
Current `example.py` is restored from `example_v4_backup.py`: V4 tracker + original horizontal camera + unchanged V2 detector weights.
Best reported online validation: **V4 0.0111**, zero errors. V5 **0.0091** regressed and was reverted.
Leader score reported by the user: **0.908**. Step 3 and further experiments are held pending the user's next direction.

## Attempt log
| # | Date | Local mAP | Online score | Model | What changed |
|---|------|-----------|--------------|-------|--------------|
| 0 | 2026-09-17 | ~0 | - | - | Baseline (edge detection) |
| 1 | 2026-09-18 | 0.069 | 0.007506 | drone_yolo_v1_validated.pt | YOLO11n single-stage, horizontal L2 sweep |
| 2 | 2026-09-18 | - | 0.002430 | objectness+classifier (two-stage) | Two-stage HURTS online — overfit |
| 3 | 2026-09-19 | 0.090 | NOT SUBMITTED | two-stage V3 + safe camera | Local better, online not tested yet |
| 4 | 2026-09-19 | 0.075 (was 0.069 V1) | NOT SUBMITTED | drone_yolo_v2_augmented.pt | Direction A: YOLO11n single-stage retrained on copy-paste diversified set (500 syn tiles, 60% zero-AP oversample); example.py now points at drone_detector_v2.py; fixed train script save-path bug (weights were in runs/detect/runs/detect/) |

## BEST ONLINE: V4, score 0.0111 (user-reported 2026-09-19)

Historical verified V1 score: 0.007506. The latest user report separately identifies
the previous V2 online baseline as 0.0075; older attempt records remain unchanged.

## Dead ends
- Two-stage pipeline: better local mAP but much worse online generalization
- Camera without movement guards: refused moves (618, 619, 697 px violations)

## Next ideas (ranked by expected impact)
1. REVERT example.py to V1 single-stage (do this first!)
2. Copy-paste augmentation + background replacement → better generalization
3. Zero-shot Grounding DINO / OWL-ViT (no retraining needed)
4. Fix zero-AP classes: hangar, mine_roller, medium_plane, small_launcher
5. Ensemble V1+V3 with NMS

## Known model files
- models/drone_yolo_v1_validated.pt  ← BEST, SHA 05d1506d
- models/drone_objectness_yolo11s_v1.pt
- models/drone_classifier_yolo11s_v1.pt

---
*Append new attempts below this line*
- Attempt #4 detail: V1 baseline re-verified (realtime 0.069 = logged 0.069; direct 0.0687). V2 realtime 0.075, direct 0.0755 — repeatable +0.006 gain (helicopter 0.109→0.158, small_plane 0→0.059; hangar/medium_plane/mine_roller still 0). Tile-level probe @conf0.25: V2 prec 0.776/rec 0.963 vs V1 0.679/0.981; detections stable under color-shift+flip. Generalization note: expected to generalize slightly BETTER than the bare +0.006 suggests (augmentation targets scene-overfit: new backgrounds/scales/lighting), BUT pastes are same-scene crops with visible seams and train-val mAP50 was 0.926 (tight scene fit), so online gain may be smaller than local — submit to verify; V1 checkpoint untouched (SHA 05d1506d) as fallback.


## V4 online verification — 2026-09-19

- User-reported V4 online validation: **0.0111**, zero errors, versus user-reported V2 **0.0075** (approximately +48% using rounded values; user reports +47%). **KEEP**. This is evidence of positive tracker transfer on this validation sequence.
- Worktree attempt #5 local V4 mAP: 0.1683168317, realtime 0.168. Worktree attempt #6 V5 local mAP: 0.2191766052, realtime 0.219. Those results preceded this main-tree promotion; no new experiment was started here.

## 2026-09-19 — promoted v5 to production — local mAP 0.219 — KEEP

- Copied files from the isolated worktree into the main repo working tree; no branch merge or cherry-pick. Copied `drone_camera_v5.py`, `example_drone_v5.py`, and their required unchanged `drone_tracker_v4.py` dependency.
- Main `example.py` now exports the V5 pipeline's predictor; detector remains `models/drone_yolo_v2_augmented.pt` (SHA-256 `4c98a7d9f21e34bc879df3636c003878a5e028732065cb885f82a6191dd13c22`).
- Observed checkout discrepancy: main-tree `example.py` was still V2, byte-identical to the saved V2 source, not V4. Preserved that exact file as `example_v2_backup.py` before overwriting. Saved the verified worktree V4 production entry point byte-for-byte as `example_v4_backup.py`; it uses the copied V2 backup's unchanged camera policy. No existing backup was overwritten.
- Final test used the main-tree, unmodified `api.py`, served on temporary port 9056, and unmodified `local_evaluator.py --url http://127.0.0.1:9056/predict --realtime --verbose`.
- Result: **COCO mAP@0.50 0.219; KEEP**. Frames accepted 25/25; skipped 0; unanswered 0; timeouts 0; HTTP errors 0; invalid responses 0; refused moves 0. Round trip mean/median/max 90/83/326 ms. Temporary verification server stopped after testing; existing production server processes were not restarted.
- `api.py`, `dtos.py`, `local_evaluator.py`, `utils.py`, `visualize.py`, and both V1/V2 model weights verified unchanged by SHA-256. Unrelated main-tree medical work preserved and excluded from the promotion commit.
- V5 online validation **PENDING**. Restart the existing production API to load the updated `example.py` before submitting its endpoint. Online comparison target is now **>0.0111**, and competition target **>0.908**.
- **Do not start roadmap step 3 or another experiment until the user provides the V5 online number.**


## 2026-09-19 — V5 online 0.0091 vs V4 0.0111 — REVERTED

- Online validation reported by the user: V5 **0.0091**, V4 **0.0111**; V5 regressed approximately **18.0%** despite the local increase from 0.168 to 0.219. V5 also produced a reported camera-limit error absent from V4's online run. The combined V5 pipeline failed the online keep gate.
- **Decision: REVERT V5 / KEEP V4.** Restored main-tree `drone-flyby/example.py` byte-for-byte from `example_v4_backup.py`. V5 reference files (`example_drone_v5.py` and `drone_camera_v5.py`) are retained unchanged. There is no `example_v5.py` in this checkout; the existing versioned entry point is `example_drone_v5.py`.
- Required safety note, as reported/instructed by the user: **"V5 camera bypassed L2 551px safe-move limit. Any future camera change MUST route through the same safety wrapper used by V4."**
- Code-provenance qualification: V4 currently imports `choose_next_view` from `example_v2_backup.py`, whose horizontal step uses 0.9 times the supplied movement limit. It does not import `drone_camera_safe.py`. V5 uses separate inline distance checks against the supplied limit. The triggering online request/command was not supplied, so the exact failure mechanism has not been independently reproduced. This records the incident and mandatory future shared-safety requirement without claiming a verified root cause. No safety code was changed in this rollback.
- Restored V4 final realtime check through unchanged `api.py` on temporary port 9056: **COCO mAP@0.50 0.168**, accepted 25/25, skipped 0, unanswered 0, timeouts 0, HTTP errors 0, invalid responses 0, refused moves 0. Round trip mean/median/max **66/60/218 ms**.
- Protected framework files, V5 reference files, V4 backup, and V1/V2 model weights verified unchanged by SHA-256. No files deleted. Unrelated work excluded from the commit.
- Temporary verification API stopped after testing; existing production processes were not restarted. Restart the production API to load restored V4 before its next use.
- **Do NOT start roadmap step 3 or any new experiment. Wait for the user's next direction.**
