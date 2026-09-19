# Attempt History — drone-flyby

## Scoring formula
mAP-based (COCO mAP@0.50 on hidden validation frames).
Score depends on: correct class labels + tight bounding boxes + camera steering.

## CRITICAL: Best online score uses single-stage V1, NOT current example.py!
Current example.py runs two-stage V3 which scores 3x WORSE online.
Best example: example_drone_best.py → copy to example.py before submitting.

## Attempt log
| # | Date | Local mAP | Online score | Model | What changed |
|---|------|-----------|--------------|-------|--------------|
| 0 | 2026-09-17 | ~0 | - | - | Baseline (edge detection) |
| 1 | 2026-09-18 | 0.069 | 0.007506 | drone_yolo_v1_validated.pt | YOLO11n single-stage, horizontal L2 sweep |
| 2 | 2026-09-18 | - | 0.002430 | objectness+classifier (two-stage) | Two-stage HURTS online — overfit |
| 3 | 2026-09-19 | 0.090 | NOT SUBMITTED | two-stage V3 + safe camera | Local better, online not tested yet |
| 4 | 2026-09-19 | 0.075 (was 0.069 V1) | NOT SUBMITTED | drone_yolo_v2_augmented.pt | Direction A: YOLO11n single-stage retrained on copy-paste diversified set (500 syn tiles, 60% zero-AP oversample); example.py now points at drone_detector_v2.py; fixed train script save-path bug (weights were in runs/detect/runs/detect/) |

## BEST ONLINE: Attempt #1 (single-stage V1, score 0.007506)

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
