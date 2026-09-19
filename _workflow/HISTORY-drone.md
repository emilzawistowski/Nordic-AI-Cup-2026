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
