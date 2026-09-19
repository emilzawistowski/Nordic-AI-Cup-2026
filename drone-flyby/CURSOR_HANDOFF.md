# Drone Flyby handoff

Work only inside drone-flyby.
Do not modify medical-appointment or survival-simulator.

## Best online checkpoint

Score: 0.007505635095660476
Attempt UUID: 2593fd809f634defab094b73a9c0d607
Model: models/drone_yolo_v1_validated.pt
SHA-256: 05d1506da89edf612c2556cab0f2274767bef3b3d80e885104f701e5b180b2cb

Pipeline:
- single-stage YOLO11n multiclass detector
- horizontal Level-2 sweep
- model warmup

## Experimental two-stage V3

Models:
- models/drone_objectness_yolo11s_v1.pt
- models/drone_classifier_yolo11s_v1.pt

Local realtime mAP after camera reset: 0.090
Online validation score: 0.0024299133210024296
Attempt UUID: 4d9bba5eedd741e0ae38f4fe2c1adbb1

The two-stage model improved local mAP but generalized worse online.

## Current camera work

File: drone_camera_safe.py

Goal:
- reset route at frame_index == 0
- never exceed request.camera_constraints.maximum_center_delta
- do not advance to the next target until the current target is reached
- complete a repeating serpentine raster on long hidden sequences
- produce zero refused camera moves

Confirm example.py uses:
requested_view=choose_next_view_safe(request)

## Known hidden-validation facts

- Hidden validation contains at least 246 frames.
- Level-2 movement limit observed: 551 pixels.
- Previous camera generated refused moves of 618, 719 and 697 pixels.
- Validation appears to retain the best historical score, but final evaluation is one attempt.

## Local references

Single-stage V1 realtime mAP: 0.069
Two-stage V3 repeatable realtime mAP: 0.090

Objectness model local validation:
- precision 0.9773
- recall 0.9630
- mAP50 0.9807
- mAP50-95 0.6782

Classifier local validation:
- top1 1.0
- top5 1.0

These local metrics are optimistic because all frames show the same physical instance of each class.

## Priorities

1. Finish and test safe camera.
2. Keep the validated V1 checkpoint immutable.
3. Add tests for maximum camera movement.
4. Improve generalization using predicted-box crops, copy-paste, background replacement and independent class examples.
5. Use frequent online validations, but never trigger final evaluation yet.
