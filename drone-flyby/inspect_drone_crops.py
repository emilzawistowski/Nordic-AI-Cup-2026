import json
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO


model = YOLO("models/drone_yolo_v1.pt")
source = Path("src/helsinki")
stats = defaultdict(lambda: [0, 0, 0.0])

for annotation_path in sorted((source / "annotations").glob("*.json")):
    data = json.loads(annotation_path.read_text())
    frame = int(data["frame"])

    image = cv2.imread(
        str(source / "images" / f"frame_{frame:06d}.png")
    )

    for annotation in data["annotations"]:
        expected = annotation["object_id"]
        x1, y1, x2, y2 = map(int, annotation["bbox"])

        size = max(x2 - x1, y2 - y1)
        padding = max(20, int(size * 1.5))

        left = max(0, x1 - padding)
        top = max(0, y1 - padding)
        right = min(image.shape[1], x2 + padding)
        bottom = min(image.shape[0], y2 + padding)

        crop = image[top:bottom, left:right]

        result = model.predict(
            crop,
            imgsz=640,
            conf=0.01,
            device="mps",
            verbose=False,
        )[0]

        predicted = None
        confidence = 0.0

        if result.boxes is not None and len(result.boxes) > 0:
            best = int(result.boxes.conf.argmax().item())
            class_id = int(result.boxes.cls[best].item())
            predicted = result.names[class_id]
            confidence = float(result.boxes.conf[best].item())

        stats[expected][0] += 1
        stats[expected][2] += confidence

        if predicted == expected:
            stats[expected][1] += 1

for object_id in sorted(stats):
    count, correct, confidence_sum = stats[object_id]

    print(
        f"{object_id:20s} "
        f"correct={correct:2d}/{count:2d} "
        f"accuracy={correct / count:6.3f} "
        f"mean_conf={confidence_sum / count:6.3f}"
    )
