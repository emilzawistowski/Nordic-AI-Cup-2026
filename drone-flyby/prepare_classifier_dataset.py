import json
import random
import shutil
from pathlib import Path

import cv2


SOURCE = Path("src/helsinki")
OUTPUT = Path("datasets/drone_classifier")
SIZE = 224
RNG = random.Random(2026)

VARIANTS = (
    (0.25, 0.00),
    (0.45, 0.10),
    (0.75, 0.15),
    (1.10, 0.20),
    (1.60, 0.20),
)


def make_crop(image, bbox, padding, jitter):
    x1, y1, x2, y2 = map(float, bbox)

    width = x2 - x1
    height = y2 - y1
    object_size = max(width, height)

    side = max(24.0, object_size * (1.0 + 2.0 * padding))

    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0

    center_x += RNG.uniform(-jitter, jitter) * object_size
    center_y += RNG.uniform(-jitter, jitter) * object_size

    left = int(round(center_x - side / 2.0))
    top = int(round(center_y - side / 2.0))
    right = int(round(center_x + side / 2.0))
    bottom = int(round(center_y + side / 2.0))

    pad_left = max(0, -left)
    pad_top = max(0, -top)
    pad_right = max(0, right - image.shape[1])
    pad_bottom = max(0, bottom - image.shape[0])

    left = max(0, left)
    top = max(0, top)
    right = min(image.shape[1], right)
    bottom = min(image.shape[0], bottom)

    crop = image[top:bottom, left:right]

    if crop.size == 0:
        return None

    if pad_left or pad_top or pad_right or pad_bottom:
        crop = cv2.copyMakeBorder(
            crop,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_REFLECT_101,
        )

    return cv2.resize(
        crop,
        (SIZE, SIZE),
        interpolation=cv2.INTER_CUBIC,
    )


def main():
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)

    counts = {"train": {}, "val": {}}

    annotation_files = sorted(
        (SOURCE / "annotations").glob("frame_*.json")
    )

    for annotation_path in annotation_files:
        data = json.loads(
            annotation_path.read_text(encoding="utf-8")
        )

        frame = int(data["frame"])
        image_path = SOURCE / "images" / f"frame_{frame:06d}.png"
        image = cv2.imread(str(image_path))

        if image is None:
            raise RuntimeError(f"Cannot read {image_path}")

        for object_index, annotation in enumerate(data["annotations"]):
            class_name = annotation["object_id"]

            for variant_index, (padding, jitter) in enumerate(VARIANTS):
                split = "val" if variant_index == 2 else "train"

                output_directory = OUTPUT / split / class_name
                output_directory.mkdir(parents=True, exist_ok=True)

                crop = make_crop(
                    image,
                    annotation["bbox"],
                    padding,
                    jitter,
                )

                if crop is None:
                    continue

                filename = (
                    f"frame_{frame:06d}"
                    f"_object_{object_index:02d}"
                    f"_variant_{variant_index:02d}.jpg"
                )

                output_path = output_directory / filename

                if not cv2.imwrite(
                    str(output_path),
                    crop,
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                ):
                    raise RuntimeError(f"Cannot save {output_path}")

                counts[split][class_name] = (
                    counts[split].get(class_name, 0) + 1
                )

    for split in ("train", "val"):
        print()
        print(split.upper())

        total = 0

        for class_name in sorted(counts[split]):
            count = counts[split][class_name]
            total += count
            print(f"{class_name:20s} {count:4d}")

        print(f"TOTAL                {total:4d}")


if __name__ == "__main__":
    main()
