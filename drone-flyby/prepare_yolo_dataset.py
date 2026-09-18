import json
import random
import shutil
from pathlib import Path

import cv2


CLASSES = [
    "hangar",
    "helicopter",
    "jet_plane",
    "large_launcher",
    "large_tower",
    "medium_launcher",
    "medium_plane",
    "mine_roller",
    "small_launcher",
    "small_plane",
    "small_tower",
    "ta-ta",
    "tank",
    "condor",
    "jammer",
    "spacecraft",
]

SOURCE = Path("src/helsinki")
OUTPUT = Path("datasets/drone_yolo")

TILE_WIDTH = 960
TILE_HEIGHT = 540

CLASS_TO_ID = {
    name: index
    for index, name in enumerate(CLASSES)
}


def clip_box(box, tile):
    x1, y1, x2, y2 = map(float, box)
    tx1, ty1, tx2, ty2 = tile

    clipped_x1 = max(x1, tx1)
    clipped_y1 = max(y1, ty1)
    clipped_x2 = min(x2, tx2)
    clipped_y2 = min(y2, ty2)

    if clipped_x2 <= clipped_x1 or clipped_y2 <= clipped_y1:
        return None

    original_area = max(
        1.0,
        (x2 - x1) * (y2 - y1),
    )

    clipped_area = (
        (clipped_x2 - clipped_x1)
        * (clipped_y2 - clipped_y1)
    )

    if clipped_area / original_area < 0.45:
        return None

    return (
        clipped_x1 - tx1,
        clipped_y1 - ty1,
        clipped_x2 - tx1,
        clipped_y2 - ty1,
    )


def yolo_line(class_id, box):
    x1, y1, x2, y2 = box

    center_x = ((x1 + x2) / 2.0) / TILE_WIDTH
    center_y = ((y1 + y2) / 2.0) / TILE_HEIGHT
    width = (x2 - x1) / TILE_WIDTH
    height = (y2 - y1) / TILE_HEIGHT

    return (
        f"{class_id} "
        f"{center_x:.8f} "
        f"{center_y:.8f} "
        f"{width:.8f} "
        f"{height:.8f}"
    )


def main():
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)

    for split in ("train", "val"):
        (OUTPUT / "images" / split).mkdir(
            parents=True,
            exist_ok=True,
        )
        (OUTPUT / "labels" / split).mkdir(
            parents=True,
            exist_ok=True,
        )

    annotation_files = sorted(
        (SOURCE / "annotations").glob("frame_*.json")
    )

    frame_numbers = [
        int(path.stem.split("_")[-1])
        for path in annotation_files
    ]

    random.Random(2026).shuffle(frame_numbers)

    validation_frames = set(frame_numbers[:5])

    tile_count = 0
    labelled_tiles = 0
    box_count = 0

    for annotation_path in annotation_files:
        data = json.loads(
            annotation_path.read_text(encoding="utf-8")
        )

        frame = int(data["frame"])
        split = "val" if frame in validation_frames else "train"

        image_path = (
            SOURCE / "images" / f"frame_{frame:06d}.png"
        )

        image = cv2.imread(str(image_path))

        if image is None:
            raise RuntimeError(
                f"Failed to read image: {image_path}"
            )

        height, width = image.shape[:2]

        if width != 3840 or height != 2160:
            raise ValueError(
                f"Unexpected image size: {width}x{height}"
            )

        for row in range(4):
            for column in range(4):
                left = column * TILE_WIDTH
                top = row * TILE_HEIGHT
                right = left + TILE_WIDTH
                bottom = top + TILE_HEIGHT

                tile = image[top:bottom, left:right]
                tile_region = (
                    left,
                    top,
                    right,
                    bottom,
                )

                tile_name = (
                    f"frame_{frame:06d}"
                    f"_r{row}_c{column}"
                )

                image_output = (
                    OUTPUT
                    / "images"
                    / split
                    / f"{tile_name}.jpg"
                )

                label_output = (
                    OUTPUT
                    / "labels"
                    / split
                    / f"{tile_name}.txt"
                )

                lines = []

                for annotation in data["annotations"]:
                    clipped = clip_box(
                        annotation["bbox"],
                        tile_region,
                    )

                    if clipped is None:
                        continue

                    object_id = annotation["object_id"]
                    class_id = CLASS_TO_ID[object_id]

                    lines.append(
                        yolo_line(class_id, clipped)
                    )

                    box_count += 1

                saved = cv2.imwrite(
                    str(image_output),
                    tile,
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                )

                if not saved:
                    raise RuntimeError(
                        f"Failed to save: {image_output}"
                    )

                label_output.write_text(
                    "\n".join(lines),
                    encoding="utf-8",
                )

                tile_count += 1

                if lines:
                    labelled_tiles += 1

    yaml_lines = [
        f"path: {OUTPUT.resolve()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]

    for index, name in enumerate(CLASSES):
        yaml_lines.append(f"  {index}: {name}")

    yaml_path = OUTPUT / "dataset.yaml"

    yaml_path.write_text(
        "\n".join(yaml_lines) + "\n",
        encoding="utf-8",
    )

    print("Dataset:", OUTPUT)
    print("Tiles:", tile_count)
    print("Labelled tiles:", labelled_tiles)
    print("Boxes after clipping:", box_count)
    print("Validation frames:", sorted(validation_frames))
    print("YAML:", yaml_path)


if __name__ == "__main__":
    main()
