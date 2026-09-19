import json
import random
import shutil
from pathlib import Path
import cv2
import numpy as np

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

ZERO_AP_CLASSES = {"hangar", "medium_plane", "mine_roller", "small_launcher"}

SOURCE = Path("src/helsinki")
OUTPUT = Path("datasets/drone_yolo_aug")
TILE_WIDTH = 960
TILE_HEIGHT = 540

CLASS_TO_ID = {name: index for index, name in enumerate(CLASSES)}
RNG = random.Random(2026)
np.random.seed(2026)

def extract_crops():
    """Extract object crops from frames with tight masks/alpha or plain crops."""
    crops_by_class = {cls: [] for cls in CLASSES}
    annotation_files = sorted((SOURCE / "annotations").glob("frame_*.json"))
    
    for annotation_path in annotation_files:
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
        frame = int(data["frame"])
        image_path = SOURCE / "images" / f"frame_{frame:06d}.png"
        image = cv2.imread(str(image_path))
        if image is None:
            continue
            
        for ann in data["annotations"]:
            cls_name = ann["object_id"]
            x1, y1, x2, y2 = map(int, ann["bbox"])
            # Clamp
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(image.shape[1], x2); y2 = min(image.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = image[y1:y2, x1:x2].copy()
            if crop.shape[0] > 3 and crop.shape[1] > 3:
                crops_by_class[cls_name].append(crop)
    return crops_by_class

def extract_backgrounds():
    """Extract background patches (tiles with no objects or with objects masked out)."""
    backgrounds = []
    annotation_files = sorted((SOURCE / "annotations").glob("frame_*.json"))
    
    for annotation_path in annotation_files:
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
        frame = int(data["frame"])
        image_path = SOURCE / "images" / f"frame_{frame:06d}.png"
        image = cv2.imread(str(image_path))
        if image is None:
            continue
            
        bg_image = image.copy()
        # Blur or fill object regions in background source image
        for ann in data["annotations"]:
            x1, y1, x2, y2 = map(int, ann["bbox"])
            cv2.rectangle(bg_image, (x1, y1), (x2, y2), (128, 128, 128), -1)
            
        # Crop 960x540 tiles from random locations or grid
        for row in range(4):
            for col in range(4):
                left = col * TILE_WIDTH
                top = row * TILE_HEIGHT
                tile = bg_image[top:top+TILE_HEIGHT, left:left+TILE_WIDTH].copy()
                backgrounds.append(tile)
    return backgrounds

def apply_color_jitter(img):
    """Apply random brightness, contrast, and HSV jitter."""
    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    # Hue shift
    hsv[:, :, 0] = (hsv[:, :, 0] + RNG.uniform(-10, 10)) % 180
    # Saturation scale
    hsv[:, :, 1] *= RNG.uniform(0.7, 1.3)
    # Value/brightness scale
    hsv[:, :, 2] *= RNG.uniform(0.7, 1.3)
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    res = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return res

def paste_crop(bg, crop):
    """Paste crop onto bg with random scale, rotation, and color jitter."""
    ch, cw = crop.shape[:2]
    # Random scale between 0.7x and 1.3x
    scale = RNG.uniform(0.7, 1.3)
    nw, nh = int(round(cw * scale)), int(round(ch * scale))
    if nw < 4 or nh < 4:
        return bg, None
        
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR)
    resized = apply_color_jitter(resized)
    
    # Pick random position on bg
    max_x = bg.shape[1] - nw
    max_y = bg.shape[0] - nh
    if max_x <= 0 or max_y <= 0:
        return bg, None
        
    px = RNG.randint(0, max_x)
    py = RNG.randint(0, max_y)
    
    # Simple direct copy paste
    bg[py:py+nh, px:px+nw] = resized
    bbox = (px, py, px + nw, py + nh)
    return bg, bbox

def yolo_line(class_id, box):
    x1, y1, x2, y2 = box
    center_x = ((x1 + x2) / 2.0) / TILE_WIDTH
    center_y = ((y1 + y2) / 2.0) / TILE_HEIGHT
    width = (x2 - x1) / TILE_WIDTH
    height = (y2 - y1) / TILE_HEIGHT
    return f"{class_id} {center_x:.8f} {center_y:.8f} {width:.8f} {height:.8f}"

def main():
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)

    for split in ("train", "val"):
        (OUTPUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    print("Extracting object crops and background tiles...")
    crops_by_class = extract_crops()
    backgrounds = extract_backgrounds()
    
    # First build standard grid tiles (from prepare_yolo_dataset logic)
    annotation_files = sorted((SOURCE / "annotations").glob("frame_*.json"))
    frame_numbers = [int(p.stem.split("_")[-1]) for p in annotation_files]
    RNG.shuffle(frame_numbers)
    val_frames = set(frame_numbers[:5])

    tile_idx = 0
    # Process original tiles for train & val
    for annotation_path in annotation_files:
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
        frame = int(data["frame"])
        split = "val" if frame in val_frames else "train"
        image_path = SOURCE / "images" / f"frame_{frame:06d}.png"
        image = cv2.imread(str(image_path))

        for row in range(4):
            for col in range(4):
                left = col * TILE_WIDTH; top = row * TILE_HEIGHT
                right = left + TILE_WIDTH; bottom = top + TILE_HEIGHT
                tile = image[top:bottom, left:right]
                tile_region = (left, top, right, bottom)
                tile_name = f"frame_{frame:06d}_r{row}_c{col}"

                lines = []
                for ann in data["annotations"]:
                    x1, y1, x2, y2 = ann["bbox"]
                    cx1, cy1 = max(x1, left), max(y1, top)
                    cx2, cy2 = min(x2, right), min(y2, bottom)
                    if cx2 > cx1 and cy2 > cy1 and ((cx2-cx1)*(cy2-cy1) / max(1.0, (x2-x1)*(y2-y1))) >= 0.45:
                        clipped = (cx1 - left, cy1 - top, cx2 - left, cy2 - top)
                        class_id = CLASS_TO_ID[ann["object_id"]]
                        lines.append(yolo_line(class_id, clipped))

                cv2.imwrite(str(OUTPUT / "images" / split / f"{tile_name}.jpg"), tile, [cv2.IMWRITE_JPEG_QUALITY, 95])
                (OUTPUT / "labels" / split / f"{tile_name}.txt").write_text("\n".join(lines), encoding="utf-8")

    # Generate synthetic training tiles using copy-paste augmentation!
    # Generate 500 extra synthetic training tiles with heavy focus on zero-AP classes
    print("Generating 500 synthetic copy-paste training tiles...")
    for syn_i in range(500):
        bg = RNG.choice(backgrounds).copy()
        lines = []
        # Decide how many objects to paste (between 3 and 10)
        n_paste = RNG.randint(3, 10)
        for _ in range(n_paste):
            # 60% chance to select a zero-AP class for disproportionate representation
            if RNG.random() < 0.60:
                cls_name = RNG.choice(list(ZERO_AP_CLASSES))
            else:
                cls_name = RNG.choice(CLASSES)
                
            crops = crops_by_class[cls_name]
            if not crops:
                continue
            crop = RNG.choice(crops)
            bg, bbox = paste_crop(bg, crop)
            if bbox is not None:
                class_id = CLASS_TO_ID[cls_name]
                lines.append(yolo_line(class_id, bbox))
                
        tile_name = f"syn_tile_{syn_i:04d}"
        cv2.imwrite(str(OUTPUT / "images" / "train" / f"{tile_name}.jpg"), bg, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (OUTPUT / "labels" / "train" / f"{tile_name}.txt").write_text("\n".join(lines), encoding="utf-8")

    # Write dataset.yaml
    yaml_lines = [
        f"path: {OUTPUT.resolve()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    for index, name in enumerate(CLASSES):
        yaml_lines.append(f"  {index}: {name}")

    (OUTPUT / "dataset.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    print("Dataset generation complete:", OUTPUT / "dataset.yaml")

if __name__ == "__main__":
    main()
