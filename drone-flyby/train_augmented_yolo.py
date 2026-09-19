from pathlib import Path
from ultralytics import YOLO

def main():
    dataset_yaml = Path("datasets/drone_yolo_aug/dataset.yaml").resolve()
    
    # Train YOLO11n single-stage model
    model = YOLO("yolo11n.pt")
    
    results = model.train(
        data=str(dataset_yaml),
        epochs=30,
        imgsz=960,
        batch=16,
        device="mps",
        workers=4,
        project="runs/detect",
        name="yolo_v2_augmented",
        exist_ok=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
    )
    
    # Save best model to models/drone_yolo_v2_augmented.pt
    # NOTE: run from repo root, YOLO resolves project= relative to cwd,
    # so the weights may land in runs/detect/runs/detect/... — check both.
    candidates = [
        Path("runs/detect/yolo_v2_augmented/weights/best.pt"),
        Path("runs/detect/runs/detect/yolo_v2_augmented/weights/best.pt"),
        Path("drone-flyby/models/../..").resolve() / "runs/detect/runs/detect/yolo_v2_augmented/weights/best.pt",
    ]
    target_weights = Path("models/drone_yolo_v2_augmented.pt")
    if not target_weights.is_absolute():
        # allow running from repo root: drone-flyby/models/...
        alt = Path("drone-flyby/models/drone_yolo_v2_augmented.pt")
        if not Path("models").exists() and alt.parent.exists():
            target_weights = alt

    best_weights = next((c for c in candidates if c.exists()), candidates[0])
    
    if best_weights.exists():
        import shutil
        shutil.copy(best_weights, target_weights)
        print(f"Saved best model to {target_weights}")
    else:
        print("ERROR: Best weights file not found!")

if __name__ == "__main__":
    main()
