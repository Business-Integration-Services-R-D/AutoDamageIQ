from ultralytics import YOLO
from pathlib import Path


def main():
    # Bu dosya: src/yolo/train_carparts_seg.py
    this_dir = Path(__file__).resolve().parent  # src/yolo

    # Config & base model
    data_yaml = this_dir / "configs" / "carparts-seg.yaml"
    base_model = this_dir / "weights" / "yolov8n-seg.pt"

    print("📁 Dataset yaml:", data_yaml)
    print("📦 Base seg model:", base_model)

    model = YOLO(str(base_model))

    model.train(
        data=str(data_yaml),
        task="segment",
        imgsz=640,
        epochs=40,
        batch=8,
        workers=4,
        project=str(this_dir / "runs"),
        name="carparts_seg_v1",
    )


if __name__ == "__main__":
    main()
