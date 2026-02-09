from ultralytics import YOLO
from pathlib import Path
import cv2


def main():
    this_dir = Path(__file__).resolve().parent
    project_root = this_dir.parents[1]
    repo_root = project_root.parent

    weights_path = this_dir / "runs" / "carparts_seg_v1" / "weights" / "best.pt"  # Eğitim sonrası oluşacak
    assets_dir = repo_root / "AutoDamageIQ" / "assets"

    test_img = assets_dir / "crash_1.jpg"  # test görseli

    model = YOLO(str(weights_path))

    results = model.predict(
        source=str(test_img),
        imgsz=640,
        save=True,
        conf=0.3,
        project=str(this_dir / "runs"),
        name="carparts_predict",
    )

    print("Sonuç klasörü:", results[0].save_dir)


if __name__ == "__main__":
    main()
