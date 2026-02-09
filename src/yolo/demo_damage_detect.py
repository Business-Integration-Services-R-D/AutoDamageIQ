from ultralytics import YOLO
from pathlib import Path


def main():
    # Bu dosyanın bulunduğu klasör: .../src/yolo
    this_file = Path(__file__).resolve()
    yolo_dir = this_file.parent                        # src/yolo
    project_root = yolo_dir.parents[1]                 # src/
    repo_root = project_root                    # AutoDamageIQ/

    # MODELLER → src/yolo/weights/best.pt
    weights_path = yolo_dir / "weights" / "best.pt"

    # TEST GÖRSELLERİ → AutoDamageIQ/assets/
    assets_dir = repo_root / "assets"
    sample_img = assets_dir / "crash_2.jpg"
    clean_img = assets_dir / "crash_3.jpg"

    print("🔍 Yüklenen model:", weights_path)
    print("🔍 Test görselleri:", sample_img, clean_img)

    model = YOLO(str(weights_path))

    results = model.predict(
        source=[str(sample_img), str(clean_img)],
        imgsz=640,
        conf=0.05,
        iou=0.5,
        save=True,
        project=str(yolo_dir / "runs"),
        name="demo_predict",
        verbose=True
    )

    print("Ham kutular:", results[0].boxes)
    print("Kaydedilen klasör:", results[0].save_dir)


if __name__ == "__main__":
    main()
