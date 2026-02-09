from ultralytics import YOLO

def main():
    model = YOLO("weights/yolo11n.pt")  # hafif model, hızlı eğitim

    model.train(
        data="data/damage_yolo.yaml",
        task="detect",
        epochs=30,
        imgsz=640,
        batch=8,
        workers=4
    )

if __name__ == "__main__":
    main()
