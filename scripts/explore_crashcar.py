import os
import sys

# src klasörünü Python path'e ekleyelim (basit yöntem)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from src.autodamageid.data.crashcar import CrashCarDataset
from src.yolo.show_masks import show_image_and_mask


def main():
    ds = CrashCarDataset(split="train", image_size=256)
    print("Toplam örnek sayısı:", len(ds))

    # İlk 3 örneği gösterelim
    for i in range(3):
        sample = ds[i]
        image = sample["image"]
        damage_mask = sample["damage_mask"]
        print(f"Örnek {i}: image shape={image.shape}, mask shape={damage_mask.shape}")
        show_image_and_mask(image, damage_mask)


if __name__ == "__main__":
    main()
