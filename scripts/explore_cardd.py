# scripts/explore_cardd.py

import os
import sys

# src klasörünü Python path'e ekle (basit yöntem)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from datasets import load_dataset
import matplotlib.pyplot as plt


def main():
    # Gerekirse token kullanabilirsin: load_dataset("harpreetsahota/CarDD", token="hf_xxx", split="train")
    ds = load_dataset("harpreetsahota/CarDD", split="train")

    print("Toplam örnek sayısı:", len(ds))

    sample = ds[0]
    print("Sample keys:", sample.keys())

    # Muhtemel alan isimleri:
    #  - 'image'        : asıl fotoğraf
    #  - 'sod_mask'     : saliency / binary damage map
    #  - 'objects' veya 'annotations' : bbox + instance segmentation
    #   (Bunların hangisi olduğunu burada göreceğiz.)

    # Görseli göster
    img = sample["image"]
    plt.imshow(img)
    plt.title("CarDD örnek görüntü")
    plt.axis("off")
    plt.show()

    # Eğer sample içinde 'sod_mask' gibi bir alan varsa:
    if "sod_mask" in sample:
        m = sample["sod_mask"]
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title("Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(m, cmap="gray")
        plt.title("SOD / Binary Mask")
        plt.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
