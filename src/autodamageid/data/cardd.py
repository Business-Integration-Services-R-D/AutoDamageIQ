# src/autodamageiq/data/cardd.py

from typing import Dict, Any
from datasets import load_dataset
from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np


class CarDDDamageDataset(Dataset):
    """
    harpreetsahota/CarDD dataset'ini DamageSeg için saran PyTorch Dataset sınıfı.

    Bu sınıfın amacı:
      - image: [3, H, W], float32, 0-1
      - damage_mask: [1, H, W], float32, 0-1  (binary maske)

    NOT: HF'den gelen sample'ın key'lerini explore_cardd.py'de bakıp,
         burada 'image' ve 'mask' alan isimlerini ona göre uyarlayacağız.
    """

    def __init__(self, split: str = "train", image_size: int = 512, hf_token: str | None = None):
        super().__init__()
        # Eğer private / lisanslı olursa token parametresiyle kullanabiliriz.
        load_kwargs: Dict[str, Any] = {"split": split}
        if hf_token is not None:
            load_kwargs["token"] = hf_token

        self.ds = load_dataset("harpreetsahota/CarDD", **load_kwargs)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.ds)

    @staticmethod
    def _pil_to_chw_tensor(img: Image.Image, image_size: int) -> torch.Tensor:
        # Resize
        img = img.resize((image_size, image_size))
        arr = np.array(img).astype(np.float32)

        # Eğer grayscale ise kanal ekle
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=-1)

        # 0-255 -> 0-1
        arr /= 255.0

        # HWC -> CHW
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.ds[idx]

        # --- BU KISIMI explore_cardd.py çıktısına göre netleştireceğiz ---
        # Büyük ihtimalle:
        #   image    -> sample["image"]
        #   sod_mask -> sample["sod"] veya sample["mask"] veya benzeri
        # Şimdilik 'image' sabit, mask'i placeholder bırakıyoruz.

        img_pil: Image.Image = sample["image"]

        # Şimdilik mask'i None bırakıp explore_cardd.py'de anahtar isimlerini öğreneceğiz.
        raise NotImplementedError("Mask alanının ismini dataset'ten kontrol edip burada dolduracağız.")
