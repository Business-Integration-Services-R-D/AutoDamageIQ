from typing import Dict, Any
from datasets import load_dataset
from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np


class CrashCarDataset(Dataset):
    """
    JensParslov/CrashCar dataset'ini PyTorch Dataset formatına çevirir.
    Bu versiyonda sadece:
      - image
      - damage_mask
    kullanıyoruz. Part mask'i ileride ekleyeceğiz.
    """

    def __init__(self, split: str = "train", image_size: int = 512):
        super().__init__()
        self.ds = load_dataset("JensParslov/CrashCar", split=split)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.ds)

    def _pil_to_tensor(self, img: Image.Image) -> torch.Tensor:
        # Resize
        img = img.resize((self.image_size, self.image_size))
        # [H, W, C] numpy
        arr = np.array(img).astype(np.float32)
        # Normalizasyon: 0-255 -> 0-1
        arr /= 255.0
        # Eğer grayscale ise kanal boyutu ekle
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=-1)
        # HWC -> CHW
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.ds[idx]

        img: Image.Image = item["image"]
        damage: Image.Image = item["damage"]

        img_tensor = self._pil_to_tensor(img)              # [3, H, W]
        damage_tensor = self._pil_to_tensor(damage)        # [1, H, W] olmalı

        # Damage mask'i 0/1'e yuvarlayalım (bazı datasetlerde 0-255 gelebilir)
        damage_tensor = (damage_tensor > 0.5).float()      # [1, H, W]

        return {
            "image": img_tensor,
            "damage_mask": damage_tensor,
        }
