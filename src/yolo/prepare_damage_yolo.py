from datasets import load_dataset
from pathlib import Path
from PIL import Image

# 1) HF dataset'i yükle
ds = load_dataset("gigwegbe/damaged-car-dataset-annotated", split="train")

# 2) YOLO klasör yapısı
root = Path("../autodamageid/data/damage_yolo")
img_dir = root / "images" / "train"
lbl_dir = root / "labels" / "train"
img_dir.mkdir(parents=True, exist_ok=True)
lbl_dir.mkdir(parents=True, exist_ok=True)

# 3) class mapping
classes = ["crack", "dent", "glass shatter", "lamp broken", "scratch", "tire flat"]
cls_to_id = {c: i for i, c in enumerate(classes)}

for i, sample in enumerate(ds):
    img = sample["image"]              # PIL Image
    w, h = img.size
    bbox = sample["bbox"]              # [x, y, width, height] (pixel)
    cat = sample["category_id"]        # string

    # bbox -> YOLO normalize
    x, y, bw, bh = bbox
    x_c = (x + bw / 2) / w
    y_c = (y + bh / 2) / h
    bw_n = bw / w
    bh_n = bh / h

    cls_id = cls_to_id[cat]

    # dosya isimleri
    img_name = f"damage_{i:05d}.jpg"
    lbl_name = f"damage_{i:05d}.txt"

    img.save(img_dir / img_name)

    with open(lbl_dir / lbl_name, "w") as f:
        f.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {bw_n:.6f} {bh_n:.6f}\n")

print("Bitti. Klasör:", root.resolve())
