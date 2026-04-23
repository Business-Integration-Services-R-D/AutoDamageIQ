#!/bin/bash
# AutoDamageID Training Script v8
# YOLOv8m + AdamW + Early Stopping + VehiDE + CarDD Unified
# Docker image: ultralytics/ultralytics:latest-gpu

echo "========================================"
echo "  AutoDamageID Training Script v8"
echo "  Model: YOLOv8m (25M params)"
echo "  Dataset: CarDD + VehiDE (~14K images)"
echo "  Optimizer: AdamW + Early Stopping"
echo "========================================"

cd /workspace

# Kaggle CLI kur
pip install -q kaggle

echo "[1/6] Downloading CarDD dataset from server..."
wget -q --timeout=120 "DATASET_URL_PLACEHOLDER/api/training/download-dataset" -O cardd_dataset.tar.gz || true

echo "[2/6] Downloading VehiDE from Kaggle..."
mkdir -p ~/.kaggle
echo '{"username":"KAGGLE_USER","key":"KAGGLE_KEY"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d hendrichscullen/vehide-dataset-automatic-vehicle-damage-detection -p /workspace/vehide_raw --force 2>/dev/null || echo "VehiDE Kaggle download failed (will use CarDD only)"

echo "[3/6] Extracting datasets..."
# CarDD extract
if [ -f cardd_dataset.tar.gz ]; then
    tar -xzf cardd_dataset.tar.gz 2>/dev/null || true
    echo "CarDD extracted: $(ls images/train 2>/dev/null | wc -l) images"
fi

# VehiDE extract + convert
if [ -f /workspace/vehide_raw/*.zip ]; then
    cd /workspace/vehide_raw
    unzip -q -o *.zip -d extracted 2>/dev/null
    cd /workspace
    
    echo "Converting VehiDE to YOLO format..."
    python3 << 'CONVERTPY'
import json, cv2, os, shutil
from pathlib import Path
from collections import Counter

CLASS_MAP = {'tray_son':4, 'rach':4, 'mop_lom':1, 'be_den':1, 'vo_kinh':2, 'thung':0}

def convert_split(json_path, img_dir, dest_img, dest_lbl):
    os.makedirs(dest_img, exist_ok=True)
    os.makedirs(dest_lbl, exist_ok=True)
    with open(json_path) as f:
        data = json.load(f)
    copied = 0
    for key, entry in data.items():
        name = entry.get('name', key)
        src = Path(img_dir) / name
        if not src.exists():
            continue
        img = cv2.imread(str(src))
        if img is None:
            continue
        h, w = img.shape[:2]
        lines = []
        for r in entry.get('regions', []):
            cls = r.get('class','').strip().lower()
            if cls not in CLASS_MAP:
                continue
            xs, ys = r.get('all_x',[]), r.get('all_y',[])
            if len(xs) < 3:
                continue
            x1,x2,y1,y2 = min(xs),max(xs),min(ys),max(ys)
            cx,cy = ((x1+x2)/2)/w, ((y1+y2)/2)/h
            bw,bh = (x2-x1)/w, (y2-y1)/h
            if 0<bw<1 and 0<bh<1:
                lines.append(f'{CLASS_MAP[cls]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}')
        if lines:
            shutil.copy2(src, f'{dest_img}/vehide_{name}')
            with open(f'{dest_lbl}/vehide_{Path(name).stem}.txt', 'w') as f:
                f.write('\n'.join(lines))
            copied += 1
    return copied

base = '/workspace/vehide_raw/extracted'
t = convert_split(f'{base}/0Train_via_annos.json', f'{base}/image/image', 'images/train', 'labels/train')
v = convert_split(f'{base}/0Val_via_annos.json', f'{base}/validation/validation', 'images/val', 'labels/val')
print(f'VehiDE converted: train={t}, val={v}')
CONVERTPY
    
    # Temizlik
    rm -rf /workspace/vehide_raw
fi

echo "[4/6] Dataset stats..."
echo "Train images: $(ls images/train 2>/dev/null | wc -l)"
echo "Train labels: $(ls labels/train 2>/dev/null | wc -l)"
echo "Val images: $(ls images/val 2>/dev/null | wc -l)"
echo "Val labels: $(ls labels/val 2>/dev/null | wc -l)"

# Val yoksa olustur
if [ ! -d "images/val" ] || [ "$(ls images/val 2>/dev/null | wc -l)" -lt 10 ]; then
    echo "Creating validation set (15%)..."
    mkdir -p images/val labels/val
    TOTAL=$(ls images/train | wc -l)
    VAL_COUNT=$((TOTAL * 15 / 100))
    ls images/train | shuf -n $VAL_COUNT | while read f; do
        mv "images/train/$f" "images/val/" 2>/dev/null
        mv "labels/train/${f%.*}.txt" "labels/val/" 2>/dev/null
    done
fi

echo "[5/6] Creating data.yaml..."
cat > data.yaml << 'EOF'
path: /workspace
train: images/train
val: images/val
nc: 6
names: [crack, dent, glass_shatter, lamp_broken, scratch, tire_flat]
EOF

echo "[6/6] Starting training..."
python3 << 'TRAINPY'
from ultralytics import YOLO
import torch, os, glob

gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
batch = 32 if gpu_mem >= 24 else (16 if gpu_mem >= 16 else (12 if gpu_mem >= 12 else 8))
print(f"GPU: {torch.cuda.get_device_name(0)} ({gpu_mem:.1f} GB) | Batch: {batch}")

train_count = len(glob.glob("/workspace/images/train/*"))
val_count = len(glob.glob("/workspace/images/val/*"))
print(f"Dataset: {train_count} train, {val_count} val")

model = YOLO("yolov8m.pt")
results = model.train(
    data="data.yaml",
    epochs=200,
    batch=batch,
    imgsz=640,
    project="runs",
    name="autodamage_v8m",
    exist_ok=True,
    patience=25,
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    weight_decay=0.0005,
    warmup_epochs=5,
    mosaic=1.0,
    mixup=0.15,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    flipud=0.0,
    fliplr=0.5,
    scale=0.5,
    translate=0.1,
    degrees=0.0,
    perspective=0.0001,
    device=0,
    workers=4,
    verbose=True,
    cos_lr=True,
    close_mosaic=15,
    amp=True,
    cache=True,
)

print("=" * 60)
best = "/workspace/runs/autodamage_v8m/weights/best.pt"
if os.path.exists(best):
    print(f"Model: {best} ({os.path.getsize(best)/1024/1024:.1f} MB)")
try:
    m = results.results_dict
    print(f"mAP@50: {m.get('metrics/mAP50(B)','N/A')}")
    print(f"mAP@50-95: {m.get('metrics/mAP50-95(B)','N/A')}")
    print(f"Precision: {m.get('metrics/precision(B)','N/A')}")
    print(f"Recall: {m.get('metrics/recall(B)','N/A')}")
except: pass
print("=" * 60)
TRAINPY

echo "DONE! Model: /workspace/runs/autodamage_v8m/weights/best.pt"
