#!/bin/bash
# AutoDamageID Training Script v8
# YOLOv8m + AdamW + Early Stopping + Agresif Augmentation
# ultralytics/ultralytics:latest-gpu image

echo "========================================"
echo "  AutoDamageID Training Script v8"
echo "  Model: YOLOv8m (25M params)"
echo "  Optimizer: AdamW"
echo "  Early Stopping: patience=25"
echo "========================================"

cd /workspace

echo "[1/4] Downloading dataset..."
wget -q "https://damage-vision.preview.emergentagent.com/api/training/download-dataset" -O dataset.tar.gz
if [ $? -ne 0 ]; then
    echo "ERROR: Dataset download failed!"
    exit 1
fi
echo "Done!"

echo "[2/4] Extracting dataset..."
tar -xzf dataset.tar.gz
echo "Train images: $(ls images/train 2>/dev/null | wc -l)"
echo "Train labels: $(ls labels/train 2>/dev/null | wc -l)"

# Val yoksa olustur (%15 ayir)
if [ ! -d "images/val" ] || [ -z "$(ls -A images/val 2>/dev/null)" ]; then
    echo "Creating validation set (15%)..."
    mkdir -p images/val labels/val
    TOTAL=$(ls images/train | wc -l)
    VAL_COUNT=$((TOTAL * 15 / 100))
    ls images/train | shuf -n $VAL_COUNT | while read f; do
        mv "images/train/$f" "images/val/" 2>/dev/null
        mv "labels/train/${f%.*}.txt" "labels/val/" 2>/dev/null
    done
    echo "Val images: $(ls images/val 2>/dev/null | wc -l)"
fi

echo "[3/4] Creating data.yaml..."
cat > data.yaml << 'EOF'
path: /workspace
train: images/train
val: images/val
nc: 6
names: [crack, dent, glass_shatter, lamp_broken, scratch, tire_flat]
EOF
echo "Done!"

echo "[4/4] Verifying environment..."
python3 << 'CHECKPY'
import numpy as np
import torch
import cv2
from ultralytics import YOLO

print(f"NumPy: {np.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"OpenCV: {cv2.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    mem_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f"GPU Memory: {mem_gb:.1f} GB")

print("Environment ready!")
CHECKPY

if [ $? -ne 0 ]; then
    echo "ERROR: Environment check failed!"
    exit 1
fi

echo ""
echo "========================================"
echo "  Starting YOLOv8m Training"
echo "  (medium model — 25M parameters)"
echo "========================================"

python3 << 'TRAINPY'
from ultralytics import YOLO
import torch
import os

gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
print(f"GPU Memory: {gpu_mem:.1f} GB")

# GPU bellegine gore batch size ayarla
if gpu_mem >= 24:
    batch_size = 32
elif gpu_mem >= 16:
    batch_size = 16
elif gpu_mem >= 12:
    batch_size = 12
else:
    batch_size = 8

print(f"Batch size (auto): {batch_size}")

# YOLOv8m — medium model (25.3M params, mAP daha yuksek)
model = YOLO("yolov8m.pt")
print("YOLOv8m model loaded!")

# Egitim verisini say
import glob
train_count = len(glob.glob("/workspace/images/train/*"))
val_count = len(glob.glob("/workspace/images/val/*"))
print(f"Train: {train_count} images, Val: {val_count} images")

results = model.train(
    data="data.yaml",
    epochs=200,               # Max epoch — early stopping durduracak
    batch=batch_size,
    imgsz=640,
    project="runs",
    name="autodamage_v8m",
    exist_ok=True,
    
    # Early stopping
    patience=25,              # 25 epoch iyilesme yoksa dur
    
    # Optimizer
    optimizer="AdamW",        # SGD yerine AdamW
    lr0=0.001,                # Baslangic lr
    lrf=0.01,                 # Final lr factor
    weight_decay=0.0005,
    warmup_epochs=5,          # Warmup
    
    # Augmentation (makale referansli)
    mosaic=1.0,               # Mosaic augmentation
    mixup=0.15,               # MixUp augmentation
    hsv_h=0.015,              # HSV hue jitter
    hsv_s=0.7,                # HSV saturation jitter
    hsv_v=0.4,                # HSV value jitter
    flipud=0.0,               # Dikey flip (arac icin mantikli degil)
    fliplr=0.5,               # Yatay flip
    scale=0.5,                # Scale augmentation
    translate=0.1,            # Translation
    degrees=0.0,              # Rotation (arac icin sifir)
    perspective=0.0001,       # Hafif perspektif
    
    # Diger
    device=0,
    workers=4,
    verbose=True,
    cos_lr=True,              # Cosine lr scheduler
    close_mosaic=15,          # Son 15 epoch'ta mosaic kapat
    
    # Performans
    amp=True,                 # Mixed precision
    cache=True,               # Veriyi RAM'e cache'le
)

print("")
print("=" * 60)
print("  TRAINING COMPLETED!")
print("=" * 60)

# Sonuclari yazdir
best_path = "/workspace/runs/autodamage_v8m/weights/best.pt"
if os.path.exists(best_path):
    size_mb = os.path.getsize(best_path) / (1024*1024)
    print(f"Model: {best_path} ({size_mb:.1f} MB)")
else:
    print("WARNING: best.pt bulunamadi!")

# Metrikleri goster
try:
    metrics = results.results_dict
    print(f"mAP@50:    {metrics.get('metrics/mAP50(B)', 'N/A')}")
    print(f"mAP@50-95: {metrics.get('metrics/mAP50-95(B)', 'N/A')}")
    print(f"Precision: {metrics.get('metrics/precision(B)', 'N/A')}")
    print(f"Recall:    {metrics.get('metrics/recall(B)', 'N/A')}")
except:
    print("(Metrikler alinamadi)")

print("=" * 60)
TRAINPY

echo ""
echo "========================================"
echo "DONE! Model: /workspace/runs/autodamage_v8m/weights/best.pt"
echo "========================================"
