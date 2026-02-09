#!/bin/bash
# AutoDamageID Training Script v7
# ultralytics/ultralytics:latest-gpu image kullanıyor
# Tüm dependency'ler zaten kurulu!

echo "========================================"
echo "  AutoDamageID Training Script v7"
echo "  Using ultralytics/ultralytics image"
echo "========================================"

cd /workspace

echo "[1/4] Downloading dataset..."
wget -q "https://cardamage-ai-1.preview.emergentagent.com/api/training/download-dataset" -O dataset.tar.gz
if [ $? -ne 0 ]; then
    echo "ERROR: Dataset download failed!"
    exit 1
fi
echo "Done!"

echo "[2/4] Extracting dataset..."
tar -xzf dataset.tar.gz
echo "Train images: $(ls images/train 2>/dev/null | wc -l)"

# Val yoksa oluştur
if [ ! -d "images/val" ] || [ -z "$(ls -A images/val 2>/dev/null)" ]; then
    echo "Creating validation set..."
    mkdir -p images/val labels/val
    ls images/train | shuf -n 500 | while read f; do
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

# Quick test
x = torch.tensor([1.0, 2.0, 3.0])
y = x.numpy()
print(f"Torch->NumPy test: OK")
print("Environment ready!")
CHECKPY

if [ $? -ne 0 ]; then
    echo "ERROR: Environment check failed!"
    exit 1
fi

echo ""
echo "========================================"
echo "  Starting YOLO Training"
echo "========================================"

python3 << 'TRAINPY'
from ultralytics import YOLO
import torch

print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

model = YOLO("yolov8n.pt")
print("Model loaded!")

results = model.train(
    data="data.yaml",
    epochs=50,
    batch=16,
    imgsz=640,
    project="runs",
    name="autodamage",
    exist_ok=True,
    patience=20,
    device=0,
    workers=4,
    verbose=True
)

print("")
print("========================================")
print("  TRAINING COMPLETED!")
print("========================================")
print("Model saved to: /workspace/runs/autodamage/weights/best.pt")
TRAINPY

echo ""
echo "========================================"
echo "DONE! Model: /workspace/runs/autodamage/weights/best.pt"
echo "========================================"
