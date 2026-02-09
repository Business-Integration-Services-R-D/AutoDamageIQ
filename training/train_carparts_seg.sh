#!/bin/bash
# CarParts Segmentation Training Script
# 100 Epoch - High Quality

echo "========================================"
echo "  CarParts Segmentation Training"
echo "  100 Epoch - High Quality"
echo "========================================"

cd /workspace

echo "[1/2] Environment check..."
python3 << 'CHECKPY'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print("Environment ready!")
CHECKPY

if [ $? -ne 0 ]; then
    echo "ERROR: Environment check failed!"
    exit 1
fi

echo ""
echo "[2/2] Starting SEGMENTATION Training (100 epochs)..."
echo "Dataset: carparts-seg (will be auto-downloaded by Ultralytics)"
echo ""

python3 << 'TRAINPY'
from ultralytics import YOLO
import torch

print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

model = YOLO("yolov8n-seg.pt")
print("Model loaded: yolov8n-seg.pt")

results = model.train(
    data="carparts-seg.yaml",
    task="segment",
    epochs=100,
    batch=16,
    imgsz=640,
    project="runs",
    name="carparts_seg",
    exist_ok=True,
    patience=30,
    device=0,
    workers=4,
    verbose=True
)

print("")
print("========================================")
print("  SEGMENTATION TRAINING COMPLETED!")
print("========================================")
print("Model saved to: /workspace/runs/carparts_seg/weights/best.pt")
TRAINPY

echo ""
echo "========================================"
echo "DONE! Model: /workspace/runs/carparts_seg/weights/best.pt"
echo "========================================"
