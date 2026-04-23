#!/bin/bash
# AutoDamageID Training Script v8.1
# YOLOv8m + AdamW + Early Stopping + VehiDE + CarDD
# Docker: ultralytics/ultralytics:latest

echo "========================================"
echo "  AutoDamageID Training Script v8.1"
echo "  Model: YOLOv8m (25M params)"
echo "  Optimizer: AdamW + Early Stopping"
echo "========================================"

cd /workspace

# Dataset URL env variable'dan veya hardcoded
DATASET_DL_URL="${DATASET_URL:-https://damage-vision.preview.emergentagent.com/api/training/download-dataset}"
echo "Dataset URL: $DATASET_DL_URL"

echo "[1/4] Downloading dataset..."
wget -q --timeout=300 "$DATASET_DL_URL" -O dataset.tar.gz
if [ $? -ne 0 ] || [ ! -s dataset.tar.gz ]; then
    echo "ERROR: Dataset download failed!"
    exit 1
fi
echo "Dataset downloaded: $(ls -lh dataset.tar.gz | awk '{print $5}')"

echo "[2/4] Extracting dataset..."
tar -xzf dataset.tar.gz 2>/dev/null
rm -f dataset.tar.gz

echo "Train images: $(ls images/train 2>/dev/null | wc -l)"
echo "Train labels: $(ls labels/train 2>/dev/null | wc -l)"
echo "Val images: $(ls images/val 2>/dev/null | wc -l)"
echo "Val labels: $(ls labels/val 2>/dev/null | wc -l)"

# Val yoksa veya azsa olustur
VAL_COUNT=$(ls images/val 2>/dev/null | wc -l)
if [ "$VAL_COUNT" -lt 10 ]; then
    echo "Creating validation set (15%)..."
    mkdir -p images/val labels/val
    TOTAL=$(ls images/train | wc -l)
    SPLIT=$((TOTAL * 15 / 100))
    ls images/train | shuf -n $SPLIT | while read f; do
        mv "images/train/$f" "images/val/" 2>/dev/null
        mv "labels/train/${f%.*}.txt" "labels/val/" 2>/dev/null
    done
    echo "Val created: $(ls images/val | wc -l) images"
fi

# Bozuk JPEG'leri temizle
echo "Checking for corrupt images..."
python3 << 'CLEANPY'
import cv2, os, glob
removed = 0
for split in ["train", "val"]:
    img_dir = f"/workspace/images/{split}"
    lbl_dir = f"/workspace/labels/{split}"
    for img_path in glob.glob(f"{img_dir}/*"):
        img = cv2.imread(img_path)
        if img is None:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            os.remove(img_path)
            lbl_path = os.path.join(lbl_dir, stem + ".txt")
            if os.path.exists(lbl_path):
                os.remove(lbl_path)
            removed += 1
print(f"Removed {removed} corrupt images")
CLEANPY

echo "Clean train: $(ls images/train 2>/dev/null | wc -l) images"
echo "Clean val: $(ls images/val 2>/dev/null | wc -l) images"

echo "[3/4] Creating data.yaml..."
cat > data.yaml << 'EOF'
path: /workspace
train: images/train
val: images/val
nc: 6
names: [crack, dent, glass_shatter, lamp_broken, scratch, tire_flat]
EOF

echo "[4/4] Starting YOLOv8m training..."
python3 << 'TRAINPY'
from ultralytics import YOLO
import torch, os, glob

# GPU bilgisi - total_memory (yeni PyTorch) veya total_mem (eski)
props = torch.cuda.get_device_properties(0)
try:
    gpu_mem = props.total_memory / 1e9
except AttributeError:
    gpu_mem = props.total_mem / 1e9

batch = 32 if gpu_mem >= 24 else (16 if gpu_mem >= 16 else (12 if gpu_mem >= 12 else 8))
print(f"GPU: {torch.cuda.get_device_name(0)} ({gpu_mem:.1f} GB) | Batch: {batch}")

train_count = len(glob.glob("/workspace/images/train/*"))
val_count = len(glob.glob("/workspace/images/val/*"))
print(f"Dataset: {train_count} train, {val_count} val")

model = YOLO("yolov8m.pt")
print("YOLOv8m loaded!")

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

print("")
print("=" * 60)
print("  TRAINING COMPLETED!")
print("=" * 60)

best = "/workspace/runs/autodamage_v8m/weights/best.pt"
if os.path.exists(best):
    size_mb = os.path.getsize(best) / (1024*1024)
    print(f"Best model: {best} ({size_mb:.1f} MB)")
else:
    print("WARNING: best.pt not found!")

try:
    m = results.results_dict
    print(f"mAP@50:    {m.get('metrics/mAP50(B)', 'N/A')}")
    print(f"mAP@50-95: {m.get('metrics/mAP50-95(B)', 'N/A')}")
    print(f"Precision: {m.get('metrics/precision(B)', 'N/A')}")
    print(f"Recall:    {m.get('metrics/recall(B)', 'N/A')}")
except Exception as e:
    print(f"Metrics error: {e}")

print("=" * 60)
TRAINPY

echo ""
echo "========================================"
echo "DONE! Best model: /workspace/runs/autodamage_v8m/weights/best.pt"
echo "========================================"
