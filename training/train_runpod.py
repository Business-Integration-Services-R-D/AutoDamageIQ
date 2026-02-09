#!/usr/bin/env python3
"""
AutoDamageID - RunPod GPU Eğitim Scripti
=========================================
Bu scripti RunPod'da GPU ile çalıştırın.

Kullanım:
1. RunPod'da yeni bir pod oluşturun (önerilen: RTX 3090/4090)
2. Bu scripti pod'a kopyalayın
3. Veri setini yükleyin
4. python train_runpod.py komutunu çalıştırın

Gereksinimler:
pip install ultralytics torch torchvision opencv-python
"""

import os
import sys
import yaml
import torch
from pathlib import Path
from datetime import datetime

# Configuration
CONFIG = {
    # Model ayarları
    "base_model": "yolov8l.pt",  # Büyük model - daha iyi doğruluk
    "task": "detect",
    
    # Eğitim ayarları
    "epochs": 100,
    "batch_size": 16,  # GPU belleğine göre ayarlayın
    "imgsz": 640,
    "patience": 20,  # Early stopping
    
    # Optimizer
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.01,
    
    # Augmentation
    "augment": True,
    "mosaic": 1.0,
    "mixup": 0.15,
    "degrees": 10.0,
    "translate": 0.1,
    "scale": 0.5,
    "flipud": 0.5,
    "fliplr": 0.5,
    
    # Diğer
    "workers": 8,
    "device": "0",  # GPU 0
    "project": "runs/damage_detection",
    "name": f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
}

# Sınıflar
CLASSES = {
    0: {"en": "crack", "tr": "Çatlak"},
    1: {"en": "dent", "tr": "Göçük"},
    2: {"en": "glass_shatter", "tr": "Cam Kırığı"},
    3: {"en": "lamp_broken", "tr": "Lamba Kırığı"},
    4: {"en": "scratch", "tr": "Çizik"},
    5: {"en": "tire_flat", "tr": "Patlak Lastik"},
}

def check_environment():
    """Ortam kontrolü"""
    print("🔍 Ortam Kontrolü")
    print(f"   Python: {sys.version}")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Bellek: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    return torch.cuda.is_available()

def create_data_yaml(data_path: str) -> str:
    """data.yaml dosyası oluştur"""
    data_yaml = {
        "path": data_path,
        "train": "images/train",
        "val": "images/val",
        "nc": 6,
        "names": [cls["en"] for cls in CLASSES.values()]
    }
    
    yaml_path = Path(data_path) / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    return str(yaml_path)

def train(data_yaml_path: str):
    """Eğitimi başlat"""
    from ultralytics import YOLO
    
    print("\n🚀 Eğitim Başlıyor...")
    print(f"   Model: {CONFIG['base_model']}")
    print(f"   Epochs: {CONFIG['epochs']}")
    print(f"   Batch Size: {CONFIG['batch_size']}")
    print(f"   Image Size: {CONFIG['imgsz']}")
    
    # Model yükle
    model = YOLO(CONFIG['base_model'])
    
    # Eğitimi başlat
    results = model.train(
        data=data_yaml_path,
        epochs=CONFIG['epochs'],
        batch=CONFIG['batch_size'],
        imgsz=CONFIG['imgsz'],
        patience=CONFIG['patience'],
        optimizer=CONFIG['optimizer'],
        lr0=CONFIG['lr0'],
        lrf=CONFIG['lrf'],
        augment=CONFIG['augment'],
        mosaic=CONFIG['mosaic'],
        mixup=CONFIG['mixup'],
        degrees=CONFIG['degrees'],
        translate=CONFIG['translate'],
        scale=CONFIG['scale'],
        flipud=CONFIG['flipud'],
        fliplr=CONFIG['fliplr'],
        workers=CONFIG['workers'],
        device=CONFIG['device'],
        project=CONFIG['project'],
        name=CONFIG['name'],
        exist_ok=True,
        verbose=True,
    )
    
    print("\n✅ Eğitim Tamamlandı!")
    print(f"   En iyi model: {CONFIG['project']}/{CONFIG['name']}/weights/best.pt")
    
    return results

def evaluate(model_path: str, data_yaml_path: str):
    """Modeli değerlendir"""
    from ultralytics import YOLO
    
    print("\n📊 Model Değerlendirmesi...")
    model = YOLO(model_path)
    
    metrics = model.val(
        data=data_yaml_path,
        split='val',
        batch=CONFIG['batch_size'],
        imgsz=CONFIG['imgsz'],
    )
    
    print("\n📈 Sonuçlar:")
    print(f"   mAP50: {metrics.box.map50:.4f}")
    print(f"   mAP50-95: {metrics.box.map:.4f}")
    print(f"   Precision: {metrics.box.mp:.4f}")
    print(f"   Recall: {metrics.box.mr:.4f}")
    
    # Sınıf bazlı sonuçlar
    print("\n📋 Sınıf Bazlı mAP50:")
    for i, (cls_name, cls_info) in enumerate(CLASSES.items()):
        if i < len(metrics.box.ap50):
            print(f"   {cls_info['tr']}: {metrics.box.ap50[i]:.4f}")
    
    return metrics

def export_model(model_path: str, formats: list = ['onnx', 'torchscript']):
    """Modeli farklı formatlara export et"""
    from ultralytics import YOLO
    
    print("\n📦 Model Export...")
    model = YOLO(model_path)
    
    for fmt in formats:
        print(f"   Exporting to {fmt}...")
        model.export(format=fmt)
    
    print("✅ Export tamamlandı!")

def main():
    """Ana fonksiyon"""
    print("="*60)
    print("🚗 AutoDamageID - Eğitim Scripti")
    print("="*60)
    
    # Ortam kontrolü
    has_gpu = check_environment()
    if not has_gpu:
        print("⚠️ GPU bulunamadı! CPU ile eğitim çok yavaş olacak.")
        CONFIG['device'] = 'cpu'
        CONFIG['batch_size'] = 4
    
    # Veri seti yolu
    data_path = os.environ.get('DATA_PATH', '/app/datasets/unified')
    data_yaml_path = os.path.join(data_path, 'data.yaml')
    
    if not os.path.exists(data_yaml_path):
        print(f"❌ data.yaml bulunamadı: {data_yaml_path}")
        print("   Lütfen veri setini yükleyin.")
        sys.exit(1)
    
    print(f"\n📁 Veri Seti: {data_path}")
    
    # Eğitim
    results = train(data_yaml_path)
    
    # Değerlendirme
    best_model = f"{CONFIG['project']}/{CONFIG['name']}/weights/best.pt"
    if os.path.exists(best_model):
        evaluate(best_model, data_yaml_path)
        
        # Export (opsiyonel)
        # export_model(best_model)
    
    print("\n🎉 İşlem tamamlandı!")

if __name__ == "__main__":
    main()
