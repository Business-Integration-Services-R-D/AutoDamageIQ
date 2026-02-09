"""
AutoDamageID - Model Yönetimi
==============================
Eğitilen modellerin yönetimi ve seçimi
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from pydantic import BaseModel

# Paths
MODELS_DIR = Path("/app/models")
WEIGHTS_DIR = Path("/app/src/yolo/weights")  # Mevcut modeller
TRAINED_DIR = MODELS_DIR / "trained"  # Eğitilen modeller

# Mevcut model bilgisi
CURRENT_MODEL_FILE = MODELS_DIR / "current_model.json"

class ModelInfo(BaseModel):
    id: str
    name: str
    path: str
    type: str  # "default", "trained", "custom"
    created_at: str
    metrics: Optional[Dict[str, float]] = None
    description: Optional[str] = None
    is_active: bool = False

def ensure_dirs():
    """Model klasörlerini oluştur"""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TRAINED_DIR.mkdir(parents=True, exist_ok=True)

def get_default_models() -> List[Dict[str, Any]]:
    """Varsayılan modelleri listele"""
    models = []
    
    # Hasar tespiti modeli (best.pt)
    damage_model = WEIGHTS_DIR / "best.pt"
    if damage_model.exists():
        models.append({
            "id": "default_damage",
            "name": "Hasar Tespiti (Varsayılan)",
            "path": str(damage_model),
            "type": "default",
            "created_at": datetime.fromtimestamp(damage_model.stat().st_mtime).isoformat(),
            "description": "Orijinal hasar tespiti modeli - 6 sınıf (crack, dent, glass_shatter, lamp_broken, scratch, tire_flat)",
            "metrics": None
        })
    
    # Parça segmentasyon modeli
    parts_model = WEIGHTS_DIR / "yolov8n-seg.pt"
    if parts_model.exists():
        models.append({
            "id": "default_parts",
            "name": "Parça Segmentasyonu (Varsayılan)",
            "path": str(parts_model),
            "type": "default",
            "created_at": datetime.fromtimestamp(parts_model.stat().st_mtime).isoformat(),
            "description": "Araç parçası segmentasyon modeli - 23 sınıf",
            "metrics": None
        })
    
    # YOLO11n modeli
    yolo11_model = WEIGHTS_DIR / "yolo11n.pt"
    if yolo11_model.exists():
        models.append({
            "id": "yolo11n_base",
            "name": "YOLO11n (Base Model)",
            "path": str(yolo11_model),
            "type": "default",
            "created_at": datetime.fromtimestamp(yolo11_model.stat().st_mtime).isoformat(),
            "description": "YOLO11 Nano - Eğitim için base model",
            "metrics": None
        })
    
    return models

def get_trained_models() -> List[Dict[str, Any]]:
    """Eğitilen modelleri listele"""
    ensure_dirs()
    models = []
    
    # trained klasöründeki modelleri tara
    for model_dir in TRAINED_DIR.glob("*/"):
        if model_dir.is_dir():
            best_pt = model_dir / "weights" / "best.pt"
            meta_file = model_dir / "meta.json"
            
            if best_pt.exists():
                # Meta bilgisi varsa oku
                meta = {}
                if meta_file.exists():
                    with open(meta_file, 'r') as f:
                        meta = json.load(f)
                
                models.append({
                    "id": model_dir.name,
                    "name": meta.get("name", f"Eğitilmiş Model - {model_dir.name}"),
                    "path": str(best_pt),
                    "type": "trained",
                    "created_at": meta.get("created_at", datetime.fromtimestamp(best_pt.stat().st_mtime).isoformat()),
                    "description": meta.get("description", "Kullanıcı eğitimi ile oluşturuldu"),
                    "metrics": meta.get("metrics", None)
                })
    
    return models

def get_all_models() -> List[Dict[str, Any]]:
    """Tüm modelleri listele"""
    models = get_default_models() + get_trained_models()
    
    # Custom modelleri ekle (models/ klasöründeki .pt dosyaları)
    for pt_file in MODELS_DIR.glob("*.pt"):
        model_id = pt_file.stem
        if not any(m["id"] == model_id for m in models):
            models.append({
                "id": model_id,
                "name": f"Özel Model - {model_id}",
                "path": str(pt_file),
                "type": "custom",
                "created_at": datetime.fromtimestamp(pt_file.stat().st_mtime).isoformat(),
                "description": "Yüklenmiş özel model",
                "metrics": None
            })
    
    # Aktif modeli işaretle
    current = get_current_model()
    for model in models:
        model["is_active"] = (model["id"] == current.get("id"))
    
    return models

def get_current_model() -> Dict[str, Any]:
    """Aktif modeli getir"""
    ensure_dirs()
    
    if CURRENT_MODEL_FILE.exists():
        with open(CURRENT_MODEL_FILE, 'r') as f:
            return json.load(f)
    
    # Varsayılan model
    default_models = get_default_models()
    if default_models:
        return default_models[0]
    
    return {"id": "none", "name": "Model Yok", "path": None}

def set_current_model(model_id: str) -> Dict[str, Any]:
    """Aktif modeli ayarla"""
    ensure_dirs()
    
    # Modeli bul
    all_models = get_all_models()
    model = next((m for m in all_models if m["id"] == model_id), None)
    
    if not model:
        raise ValueError(f"Model bulunamadı: {model_id}")
    
    # Kaydet
    with open(CURRENT_MODEL_FILE, 'w') as f:
        json.dump(model, f, indent=2)
    
    return model

def register_trained_model(
    job_id: str,
    name: str,
    model_path: str,
    metrics: Dict[str, float] = None,
    description: str = None
) -> Dict[str, Any]:
    """Yeni eğitilmiş modeli kaydet"""
    ensure_dirs()
    
    model_dir = TRAINED_DIR / job_id
    model_dir.mkdir(parents=True, exist_ok=True)
    
    meta = {
        "id": job_id,
        "name": name,
        "created_at": datetime.utcnow().isoformat(),
        "metrics": metrics or {},
        "description": description or f"Eğitim: {job_id}"
    }
    
    # Meta kaydet
    with open(model_dir / "meta.json", 'w') as f:
        json.dump(meta, f, indent=2)
    
    # Eğer model dosyası başka yerdeyse, kopyala veya symlink
    weights_dir = model_dir / "weights"
    weights_dir.mkdir(exist_ok=True)
    
    src_path = Path(model_path)
    if src_path.exists():
        dest_path = weights_dir / "best.pt"
        if not dest_path.exists():
            os.symlink(src_path.resolve(), dest_path)
    
    return {
        "id": job_id,
        "name": name,
        "path": str(weights_dir / "best.pt"),
        "type": "trained",
        **meta
    }

def delete_model(model_id: str) -> bool:
    """Eğitilmiş modeli sil (varsayılanlar silinemez)"""
    model_dir = TRAINED_DIR / model_id
    
    if model_dir.exists():
        import shutil
        shutil.rmtree(model_dir)
        
        # Eğer aktif modelse, varsayılana dön
        current = get_current_model()
        if current.get("id") == model_id:
            default_models = get_default_models()
            if default_models:
                set_current_model(default_models[0]["id"])
        
        return True
    
    return False

# Initialize
ensure_dirs()
