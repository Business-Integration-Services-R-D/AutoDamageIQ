"""
AutoDamageID - Eğitim ve Etiketleme API Modülü
==============================================
Bu modül:
- Yeni görselleri etiketleme
- Mevcut etiketleri düzeltme
- RunPod üzerinden eğitim başlatma
- Eğitim durumu takibi
"""

import os
import json
import shutil
import base64
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from pydantic import BaseModel

# RunPod API
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
RUNPOD_GRAPHQL_URL = "https://api.runpod.io/graphql"

# Paths
DATASETS_DIR = Path("/app/datasets")
UNIFIED_DIR = DATASETS_DIR / "unified"
CUSTOM_DIR = DATASETS_DIR / "custom"  # Kullanıcı etiketli veriler
TRAINING_DIR = Path("/app/training")
JOBS_DIR = TRAINING_DIR / "jobs"

# Sınıflar
DAMAGE_CLASSES = {
    0: {"en": "crack", "tr": "Çatlak", "color": "#FF6B6B"},
    1: {"en": "dent", "tr": "Göçük", "color": "#4ECDC4"},
    2: {"en": "glass_shatter", "tr": "Cam Kırığı", "color": "#45B7D1"},
    3: {"en": "lamp_broken", "tr": "Lamba Kırığı", "color": "#96CEB4"},
    4: {"en": "scratch", "tr": "Çizik", "color": "#FFEAA7"},
    5: {"en": "tire_flat", "tr": "Patlak Lastik", "color": "#DDA0DD"},
}

# Pydantic Models
class BoundingBox(BaseModel):
    x: float  # normalized (0-1)
    y: float
    width: float
    height: float
    class_id: int
    confidence: Optional[float] = None

class AnnotationRequest(BaseModel):
    image_id: str
    boxes: List[BoundingBox]
    source: str = "user"  # user, correction, auto

class TrainingConfig(BaseModel):
    model_size: str = "m"  # n, s, m, l, x (v8 default: m)
    epochs: int = 200
    batch_size: int = 16
    image_size: int = 640
    include_custom: bool = True
    optimizer: str = "AdamW"
    patience: int = 25

class TrainingJob(BaseModel):
    job_id: str
    status: str  # pending, running, completed, failed
    created_at: str
    config: Dict[str, Any]
    metrics: Optional[Dict[str, Any]] = None

def ensure_custom_dirs():
    """Custom veri seti klasörlerini oluştur"""
    dirs = [
        CUSTOM_DIR / "images" / "train",
        CUSTOM_DIR / "images" / "val",
        CUSTOM_DIR / "labels" / "train",
        CUSTOM_DIR / "labels" / "val",
        CUSTOM_DIR / "pending",  # Etiketlenmemiş görseller
        JOBS_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

def save_annotation(
    image_id: str,
    image_data: bytes,
    boxes: List[Dict],
    source: str = "user"
) -> Dict[str, Any]:
    """
    Yeni etiket kaydet
    """
    ensure_custom_dirs()
    
    # Görsel kaydet
    image_path = CUSTOM_DIR / "images" / "train" / f"{image_id}.jpg"
    with open(image_path, 'wb') as f:
        f.write(image_data)
    
    # YOLO formatında etiket oluştur
    label_lines = []
    for box in boxes:
        x_center = box['x'] + box['width'] / 2
        y_center = box['y'] + box['height'] / 2
        label_lines.append(
            f"{box['class_id']} {x_center:.6f} {y_center:.6f} {box['width']:.6f} {box['height']:.6f}"
        )
    
    label_path = CUSTOM_DIR / "labels" / "train" / f"{image_id}.txt"
    with open(label_path, 'w') as f:
        f.write('\n'.join(label_lines))
    
    meta = {
        "image_id": image_id,
        "source": source,
        "created_at": datetime.utcnow().isoformat(),
        "box_count": len(boxes),
        "classes": list(set(b['class_id'] for b in boxes))
    }
    
    meta_path = CUSTOM_DIR / "labels" / "train" / f"{image_id}.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f)
    
    return {
        "success": True,
        "image_id": image_id,
        "image_path": str(image_path),
        "label_path": str(label_path),
        "box_count": len(boxes)
    }

def get_pending_images() -> List[Dict[str, Any]]:
    """Etiketlenmemiş görselleri listele"""
    ensure_custom_dirs()
    
    pending_dir = CUSTOM_DIR / "pending"
    images = []
    
    for img_path in pending_dir.glob("*"):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            with open(img_path, 'rb') as f:
                img_data = f.read()
            
            images.append({
                "id": img_path.stem,
                "filename": img_path.name,
                "size": len(img_data),
                "thumbnail": base64.b64encode(img_data).decode('utf-8')[:1000] + "..."
            })
    
    return images

def get_custom_dataset_stats() -> Dict[str, Any]:
    """Custom veri seti istatistikleri"""
    ensure_custom_dirs()
    
    train_images = list((CUSTOM_DIR / "images" / "train").glob("*"))
    train_labels = list((CUSTOM_DIR / "labels" / "train").glob("*.txt"))
    pending = list((CUSTOM_DIR / "pending").glob("*"))
    
    class_counts = {i: 0 for i in range(6)}
    for label_path in train_labels:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    if class_id in class_counts:
                        class_counts[class_id] += 1
    
    return {
        "train_images": len(train_images),
        "train_labels": len(train_labels),
        "pending_images": len(pending),
        "class_distribution": class_counts,
        "total_annotations": sum(class_counts.values())
    }

# =============================================================================
# RUNPOD ENTEGRASYONU
# =============================================================================

def runpod_query(query: str, variables: Dict = None) -> Dict:
    """RunPod GraphQL API çağrısı"""
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    
    response = requests.post(RUNPOD_GRAPHQL_URL, headers=headers, json=payload)
    return response.json()

def get_runpod_gpus() -> List[Dict]:
    """Kullanılabilir GPU'ları listele"""
    query = """
    query {
      gpuTypes {
        id
        displayName
        memoryInGb
        securePrice
      }
    }
    """
    result = runpod_query(query)
    if 'data' in result:
        return result['data'].get('gpuTypes', [])
    return []

def get_runpod_pods() -> List[Dict]:
    """Mevcut pod'ları listele"""
    query = """
    query {
      myself {
        pods {
          id
          name
          podType
          desiredStatus
          runtime {
            uptimeInSeconds
          }
          machine {
            gpuDisplayName
          }
        }
      }
    }
    """
    result = runpod_query(query)
    if 'data' in result and result['data']['myself']:
        return result['data']['myself'].get('pods', [])
    return []

def get_runpod_endpoints() -> List[Dict]:
    """Serverless endpoint'leri listele"""
    query = """
    query {
      myself {
        endpoints {
          id
          name
          templateId
          gpuIds
          workersMax
          workersMin
          idleTimeout
        }
      }
    }
    """
    result = runpod_query(query)
    if 'data' in result and result['data']['myself']:
        return result['data']['myself'].get('endpoints', [])
    return []

def create_runpod_pod(config: TrainingConfig) -> Dict[str, Any]:
    """
    RunPod'da yeni bir GPU pod oluştur ve eğitimi otomatik başlat
    """
    job_id = f"train_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    
    # URLs
    dataset_url = "https://damage-vision.preview.emergentagent.com/api/training/download-dataset"
    script_url = "https://damage-vision.preview.emergentagent.com/api/training/download-script"
    
    # Startup komutu - bash -c ile script'i indir ve çalıştır, sonra container'ı canlı tut
    startup_cmd = f'bash -c "wget -qO /workspace/train.sh {script_url} && chmod +x /workspace/train.sh && /workspace/train.sh 2>&1 | tee /workspace/training.log; tail -f /dev/null"'
    
    # Pod oluşturma mutation
    mutation = """
    mutation createPod($input: PodFindAndDeployOnDemandInput!) {
      podFindAndDeployOnDemand(input: $input) {
        id
        name
        desiredStatus
        imageName
        machineId
      }
    }
    """
    
    variables = {
        "input": {
            "cloudType": "COMMUNITY",
            "gpuCount": 1,
            "volumeInGb": 50,
            "containerDiskInGb": 20,
            "minVcpuCount": 4,
            "minMemoryInGb": 16,
            "gpuTypeId": "NVIDIA RTX A4000",
            "name": f"autodamage-{job_id}",
            "imageName": "ultralytics/ultralytics:latest",
            "dockerArgs": startup_cmd,
            "ports": "8888/http",
            "volumeMountPath": "/workspace",
            "startJupyter": True,
            "startSsh": True,
            "env": [
                {"key": "JOB_ID", "value": job_id},
                {"key": "MODEL_SIZE", "value": config.model_size},
                {"key": "EPOCHS", "value": str(config.epochs)},
                {"key": "BATCH_SIZE", "value": str(config.batch_size)},
                {"key": "IMAGE_SIZE", "value": str(config.image_size)},
                {"key": "DATASET_URL", "value": dataset_url},
                {"key": "WANDB_MODE", "value": "offline"}
            ]
        }
    }
    
    result = runpod_query(mutation, variables)
    
    # Job bilgisini kaydet
    job_info = {
        "job_id": job_id,
        "status": "creating",
        "created_at": datetime.utcnow().isoformat(),
        "config": config.dict(),
        "runpod_response": result,
        "dataset_url": dataset_url,
        "script_url": script_url
    }
    
    ensure_custom_dirs()
    job_file = JOBS_DIR / f"{job_id}.json"
    with open(job_file, 'w') as f:
        json.dump(job_info, f, indent=2)
    
    if 'data' in result and result['data'].get('podFindAndDeployOnDemand'):
        pod = result['data']['podFindAndDeployOnDemand']
        job_info["status"] = "running"
        job_info["pod_id"] = pod['id']
        job_info["pod_name"] = pod['name']
        
        with open(job_file, 'w') as f:
            json.dump(job_info, f, indent=2)
        
        return {
            "success": True,
            "job_id": job_id,
            "pod_id": pod['id'],
            "pod_name": pod['name'],
            "status": "running",
            "message": f"🚀 Eğitim otomatik başlatılıyor!",
            "dashboard_url": f"https://www.runpod.io/console/pods/{pod['id']}",
            "info": {
                "model": f"yolov8{config.model_size}",
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "dataset_size": "5,370 görsel",
                "estimated_time": "1-2 saat"
            }
        }
    else:
        error_msg = result.get('errors', [{}])[0].get('message', 'Bilinmeyen hata')
        job_info["status"] = "failed"
        job_info["error"] = error_msg
        
        with open(job_file, 'w') as f:
            json.dump(job_info, f, indent=2)
        
        return {
            "success": False,
            "job_id": job_id,
            "status": "failed",
            "error": error_msg,
            "message": f"Pod oluşturulamadı: {error_msg}"
        }

def start_runpod_training(config: TrainingConfig) -> Dict[str, Any]:
    """RunPod'da eğitim başlat - Direkt pod oluştur"""
    
    # Direkt yeni pod oluştur
    return create_runpod_pod(config)

def get_training_status(job_id: str) -> Dict[str, Any]:
    """Eğitim durumunu kontrol et"""
    ensure_custom_dirs()
    
    job_file = JOBS_DIR / f"{job_id}.json"
    
    if job_file.exists():
        with open(job_file, 'r') as f:
            job_info = json.load(f)
        
        # Pod durumunu güncelle
        if "pod_id" in job_info:
            pods = get_runpod_pods()
            for pod in pods:
                if pod['id'] == job_info['pod_id']:
                    job_info['pod_status'] = pod.get('desiredStatus', 'unknown')
                    runtime = pod.get('runtime')
                    job_info['uptime'] = runtime.get('uptimeInSeconds', 0) if runtime else 0
                    job_info['gpu'] = pod.get('machine', {}).get('gpuDisplayName', 'N/A')
                    break
        
        return job_info
    
    return {
        "job_id": job_id,
        "status": "not_found",
        "message": "Eğitim job'ı bulunamadı"
    }

def get_all_training_jobs() -> List[Dict[str, Any]]:
    """Tüm eğitim job'larını listele"""
    ensure_custom_dirs()
    
    jobs = []
    for job_file in JOBS_DIR.glob("*.json"):
        with open(job_file, 'r') as f:
            jobs.append(json.load(f))
    
    return sorted(jobs, key=lambda x: x.get('created_at', ''), reverse=True)

# Initialize
ensure_custom_dirs()
