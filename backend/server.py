import os
import sys
import uuid
import base64
import torch
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from io import BytesIO

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from pymongo import MongoClient
from dotenv import load_dotenv
import numpy as np
import cv2
from PIL import Image

# Load environment variables
load_dotenv()

# Fix for PyTorch 2.6+ weights_only issue
torch.serialization.add_safe_globals([])

# Add src path for YOLO models
SRC_PATH = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_PATH))

# Set environment variable to allow unsafe loading for YOLO models
os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'

from ultralytics import YOLO

# Initialize FastAPI
app = FastAPI(
    title="AutoDamageID API",
    description="Araç Hasar Tespit ve Analiz API'si",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB connection
MONGO_URL = os.environ.get("MONGO_URL", "mongodb://localhost:27017/autodamageid")
client = MongoClient(MONGO_URL)
db = client.autodamageid
analyses_collection = db.analyses

# Model paths
YOLO_DIR = Path(__file__).parent.parent / "src" / "yolo"
MODELS_DIR = Path("/app/models")
DEFAULT_DAMAGE_MODEL_PATH = YOLO_DIR / "weights" / "best.pt"
# Yeni eğitilmiş parça segmentasyonu modeli
CUSTOM_PARTS_MODEL_PATH = MODELS_DIR / "carparts_seg_best.pt"
DEFAULT_PARTS_MODEL_PATH = YOLO_DIR / "runs" / "carparts_seg_v1" / "weights" / "best.pt"

# Load models (lazy loading)
damage_model = None
damage_model_path = None  # Track current model path
parts_model = None
parts_model_path = None  # Track current parts model path

def get_active_damage_model_path():
    """Aktif hasar modelinin yolunu al"""
    current_model_file = MODELS_DIR / "current_model.json"
    if current_model_file.exists():
        import json
        with open(current_model_file, 'r') as f:
            model_info = json.load(f)
            path = model_info.get("path")
            if path and Path(path).exists():
                return Path(path)
    return DEFAULT_DAMAGE_MODEL_PATH

def get_active_parts_model_path():
    """Aktif parça segmentasyonu modelinin yolunu al"""
    # Önce custom eğitilmiş modeli kontrol et
    if CUSTOM_PARTS_MODEL_PATH.exists():
        return CUSTOM_PARTS_MODEL_PATH
    return DEFAULT_PARTS_MODEL_PATH

def get_damage_model():
    global damage_model, damage_model_path
    active_path = get_active_damage_model_path()
    
    # Reload if model changed
    if damage_model is None or damage_model_path != str(active_path):
        print(f"Loading damage model from {active_path}")
        # Use weights_only=False for YOLO custom models
        import torch
        original_load = torch.load
        def patched_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        torch.load = patched_load
        damage_model = YOLO(str(active_path))
        damage_model_path = str(active_path)
        torch.load = original_load
    return damage_model

def get_parts_model():
    global parts_model, parts_model_path
    active_path = get_active_parts_model_path()
    
    # Reload if model changed
    if parts_model is None or parts_model_path != str(active_path):
        print(f"Loading parts model from {active_path}")
        # Use weights_only=False for YOLO custom models
        import torch
        original_load = torch.load
        def patched_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        torch.load = patched_load
        parts_model = YOLO(str(active_path))
        parts_model_path = str(active_path)
        torch.load = original_load
    return parts_model

# Damage type translations
DAMAGE_TR = {
    "crack": "Çatlak",
    "dent": "Göçük",
    "glass_shatter": "Cam Kırığı",
    "lamp_broken": "Lamba Kırığı",
    "scratch": "Çizik",
    "tire_flat": "Patlak Lastik"
}

# Part name translations
PARTS_TR = {
    "back_bumper": "Arka Tampon",
    "back_door": "Arka Kapı",
    "back_glass": "Arka Cam",
    "back_left_door": "Arka Sol Kapı",
    "back_left_light": "Arka Sol Far",
    "back_light": "Arka Far",
    "back_right_door": "Arka Sağ Kapı",
    "back_right_light": "Arka Sağ Far",
    "front_bumper": "Ön Tampon",
    "front_door": "Ön Kapı",
    "front_glass": "Ön Cam",
    "front_left_door": "Ön Sol Kapı",
    "front_left_light": "Ön Sol Far",
    "front_light": "Ön Far",
    "front_right_door": "Ön Sağ Kapı",
    "front_right_light": "Ön Sağ Far",
    "hood": "Kaput",
    "left_mirror": "Sol Ayna",
    "object": "Nesne",
    "right_mirror": "Sağ Ayna",
    "tailgate": "Bagaj Kapağı",
    "trunk": "Bagaj",
    "wheel": "Tekerlek"
}

# Severity mapping based on damage type
SEVERITY_MAP = {
    "crack": 3,
    "dent": 3,
    "glass_shatter": 5,
    "lamp_broken": 4,
    "scratch": 2,
    "tire_flat": 4
}

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON/MongoDB serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

def convert_to_native_types(obj):
    """Recursively convert all numpy types to native Python types"""
    import json
    
    def default_converter(o):
        if isinstance(o, (np.integer, np.int64, np.int32)):
            return int(o)
        elif isinstance(o, (np.floating, np.float64, np.float32)):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, np.bool_):
            return bool(o)
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")
    
    # Convert to JSON string and back to ensure all numpy types are converted
    json_str = json.dumps(obj, default=default_converter)
    return json.loads(json_str)

def box_iou(box_a, box_b):
    """Calculate IoU between two boxes [x1, y1, x2, y2]"""
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    
    area_a = max(0.0, (float(box_a[2]) - float(box_a[0])) * (float(box_a[3]) - float(box_a[1])))
    area_b = max(0.0, (float(box_b[2]) - float(box_b[0])) * (float(box_b[3]) - float(box_b[1])))
    
    union = area_a + area_b - inter_area + 1e-6
    return float(inter_area / union)

def analyze_image(image_np: np.ndarray) -> Dict[str, Any]:
    """Run damage detection and parts segmentation on image"""
    
    damage_mod = get_damage_model()
    parts_mod = get_parts_model()
    
    h, w = image_np.shape[:2]
    
    # Run damage detection
    damage_results = damage_mod.predict(
        source=image_np,
        imgsz=640,
        conf=0.05,
        verbose=False
    )[0]
    
    # Run parts segmentation
    parts_results = parts_mod.predict(
        source=image_np,
        imgsz=640,
        conf=0.05,
        verbose=False
    )[0]
    
    # Extract damage boxes
    dmg_boxes = damage_results.boxes.xyxy.cpu().numpy() if damage_results.boxes is not None else np.zeros((0, 4))
    dmg_cls = damage_results.boxes.cls.cpu().numpy().astype(int) if damage_results.boxes is not None else np.zeros((0,), int)
    dmg_conf = damage_results.boxes.conf.cpu().numpy() if damage_results.boxes is not None else np.zeros((0,))
    dmg_names = damage_mod.names
    
    # Extract part boxes
    part_boxes = parts_results.boxes.xyxy.cpu().numpy() if parts_results.boxes is not None else np.zeros((0, 4))
    part_cls = parts_results.boxes.cls.cpu().numpy().astype(int) if parts_results.boxes is not None else np.zeros((0,), int)
    part_names = parts_mod.names
    
    # Match damages to parts
    damages = []
    for i, dmg_box in enumerate(dmg_boxes):
        best_iou = 0.0
        best_part = None
        best_part_box = None
        
        for j, part_box in enumerate(part_boxes):
            iou = box_iou(dmg_box, part_box)
            if iou > best_iou:
                best_iou = iou
                best_part = part_names[int(part_cls[j])]
                best_part_box = [float(x) for x in part_box.tolist()]
        
        damage_type = dmg_names[int(dmg_cls[i])]
        confidence = float(dmg_conf[i])
        
        damage_entry = {
            "id": str(uuid.uuid4())[:8],
            "type": damage_type,
            "type_tr": DAMAGE_TR.get(damage_type, damage_type),
            "confidence": float(round(confidence * 100, 1)),
            "severity": int(SEVERITY_MAP.get(damage_type, 3)),
            "box": [float(x) for x in dmg_box.tolist()],
            "part": best_part if best_iou > 0.1 else None,
            "part_tr": PARTS_TR.get(best_part, best_part) if best_iou > 0.1 else None,
            "part_box": best_part_box if best_iou > 0.1 else None,
            "iou_with_part": float(round(best_iou, 3))
        }
        damages.append(damage_entry)
    
    # Extract unique parts detected
    parts = []
    for j, part_box in enumerate(part_boxes):
        part_name = part_names[int(part_cls[j])]
        parts.append({
            "name": part_name,
            "name_tr": PARTS_TR.get(part_name, part_name),
            "box": [float(x) for x in part_box.tolist()]
        })
    
    # Calculate summary
    total_damages = len(damages)
    affected_parts = len(set([d["part"] for d in damages if d["part"]]))
    avg_severity = float(round(sum([d["severity"] for d in damages]) / max(1, total_damages), 1))
    
    risk_level = "Düşük"
    if avg_severity >= 4 or total_damages >= 4:
        risk_level = "Yüksek"
    elif avg_severity >= 2.5 or total_damages >= 2:
        risk_level = "Orta"
    
    result = {
        "damages": damages,
        "parts": parts,
        "summary": {
            "total_damages": int(total_damages),
            "affected_parts": int(affected_parts),
            "average_severity": float(avg_severity),
            "risk_level": risk_level
        },
        "image_size": {"width": int(w), "height": int(h)}
    }
    
    # Convert all numpy types to native Python types
    return convert_numpy_types(result)

# Pydantic models
class AnalysisResponse(BaseModel):
    id: str
    created_at: str
    image_base64: str
    results: Dict[str, Any]

class AnalysisListItem(BaseModel):
    id: str
    created_at: str
    thumbnail: str
    summary: Dict[str, Any]

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "AutoDamageID"}

@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_vehicle(file: UploadFile = File(...)):
    """Upload and analyze a vehicle image for damage detection"""
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Sadece resim dosyaları kabul edilir")
    
    # Read image
    contents = await file.read()
    
    # Convert to numpy array
    nparr = np.frombuffer(contents, np.uint8)
    image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image_np is None:
        raise HTTPException(status_code=400, detail="Resim okunamadı")
    
    # Analyze
    results = analyze_image(image_np)
    
    # Ensure all numpy types are converted to native Python types
    results = convert_to_native_types(results)
    
    # Convert image to base64 for storage/display
    _, buffer = cv2.imencode('.jpg', image_np, [cv2.IMWRITE_JPEG_QUALITY, 85])
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Create thumbnail
    thumb_size = 200
    h, w = image_np.shape[:2]
    scale = thumb_size / max(h, w)
    thumb = cv2.resize(image_np, (int(w * scale), int(h * scale)))
    _, thumb_buffer = cv2.imencode('.jpg', thumb, [cv2.IMWRITE_JPEG_QUALITY, 60])
    thumbnail_base64 = base64.b64encode(thumb_buffer).decode('utf-8')
    
    # Create analysis record
    analysis_id = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat()
    
    analysis_doc = {
        "_id": analysis_id,
        "created_at": created_at,
        "image_base64": image_base64,
        "thumbnail": thumbnail_base64,
        "results": results,
        "filename": file.filename
    }
    
    # Save to MongoDB
    analyses_collection.insert_one(analysis_doc)
    
    return AnalysisResponse(
        id=analysis_id,
        created_at=created_at,
        image_base64=image_base64,
        results=results
    )

@app.get("/api/analyses")
async def get_analyses(limit: int = 20):
    """Get list of past analyses"""
    analyses = list(analyses_collection.find().sort("created_at", -1).limit(limit))
    
    return [
        {
            "id": str(a["_id"]),
            "created_at": a["created_at"],
            "thumbnail": a.get("thumbnail", ""),
            "summary": a["results"]["summary"],
            "filename": a.get("filename", "Bilinmeyen")
        }
        for a in analyses
    ]

@app.get("/api/analyses/{analysis_id}")
async def get_analysis(analysis_id: str):
    """Get a specific analysis by ID"""
    analysis = analyses_collection.find_one({"_id": analysis_id})
    
    if not analysis:
        raise HTTPException(status_code=404, detail="Analiz bulunamadı")
    
    return {
        "id": str(analysis["_id"]),
        "created_at": analysis["created_at"],
        "image_base64": analysis["image_base64"],
        "results": analysis["results"],
        "filename": analysis.get("filename", "Bilinmeyen")
    }

@app.delete("/api/analyses/{analysis_id}")
async def delete_analysis(analysis_id: str):
    """Delete an analysis"""
    result = analyses_collection.delete_one({"_id": analysis_id})
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Analiz bulunamadı")
    
    return {"message": "Analiz silindi"}

@app.get("/api/analyses/{analysis_id}/pdf")
async def download_pdf(analysis_id: str):
    """Generate and download PDF report"""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    
    analysis = analyses_collection.find_one({"_id": analysis_id})
    
    if not analysis:
        raise HTTPException(status_code=404, detail="Analiz bulunamadı")
    
    # Create PDF in memory
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1.5*cm, bottomMargin=1.5*cm)
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=24, spaceAfter=20)
    heading_style = ParagraphStyle('Heading', parent=styles['Heading2'], fontSize=14, spaceAfter=10, spaceBefore=15)
    normal_style = styles['Normal']
    
    elements = []
    
    # Title
    elements.append(Paragraph("Araç Hasar Analiz Raporu", title_style))
    elements.append(Paragraph(f"Tarih: {analysis['created_at'][:10]}", normal_style))
    elements.append(Paragraph(f"Rapor ID: {analysis_id[:8]}", normal_style))
    elements.append(Spacer(1, 20))
    
    # Add image
    img_data = base64.b64decode(analysis['image_base64'])
    img_buffer = BytesIO(img_data)
    img = RLImage(img_buffer, width=14*cm, height=10*cm, kind='proportional')
    elements.append(img)
    elements.append(Spacer(1, 20))
    
    # Summary
    results = analysis['results']
    summary = results['summary']
    
    elements.append(Paragraph("Özet", heading_style))
    summary_data = [
        ["Toplam Hasar", str(summary['total_damages'])],
        ["Etkilenen Parça", str(summary['affected_parts'])],
        ["Ortalama Şiddet", f"{summary['average_severity']}/5"],
        ["Risk Seviyesi", summary['risk_level']]
    ]
    summary_table = Table(summary_data, colWidths=[6*cm, 6*cm])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F5F5F7')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#1D1D1F')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E5E5E7'))
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 20))
    
    # Damage details
    if results['damages']:
        elements.append(Paragraph("Hasar Detayları", heading_style))
        damage_data = [["Hasar Tipi", "Parça", "Güven", "Şiddet"]]
        for d in results['damages']:
            severity_dots = "●" * d['severity'] + "○" * (5 - d['severity'])
            damage_data.append([
                d['type_tr'],
                d['part_tr'] or "Belirsiz",
                f"%{d['confidence']}",
                severity_dots
            ])
        
        damage_table = Table(damage_data, colWidths=[4*cm, 4*cm, 2.5*cm, 3.5*cm])
        damage_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#000000')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E5E5E7')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#FAFAFA')])
        ]))
        elements.append(damage_table)
    else:
        elements.append(Paragraph("Hasar tespit edilmedi.", normal_style))
    
    elements.append(Spacer(1, 30))
    elements.append(Paragraph("Bu rapor AutoDamageID yapay zeka sistemi tarafından otomatik olarak oluşturulmuştur.", 
                              ParagraphStyle('Footer', parent=normal_style, fontSize=9, textColor=colors.HexColor('#86868B'))))
    
    doc.build(elements)
    buffer.seek(0)
    
    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=hasar-raporu-{analysis_id[:8]}.pdf"}
    )

# =============================================================================
# EĞİTİM VE ETİKETLEME API'LERİ
# =============================================================================

from training_api import (
    save_annotation, get_pending_images, get_custom_dataset_stats,
    start_runpod_training, get_training_status, DAMAGE_CLASSES,
    TrainingConfig, AnnotationRequest
)

# Sınıf bilgilerini döndür
@app.get("/api/training/classes")
async def get_damage_classes():
    """Hasar sınıflarını ve renklerini döndür"""
    return {
        "classes": DAMAGE_CLASSES,
        "count": len(DAMAGE_CLASSES)
    }

# Custom veri seti istatistikleri
@app.get("/api/training/stats")
async def get_training_stats():
    """Eğitim veri seti istatistikleri"""
    return get_custom_dataset_stats()

# Etiketlenmemiş görseller
@app.get("/api/training/pending")
async def get_pending():
    """Etiketlenmemiş görselleri listele"""
    return {"images": get_pending_images()}

# Yeni görsel yükle (etiketleme için)
@app.post("/api/training/upload")
async def upload_for_annotation(file: UploadFile = File(...)):
    """Etiketleme için yeni görsel yükle"""
    from pathlib import Path
    
    # Dosyayı kaydet
    pending_dir = Path("/app/datasets/custom/pending")
    pending_dir.mkdir(parents=True, exist_ok=True)
    
    # Unique isim oluştur
    file_id = str(uuid.uuid4())[:8]
    ext = Path(file.filename).suffix or ".jpg"
    file_path = pending_dir / f"{file_id}{ext}"
    
    contents = await file.read()
    with open(file_path, 'wb') as f:
        f.write(contents)
    
    # Base64 thumbnail
    thumbnail = base64.b64encode(contents).decode('utf-8')
    
    return {
        "id": file_id,
        "filename": file.filename,
        "path": str(file_path),
        "thumbnail": thumbnail[:500] + "..." if len(thumbnail) > 500 else thumbnail
    }

# Etiket kaydet
@app.post("/api/training/annotate")
async def save_annotation_endpoint(
    image_id: str,
    boxes: List[Dict[str, Any]],
    source: str = "user"
):
    """Görsel için etiket kaydet"""
    from pathlib import Path
    
    # Pending klasöründen görseli bul
    pending_dir = Path("/app/datasets/custom/pending")
    image_path = None
    
    for ext in ['.jpg', '.jpeg', '.png']:
        potential_path = pending_dir / f"{image_id}{ext}"
        if potential_path.exists():
            image_path = potential_path
            break
    
    if not image_path:
        raise HTTPException(status_code=404, detail="Görsel bulunamadı")
    
    # Görseli oku
    with open(image_path, 'rb') as f:
        image_data = f.read()
    
    # Etiketi kaydet
    result = save_annotation(image_id, image_data, boxes, source)
    
    # Pending'den sil
    image_path.unlink()
    
    return result

# Analiz sonucunu düzelt ve eğitime ekle
@app.post("/api/training/correct/{analysis_id}")
async def correct_analysis(analysis_id: str, boxes: List[Dict[str, Any]]):
    """Mevcut analiz sonucunu düzelt ve eğitim verisine ekle"""
    
    # Analizi bul
    analysis = analyses_collection.find_one({"_id": analysis_id})
    if not analysis:
        raise HTTPException(status_code=404, detail="Analiz bulunamadı")
    
    # Görseli decode et
    image_data = base64.b64decode(analysis['image_base64'])
    
    # Yeni etiket kaydet
    result = save_annotation(
        f"corrected_{analysis_id[:8]}", 
        image_data, 
        boxes, 
        source="correction"
    )
    
    return {
        "success": True,
        "message": "Düzeltme kaydedildi",
        "original_id": analysis_id,
        **result
    }

# Eğitim başlat
@app.post("/api/training/start")
async def start_training(config: TrainingConfig):
    """RunPod'da eğitim başlat"""
    return start_runpod_training(config)

# Eğitim durumu
@app.get("/api/training/status/{job_id}")
async def training_status(job_id: str):
    """Eğitim durumunu kontrol et"""
    return get_training_status(job_id)

# Tüm eğitim job'larını listele
@app.get("/api/training/jobs")
async def list_training_jobs():
    """Tüm eğitim job'larını listele"""
    from training_api import get_all_training_jobs
    return {"jobs": get_all_training_jobs()}

# RunPod durumu
@app.get("/api/training/runpod-status")
async def runpod_status():
    """RunPod hesap durumunu göster"""
    from training_api import get_runpod_pods, get_runpod_endpoints
    return {
        "pods": get_runpod_pods(),
        "endpoints": get_runpod_endpoints()
    }

# Tüm unified veri seti istatistikleri
@app.get("/api/training/unified-stats")
async def get_unified_stats():
    """Birleşik veri seti istatistikleri"""
    from pathlib import Path
    
    unified_dir = Path("/app/datasets/unified")
    
    stats = {
        "train_images": 0,
        "val_images": 0,
        "class_distribution": {i: 0 for i in range(6)}
    }
    
    # Train
    train_labels = unified_dir / "labels" / "train"
    if train_labels.exists():
        for lbl in train_labels.glob("*.txt"):
            stats["train_images"] += 1
            with open(lbl, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        if class_id in stats["class_distribution"]:
                            stats["class_distribution"][class_id] += 1
    
    # Val
    val_labels = unified_dir / "labels" / "val"
    if val_labels.exists():
        stats["val_images"] = len(list(val_labels.glob("*.txt")))
    
    return stats

# Veri seti indirme endpoint'i (RunPod için)
@app.get("/api/training/download-dataset")
async def download_dataset():
    """Eğitim veri setini indir"""
    dataset_path = Path("/app/training/dataset.tar.gz")
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found. Please prepare it first.")
    return FileResponse(
        path=str(dataset_path),
        media_type="application/gzip",
        filename="dataset.tar.gz"
    )

# Training script indirme endpoint'i
@app.get("/api/training/download-script")
async def download_script():
    """Training script'i indir"""
    from fastapi.responses import PlainTextResponse
    script_path = Path("/app/training/train_script.sh")
    if not script_path.exists():
        raise HTTPException(status_code=404, detail="Script not found.")
    with open(script_path, 'r') as f:
        content = f.read()
    return PlainTextResponse(content=content, media_type="text/plain")

@app.get("/api/training/download-script-seg")
async def download_script_seg():
    """Segmentation training script'i indir"""
    from fastapi.responses import PlainTextResponse
    script_path = Path("/app/training/train_carparts_seg.sh")
    if not script_path.exists():
        raise HTTPException(status_code=404, detail="Segmentation script not found.")
    with open(script_path, 'r') as f:
        content = f.read()
    return PlainTextResponse(content=content, media_type="text/plain")

# =============================================================================
# MODEL YÖNETİMİ API'LERİ
# =============================================================================

from model_manager import (
    get_all_models, get_current_model, set_current_model,
    register_trained_model, delete_model
)

@app.get("/api/models")
async def list_models():
    """Tüm modelleri listele"""
    return {"models": get_all_models()}

@app.get("/api/models/current")
async def current_model():
    """Aktif modeli getir"""
    return get_current_model()

@app.post("/api/models/{model_id}/activate")
async def activate_model(model_id: str):
    """Modeli aktif yap"""
    try:
        model = set_current_model(model_id)
        return {"success": True, "model": model, "message": "Model aktif edildi"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.delete("/api/models/{model_id}")
async def remove_model(model_id: str):
    """Eğitilmiş modeli sil"""
    success = delete_model(model_id)
    if success:
        return {"success": True, "message": "Model silindi"}
    raise HTTPException(status_code=404, detail="Model bulunamadı veya silinemez")

# =============================================================================
# ETİKETLEME DÜZELTME - Body olarak boxes al
# =============================================================================

from pydantic import BaseModel as PydanticBaseModel
from typing import List as TypingList

class BoxData(PydanticBaseModel):
    x: float
    y: float
    width: float
    height: float
    class_id: int

class AnnotateRequest(PydanticBaseModel):
    image_id: str
    boxes: TypingList[BoxData]
    source: str = "user"

@app.post("/api/training/annotate/save")
async def save_annotation_v2(request: AnnotateRequest):
    """Görsel için etiket kaydet (v2 - JSON body)"""
    from pathlib import Path
    
    # Pending klasöründen görseli bul
    pending_dir = Path("/app/datasets/custom/pending")
    image_path = None
    
    for ext in ['.jpg', '.jpeg', '.png']:
        potential_path = pending_dir / f"{request.image_id}{ext}"
        if potential_path.exists():
            image_path = potential_path
            break
    
    if not image_path:
        raise HTTPException(status_code=404, detail="Görsel bulunamadı")
    
    # Görseli oku
    with open(image_path, 'rb') as f:
        image_data = f.read()
    
    # Box'ları dict'e çevir
    boxes_dict = [{"x": b.x, "y": b.y, "width": b.width, "height": b.height, "class_id": b.class_id} for b in request.boxes]
    
    # Etiketi kaydet
    result = save_annotation(request.image_id, image_data, boxes_dict, request.source)
    
    # Pending'den sil
    image_path.unlink()
    
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
