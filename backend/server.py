import os
import sys
import uuid
import base64
import logging
from datetime import datetime, timezone
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

logger = logging.getLogger("autodamageid")

# --- Graceful ML imports (production-safe) ---
_ml_available = False
_ml_import_error = None
YOLO = None

try:
    import torch
    torch.serialization.add_safe_globals([])
    os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'

    SRC_PATH = Path(__file__).parent.parent / "src"
    sys.path.insert(0, str(SRC_PATH))

    from ultralytics import YOLO as _YOLO
    YOLO = _YOLO
    _ml_available = True
    logger.info("ML libraries (PyTorch + YOLO) loaded successfully")
except Exception as e:
    _ml_import_error = str(e)
    logger.warning(f"ML libraries not available: {e}. Analysis endpoints will return errors.")

# Import lightweight modules (always available)
from image_quality import assess_image_quality
from repair_engine import get_repair_recommendation
from anomaly_detector import compute_phash, check_duplicate_image, generate_anomaly_score
from before_after import compare_analyses

# Conditional SAM import
try:
    from sam_integration import is_sam_available, enhance_damages_with_sam, get_sam_status
except Exception:
    def is_sam_available(): return False
    def enhance_damages_with_sam(img, damages): return damages
    def get_sam_status(): return {"available": False, "reason": "SAM module not loaded"}

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
MONGO_URL = os.environ.get("MONGO_URL")
DB_NAME = os.environ.get("DB_NAME")
if not MONGO_URL or not DB_NAME:
    raise RuntimeError("MONGO_URL and DB_NAME environment variables must be set")
client = MongoClient(MONGO_URL)
db = client[DB_NAME]
analyses_collection = db.analyses

# Model paths
YOLO_DIR = Path(__file__).parent.parent / "src" / "yolo"
MODELS_DIR = Path("/app/models")
DEFAULT_DAMAGE_MODEL_PATH = YOLO_DIR / "weights" / "best.pt"
CUSTOM_PARTS_MODEL_PATH = MODELS_DIR / "carparts_seg_best.pt"
DEFAULT_PARTS_MODEL_PATH = YOLO_DIR / "runs" / "carparts_seg_v1" / "weights" / "best.pt"

# Load models (lazy loading)
damage_model = None
damage_model_path = None
parts_model = None
parts_model_path = None

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
    if DEFAULT_DAMAGE_MODEL_PATH.exists():
        return DEFAULT_DAMAGE_MODEL_PATH
    return None

def get_active_parts_model_path():
    """Aktif parça segmentasyonu modelinin yolunu al"""
    if CUSTOM_PARTS_MODEL_PATH.exists():
        return CUSTOM_PARTS_MODEL_PATH
    if DEFAULT_PARTS_MODEL_PATH.exists():
        return DEFAULT_PARTS_MODEL_PATH
    return None

def _load_yolo_model(model_path):
    """Safely load a YOLO model with weights_only=False patch"""
    if not _ml_available or YOLO is None:
        return None
    if model_path is None or not Path(model_path).exists():
        logger.warning(f"Model file not found: {model_path}")
        return None
    try:
        import torch as _torch
        original_load = _torch.load
        def patched_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        _torch.load = patched_load
        try:
            model = YOLO(str(model_path))
        finally:
            _torch.load = original_load
        return model
    except Exception as e:
        logger.error(f"Model loading failed for {model_path}: {e}")
        return None

def get_damage_model():
    global damage_model, damage_model_path
    active_path = get_active_damage_model_path()
    if active_path is None:
        return None
    if damage_model is None or damage_model_path != str(active_path):
        logger.info(f"Loading damage model from {active_path}")
        damage_model = _load_yolo_model(active_path)
        damage_model_path = str(active_path)
    return damage_model

def get_parts_model():
    global parts_model, parts_model_path
    active_path = get_active_parts_model_path()
    if active_path is None:
        return None
    if parts_model is None or parts_model_path != str(active_path):
        logger.info(f"Loading parts model from {active_path}")
        parts_model = _load_yolo_model(active_path)
        parts_model_path = str(active_path)
    return parts_model

# Import shared constants
from constants import DAMAGE_TR, PARTS_TR, SEVERITY_MAP

# Manual review confidence threshold
REVIEW_CONFIDENCE_THRESHOLD = 35.0
REVIEW_RISK_THRESHOLD = "Yuksek"

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

def calculate_enhanced_severity(damage_type: str, box: list, image_w: int, image_h: int, confidence: float) -> Dict[str, Any]:
    """Cok degiskenli siddet skoru hesapla"""
    base_severity = SEVERITY_MAP.get(damage_type, 3)
    
    # Alan orani hesapla (hasar kutusu / gorsel alani)
    box_area = max(0, (box[2] - box[0]) * (box[3] - box[1]))
    image_area = max(1, image_w * image_h)
    area_ratio = box_area / image_area
    
    # Alan bazli siddet katkisi (buyuk hasar = daha ciddi)
    if area_ratio > 0.15:
        area_factor = 1.5
    elif area_ratio > 0.08:
        area_factor = 1.2
    elif area_ratio > 0.03:
        area_factor = 1.0
    else:
        area_factor = 0.8
    
    # Guven bazli ayarlama (dusuk guven = belirsizlik)
    confidence_factor = 1.0 if confidence > 50 else 0.9
    
    # Birlesik skor
    enhanced_score = base_severity * area_factor * confidence_factor
    enhanced_score = round(min(5.0, max(1.0, enhanced_score)), 1)
    
    # Tam sayi siddet sinifi
    severity_class = int(round(enhanced_score))
    severity_class = max(1, min(5, severity_class))
    
    # Siddet etiketi
    if enhanced_score >= 4.0:
        severity_label = "Yuksek"
    elif enhanced_score >= 2.5:
        severity_label = "Orta"
    else:
        severity_label = "Dusuk"
    
    return {
        "score": float(enhanced_score),
        "class": severity_class,
        "label": severity_label,
        "area_ratio": float(round(area_ratio * 100, 2)),
        "base_severity": base_severity,
        "area_factor": float(area_factor),
        "confidence_factor": float(confidence_factor)
    }


def analyze_image(image_np: np.ndarray, enable_sam: bool = True) -> Dict[str, Any]:
    """Run damage detection and parts segmentation on image"""
    
    damage_mod = get_damage_model()
    if damage_mod is None:
        raise HTTPException(
            status_code=503,
            detail="Hasar tespit modeli yüklenemedi. Model dosyaları mevcut değil veya ML kütüphaneleri yüklü değil."
        )
    
    parts_mod = get_parts_model()
    # parts_mod None olabilir - VLM fallback devreye girer
    
    h, w = image_np.shape[:2]
    
    # Run damage detection
    damage_results = damage_mod.predict(
        source=image_np,
        imgsz=640,
        conf=0.15,
        verbose=False
    )[0]
    
    # Run parts segmentation (eğer model varsa)
    part_boxes = np.zeros((0, 4))
    part_cls = np.zeros((0,), int)
    part_names = {}
    
    if parts_mod is not None:
        parts_results = parts_mod.predict(
            source=image_np,
            imgsz=640,
            conf=0.15,
            verbose=False
        )[0]
        part_boxes = parts_results.boxes.xyxy.cpu().numpy() if parts_results.boxes is not None else np.zeros((0, 4))
        part_cls = parts_results.boxes.cls.cpu().numpy().astype(int) if parts_results.boxes is not None else np.zeros((0,), int)
        part_names = parts_mod.names
    
    # Extract damage boxes
    dmg_boxes = damage_results.boxes.xyxy.cpu().numpy() if damage_results.boxes is not None else np.zeros((0, 4))
    dmg_cls = damage_results.boxes.cls.cpu().numpy().astype(int) if damage_results.boxes is not None else np.zeros((0,), int)
    dmg_conf = damage_results.boxes.conf.cpu().numpy() if damage_results.boxes is not None else np.zeros((0,))
    dmg_names = damage_mod.names
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
        
        # Enhanced severity calculation
        severity_info = calculate_enhanced_severity(
            damage_type, 
            [float(x) for x in dmg_box.tolist()], 
            w, h, 
            confidence * 100
        )
        
        # Repair recommendation
        repair_rec = get_repair_recommendation(
            damage_type=damage_type,
            severity=severity_info["class"],
            confidence=confidence * 100,
            panel=best_part if best_iou > 0.1 else None
        )
        
        damage_entry = {
            "id": str(uuid.uuid4())[:8],
            "type": damage_type,
            "type_tr": DAMAGE_TR.get(damage_type, damage_type),
            "confidence": float(round(confidence * 100, 1)),
            "severity": severity_info["class"],
            "severity_details": severity_info,
            "box": [float(x) for x in dmg_box.tolist()],
            "part": best_part if best_iou > 0.1 else None,
            "part_tr": PARTS_TR.get(best_part, best_part) if best_iou > 0.1 else None,
            "part_box": best_part_box if best_iou > 0.1 else None,
            "iou_with_part": float(round(best_iou, 3)),
            "repair": repair_rec
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
    avg_confidence = float(round(sum([d["confidence"] for d in damages]) / max(1, total_damages), 1))
    
    risk_level = "Dusuk"
    if avg_severity >= 4 or total_damages >= 4:
        risk_level = "Yuksek"
    elif avg_severity >= 2.5 or total_damages >= 2:
        risk_level = "Orta"
    
    # Manuel inceleme gerekli mi?
    needs_review = (
        avg_confidence < REVIEW_CONFIDENCE_THRESHOLD or
        risk_level == "Yuksek" or
        any(d["confidence"] < 20 for d in damages)
    )
    
    review_reasons = []
    if avg_confidence < REVIEW_CONFIDENCE_THRESHOLD:
        review_reasons.append("Dusuk ortalama guven skoru")
    if risk_level == "Yuksek":
        review_reasons.append("Yuksek risk seviyesi")
    if any(d["confidence"] < 20 for d in damages):
        review_reasons.append("Cok dusuk guvenli hasar tespiti mevcut")
    
    result = {
        "damages": damages,
        "parts": parts,
        "summary": {
            "total_damages": int(total_damages),
            "affected_parts": int(affected_parts),
            "average_severity": float(avg_severity),
            "average_confidence": float(avg_confidence),
            "risk_level": risk_level,
            "needs_review": bool(needs_review),
            "review_reasons": review_reasons
        },
        "image_size": {"width": int(w), "height": int(h)},
        "sam_available": is_sam_available()
    }
    
    # SAM enhancement (optional)
    if enable_sam and is_sam_available() and len(damages) > 0:
        try:
            result["damages"] = enhance_damages_with_sam(image_np, damages)
            result["sam_used"] = True
        except Exception as e:
            print(f"SAM enhancement error: {e}")
            result["sam_used"] = False
    else:
        result["sam_used"] = False
    
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
    damage_path = get_active_damage_model_path()
    parts_path = get_active_parts_model_path()
    return {
        "status": "healthy",
        "service": "AutoDamageID",
        "ml_available": _ml_available,
        "ml_error": _ml_import_error,
        "damage_model": str(damage_path) if damage_path else None,
        "parts_model": str(parts_path) if parts_path else None,
        "sam_available": is_sam_available(),
        "vlm_available": bool(os.environ.get("EMERGENT_LLM_KEY"))
    }

@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_vehicle(file: UploadFile = File(...)):
    """Upload and analyze a vehicle image for damage detection"""
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Sadece resim dosyalari kabul edilir")
    
    # Read image
    contents = await file.read()
    
    # Convert to numpy array
    nparr = np.frombuffer(contents, np.uint8)
    image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image_np is None:
        raise HTTPException(status_code=400, detail="Resim okunamadi")
    
    # Step 1: Image quality assessment
    quality_result = assess_image_quality(image_np)
    
    # Step 2: Compute perceptual hash
    phash = compute_phash(image_np)
    
    # Step 3: Check for duplicate images
    duplicate_result = check_duplicate_image(image_np, phash, db)
    
    # Step 4: Generate anomaly score
    anomaly_result = generate_anomaly_score(image_np, duplicate_result, quality_result)
    
    # Step 5: Run damage analysis (even if quality is low, but add warning)
    results = analyze_image(image_np)
    
    # Step 6: VLM Fallback - belirsiz parça eşleşmeleri için GPT-4o
    try:
        from vlm_parts_fallback import enhance_damages_with_vlm
        unmatched = [d for d in results["damages"] if d.get("part") is None]
        if unmatched:
            results["damages"] = await enhance_damages_with_vlm(image_np, results["damages"])
            vlm_enhanced_count = sum(1 for d in results["damages"] if d.get("part_source") == "vlm")
            results["vlm_enhanced"] = vlm_enhanced_count
            # Etkilenen parça sayısını güncelle
            results["summary"]["affected_parts"] = len(set(
                d["part"] for d in results["damages"] if d.get("part")
            ))
            logger.info(f"VLM enhanced {vlm_enhanced_count} damages")
        else:
            results["vlm_enhanced"] = 0
            for d in results["damages"]:
                d["part_source"] = "yolo"
    except Exception as e:
        logger.warning(f"VLM fallback skipped: {e}")
        results["vlm_enhanced"] = 0
    
    # Ensure all numpy types are converted to native Python types
    results = convert_to_native_types(results)
    
    # Add quality, anomaly and review data to results
    results["quality"] = quality_result
    results["anomaly"] = anomaly_result
    
    # Update review status based on quality and anomaly
    if quality_result.get("warnings"):
        if not results["summary"]["needs_review"]:
            high_quality_warnings = [w for w in quality_result["warnings"] if w["severity"] == "high"]
            if high_quality_warnings:
                results["summary"]["needs_review"] = True
                results["summary"]["review_reasons"].append("Dusuk goruntu kalitesi")
    
    if anomaly_result.get("anomaly_score", 0) >= 30:
        results["summary"]["needs_review"] = True
        results["summary"]["review_reasons"].append("Anomali sinyali tespit edildi")
    
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
    created_at = datetime.now(timezone.utc).isoformat()
    
    analysis_doc = {
        "_id": analysis_id,
        "created_at": created_at,
        "image_base64": image_base64,
        "thumbnail": thumbnail_base64,
        "results": results,
        "filename": file.filename,
        "phash": phash,
        "needs_review": results["summary"]["needs_review"]
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
        raise HTTPException(status_code=404, detail="Analiz bulunamadi")
    
    return {"message": "Analiz silindi"}

@app.get("/api/review-queue")
async def get_review_queue(limit: int = 50):
    """Manuel inceleme gerektiren analizleri listele"""
    analyses = list(
        analyses_collection.find(
            {"needs_review": True},
            {"image_base64": 0}
        ).sort("created_at", -1).limit(limit)
    )
    
    return [
        {
            "id": str(a["_id"]),
            "created_at": a["created_at"],
            "thumbnail": a.get("thumbnail", ""),
            "summary": a["results"]["summary"],
            "quality": a["results"].get("quality", {}),
            "anomaly": a["results"].get("anomaly", {}),
            "filename": a.get("filename", "Bilinmeyen"),
            "review_reasons": a["results"]["summary"].get("review_reasons", [])
        }
        for a in analyses
    ]

@app.post("/api/analyses/{analysis_id}/review")
async def mark_reviewed(analysis_id: str):
    """Analizi incelendi olarak isaretle"""
    result = analyses_collection.update_one(
        {"_id": analysis_id},
        {"$set": {"needs_review": False, "reviewed_at": datetime.utcnow().isoformat()}}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Analiz bulunamadi")
    
    return {"message": "Analiz incelendi olarak isaretlendi"}

@app.post("/api/quality-check")
async def quality_check(file: UploadFile = File(...)):
    """Goruntu kalite kontrolu yap (analiz yapmadan)"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Sadece resim dosyalari kabul edilir")
    
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image_np is None:
        raise HTTPException(status_code=400, detail="Resim okunamadi")
    
    quality_result = assess_image_quality(image_np)
    return quality_result

# =============================================================================
# SAM STATUS ENDPOINT
# =============================================================================

@app.get("/api/sam/status")
async def sam_status():
    """SAM model durumunu dondur"""
    return get_sam_status()

# =============================================================================
# BEFORE/AFTER KARSILASTIRMA ENDPOINT'LERI
# =============================================================================

class CompareRequest(BaseModel):
    before_id: str
    after_id: str

@app.post("/api/compare")
async def compare_before_after(request: CompareRequest):
    """Iki analizi karsilastir - yeni hasar tespiti"""
    
    # Before analizi
    before_doc = analyses_collection.find_one({"_id": request.before_id})
    if not before_doc:
        raise HTTPException(status_code=404, detail="Onceki analiz bulunamadi")
    
    # After analizi
    after_doc = analyses_collection.find_one({"_id": request.after_id})
    if not after_doc:
        raise HTTPException(status_code=404, detail="Sonraki analiz bulunamadi")
    
    # Gorselleri decode et
    before_img_data = base64.b64decode(before_doc["image_base64"])
    after_img_data = base64.b64decode(after_doc["image_base64"])
    
    before_np = cv2.imdecode(np.frombuffer(before_img_data, np.uint8), cv2.IMREAD_COLOR)
    after_np = cv2.imdecode(np.frombuffer(after_img_data, np.uint8), cv2.IMREAD_COLOR)
    
    if before_np is None or after_np is None:
        raise HTTPException(status_code=500, detail="Gorseller okunamadi")
    
    # Hasar listelerini al
    before_damages = before_doc.get("results", {}).get("damages", [])
    after_damages = after_doc.get("results", {}).get("damages", [])
    
    # Karsilastirma yap
    comparison = compare_analyses(before_np, after_np, before_damages, after_damages)
    comparison = convert_to_native_types(comparison)
    
    # Karsilastirma kaydini MongoDB'ye kaydet
    comparison_id = str(uuid.uuid4())
    comparison_doc = {
        "_id": comparison_id,
        "before_id": request.before_id,
        "after_id": request.after_id,
        "created_at": datetime.utcnow().isoformat(),
        "result": comparison
    }
    db.comparisons.insert_one(comparison_doc)
    
    return {
        "id": comparison_id,
        "before_id": request.before_id,
        "after_id": request.after_id,
        **comparison
    }

@app.post("/api/compare/upload")
async def compare_upload(
    before_file: UploadFile = File(...),
    after_file: UploadFile = File(...)
):
    """Iki gorsel yukleyerek dogrudan karsilastirma yap"""
    
    # Before gorseli
    before_contents = await before_file.read()
    before_np = cv2.imdecode(np.frombuffer(before_contents, np.uint8), cv2.IMREAD_COLOR)
    if before_np is None:
        raise HTTPException(status_code=400, detail="Onceki gorsel okunamadi")
    
    # After gorseli
    after_contents = await after_file.read()
    after_np = cv2.imdecode(np.frombuffer(after_contents, np.uint8), cv2.IMREAD_COLOR)
    if after_np is None:
        raise HTTPException(status_code=400, detail="Sonraki gorsel okunamadi")
    
    # Her iki gorseli analiz et
    before_results = analyze_image(before_np, enable_sam=False)
    after_results = analyze_image(after_np, enable_sam=False)
    
    before_results = convert_to_native_types(before_results)
    after_results = convert_to_native_types(after_results)
    
    # Karsilastirma
    comparison = compare_analyses(
        before_np, after_np,
        before_results.get("damages", []),
        after_results.get("damages", [])
    )
    comparison = convert_to_native_types(comparison)
    
    return {
        "before_analysis": {
            "damage_count": before_results["summary"]["total_damages"],
            "risk_level": before_results["summary"]["risk_level"],
            "damages": before_results["damages"]
        },
        "after_analysis": {
            "damage_count": after_results["summary"]["total_damages"],
            "risk_level": after_results["summary"]["risk_level"],
            "damages": after_results["damages"]
        },
        **comparison
    }

@app.get("/api/comparisons")
async def list_comparisons(limit: int = 20):
    """Gecmis karsilastirmalari listele"""
    comparisons = list(
        db.comparisons.find().sort("created_at", -1).limit(limit)
    )
    
    return [
        {
            "id": str(c["_id"]),
            "before_id": c["before_id"],
            "after_id": c["after_id"],
            "created_at": c["created_at"],
            "has_new_damage": c["result"]["has_new_damage"],
            "verdict": c["result"]["verdict"],
            "new_damage_count": c["result"]["new_damage_count"]
        }
        for c in comparisons
    ]

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
