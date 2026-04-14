"""
AutoDamageIQ - SAM (Segment Anything Model) + YOLO Hibrit Pipeline
===================================================================
YOLO bounding box'larindan SAM ile piksel duzeyinde hassas maske uretir.
Hasar alani (piksel ve tahmini cm2) hesaplar.
"""

import numpy as np
import torch
import cv2
from pathlib import Path
from typing import Dict, List, Any, Optional

# SAM state
_sam_model = None
_sam_predictor = None
_sam_available = None  # None = not checked yet


SAM_CHECKPOINT_PATH = Path("/app/models/sam_vit_b_01ec64.pth")
SAM_MODEL_TYPE = "vit_b"


def is_sam_available() -> bool:
    """SAM kullanilabilirligini kontrol et (lazy check)"""
    global _sam_available
    if _sam_available is not None:
        return _sam_available
    
    try:
        from segment_anything import sam_model_registry, SamPredictor
        if SAM_CHECKPOINT_PATH.exists():
            _sam_available = True
        else:
            _sam_available = False
            print(f"SAM checkpoint bulunamadi: {SAM_CHECKPOINT_PATH}")
    except ImportError:
        _sam_available = False
        print("segment-anything kutuphanesi yuklu degil")
    
    return _sam_available


def _get_predictor():
    """SAM predictor'u lazy-load et"""
    global _sam_model, _sam_predictor
    
    if _sam_predictor is not None:
        return _sam_predictor
    
    if not is_sam_available():
        return None
    
    try:
        from segment_anything import sam_model_registry, SamPredictor
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"SAM yukleniyor ({SAM_MODEL_TYPE}) - Device: {device}")
        
        _sam_model = sam_model_registry[SAM_MODEL_TYPE](checkpoint=str(SAM_CHECKPOINT_PATH))
        _sam_model.to(device)
        _sam_predictor = SamPredictor(_sam_model)
        
        print("SAM basariyla yuklendi")
        return _sam_predictor
    except Exception as e:
        print(f"SAM yukleme hatasi: {e}")
        _sam_available = False  # Don't retry
        return None


def generate_mask_for_box(
    image_rgb: np.ndarray,
    box: List[float]
) -> Optional[Dict[str, Any]]:
    """
    Verilen bounding box icin SAM ile segmentasyon maskesi olustur.
    
    Args:
        image_rgb: RGB formatinda gorsel (H, W, 3)
        box: [x1, y1, x2, y2] formatinda bounding box
    
    Returns:
        mask_data dict veya None
    """
    predictor = _get_predictor()
    if predictor is None:
        return None
    
    try:
        predictor.set_image(image_rgb)
        
        box_np = np.array(box, dtype=np.float32)
        
        masks, scores, _ = predictor.predict(
            box=box_np,
            multimask_output=True
        )
        
        # En iyi maskeyi sec
        best_idx = int(np.argmax(scores))
        mask = masks[best_idx]
        score = float(scores[best_idx])
        
        # Maske alanini hesapla
        area_pixels = int(np.sum(mask))
        
        # Kontur bul ve polygon olustur
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        polygon = []
        if contours:
            largest = max(contours, key=cv2.contourArea)
            epsilon = 0.005 * cv2.arcLength(largest, True)
            approx = cv2.approxPolyDP(largest, epsilon, True)
            pts = approx.squeeze()
            if pts.ndim == 2 and len(pts) >= 3:
                polygon = pts.tolist()
        
        return {
            "mask_score": score,
            "area_pixels": area_pixels,
            "polygon": polygon,
            "polygon_point_count": len(polygon)
        }
    
    except Exception as e:
        print(f"SAM maske hatasi: {e}")
        return None


def calculate_damage_measurements(
    mask_data: Dict[str, Any],
    image_width: int,
    image_height: int,
    reference_length_cm: float = 450.0
) -> Dict[str, Any]:
    """
    Maske verisinden hasar olcu bilgisi hesapla.
    
    Varsayim: Gorsel aracin buyuk bir bolumunu kapsiyor.
    reference_length_cm: Ortalama arac uzunlugu (varsayilan 450cm)
    """
    area_pixels = mask_data["area_pixels"]
    total_pixels = image_width * image_height
    
    # Piksel basina cm orani
    px_per_cm = image_width / reference_length_cm
    area_cm2 = area_pixels / (px_per_cm ** 2) if px_per_cm > 0 else 0
    
    # Gorsel yuzdesine gore alan
    area_percentage = (area_pixels / max(1, total_pixels)) * 100
    
    # Boyut bandi
    if area_cm2 < 5:
        size_band = "Cok Kucuk"
    elif area_cm2 < 25:
        size_band = "Kucuk"
    elif area_cm2 < 100:
        size_band = "Orta"
    elif area_cm2 < 500:
        size_band = "Buyuk"
    else:
        size_band = "Cok Buyuk"
    
    return {
        "area_pixels": area_pixels,
        "area_cm2": round(float(area_cm2), 2),
        "area_percentage": round(float(area_percentage), 3),
        "size_band": size_band,
        "reference_length_cm": reference_length_cm,
        "measurement_confidence": "tahmini"
    }


def enhance_damages_with_sam(
    image_np: np.ndarray,
    damages: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    YOLO hasar listesini SAM maskeleri ile zenginlestir.
    SAM kullanilabilir degilse hasarlari degistirmeden dondurur.
    
    Args:
        image_np: BGR gorsel (OpenCV format)
        damages: YOLO'dan gelen hasar listesi
    
    Returns:
        SAM verileri eklenmis hasar listesi
    """
    if not is_sam_available():
        # SAM yok — her hasara fallback bilgisi ekle
        for dmg in damages:
            dmg["sam_data"] = {"available": False, "reason": "SAM modeli yuklu degil"}
        return damages
    
    # BGR -> RGB
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    h, w = image_np.shape[:2]
    
    for dmg in damages:
        box = dmg.get("box", [])
        if len(box) != 4:
            dmg["sam_data"] = {"available": False, "reason": "Gecersiz bounding box"}
            continue
        
        mask_data = generate_mask_for_box(image_rgb, box)
        
        if mask_data:
            measurements = calculate_damage_measurements(mask_data, w, h)
            dmg["sam_data"] = {
                "available": True,
                "mask_score": mask_data["mask_score"],
                "measurements": measurements,
                "polygon_point_count": mask_data["polygon_point_count"]
            }
        else:
            dmg["sam_data"] = {"available": False, "reason": "Maske uretilemedi"}
    
    return damages


def get_sam_status() -> Dict[str, Any]:
    """SAM durumunu raporla"""
    checkpoint_exists = SAM_CHECKPOINT_PATH.exists()
    checkpoint_size_mb = round(SAM_CHECKPOINT_PATH.stat().st_size / (1024*1024), 1) if checkpoint_exists else 0
    
    return {
        "library_installed": _check_library(),
        "checkpoint_exists": checkpoint_exists,
        "checkpoint_path": str(SAM_CHECKPOINT_PATH),
        "checkpoint_size_mb": checkpoint_size_mb,
        "model_type": SAM_MODEL_TYPE,
        "model_loaded": _sam_predictor is not None,
        "device": str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    }


def _check_library() -> bool:
    try:
        from segment_anything import sam_model_registry
        return True
    except ImportError:
        return False
