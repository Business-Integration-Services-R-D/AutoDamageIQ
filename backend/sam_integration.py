"""
SAM (Segment Anything Model) + YOLO Hibrit Pipeline
====================================================
Bu modül YOLO detection sonuçlarını SAM ile pixel-level segmentasyona dönüştürür.
"""

import os
import numpy as np
import torch
import cv2
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# SAM availability flag
SAM_AVAILABLE = False
sam_model = None
sam_predictor = None

def check_sam_availability() -> bool:
    """SAM kullanılabilirliğini kontrol et"""
    global SAM_AVAILABLE
    try:
        from segment_anything import sam_model_registry, SamPredictor
        SAM_AVAILABLE = True
        return True
    except ImportError:
        SAM_AVAILABLE = False
        return False

def get_sam_checkpoint_path() -> Optional[Path]:
    """SAM checkpoint dosyasını bul veya indir"""
    # Olası konumlar
    possible_paths = [
        Path("/app/models/sam_vit_b_01ec64.pth"),  # ViT-B (daha küçük)
        Path("/app/models/sam_vit_h_4b8939.pth"),  # ViT-H (daha büyük)
        Path.home() / ".cache" / "sam" / "sam_vit_b_01ec64.pth",
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    return None

def initialize_sam(model_type: str = "vit_b") -> bool:
    """SAM modelini başlat"""
    global sam_model, sam_predictor, SAM_AVAILABLE
    
    if not check_sam_availability():
        print("⚠️ SAM kütüphanesi yüklü değil")
        return False
    
    checkpoint_path = get_sam_checkpoint_path()
    if checkpoint_path is None:
        print("⚠️ SAM checkpoint dosyası bulunamadı")
        print("   İndirmek için: https://github.com/facebookresearch/segment-anything#model-checkpoints")
        return False
    
    try:
        from segment_anything import sam_model_registry, SamPredictor
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔄 SAM yükleniyor ({model_type}) - Device: {device}")
        
        sam_model = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
        sam_model.to(device)
        sam_predictor = SamPredictor(sam_model)
        
        print(f"✅ SAM başarıyla yüklendi")
        return True
    except Exception as e:
        print(f"❌ SAM yükleme hatası: {e}")
        return False

def get_sam_mask_for_box(
    image_rgb: np.ndarray, 
    box: List[float],
    multimask: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Verilen bounding box için SAM ile segmentasyon maskesi oluştur
    
    Args:
        image_rgb: RGB formatında görsel (numpy array)
        box: [x1, y1, x2, y2] formatında bounding box
        multimask: Birden fazla maske üret
    
    Returns:
        Dict: mask, score, area bilgileri
    """
    global sam_predictor
    
    if sam_predictor is None:
        if not initialize_sam():
            return None
    
    try:
        # Görseli SAM'e set et
        sam_predictor.set_image(image_rgb)
        
        # Box'ı numpy array'e çevir
        box_np = np.array(box).astype(np.int32)
        
        # Maske tahmin et
        masks, scores, logits = sam_predictor.predict(
            box=box_np,
            multimask_output=multimask
        )
        
        # En iyi maskeyi seç
        if multimask:
            best_idx = np.argmax(scores)
            mask = masks[best_idx]
            score = scores[best_idx]
        else:
            mask = masks[0]
            score = scores[0]
        
        # Maske alanını hesapla
        area_pixels = int(np.sum(mask))
        
        # Maske konturu bul
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # En büyük konturu al
        polygon = []
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            # Konturu basitleştir
            epsilon = 0.005 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            polygon = approx.squeeze().tolist()
            if isinstance(polygon[0], int):
                polygon = [polygon]
        
        return {
            "mask": mask,
            "score": float(score),
            "area_pixels": area_pixels,
            "polygon": polygon,
            "bbox": box
        }
    
    except Exception as e:
        print(f"SAM maske hatası: {e}")
        return None

def calculate_damage_area(
    mask_data: Dict[str, Any],
    image_width: int,
    image_height: int,
    reference_size_cm: float = 450.0  # Ortalama araç uzunluğu
) -> Dict[str, float]:
    """
    Hasar alanını hesapla (tahmini cm²)
    
    Args:
        mask_data: SAM maske verisi
        image_width: Görsel genişliği
        image_height: Görsel yüksekliği
        reference_size_cm: Referans boyut (araç uzunluğu varsayılan 450cm)
    
    Returns:
        Dict: Alan hesaplamaları
    """
    area_pixels = mask_data["area_pixels"]
    total_pixels = image_width * image_height
    
    # Pikselden cm²'ye dönüşüm (yaklaşık)
    # Varsayım: Görsel arabanın tamamını kapsıyor
    px_per_cm = image_width / reference_size_cm
    area_cm2 = area_pixels / (px_per_cm ** 2)
    
    # Yüzde hesapla
    percentage = (area_pixels / total_pixels) * 100
    
    return {
        "area_pixels": area_pixels,
        "area_cm2": round(area_cm2, 2),
        "percentage": round(percentage, 3),
        "reference_used_cm": reference_size_cm
    }

def enhance_damage_with_sam(
    image_rgb: np.ndarray,
    damages: List[Dict[str, Any]],
    enable_sam: bool = True
) -> List[Dict[str, Any]]:
    """
    YOLO detection sonuçlarını SAM ile zenginleştir
    
    Args:
        image_rgb: RGB görsel
        damages: YOLO'dan gelen hasar listesi
        enable_sam: SAM'i etkinleştir/devre dışı bırak
    
    Returns:
        Zenginleştirilmiş hasar listesi
    """
    if not enable_sam or not SAM_AVAILABLE:
        return damages
    
    h, w = image_rgb.shape[:2]
    enhanced_damages = []
    
    for damage in damages:
        enhanced = damage.copy()
        
        # SAM ile maske oluştur
        box = damage.get("box", [])
        if len(box) == 4:
            mask_data = get_sam_mask_for_box(image_rgb, box)
            
            if mask_data:
                # Alan hesapla
                area_info = calculate_damage_area(mask_data, w, h)
                
                enhanced["sam_data"] = {
                    "score": mask_data["score"],
                    "area": area_info,
                    "polygon": mask_data["polygon"],
                    "has_precise_mask": True
                }
            else:
                enhanced["sam_data"] = {
                    "has_precise_mask": False,
                    "reason": "SAM mask generation failed"
                }
        
        enhanced_damages.append(enhanced)
    
    return enhanced_damages

def draw_sam_masks_on_image(
    image_rgb: np.ndarray,
    damages: List[Dict[str, Any]],
    alpha: float = 0.4
) -> np.ndarray:
    """
    SAM maskelerini görsel üzerine çiz
    
    Args:
        image_rgb: Orijinal görsel
        damages: SAM verisi içeren hasar listesi
        alpha: Maske transparanlığı
    
    Returns:
        Maskelerin çizildiği görsel
    """
    overlay = image_rgb.copy()
    output = image_rgb.copy()
    
    # Renk paleti
    colors = [
        (255, 0, 0),    # Kırmızı - Çatlak
        (0, 255, 0),    # Yeşil - Göçük
        (0, 0, 255),    # Mavi - Cam kırığı
        (255, 255, 0),  # Sarı - Lamba kırığı
        (255, 0, 255),  # Magenta - Çizik
        (0, 255, 255),  # Cyan - Patlak lastik
    ]
    
    damage_type_to_color = {
        "crack": 0, "dent": 1, "glass_shatter": 2,
        "lamp_broken": 3, "scratch": 4, "tire_flat": 5
    }
    
    for damage in damages:
        sam_data = damage.get("sam_data", {})
        polygon = sam_data.get("polygon", [])
        damage_type = damage.get("type", "unknown")
        
        color_idx = damage_type_to_color.get(damage_type, 0)
        color = colors[color_idx % len(colors)]
        
        if polygon and len(polygon) > 2:
            pts = np.array(polygon, dtype=np.int32)
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(output, [pts], True, color, 2)
    
    # Overlay blend
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    
    return output

# Modül yüklendiğinde SAM durumunu kontrol et
check_sam_availability()
