"""
AutoDamageIQ - SAM (Segment Anything Model) + YOLO Hibrit Pipeline
===================================================================
YOLO bounding box'larindan SAM ile piksel duzeyinde hassas maske uretir.
Panel bazli kalibrasyon ile hasar alani hesaplar.
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

# Panel fiziksel boyutlari (cm): genislik x yukseklik (yaklasik)
PANEL_DIMENSIONS_CM = {
    "hood":             (120, 100),
    "front_bumper":     (180,  50),
    "back_bumper":      (180,  50),
    "front_door":       (100,  80),
    "back_door":        ( 90,  80),
    "front_left_door":  (100,  80),
    "front_right_door": (100,  80),
    "back_left_door":   ( 90,  80),
    "back_right_door":  ( 90,  80),
    "front_glass":      (120,  70),
    "back_glass":       (110,  60),
    "front_light":      ( 35,  25),
    "front_left_light": ( 35,  25),
    "front_right_light":( 35,  25),
    "back_light":       ( 30,  20),
    "back_left_light":  ( 30,  20),
    "back_right_light": ( 30,  20),
    "left_mirror":      ( 20,  15),
    "right_mirror":     ( 20,  15),
    "tailgate":         (120,  90),
    "trunk":            (120,  90),
    "wheel":            ( 65,  65),
}

# Varsayilan arac boyutlari (cm)
DEFAULT_VEHICLE_LENGTH_CM = 450
DEFAULT_VEHICLE_HEIGHT_CM = 150


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
        _sam_available = False
        return None


def generate_mask_for_box(
    predictor_with_image,
    box: List[float]
) -> Optional[Dict[str, Any]]:
    """
    Verilen bounding box icin SAM ile segmentasyon maskesi olustur.
    predictor_with_image: set_image() zaten cagrilmis predictor
    """
    try:
        box_np = np.array(box, dtype=np.float32)
        masks, scores, _ = predictor_with_image.predict(box=box_np, multimask_output=True)

        best_idx = int(np.argmax(scores))
        mask = masks[best_idx]
        score = float(scores[best_idx])
        area_pixels = int(np.sum(mask))

        # Bounding box alani
        box_w = max(1, box[2] - box[0])
        box_h = max(1, box[3] - box[1])
        box_area = box_w * box_h

        # Maske / bbox doluluk orani
        fill_ratio = area_pixels / max(1, box_area)

        # Kontur ve polygon
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        polygon = []
        if contours:
            largest = max(contours, key=cv2.contourArea)
            epsilon = 0.005 * cv2.arcLength(largest, True)
            approx = cv2.approxPolyDP(largest, epsilon, True)
            pts = approx.squeeze()
            if pts.ndim == 2 and len(pts) >= 3:
                polygon = pts.tolist()

        # Maskenin minimum bounding rect'i (piksel boyut)
        mask_coords = np.where(mask)
        if len(mask_coords[0]) > 0:
            mask_min_y, mask_max_y = int(mask_coords[0].min()), int(mask_coords[0].max())
            mask_min_x, mask_max_x = int(mask_coords[1].min()), int(mask_coords[1].max())
            mask_width_px = mask_max_x - mask_min_x
            mask_height_px = mask_max_y - mask_min_y
        else:
            mask_width_px = 0
            mask_height_px = 0

        return {
            "mask_score": score,
            "area_pixels": area_pixels,
            "box_area_pixels": int(box_area),
            "fill_ratio": round(float(fill_ratio), 3),
            "mask_width_px": mask_width_px,
            "mask_height_px": mask_height_px,
            "polygon": polygon,
            "polygon_point_count": len(polygon)
        }
    except Exception as e:
        print(f"SAM maske hatasi: {e}")
        return None


# -------------------------------------------------------------------------
# KALIBRASYON: Panel bazli, bbox bazli ve gorsel bazli 3 katmanli olcum
# -------------------------------------------------------------------------

# Hasar tipi duzeltme faktoru: bbox alaninin ne kadari gercek hasar alani?
# Scratch/crack = ince cizgi → bbox'un kucuk bir kismi
# Dent = orta doluluk
# Glass/lamp = buyuk doluluk
DAMAGE_TYPE_AREA_FACTOR = {
    "scratch": 0.10,     # Cizik: ince cizgi, bbox'un ~%10'u
    "crack":   0.12,     # Catlak: ince cizgi, biraz daha genis
    "dent":    0.45,     # Gocuk: bolgenin yaklasik yarisi
    "glass_shatter": 0.80,  # Cam kirigi: buyuk alan
    "lamp_broken":   0.65,  # Lamba kirigi: buyuk bolum
    "tire_flat":     0.30,  # Patlak lastik: gorunur alan
}


def _estimate_via_panel(
    mask_data: Dict[str, Any],
    panel_name: Optional[str],
    panel_box: Optional[List[float]],
    damage_box: List[float],
    damage_type: str,
    image_w: int, image_h: int
) -> Optional[Dict[str, Any]]:
    """
    Panel fiziksel boyutunu referans alarak hasar boyutunu hesapla.
    Bounding box boyutlari + hasar tipi duzeltme faktoru kullanir.
    """
    if not panel_name or not panel_box or panel_name not in PANEL_DIMENSIONS_CM:
        return None

    panel_w_cm, panel_h_cm = PANEL_DIMENSIONS_CM[panel_name]

    # Panelin gorseldeki piksel boyutu
    panel_px_w = max(1, panel_box[2] - panel_box[0])
    panel_px_h = max(1, panel_box[3] - panel_box[1])

    # Piksel/cm oranlari
    px_per_cm_x = panel_px_w / panel_w_cm
    px_per_cm_y = panel_px_h / panel_h_cm
    px_per_cm = (px_per_cm_x + px_per_cm_y) / 2

    # Hasar BOUNDING BOX boyutlari (piksel) → cm
    bbox_w_px = max(1, damage_box[2] - damage_box[0])
    bbox_h_px = max(1, damage_box[3] - damage_box[1])
    bbox_w_cm = bbox_w_px / max(0.1, px_per_cm)
    bbox_h_cm = bbox_h_px / max(0.1, px_per_cm)
    bbox_area_cm2 = bbox_w_cm * bbox_h_cm

    # Hasar tipi duzeltme faktoru
    type_factor = DAMAGE_TYPE_AREA_FACTOR.get(damage_type, 0.40)
    estimated_damage_cm2 = bbox_area_cm2 * type_factor

    return {
        "method": "panel_reference",
        "confidence": "Yuksek",
        "panel_reference": panel_name,
        "panel_size_cm": f"{panel_w_cm}x{panel_h_cm}",
        "bbox_width_cm": round(float(bbox_w_cm), 1),
        "bbox_height_cm": round(float(bbox_h_cm), 1),
        "bbox_area_cm2": round(float(bbox_area_cm2), 1),
        "type_factor": type_factor,
        "damage_area_cm2": round(float(estimated_damage_cm2), 1),
        "px_per_cm": round(float(px_per_cm), 2),
    }


def _estimate_via_bbox_ratio(
    mask_data: Dict[str, Any],
    damage_type: str,
    damage_box: List[float],
    image_w: int, image_h: int
) -> Dict[str, Any]:
    """
    BBox boyutu + hasar tipi faktoru ile goresel olcum.
    Panel bilgisi yoksa kullanilir.
    """
    # BBox'un gorsel icindeki orani
    bbox_w = max(1, damage_box[2] - damage_box[0])
    bbox_h = max(1, damage_box[3] - damage_box[1])
    bbox_ratio = (bbox_w * bbox_h) / max(1, image_w * image_h)

    type_factor = DAMAGE_TYPE_AREA_FACTOR.get(damage_type, 0.40)
    effective_ratio = bbox_ratio * type_factor

    # Goreceli buyukluk (gorsel alanina gore)
    if effective_ratio < 0.005:
        relative_size = "Cok Kucuk"
    elif effective_ratio < 0.02:
        relative_size = "Kucuk"
    elif effective_ratio < 0.06:
        relative_size = "Orta"
    elif effective_ratio < 0.15:
        relative_size = "Buyuk"
    else:
        relative_size = "Cok Buyuk"

    return {
        "method": "bbox_ratio",
        "confidence": "Orta",
        "fill_ratio": mask_data["fill_ratio"],
        "bbox_image_ratio": round(float(bbox_ratio * 100), 2),
        "type_factor": type_factor,
        "effective_ratio": round(float(effective_ratio * 100), 2),
        "relative_size": relative_size,
    }


def _estimate_via_image_ratio(
    mask_data: Dict[str, Any],
    damage_type: str,
    damage_box: List[float],
    image_w: int, image_h: int
) -> Dict[str, Any]:
    """
    Gorsel alanina gore kaba tahmini (varsayilan gorunur alan 200x150 cm).
    """
    bbox_w = max(1, damage_box[2] - damage_box[0])
    bbox_h = max(1, damage_box[3] - damage_box[1])
    bbox_area = bbox_w * bbox_h
    total_px = max(1, image_w * image_h)
    bbox_pct = (bbox_area / total_px) * 100

    # Kaba varsayim: gorsel ~200x150 cm alan kapsiyor
    visible_area_cm2 = 200 * 150
    type_factor = DAMAGE_TYPE_AREA_FACTOR.get(damage_type, 0.40)
    approx_cm2 = (bbox_pct / 100) * visible_area_cm2 * type_factor

    if approx_cm2 < 5:
        size_band = "Cok Kucuk"
    elif approx_cm2 < 30:
        size_band = "Kucuk"
    elif approx_cm2 < 150:
        size_band = "Orta"
    elif approx_cm2 < 600:
        size_band = "Buyuk"
    else:
        size_band = "Cok Buyuk"

    return {
        "method": "image_ratio",
        "confidence": "Dusuk",
        "bbox_percentage": round(float(bbox_pct), 2),
        "approx_area_cm2": round(float(approx_cm2), 1),
        "size_band": size_band,
    }


def calculate_damage_measurements(
    mask_data: Dict[str, Any],
    image_width: int,
    image_height: int,
    panel_name: Optional[str] = None,
    panel_box: Optional[List[float]] = None,
    damage_type: str = "",
    damage_box: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Cok katmanli kalibrasyon ile hasar olcu bilgisi hesapla.
    """
    box = damage_box or [0, 0, image_width, image_height]

    # 1) Panel bazli
    panel_est = _estimate_via_panel(
        mask_data, panel_name, panel_box, box, damage_type, image_width, image_height
    )

    # 2) BBox ratio bazli
    bbox_est = _estimate_via_bbox_ratio(mask_data, damage_type, box, image_width, image_height)

    # 3) Gorsel oran bazli
    img_est = _estimate_via_image_ratio(mask_data, damage_type, box, image_width, image_height)

    # Birincil tahmini sec
    if panel_est:
        primary = panel_est
        size_band = _cm2_to_band(panel_est["damage_area_cm2"])
    else:
        primary = bbox_est
        size_band = bbox_est["relative_size"]

    return {
        "primary": primary,
        "bbox_analysis": bbox_est,
        "image_analysis": img_est,
        "area_pixels": mask_data["area_pixels"],
        "fill_ratio": mask_data["fill_ratio"],
        "size_band": size_band,
        "has_panel_calibration": panel_est is not None,
    }


def _cm2_to_band(area_cm2: float) -> str:
    if area_cm2 < 5:
        return "Cok Kucuk"
    elif area_cm2 < 30:
        return "Kucuk"
    elif area_cm2 < 150:
        return "Orta"
    elif area_cm2 < 600:
        return "Buyuk"
    return "Cok Buyuk"


# -------------------------------------------------------------------------
# ANA ENTEGRASYON
# -------------------------------------------------------------------------

MAX_SAM_DAMAGES = 5  # CPU'da makul sure icin max hasar sayisi


def enhance_damages_with_sam(
    image_np: np.ndarray,
    damages: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    YOLO hasar listesini SAM maskeleri ile zenginlestir.
    """
    if not is_sam_available():
        for dmg in damages:
            dmg["sam_data"] = {"available": False, "reason": "SAM modeli yuklu degil"}
        return damages

    predictor = _get_predictor()
    if predictor is None:
        for dmg in damages:
            dmg["sam_data"] = {"available": False, "reason": "SAM predictor yuklenemedi"}
        return damages

    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    h, w = image_np.shape[:2]

    # set_image BIR KEZ cagir (en pahali islem)
    predictor.set_image(image_rgb)

    # En yuksek guvenli hasarlardan basla, max N tane isle
    sorted_indices = sorted(range(len(damages)), key=lambda i: damages[i].get("confidence", 0), reverse=True)

    for rank, idx in enumerate(sorted_indices):
        dmg = damages[idx]
        box = dmg.get("box", [])

        if len(box) != 4:
            dmg["sam_data"] = {"available": False, "reason": "Gecersiz bounding box"}
            continue

        if rank >= MAX_SAM_DAMAGES:
            dmg["sam_data"] = {"available": False, "reason": f"SAM limiti ({MAX_SAM_DAMAGES}) asildi"}
            continue

        mask_data = generate_mask_for_box(predictor, box)
        if not mask_data:
            dmg["sam_data"] = {"available": False, "reason": "Maske uretilemedi"}
            continue

        measurements = calculate_damage_measurements(
            mask_data, w, h,
            panel_name=dmg.get("part"),
            panel_box=dmg.get("part_box"),
            damage_type=dmg.get("type", ""),
            damage_box=box,
        )

        dmg["sam_data"] = {
            "available": True,
            "mask_score": mask_data["mask_score"],
            "measurements": measurements,
            "polygon_point_count": mask_data["polygon_point_count"],
        }

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
        "device": str(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
    }


def _check_library() -> bool:
    try:
        from segment_anything import sam_model_registry
        return True
    except ImportError:
        return False
