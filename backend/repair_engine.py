"""
AutoDamageIQ - Onarim Tipi Oneri Motoru
========================================
Hasar tipi, siddet ve panel bilgisinden kural tabanli
onarim tipi onerisi ureten hibrit karar motoru.
"""

from typing import Dict, Any, Optional


# Onarim tipleri
REPAIR_TYPES = {
    "local_paint": {
        "tr": "Lokal Boya",
        "description_tr": "Yuzeysel boya hasari - lokal boya islemi yeterli",
        "cost_level": 1
    },
    "panel_repair": {
        "tr": "Kaporta Duzeltme",
        "description_tr": "Gocuk veya deformasyon - boyasiz gocuk duzeltme veya kaporta islemi",
        "cost_level": 2
    },
    "part_replacement": {
        "tr": "Parca Degisimi",
        "description_tr": "Parca butunlugu bozulmus - komple parca degisimi gerekli",
        "cost_level": 3
    },
    "glass_replacement": {
        "tr": "Cam Degisimi",
        "description_tr": "Cam kirigi - cam degisimi gerekli",
        "cost_level": 3
    },
    "lamp_replacement": {
        "tr": "Far/Lamba Degisimi",
        "description_tr": "Far veya lamba hasari - unite degisimi gerekli",
        "cost_level": 2
    },
    "tire_replacement": {
        "tr": "Lastik Degisimi",
        "description_tr": "Patlak veya hasarli lastik - lastik degisimi gerekli",
        "cost_level": 2
    },
    "detailed_inspection": {
        "tr": "Detayli Inceleme Gerekli",
        "description_tr": "Otomatik oneri uretilemiyor - uzman incelemesi gerekli",
        "cost_level": 0
    }
}

# Hasar tipi x siddet -> onarim tipi matrisi
REPAIR_MATRIX = {
    "scratch": {
        1: "local_paint",
        2: "local_paint",
        3: "panel_repair",
        4: "panel_repair",
        5: "part_replacement"
    },
    "dent": {
        1: "panel_repair",
        2: "panel_repair",
        3: "panel_repair",
        4: "part_replacement",
        5: "part_replacement"
    },
    "crack": {
        1: "local_paint",
        2: "panel_repair",
        3: "panel_repair",
        4: "part_replacement",
        5: "part_replacement"
    },
    "glass_shatter": {
        1: "glass_replacement",
        2: "glass_replacement",
        3: "glass_replacement",
        4: "glass_replacement",
        5: "glass_replacement"
    },
    "lamp_broken": {
        1: "lamp_replacement",
        2: "lamp_replacement",
        3: "lamp_replacement",
        4: "lamp_replacement",
        5: "lamp_replacement"
    },
    "tire_flat": {
        1: "tire_replacement",
        2: "tire_replacement",
        3: "tire_replacement",
        4: "tire_replacement",
        5: "tire_replacement"
    }
}

# Panel bazli onarim maliyet carpani
PANEL_COST_MULTIPLIER = {
    "hood": 1.3,
    "front_bumper": 1.0,
    "back_bumper": 1.0,
    "front_door": 1.2,
    "back_door": 1.2,
    "front_left_door": 1.2,
    "front_right_door": 1.2,
    "back_left_door": 1.2,
    "back_right_door": 1.2,
    "front_glass": 1.5,
    "back_glass": 1.3,
    "front_light": 1.1,
    "front_left_light": 1.1,
    "front_right_light": 1.1,
    "back_light": 1.0,
    "back_left_light": 1.0,
    "back_right_light": 1.0,
    "left_mirror": 0.8,
    "right_mirror": 0.8,
    "tailgate": 1.3,
    "trunk": 1.3,
    "wheel": 1.0,
}


def get_repair_recommendation(
    damage_type: str,
    severity: int,
    confidence: float,
    panel: Optional[str] = None
) -> Dict[str, Any]:
    """
    Hasar bilgilerinden onarim tipi onerisi uretir.
    
    Args:
        damage_type: Hasar tipi (crack, dent, scratch, vb.)
        severity: Siddet skoru (1-5)
        confidence: Guven skoru (0-100)
        panel: Panel adi (opsiyonel)
    
    Returns:
        Onarim tipi onerisi
    """
    # Siddet araligini kısıtla
    severity = max(1, min(5, severity))
    
    # Matristen onarim tipi al
    type_matrix = REPAIR_MATRIX.get(damage_type)
    if type_matrix is None:
        repair_key = "detailed_inspection"
    else:
        repair_key = type_matrix.get(severity, "detailed_inspection")
    
    # Dusuk guven durumunda detayli inceleme oner
    if confidence < 30:
        repair_key = "detailed_inspection"
    
    repair_info = REPAIR_TYPES[repair_key]
    
    # Panel bazli maliyet carpani
    cost_multiplier = PANEL_COST_MULTIPLIER.get(panel, 1.0) if panel else 1.0
    
    # Oneri guveni hesapla
    recommendation_confidence = min(95, confidence * 0.8 + (30 if confidence > 50 else 0))
    
    return {
        "repair_type": repair_key,
        "repair_type_tr": repair_info["tr"],
        "description_tr": repair_info["description_tr"],
        "cost_level": int(repair_info["cost_level"]),
        "cost_level_tr": ["", "Dusuk", "Orta", "Yuksek"][min(3, repair_info["cost_level"])] if repair_info["cost_level"] > 0 else "Belirsiz",
        "panel_multiplier": float(round(cost_multiplier, 2)),
        "recommendation_confidence": float(round(recommendation_confidence, 1))
    }
