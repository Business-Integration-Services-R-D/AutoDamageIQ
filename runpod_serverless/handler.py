"""
AutoDamageIQ - RunPod Serverless Handler
=========================================
Bu handler RunPod serverless worker olarak çalışır.
YOLO hasar + parça modelleri yükler, görsel analiz yapar.
"""

import runpod
import os
import base64
import json
import uuid
import numpy as np
import cv2
from pathlib import Path

# --- Model yükleme (cold start'ta bir kere) ---
print("Loading YOLO models...")

from ultralytics import YOLO

MODELS_DIR = Path("/models")
DAMAGE_MODEL_PATH = MODELS_DIR / "autodamage_best.pt"
PARTS_MODEL_PATH = MODELS_DIR / "carparts_seg_best.pt"

# Modelleri yükle
damage_model = None
parts_model = None

if DAMAGE_MODEL_PATH.exists():
    damage_model = YOLO(str(DAMAGE_MODEL_PATH))
    print(f"Damage model loaded: {DAMAGE_MODEL_PATH}")
else:
    print(f"WARNING: Damage model not found at {DAMAGE_MODEL_PATH}")

if PARTS_MODEL_PATH.exists():
    parts_model = YOLO(str(PARTS_MODEL_PATH))
    print(f"Parts model loaded: {PARTS_MODEL_PATH}")
else:
    print(f"WARNING: Parts model not found at {PARTS_MODEL_PATH}")

# Sabitler
DAMAGE_TR = {
    "crack": "Çatlak", "dent": "Göçük", "glass_shatter": "Cam Kırığı",
    "lamp_broken": "Lamba Kırığı", "scratch": "Çizik", "tire_flat": "Patlak Lastik"
}

PARTS_TR = {
    "back_bumper": "Arka Tampon", "back_door": "Arka Kapı", "back_glass": "Arka Cam",
    "back_left_door": "Arka Sol Kapı", "back_left_light": "Arka Sol Far",
    "back_light": "Arka Far", "back_right_door": "Arka Sağ Kapı",
    "back_right_light": "Arka Sağ Far", "front_bumper": "Ön Tampon",
    "front_door": "Ön Kapı", "front_glass": "Ön Cam", "front_left_door": "Ön Sol Kapı",
    "front_left_light": "Ön Sol Far", "front_light": "Ön Far",
    "front_right_door": "Ön Sağ Kapı", "front_right_light": "Ön Sağ Far",
    "hood": "Kaput", "left_mirror": "Sol Ayna", "right_mirror": "Sağ Ayna",
    "tailgate": "Bagaj Kapağı", "trunk": "Bagaj", "wheel": "Tekerlek"
}

SEVERITY_MAP = {"crack": 3, "dent": 3, "glass_shatter": 5, "lamp_broken": 4, "scratch": 2, "tire_flat": 4}


def box_iou(box_a, box_b):
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, (float(box_a[2]) - float(box_a[0])) * (float(box_a[3]) - float(box_a[1])))
    area_b = max(0, (float(box_b[2]) - float(box_b[0])) * (float(box_b[3]) - float(box_b[1])))
    return float(inter / (area_a + area_b - inter + 1e-6))


def calculate_severity(damage_type, box, w, h, confidence):
    base = SEVERITY_MAP.get(damage_type, 3)
    area_ratio = max(0, (box[2]-box[0])*(box[3]-box[1])) / max(1, w*h)
    if area_ratio > 0.15: af = 1.5
    elif area_ratio > 0.08: af = 1.2
    elif area_ratio > 0.03: af = 1.0
    else: af = 0.8
    cf = 1.0 if confidence > 50 else 0.9
    score = round(min(5.0, max(1.0, base * af * cf)), 1)
    cls = max(1, min(5, int(round(score))))
    label = "Yuksek" if score >= 4.0 else ("Orta" if score >= 2.5 else "Dusuk")
    return {"score": score, "class": cls, "label": label, "area_ratio": round(area_ratio*100, 2)}


def handler(event):
    """RunPod serverless handler - araç hasar analizi"""
    input_data = event.get("input", {})
    
    # Görsel al (base64)
    image_b64 = input_data.get("image_base64")
    if not image_b64:
        return {"error": "image_base64 field required"}
    
    # Decode image
    try:
        img_bytes = base64.b64decode(image_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image_np is None:
            return {"error": "Could not decode image"}
    except Exception as e:
        return {"error": f"Image decode failed: {str(e)}"}
    
    h, w = image_np.shape[:2]
    
    # Model kontrolü
    if damage_model is None:
        return {"error": "Damage model not loaded"}
    
    # Hasar tespiti
    damage_results = damage_model.predict(source=image_np, imgsz=640, conf=0.15, verbose=False)[0]
    
    dmg_boxes = damage_results.boxes.xyxy.cpu().numpy() if damage_results.boxes is not None else np.zeros((0, 4))
    dmg_cls = damage_results.boxes.cls.cpu().numpy().astype(int) if damage_results.boxes is not None else np.zeros((0,), int)
    dmg_conf = damage_results.boxes.conf.cpu().numpy() if damage_results.boxes is not None else np.zeros((0,))
    dmg_names = damage_model.names
    
    # Parça segmentasyonu
    part_boxes = np.zeros((0, 4))
    part_cls_arr = np.zeros((0,), int)
    part_names = {}
    
    if parts_model is not None:
        parts_results = parts_model.predict(source=image_np, imgsz=640, conf=0.15, verbose=False)[0]
        part_boxes = parts_results.boxes.xyxy.cpu().numpy() if parts_results.boxes is not None else np.zeros((0, 4))
        part_cls_arr = parts_results.boxes.cls.cpu().numpy().astype(int) if parts_results.boxes is not None else np.zeros((0,), int)
        part_names = parts_model.names
    
    # Hasar-parça eşleme
    damages = []
    for i, dmg_box in enumerate(dmg_boxes):
        best_iou = 0.0
        best_part = None
        best_part_box = None
        
        for j, part_box in enumerate(part_boxes):
            iou = box_iou(dmg_box, part_box)
            if iou > best_iou:
                best_iou = iou
                best_part = part_names[int(part_cls_arr[j])]
                best_part_box = [float(x) for x in part_box.tolist()]
        
        damage_type = dmg_names[int(dmg_cls[i])]
        conf = float(dmg_conf[i])
        severity = calculate_severity(damage_type, [float(x) for x in dmg_box.tolist()], w, h, conf*100)
        
        damages.append({
            "id": str(uuid.uuid4())[:8],
            "type": damage_type,
            "type_tr": DAMAGE_TR.get(damage_type, damage_type),
            "confidence": round(conf * 100, 1),
            "severity": severity["class"],
            "severity_details": severity,
            "box": [float(x) for x in dmg_box.tolist()],
            "part": best_part if best_iou > 0.1 else None,
            "part_tr": PARTS_TR.get(best_part, best_part) if best_iou > 0.1 else None,
            "part_box": best_part_box if best_iou > 0.1 else None,
            "iou_with_part": round(best_iou, 3),
            "part_source": "yolo" if best_iou > 0.1 else "none"
        })
    
    # Parçalar
    parts = []
    for j, part_box in enumerate(part_boxes):
        pn = part_names[int(part_cls_arr[j])]
        parts.append({"name": pn, "name_tr": PARTS_TR.get(pn, pn), "box": [float(x) for x in part_box.tolist()]})
    
    # Özet
    total = len(damages)
    affected = len(set(d["part"] for d in damages if d["part"]))
    avg_sev = round(sum(d["severity"] for d in damages) / max(1, total), 1)
    avg_conf = round(sum(d["confidence"] for d in damages) / max(1, total), 1)
    risk = "Yuksek" if (avg_sev >= 4 or total >= 4) else ("Orta" if (avg_sev >= 2.5 or total >= 2) else "Dusuk")
    
    return {
        "damages": damages,
        "parts": parts,
        "summary": {
            "total_damages": total,
            "affected_parts": affected,
            "average_severity": avg_sev,
            "average_confidence": avg_conf,
            "risk_level": risk,
            "needs_review": avg_conf < 35 or risk == "Yuksek",
            "review_reasons": []
        },
        "image_size": {"width": w, "height": h},
        "inference_source": "runpod_serverless"
    }


# RunPod serverless başlat
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
