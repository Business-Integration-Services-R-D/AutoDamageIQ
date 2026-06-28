"""
AutoDamageIQ - RunPod Serverless Handler
=========================================
Cold start'ta modelleri Emergent Object Storage'dan indirir,
YOLO hasar + parça tespiti yapar.
"""

import runpod
import os
import base64
import uuid
import requests
import numpy as np
import cv2
from pathlib import Path

# ============================================================
# MODEL İNDİRME (Cold start'ta bir kere çalışır)
# ============================================================
STORAGE_URL = "https://integrations.emergentagent.com/objstore/api/v1/storage"
EMERGENT_KEY = os.environ.get("EMERGENT_LLM_KEY", "")
MODELS_DIR = Path("/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

storage_key = None

def init_storage():
    global storage_key
    if storage_key:
        return storage_key
    resp = requests.post(f"{STORAGE_URL}/init", json={"emergent_key": EMERGENT_KEY}, timeout=30)
    resp.raise_for_status()
    storage_key = resp.json()["storage_key"]
    return storage_key

def download_model(storage_path: str, local_path: Path):
    """Modeli Emergent Object Storage'dan indir"""
    if local_path.exists():
        print(f"Model already cached: {local_path}")
        return
    print(f"Downloading model: {storage_path} -> {local_path}")
    key = init_storage()
    resp = requests.get(
        f"{STORAGE_URL}/objects/{storage_path}",
        headers={"X-Storage-Key": key},
        timeout=300
    )
    resp.raise_for_status()
    local_path.write_bytes(resp.content)
    print(f"Downloaded: {local_path} ({local_path.stat().st_size / 1e6:.1f} MB)")

# Modelleri indir
print("=" * 50)
print("AutoDamageIQ Worker - Initializing...")
print("=" * 50)

DAMAGE_MODEL_PATH = MODELS_DIR / "autodamage_best.pt"
PARTS_MODEL_PATH = MODELS_DIR / "carparts_seg_best.pt"

try:
    download_model("autodamageiq/models/autodamage_best.pt", DAMAGE_MODEL_PATH)
    download_model("autodamageiq/models/carparts_seg_best.pt", PARTS_MODEL_PATH)
except Exception as e:
    print(f"WARNING: Model download failed: {e}")

# ============================================================
# YOLO YÜKLEME
# ============================================================
from ultralytics import YOLO

damage_model = None
parts_model = None

if DAMAGE_MODEL_PATH.exists():
    damage_model = YOLO(str(DAMAGE_MODEL_PATH))
    print(f"Damage model loaded: {DAMAGE_MODEL_PATH}")

if PARTS_MODEL_PATH.exists():
    parts_model = YOLO(str(PARTS_MODEL_PATH))
    print(f"Parts model loaded: {PARTS_MODEL_PATH}")

print("Worker ready!")

# ============================================================
# SABİTLER
# ============================================================
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


def box_iou(a, b):
    x1, y1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    x2, y2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    inter = max(0, x2-x1) * max(0, y2-y1)
    aa = max(0, (float(a[2])-float(a[0]))*(float(a[3])-float(a[1])))
    ab = max(0, (float(b[2])-float(b[0]))*(float(b[3])-float(b[1])))
    return float(inter / (aa + ab - inter + 1e-6))


def calc_severity(dtype, box, w, h, conf):
    base = SEVERITY_MAP.get(dtype, 3)
    ar = max(0, (box[2]-box[0])*(box[3]-box[1])) / max(1, w*h)
    af = 1.5 if ar > 0.15 else (1.2 if ar > 0.08 else (1.0 if ar > 0.03 else 0.8))
    cf = 1.0 if conf > 50 else 0.9
    score = round(min(5.0, max(1.0, base*af*cf)), 1)
    cls = max(1, min(5, int(round(score))))
    return {"score": score, "class": cls, "label": "Yuksek" if score>=4 else ("Orta" if score>=2.5 else "Dusuk"), "area_ratio": round(ar*100, 2)}


# ============================================================
# HANDLER
# ============================================================
def handler(event):
    inp = event.get("input", {})
    image_b64 = inp.get("image_base64")
    if not image_b64:
        return {"error": "image_base64 required"}

    try:
        img_bytes = base64.b64decode(image_b64)
        image_np = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image_np is None:
            return {"error": "Image decode failed"}
    except Exception as e:
        return {"error": f"Image error: {str(e)}"}

    if damage_model is None:
        return {"error": "Damage model not loaded"}

    h, w = image_np.shape[:2]

    # Hasar tespiti
    dr = damage_model.predict(source=image_np, imgsz=640, conf=0.15, verbose=False)[0]
    d_boxes = dr.boxes.xyxy.cpu().numpy() if dr.boxes is not None else np.zeros((0,4))
    d_cls = dr.boxes.cls.cpu().numpy().astype(int) if dr.boxes is not None else np.zeros((0,),int)
    d_conf = dr.boxes.conf.cpu().numpy() if dr.boxes is not None else np.zeros((0,))
    d_names = damage_model.names

    # Parça
    p_boxes, p_cls, p_names = np.zeros((0,4)), np.zeros((0,),int), {}
    if parts_model:
        pr = parts_model.predict(source=image_np, imgsz=640, conf=0.15, verbose=False)[0]
        p_boxes = pr.boxes.xyxy.cpu().numpy() if pr.boxes is not None else np.zeros((0,4))
        p_cls = pr.boxes.cls.cpu().numpy().astype(int) if pr.boxes is not None else np.zeros((0,),int)
        p_names = parts_model.names

    damages = []
    for i, db in enumerate(d_boxes):
        best_iou, best_part, best_pbox = 0.0, None, None
        for j, pb in enumerate(p_boxes):
            iou = box_iou(db, pb)
            if iou > best_iou:
                best_iou, best_part = iou, p_names[int(p_cls[j])]
                best_pbox = [float(x) for x in pb.tolist()]
        
        dt = d_names[int(d_cls[i])]
        c = float(d_conf[i])
        sev = calc_severity(dt, [float(x) for x in db.tolist()], w, h, c*100)
        damages.append({
            "id": str(uuid.uuid4())[:8], "type": dt, "type_tr": DAMAGE_TR.get(dt, dt),
            "confidence": round(c*100, 1), "severity": sev["class"], "severity_details": sev,
            "box": [float(x) for x in db.tolist()],
            "part": best_part if best_iou > 0.1 else None,
            "part_tr": PARTS_TR.get(best_part, best_part) if best_iou > 0.1 else None,
            "part_box": best_pbox if best_iou > 0.1 else None,
            "iou_with_part": round(best_iou, 3),
            "part_source": "yolo" if best_iou > 0.1 else "none"
        })

    parts = [{"name": p_names[int(p_cls[j])], "name_tr": PARTS_TR.get(p_names[int(p_cls[j])], p_names[int(p_cls[j])]), "box": [float(x) for x in pb.tolist()]} for j, pb in enumerate(p_boxes)]

    total = len(damages)
    aff = len(set(d["part"] for d in damages if d["part"]))
    avg_s = round(sum(d["severity"] for d in damages)/max(1,total), 1)
    avg_c = round(sum(d["confidence"] for d in damages)/max(1,total), 1)
    risk = "Yuksek" if (avg_s>=4 or total>=4) else ("Orta" if (avg_s>=2.5 or total>=2) else "Dusuk")

    return {
        "damages": damages, "parts": parts,
        "summary": {"total_damages": total, "affected_parts": aff, "average_severity": avg_s,
                     "average_confidence": avg_c, "risk_level": risk,
                     "needs_review": avg_c < 35 or risk == "Yuksek", "review_reasons": []},
        "image_size": {"width": w, "height": h},
        "inference_source": "runpod_serverless"
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
