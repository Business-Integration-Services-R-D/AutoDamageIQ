"""
AutoDamageIQ — VLM Etiketleme Pipeline
========================================
GPT-4o Vision ile arac hasar gorsellerini analiz eder,
YOLO formatinda bounding box etiketleri uretir.
"""

import os
import json
import base64
import asyncio
import glob
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from emergentintegrations.llm.chat import LlmChat, UserMessage, ImageContent

EMERGENT_KEY = os.environ.get("EMERGENT_LLM_KEY")
VLM_QUEUE_DIR = Path("/app/datasets/vlm_queue/images")
VLM_LABELS_DIR = Path("/app/datasets/vlm_queue/labels")
UNIFIED_TRAIN_IMG = Path("/app/datasets/unified/images/train")
UNIFIED_TRAIN_LBL = Path("/app/datasets/unified/labels/train")

DAMAGE_CLASSES = {
    "crack": 0,
    "dent": 1,
    "glass_shatter": 2,
    "lamp_broken": 3,
    "scratch": 4,
    "tire_flat": 5,
}

SYSTEM_PROMPT = """You are an expert vehicle damage detection system. Analyze the given car image and detect ALL visible damages.

For EACH damage found, provide:
1. "type": One of: crack, dent, glass_shatter, lamp_broken, scratch, tire_flat
2. "bbox": [x_center, y_center, width, height] — ALL values NORMALIZED (0.0 to 1.0 relative to image dimensions)
3. "confidence": Your confidence (0.0 to 1.0)

Rules:
- x_center, y_center = center of the damage bounding box, normalized by image width/height
- width, height = size of bounding box, normalized
- Be precise with bounding boxes — they should tightly surround only the damaged area
- scratch = any surface scratch, paint scratch, abrasion
- dent = any dent, deformation, depression in body panels
- crack = crack lines on body panels (NOT glass)
- glass_shatter = broken/cracked glass (windshield, windows)
- lamp_broken = broken headlight, taillight, fog light
- tire_flat = flat or damaged tire
- If NO damage is visible, return empty damages array

Respond ONLY with valid JSON:
{"damages": [{"type": "scratch", "bbox": [0.5, 0.3, 0.2, 0.05], "confidence": 0.85}]}"""


def encode_image(image_path: str) -> str:
    """Gorseli base64'e cevir"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


async def label_single_image(image_path: str) -> dict:
    """Tek bir gorseli GPT-4o ile etiketle"""
    img_b64 = encode_image(image_path)

    chat = LlmChat(
        api_key=EMERGENT_KEY,
        session_id=f"vlm-label-{Path(image_path).stem}",
        system_message=SYSTEM_PROMPT,
    ).with_model("openai", "gpt-4o")

    image_content = ImageContent(image_base64=img_b64)
    msg = UserMessage(
        text="Analyze this vehicle image for damages. Return JSON only.",
        file_contents=[image_content],
    )

    # Non-streaming — sadece JSON sonuç lazım
    response = await chat.send_message(msg)
    text = response.strip() if isinstance(response, str) else response.text.strip()

    # JSON parse
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        # Bazen ekstra text olabiliyor, JSON parcasini bul
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            result = json.loads(text[start:end])
        else:
            result = {"damages": [], "error": "JSON parse failed"}

    return result


def vlm_result_to_yolo(result: dict) -> list:
    """VLM sonucunu YOLO label satirlarina cevir"""
    lines = []
    for dmg in result.get("damages", []):
        dtype = dmg.get("type", "").lower().strip()
        if dtype not in DAMAGE_CLASSES:
            continue

        bbox = dmg.get("bbox", [])
        if len(bbox) != 4:
            continue

        cx, cy, w, h = [float(v) for v in bbox]

        # Gecerlilik kontrolu
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
            continue

        class_id = DAMAGE_CLASSES[dtype]
        conf = dmg.get("confidence", 0.5)

        if conf >= 0.3:  # Dusuk guvenli etiketleri atla
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return lines


async def label_batch(image_paths: list, save_labels: bool = True) -> dict:
    """Birden fazla gorseli etiketle"""
    VLM_LABELS_DIR.mkdir(parents=True, exist_ok=True)

    results = {"processed": 0, "with_damage": 0, "total_labels": 0, "errors": 0, "details": []}

    for i, img_path in enumerate(image_paths):
        stem = Path(img_path).stem
        label_path = VLM_LABELS_DIR / f"{stem}.txt"

        # Zaten etiketlenmis mi?
        if label_path.exists():
            results["processed"] += 1
            continue

        try:
            print(f"[{i+1}/{len(image_paths)}] {Path(img_path).name}...", end=" ", flush=True)
            vlm_result = await label_single_image(img_path)
            yolo_lines = vlm_result_to_yolo(vlm_result)

            if save_labels and yolo_lines:
                with open(label_path, "w") as f:
                    f.write("\n".join(yolo_lines))

            damage_count = len(yolo_lines)
            results["processed"] += 1
            if damage_count > 0:
                results["with_damage"] += 1
            results["total_labels"] += damage_count
            results["details"].append({
                "file": Path(img_path).name,
                "damages": vlm_result.get("damages", []),
                "yolo_labels": damage_count,
            })
            print(f"{damage_count} hasar")

        except Exception as e:
            print(f"HATA: {e}")
            results["errors"] += 1
            results["details"].append({"file": Path(img_path).name, "error": str(e)})

    return results


def merge_vlm_labels_to_unified():
    """VLM etiketlerini unified egitim veri setine tasi"""
    UNIFIED_TRAIN_IMG.mkdir(parents=True, exist_ok=True)
    UNIFIED_TRAIN_LBL.mkdir(parents=True, exist_ok=True)

    copied = 0
    for label_file in VLM_LABELS_DIR.glob("*.txt"):
        stem = label_file.stem

        # Image'i bul
        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = VLM_QUEUE_DIR / (stem + ext)
            if candidate.exists():
                img_path = candidate
                break

        if not img_path:
            continue

        dest_img = UNIFIED_TRAIN_IMG / f"vlm_{img_path.name}"
        dest_lbl = UNIFIED_TRAIN_LBL / f"vlm_{stem}.txt"

        if not dest_img.exists():
            import shutil
            shutil.copy2(img_path, dest_img)
            shutil.copy2(label_file, dest_lbl)
            copied += 1

    print(f"VLM etiketli {copied} gorsel unified'a eklendi")
    return copied


async def run_vlm_pipeline(max_images: int = 50):
    """Tam VLM etiketleme pipeline'i"""
    print("=" * 50)
    print("  VLM Etiketleme Pipeline (GPT-4o)")
    print("=" * 50)

    # Kuyruktan gorselleri al
    all_images = sorted(glob.glob(str(VLM_QUEUE_DIR / "*")))
    all_images = [p for p in all_images if Path(p).suffix.lower() in [".jpg", ".jpeg", ".png"]]

    # Zaten etiketlenmisleri filtrele
    labeled = set(f.stem for f in VLM_LABELS_DIR.glob("*.txt")) if VLM_LABELS_DIR.exists() else set()
    unlabeled = [p for p in all_images if Path(p).stem not in labeled]

    print(f"Kuyrukta: {len(all_images)} gorsel")
    print(f"Zaten etiketli: {len(labeled)}")
    print(f"Etiketlenecek: {len(unlabeled)}")
    print(f"Bu tur islenecek: {min(max_images, len(unlabeled))}")
    print()

    batch = unlabeled[:max_images]
    if not batch:
        print("Etiketlenecek gorsel yok!")
        return

    results = await label_batch(batch)

    print(f"\n{'=' * 50}")
    print(f"  SONUC")
    print(f"{'=' * 50}")
    print(f"  Islenen: {results['processed']}")
    print(f"  Hasarli: {results['with_damage']}")
    print(f"  Toplam etiket: {results['total_labels']}")
    print(f"  Hata: {results['errors']}")

    # Unified'a ekle
    if results["with_damage"] > 0:
        merged = merge_vlm_labels_to_unified()
        print(f"  Unified'a eklenen: {merged}")

    return results


if __name__ == "__main__":
    asyncio.run(run_vlm_pipeline(max_images=10))
