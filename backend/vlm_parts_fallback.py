"""
AutoDamageIQ - VLM Parts Fallback (GPT-4o)
===========================================
Parça segmentasyonu modelinin belirsiz kaldığı durumlarda
GPT-4o vision ile araç parçası tespiti yapar.

Kullanım senaryoları:
- YOLO parça modeli düşük IoU verdiğinde
- Hasar-parça eşleşmesi bulunamadığında
- Birden fazla parça adayı olduğunda doğrulama
"""

import os
import base64
import json
import asyncio
import logging
import cv2
import numpy as np
from typing import Optional, Dict, List, Any

logger = logging.getLogger("autodamageid.vlm_parts")

EMERGENT_LLM_KEY = os.environ.get("EMERGENT_LLM_KEY")

# Geçerli parça isimleri (YOLO modeli ile uyumlu)
VALID_PARTS = [
    "back_bumper", "back_door", "back_glass", "back_left_door",
    "back_left_light", "back_light", "back_right_door", "back_right_light",
    "front_bumper", "front_door", "front_glass", "front_left_door",
    "front_left_light", "front_light", "front_right_door", "front_right_light",
    "hood", "left_mirror", "right_mirror", "tailgate", "trunk", "wheel"
]

SYSTEM_PROMPT = """Sen bir araç hasar uzmanısın. Sana bir araç hasarının kırpılmış görselini vereceğim.
Görevlerin:
1. Bu hasarın hangi araç parçası/paneli üzerinde olduğunu tespit et.
2. Sadece şu parça isimlerinden birini seç: """ + ", ".join(VALID_PARTS) + """

SADECE JSON formatında yanıt ver, başka hiçbir şey yazma:
{"part": "parca_adi", "confidence": 0.85, "reasoning": "kısa açıklama"}

Eğer parça belirlenemiyorsa:
{"part": null, "confidence": 0.0, "reasoning": "neden belirlenemediği"}
"""


def _crop_damage_region(image_np: np.ndarray, box: List[float], padding: float = 0.15) -> np.ndarray:
    """Hasar bölgesini padding ile kırp"""
    h, w = image_np.shape[:2]
    x1, y1, x2, y2 = box

    # Padding ekle
    bw = x2 - x1
    bh = y2 - y1
    x1 = max(0, int(x1 - bw * padding))
    y1 = max(0, int(y1 - bh * padding))
    x2 = min(w, int(x2 + bw * padding))
    y2 = min(h, int(y2 + bh * padding))

    return image_np[y1:y2, x1:x2]


def _image_to_base64(image_np: np.ndarray) -> str:
    """Numpy image'ı base64 string'e çevir"""
    _, buffer = cv2.imencode('.jpg', image_np, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buffer).decode('utf-8')


async def identify_part_with_vlm(image_np: np.ndarray, damage_box: List[float]) -> Optional[Dict[str, Any]]:
    """
    GPT-4o ile hasar bölgesindeki araç parçasını tespit et.
    
    Returns: {"part": str, "confidence": float, "reasoning": str} veya None
    """
    if not EMERGENT_LLM_KEY:
        logger.warning("EMERGENT_LLM_KEY not set, VLM fallback disabled")
        return None

    try:
        from emergentintegrations.llm.chat import LlmChat, UserMessage, ImageContent, TextDelta, StreamDone

        # Hasar bölgesini kırp
        cropped = _crop_damage_region(image_np, damage_box)
        if cropped.size == 0:
            return None

        img_b64 = _image_to_base64(cropped)

        chat = LlmChat(
            api_key=EMERGENT_LLM_KEY,
            session_id=f"vlm-parts-{id(image_np)}",
            system_message=SYSTEM_PROMPT
        ).with_model("openai", "gpt-4o")

        image_content = ImageContent(image_base64=img_b64)

        # send_message for non-streaming (internal API call, no UI)
        response = await chat.send_message(UserMessage(
            text="Bu araç hasarı hangi parça/panel üzerinde? JSON olarak yanıtla.",
            file_contents=[image_content]
        ))

        # response can be string or object with .text
        response_text = response if isinstance(response, str) else getattr(response, 'text', str(response))
        response_text = response_text.strip()

        # JSON çıkar (bazen markdown code block içinde gelebilir)
        if "```" in response_text:
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            response_text = response_text[json_start:json_end]

        result = json.loads(response_text)

        # Geçerli parça mı kontrol et
        part = result.get("part")
        if part and part not in VALID_PARTS:
            # Yakın eşleşme dene
            part_lower = part.lower().replace(" ", "_")
            for vp in VALID_PARTS:
                if part_lower in vp or vp in part_lower:
                    result["part"] = vp
                    break
            else:
                result["part"] = None

        logger.info(f"VLM part identification: {result}")
        return result

    except json.JSONDecodeError as e:
        logger.error(f"VLM JSON parse error: {e}")
        return None
    except Exception as e:
        logger.error(f"VLM parts fallback error: {e}")
        return None


async def enhance_damages_with_vlm(
    image_np: np.ndarray,
    damages: List[Dict[str, Any]],
    iou_threshold: float = 0.1
) -> List[Dict[str, Any]]:
    """
    Belirsiz parça eşleşmesi olan hasarları VLM ile iyileştir.
    
    Kurallar:
    - IoU < iou_threshold olan hasarlar (parça bulunamadı) → VLM'e gönder
    - VLM sonucu confidence >= 0.5 ise parçayı güncelle
    """
    from repair_engine import get_repair_recommendation

    PARTS_TR = {
        "back_bumper": "Arka Tampon", "back_door": "Arka Kapı",
        "back_glass": "Arka Cam", "back_left_door": "Arka Sol Kapı",
        "back_left_light": "Arka Sol Far", "back_light": "Arka Far",
        "back_right_door": "Arka Sağ Kapı", "back_right_light": "Arka Sağ Far",
        "front_bumper": "Ön Tampon", "front_door": "Ön Kapı",
        "front_glass": "Ön Cam", "front_left_door": "Ön Sol Kapı",
        "front_left_light": "Ön Sol Far", "front_light": "Ön Far",
        "front_right_door": "Ön Sağ Kapı", "front_right_light": "Ön Sağ Far",
        "hood": "Kaput", "left_mirror": "Sol Ayna",
        "right_mirror": "Sağ Ayna", "tailgate": "Bagaj Kapağı",
        "trunk": "Bagaj", "wheel": "Tekerlek"
    }

    enhanced = []
    vlm_tasks = []

    for i, dmg in enumerate(damages):
        if dmg.get("part") is None or dmg.get("iou_with_part", 0) < iou_threshold:
            vlm_tasks.append((i, dmg))

    # VLM çağrılarını paralel yap
    if vlm_tasks:
        results = await asyncio.gather(
            *[identify_part_with_vlm(image_np, dmg["box"]) for _, dmg in vlm_tasks],
            return_exceptions=True
        )

        vlm_map = {}
        for (idx, _), result in zip(vlm_tasks, results):
            if isinstance(result, dict) and result.get("part") and result.get("confidence", 0) >= 0.5:
                vlm_map[idx] = result

        for i, dmg in enumerate(damages):
            if i in vlm_map:
                vlm_result = vlm_map[i]
                part_name = vlm_result["part"]
                dmg["part"] = part_name
                dmg["part_tr"] = PARTS_TR.get(part_name, part_name)
                dmg["part_source"] = "vlm"
                dmg["vlm_confidence"] = vlm_result["confidence"]
                dmg["vlm_reasoning"] = vlm_result.get("reasoning", "")
                # Repair recommendation güncelle
                dmg["repair"] = get_repair_recommendation(
                    damage_type=dmg["type"],
                    severity=dmg["severity"],
                    confidence=dmg["confidence"],
                    panel=part_name
                )
            else:
                dmg["part_source"] = "yolo" if dmg.get("part") else "none"
            enhanced.append(dmg)
    else:
        for dmg in damages:
            dmg["part_source"] = "yolo" if dmg.get("part") else "none"
            enhanced.append(dmg)

    return enhanced
