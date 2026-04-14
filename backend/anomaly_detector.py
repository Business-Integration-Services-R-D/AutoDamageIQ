"""
AutoDamageIQ - Anomali/Usulsuzluk Tespit Modulu
=================================================
Perceptual hashing ile tekrar gorsel tespiti ve
temel anomali sinyalleri uretir.
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Optional
from pymongo import MongoClient
from datetime import datetime, timezone
import os


def compute_phash(image_np: np.ndarray, hash_size: int = 16) -> str:
    """
    Perceptual hash hesapla (pHash).
    Gorselin icerigi degismeden boyut/format degisse bile benzer hash uretir.
    """
    # Gri tonlamaya cevir
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    
    # hash_size * hash_size boyutuna kucult
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    
    # Yatay gradient (DCT yerine basit fark)
    diff = resized[:, 1:] > resized[:, :-1]
    
    # Binary hash olustur
    hash_bits = diff.flatten()
    hash_hex = ''.join(['1' if b else '0' for b in hash_bits])
    
    # Hex'e cevir (her 4 bit = 1 hex karakter)
    hex_str = ''
    for i in range(0, len(hash_hex), 4):
        chunk = hash_hex[i:i+4]
        if len(chunk) == 4:
            hex_str += hex(int(chunk, 2))[2:]
    
    return hex_str


def hamming_distance(hash1: str, hash2: str) -> int:
    """Iki hash arasindaki Hamming mesafesini hesapla"""
    if len(hash1) != len(hash2):
        return max(len(hash1), len(hash2))
    
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))


def check_duplicate_image(
    image_np: np.ndarray,
    current_hash: str,
    db,
    threshold: int = 8
) -> Dict[str, Any]:
    """
    Veritabanindaki mevcut analizlerle gorsel benzerlik kontrolu yapar.
    
    Args:
        image_np: Analiz edilen gorsel
        current_hash: Mevcut gorselin pHash'i
        db: MongoDB veritabani referansi
        threshold: Benzerlik esigi (dusuk = daha kati)
    
    Returns:
        Benzerlik analiz sonucu
    """
    duplicates = []
    
    # Son 100 analizi kontrol et
    recent_analyses = list(
        db.analyses.find(
            {"phash": {"$exists": True}},
            {"_id": 1, "phash": 1, "created_at": 1, "filename": 1}
        ).sort("created_at", -1).limit(100)
    )
    
    for analysis in recent_analyses:
        stored_hash = analysis.get("phash", "")
        if not stored_hash:
            continue
        
        distance = hamming_distance(current_hash, stored_hash)
        
        if distance <= threshold:
            similarity = round((1 - distance / max(len(current_hash), 1)) * 100, 1)
            duplicates.append({
                "analysis_id": str(analysis["_id"]),
                "filename": analysis.get("filename", "Bilinmiyor"),
                "created_at": analysis.get("created_at", ""),
                "hamming_distance": int(distance),
                "similarity_percent": float(similarity)
            })
    
    return {
        "has_duplicates": len(duplicates) > 0,
        "duplicate_count": len(duplicates),
        "duplicates": duplicates[:5]  # En fazla 5 sonuc
    }


def generate_anomaly_score(
    image_np: np.ndarray,
    duplicate_result: Dict[str, Any],
    quality_result: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Coklu sinyallerden birlesik suphe skoru uretir.
    
    Sinyaller:
    1. Gorsel tekrar kullanimi (duplicate check)
    2. Goruntu kalite anomalileri
    3. Piksel dagilim anomalileri
    """
    signals = []
    score_components = []
    
    # Sinyal 1: Tekrar gorsel kullanimi
    if duplicate_result["has_duplicates"]:
        max_similarity = max(
            d["similarity_percent"] for d in duplicate_result["duplicates"]
        )
        signal_score = min(100, max_similarity)
        score_components.append(signal_score * 0.50)  # %50 agirlik
        signals.append({
            "type": "duplicate_image",
            "message_tr": f"Benzer gorsel tespit edildi (benzerlik: %{max_similarity})",
            "severity": "high" if max_similarity > 95 else "medium",
            "score": float(round(signal_score, 1))
        })
    else:
        score_components.append(0)
    
    # Sinyal 2: Kalite anomalisi
    if quality_result:
        quality_warnings = quality_result.get("warnings", [])
        high_severity_count = sum(1 for w in quality_warnings if w["severity"] == "high")
        if high_severity_count > 0:
            signal_score = min(60, high_severity_count * 30)
            score_components.append(signal_score * 0.20)
            signals.append({
                "type": "quality_anomaly",
                "message_tr": "Goruntu kalitesi normalin disinda - kasitli bozulma olabilir",
                "severity": "low",
                "score": float(round(signal_score, 1))
            })
        else:
            score_components.append(0)
    else:
        score_components.append(0)
    
    # Sinyal 3: Piksel dagilim anomalisi (asiri uniform bolge tespiti)
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    h, w_img = gray.shape
    
    # 4x4 grid'e bol, her bolgenin varyansini hesapla
    block_h, block_w = h // 4, w_img // 4
    low_variance_blocks = 0
    total_blocks = 0
    
    for i in range(4):
        for j in range(4):
            block = gray[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
            if block.size > 0:
                total_blocks += 1
                if np.std(block) < 5:  # Cok dusuk varyans = uniform (olasi manipulasyon)
                    low_variance_blocks += 1
    
    uniform_ratio = low_variance_blocks / max(1, total_blocks)
    if uniform_ratio > 0.3:  # %30'dan fazla uniform bolge
        signal_score = min(50, uniform_ratio * 100)
        score_components.append(signal_score * 0.30)
        signals.append({
            "type": "pixel_anomaly",
            "message_tr": "Goruntu iceriginde olagan disi uniform bolgeler tespit edildi",
            "severity": "low",
            "score": float(round(signal_score, 1))
        })
    else:
        score_components.append(0)
    
    # Birlesik suphe skoru
    total_score = sum(score_components)
    total_score = round(min(100, max(0, total_score)), 1)
    
    # Aksiyon karari
    if total_score >= 60:
        action = "Manuel inceleme kuyruğuna aktar"
        risk_level = "Yuksek"
    elif total_score >= 30:
        action = "Dikkatli inceleme onerilir"
        risk_level = "Orta"
    else:
        action = None
        risk_level = "Dusuk"
    
    return {
        "anomaly_score": float(total_score),
        "risk_level": risk_level,
        "signals": signals,
        "action_tr": action,
        "signal_count": len(signals)
    }
