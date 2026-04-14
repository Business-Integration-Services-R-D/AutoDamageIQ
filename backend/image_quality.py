"""
AutoDamageIQ - Görüntü Kalite Kontrol Modülü
=============================================
Bulanıklık, pozlama, çözünürlük ve yansıma metrikleri ile
görüntü kalitesini değerlendirir. Düşük kaliteli görüntülerde
uyarı üretir.
"""

import cv2
import numpy as np
from typing import Dict, Any


# Kalite eşikleri
BLUR_THRESHOLD = 80.0        # Laplacian varyansı altı = bulanık
BRIGHTNESS_LOW = 40           # Ortalama parlaklık altı = karanlık
BRIGHTNESS_HIGH = 220         # Ortalama parlaklık üstü = aşırı pozlanmış
MIN_RESOLUTION = 480          # Minimum kenar piksel sayısı
REFLECTION_THRESHOLD = 0.15   # Aşırı parlak piksel oranı


def assess_image_quality(image_np: np.ndarray) -> Dict[str, Any]:
    """
    Görüntü kalite değerlendirmesi yapar.
    
    Returns:
        quality_score: 0-100 arası genel kalite skoru
        is_acceptable: Analize uygun mu
        warnings: Uyarı listesi
        metrics: Detaylı metrikler
    """
    h, w = image_np.shape[:2]
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    
    warnings = []
    scores = []
    
    # 1. Bulanıklık kontrolü (Laplacian varyansı)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_score = min(100, (laplacian_var / BLUR_THRESHOLD) * 100)
    scores.append(blur_score)
    
    is_blurry = laplacian_var < BLUR_THRESHOLD
    if is_blurry:
        warnings.append({
            "type": "blur",
            "message_tr": "Goruntu bulanik - yeniden cekim onerilir",
            "severity": "high" if laplacian_var < BLUR_THRESHOLD * 0.5 else "medium"
        })
    
    # 2. Pozlama kontrolü (ortalama parlaklık)
    mean_brightness = float(np.mean(gray))
    
    if mean_brightness < BRIGHTNESS_LOW:
        brightness_score = (mean_brightness / BRIGHTNESS_LOW) * 70
        warnings.append({
            "type": "dark",
            "message_tr": "Goruntu cok karanlik - aydinlik ortamda yeniden cekim onerilir",
            "severity": "high" if mean_brightness < BRIGHTNESS_LOW * 0.5 else "medium"
        })
    elif mean_brightness > BRIGHTNESS_HIGH:
        brightness_score = max(0, (255 - mean_brightness) / (255 - BRIGHTNESS_HIGH) * 70)
        warnings.append({
            "type": "overexposed",
            "message_tr": "Asiri pozlanmis goruntu - parlaklik azaltilarak yeniden cekim onerilir",
            "severity": "medium"
        })
    else:
        brightness_score = 100
    scores.append(brightness_score)
    
    # 3. Çözünürlük kontrolü
    min_dim = min(h, w)
    if min_dim < MIN_RESOLUTION:
        resolution_score = (min_dim / MIN_RESOLUTION) * 70
        warnings.append({
            "type": "low_resolution",
            "message_tr": f"Dusuk cozunurluk ({w}x{h}) - daha yuksek cozunurluklu goruntu onerilir",
            "severity": "medium"
        })
    else:
        resolution_score = 100
    scores.append(resolution_score)
    
    # 4. Yansıma kontrolü (aşırı parlak piksel oranı)
    bright_pixels = np.sum(gray > 240)
    total_pixels = h * w
    reflection_ratio = float(bright_pixels / total_pixels)
    
    if reflection_ratio > REFLECTION_THRESHOLD:
        reflection_score = max(0, (1 - reflection_ratio) * 100)
        warnings.append({
            "type": "reflection",
            "message_tr": "Yuksek yansima tespit edildi - analiz sonuclari etkilenebilir",
            "severity": "low"
        })
    else:
        reflection_score = 100
    scores.append(reflection_score)
    
    # 5. Kontrast kontrolü
    contrast = float(np.std(gray))
    if contrast < 30:
        contrast_score = (contrast / 30) * 70
        warnings.append({
            "type": "low_contrast",
            "message_tr": "Dusuk kontrast - hasar detaylari net olmayabilir",
            "severity": "low"
        })
    else:
        contrast_score = 100
    scores.append(contrast_score)
    
    # Genel kalite skoru (ağırlıklı ortalama)
    weights = [0.30, 0.25, 0.15, 0.15, 0.15]  # blur, brightness, resolution, reflection, contrast
    quality_score = sum(s * w for s, w in zip(scores, weights))
    quality_score = round(min(100, max(0, quality_score)), 1)
    
    # Kabul edilebilirlik kararı
    is_acceptable = quality_score >= 40 and not any(
        w["severity"] == "high" for w in warnings
    )
    
    # Kalite seviyesi
    if quality_score >= 80:
        quality_level = "Yuksek"
    elif quality_score >= 55:
        quality_level = "Orta"
    else:
        quality_level = "Dusuk"
    
    return {
        "quality_score": float(quality_score),
        "quality_level": quality_level,
        "is_acceptable": bool(is_acceptable),
        "warnings": warnings,
        "recommendation": "Yeniden cekim onerilir" if not is_acceptable else None,
        "metrics": {
            "blur_variance": round(float(laplacian_var), 2),
            "mean_brightness": round(mean_brightness, 1),
            "resolution": f"{w}x{h}",
            "min_dimension": int(min_dim),
            "reflection_ratio": round(reflection_ratio, 4),
            "contrast": round(contrast, 1)
        }
    }
