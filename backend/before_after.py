"""
AutoDamageIQ - Before/After Yeni Hasar Analizi Modulu
======================================================
Teslim oncesi ve sonrasi gorselleri karsilastirarak
yeni hasar tespiti yapar.
"""

import cv2
import numpy as np
import uuid
from typing import Dict, List, Any, Optional, Tuple


def align_images(
    before_np: np.ndarray,
    after_np: np.ndarray,
    max_features: int = 2000,
    match_ratio: float = 0.7
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    ORB feature matching ile before gorseli after gorseline hizala.
    
    Returns:
        (hizalanmis before gorseli veya None, hizalama bilgisi)
    """
    gray_before = cv2.cvtColor(before_np, cv2.COLOR_BGR2GRAY)
    gray_after = cv2.cvtColor(after_np, cv2.COLOR_BGR2GRAY)
    
    # ORB feature detector
    orb = cv2.ORB_create(nfeatures=max_features)
    
    kp1, des1 = orb.detectAndCompute(gray_before, None)
    kp2, des2 = orb.detectAndCompute(gray_after, None)
    
    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return None, {
            "success": False,
            "reason": "Yeterli ozellik noktasi bulunamadi",
            "keypoints_before": len(kp1) if kp1 else 0,
            "keypoints_after": len(kp2) if kp2 else 0
        }
    
    # BFMatcher with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Lowe's ratio test
    good_matches = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < match_ratio * n.distance:
                good_matches.append(m)
    
    if len(good_matches) < 10:
        return None, {
            "success": False,
            "reason": "Yeterli eslesen ozellik bulunamadi",
            "total_matches": len(matches),
            "good_matches": len(good_matches)
        }
    
    # Homography hesapla
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if H is None:
        return None, {
            "success": False,
            "reason": "Homografi matrisi hesaplanamadi",
            "good_matches": len(good_matches)
        }
    
    inlier_count = int(mask.sum()) if mask is not None else 0
    
    # Before gorseli hizala
    h, w = after_np.shape[:2]
    aligned_before = cv2.warpPerspective(before_np, H, (w, h))
    
    return aligned_before, {
        "success": True,
        "keypoints_before": len(kp1),
        "keypoints_after": len(kp2),
        "good_matches": len(good_matches),
        "inlier_count": inlier_count,
        "inlier_ratio": round(inlier_count / max(1, len(good_matches)), 3)
    }


def compute_difference_map(
    before_np: np.ndarray,
    after_np: np.ndarray,
    blur_size: int = 5,
    threshold: int = 30
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Iki gorsel arasindaki fark haritasini olustur.
    
    Returns:
        (fark maskesi, istatistikler)
    """
    # Ayni boyuta getir
    h, w = after_np.shape[:2]
    before_resized = cv2.resize(before_np, (w, h))
    
    # Gri tonlama
    gray_before = cv2.cvtColor(before_resized, cv2.COLOR_BGR2GRAY)
    gray_after = cv2.cvtColor(after_np, cv2.COLOR_BGR2GRAY)
    
    # Gaussian blur
    gray_before = cv2.GaussianBlur(gray_before, (blur_size, blur_size), 0)
    gray_after = cv2.GaussianBlur(gray_after, (blur_size, blur_size), 0)
    
    # Mutlak fark
    diff = cv2.absdiff(gray_before, gray_after)
    
    # Threshold
    _, diff_mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    # Morfolojik isleme (gurultu temizle)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_OPEN, kernel)
    diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_CLOSE, kernel)
    
    # Istatistikler
    changed_pixels = int(np.sum(diff_mask > 0))
    total_pixels = h * w
    change_ratio = changed_pixels / max(1, total_pixels)
    
    return diff_mask, {
        "changed_pixels": changed_pixels,
        "total_pixels": total_pixels,
        "change_ratio": round(float(change_ratio), 4),
        "change_percentage": round(float(change_ratio * 100), 2)
    }


def find_change_regions(
    diff_mask: np.ndarray,
    min_area: int = 500
) -> List[Dict[str, Any]]:
    """
    Fark maskesinden degisim bolgelerini cikar.
    """
    contours, _ = cv2.findContours(diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        
        regions.append({
            "id": str(uuid.uuid4())[:8],
            "box": [int(x), int(y), int(x + w), int(y + h)],
            "area_pixels": int(area),
            "center": [int(x + w // 2), int(y + h // 2)]
        })
    
    # Buyukten kucuge sirala
    regions.sort(key=lambda r: r["area_pixels"], reverse=True)
    return regions


def match_damages_to_changes(
    after_damages: List[Dict[str, Any]],
    change_regions: List[Dict[str, Any]],
    before_damages: List[Dict[str, Any]],
    iou_threshold: float = 0.2
) -> List[Dict[str, Any]]:
    """
    After hasarlarini degisim bolgeleri ve before hasarlariyla eslestirir.
    Yeni hasarlari belirler.
    """
    new_damages = []
    
    for after_dmg in after_damages:
        after_box = after_dmg.get("box", [])
        if len(after_box) != 4:
            continue
        
        # Before'da benzer hasar var mi?
        matched_before = False
        for before_dmg in before_damages:
            before_box = before_dmg.get("box", [])
            if len(before_box) != 4:
                continue
            iou = _box_iou(after_box, before_box)
            if iou > iou_threshold and after_dmg.get("type") == before_dmg.get("type"):
                matched_before = True
                break
        
        # Degisim bolgesiyle ortusme kontrolu
        overlaps_change = False
        for region in change_regions:
            iou = _box_iou(after_box, region["box"])
            if iou > 0.1:
                overlaps_change = True
                break
        
        if not matched_before:
            new_damages.append({
                **after_dmg,
                "is_new": True,
                "overlaps_change_region": overlaps_change,
                "evidence_strength": "Yuksek" if overlaps_change else "Orta"
            })
    
    return new_damages


def compare_analyses(
    before_image: np.ndarray,
    after_image: np.ndarray,
    before_damages: List[Dict[str, Any]],
    after_damages: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Before/After tam karsilastirma analizi.
    
    Args:
        before_image: Teslim oncesi gorsel (BGR)
        after_image: Teslim sonrasi gorsel (BGR)
        before_damages: Onceki analiz hasarlari
        after_damages: Sonraki analiz hasarlari
    
    Returns:
        Karsilastirma raporu
    """
    # 1. Gorsel hizalama
    aligned_before, alignment_info = align_images(before_image, after_image)
    
    # Hizalama basarisizsa orijinal kullan
    source_for_diff = aligned_before if aligned_before is not None else before_image
    alignment_used = aligned_before is not None
    
    # 2. Fark haritasi
    diff_mask, diff_stats = compute_difference_map(source_for_diff, after_image)
    
    # 3. Degisim bolgeleri
    change_regions = find_change_regions(diff_mask)
    
    # 4. Yeni hasar eslestirme
    new_damages = match_damages_to_changes(
        after_damages, change_regions, before_damages
    )
    
    # 5. Ozet
    has_new_damage = len(new_damages) > 0
    
    if has_new_damage:
        high_evidence = sum(1 for d in new_damages if d.get("evidence_strength") == "Yuksek")
        if high_evidence > 0:
            verdict = "Yeni hasar tespit edildi"
            confidence = "Yuksek"
        else:
            verdict = "Yeni hasar sinyali mevcut"
            confidence = "Orta"
    else:
        if diff_stats["change_percentage"] < 5:
            verdict = "Yeni hasar tespit edilmedi"
            confidence = "Yuksek"
        else:
            verdict = "Belirgin fark var ancak yeni hasar eslestirilmedi - manuel inceleme onerilir"
            confidence = "Dusuk"
    
    return {
        "has_new_damage": has_new_damage,
        "verdict": verdict,
        "verdict_confidence": confidence,
        "new_damages": new_damages,
        "new_damage_count": len(new_damages),
        "alignment": {
            "used": alignment_used,
            **alignment_info
        },
        "difference": diff_stats,
        "change_regions": change_regions[:10],  # Max 10 bolge
        "change_region_count": len(change_regions),
        "summary": {
            "before_damage_count": len(before_damages),
            "after_damage_count": len(after_damages),
            "new_damage_count": len(new_damages),
            "change_percentage": diff_stats["change_percentage"]
        }
    }


def _box_iou(box_a: list, box_b: list) -> float:
    """IoU hesapla [x1, y1, x2, y2]"""
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, (float(box_a[2]) - float(box_a[0])) * (float(box_a[3]) - float(box_a[1])))
    area_b = max(0, (float(box_b[2]) - float(box_b[0])) * (float(box_b[3]) - float(box_b[1])))
    
    union = area_a + area_b - inter + 1e-6
    return float(inter / union)
