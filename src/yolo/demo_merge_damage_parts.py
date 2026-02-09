from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np


def box_iou(box_a, box_b):
    """
    box: [x1, y1, x2, y2]
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = max(0.0, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))

    union = area_a + area_b - inter_area + 1e-6
    return inter_area / union, inter_area, area_a, area_b


def main():
    this_dir = Path(__file__).resolve().parent        # src/yolo
    project_root = this_dir.parents[1]                # src/
    repo_root = project_root.parent                   # AutoDamageIQ

    # 🔧 BURAYI GEREKİRSE KENDİ RUN KLASÖRÜNE GÖRE DÜZELT
    carparts_weights = this_dir / "runs" / "carparts_seg_v1" / "weights" / "best.pt"
    damage_weights   = this_dir / "weights" / "best.pt"

    # Test görseli
    img_path = repo_root / "AutoDamageIQ" / "assets" / "crash_5.jpg"

    print("🧩 CarParts model:", carparts_weights)
    print("💥 Damage model  :", damage_weights)
    print("🖼️ Image        :", img_path)

    # Modelleri yükle
    carparts_model = YOLO(str(carparts_weights))
    damage_model = YOLO(str(damage_weights))

    # İnferans
    carparts_res = carparts_model.predict(
        source=str(img_path),
        imgsz=640,
        conf=0.05,
        verbose=False
    )[0]

    damage_res = damage_model.predict(
        source=str(img_path),
        imgsz=640,
        conf=0.05,   # dışarıdan aldığın görsellerde düşük conf işe yarıyordu
        verbose=False
    )[0]

    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]

    # Parçalar (segmentasyon modelinin boxes'larını kullanıyoruz)
    part_boxes = carparts_res.boxes.xyxy.cpu().numpy() if carparts_res.boxes is not None else np.zeros((0, 4))
    part_cls   = carparts_res.boxes.cls.cpu().numpy().astype(int) if carparts_res.boxes is not None else np.zeros((0,), int)
    part_names = carparts_model.names  # dict: id -> name

    # Hasarlar
    dmg_boxes = damage_res.boxes.xyxy.cpu().numpy() if damage_res.boxes is not None else np.zeros((0, 4))
    dmg_cls   = damage_res.boxes.cls.cpu().numpy().astype(int) if damage_res.boxes is not None else np.zeros((0,), int)
    dmg_conf  = damage_res.boxes.conf.cpu().numpy() if damage_res.boxes is not None else np.zeros((0,))
    dmg_names = damage_model.names

    merged_results = []

    # Her hasar kutusu için en iyi eşleşen parçayı bul
    for i, dmg_box in enumerate(dmg_boxes):
        best_iou = 0.0
        best_idx = -1
        best_inter = 0.0
        best_part_area = 0.0

        for j, part_box in enumerate(part_boxes):
            iou, inter_area, area_dmg, area_part = box_iou(dmg_box, part_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = j
                best_inter = inter_area
                best_part_area = area_part

        dmg_label = dmg_names[int(dmg_cls[i])]
        conf = float(dmg_conf[i])

        if best_idx >= 0 and best_iou > 0.1:  # eşik
            part_label = part_names[int(part_cls[best_idx])]
            damage_area = (dmg_box[2] - dmg_box[0]) * (dmg_box[3] - dmg_box[1])
            part_area = best_part_area
            area_ratio = float(damage_area / (part_area + 1e-6))
        else:
            part_label = None
            area_ratio = None

        merged_results.append({
            "damage_type": dmg_label,
            "confidence": round(conf, 3),
            "part": part_label,
            "iou_with_part": round(float(best_iou), 3),
            "damage_to_part_area_ratio": None if area_ratio is None else round(area_ratio, 3),
        })

    # 🖨️ Konsola özet bas
    print("\n==== Birleşik Hasar + Parça Analizi ====\n")
    for idx, item in enumerate(merged_results, start=1):
        print(f"{idx}. type={item['damage_type']:<10} "
              f"conf={item['confidence']:.2f}  "
              f"part={item['part'] or 'N/A':<18} "
              f"IoU={item['iou_with_part']:.2f}  "
              f"area_ratio={item['damage_to_part_area_ratio']}")

    # 🖼️ Görüntü üzerine çizim (sadece kutularla, maskesiz basit overlay)
    vis = img.copy()

    # Parça kutuları (yeşil)
    for box, cls_id in zip(part_boxes, part_cls):
        x1, y1, x2, y2 = box.astype(int)
        label = part_names[int(cls_id)]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, label, (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # Hasar kutuları (kırmızı) + eşleşen parça ismi
    for (box, cls_id, item) in zip(dmg_boxes, dmg_cls, merged_results):
        x1, y1, x2, y2 = box.astype(int)
        dmg_label = dmg_names[int(cls_id)]
        text = dmg_label
        if item["part"]:
            text += f" @ {item['part']}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(vis, text, (x1, min(y2 + 15, h - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    out_dir = this_dir / "runs" / "merge_demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "merge_result.jpg"
    cv2.imwrite(str(out_path), vis)
    print("\n📁 Birleşik görsel kaydedildi:", out_path)


if __name__ == "__main__":
    main()
