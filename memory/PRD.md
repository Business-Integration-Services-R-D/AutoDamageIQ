# AutoDamageIQ — PRD (Product Requirements Document)

## Proje Tanımı
AutoDamageIQ, araç dış ve iç görüntülerini analiz ederek filo yönetimi ve kiralama operasyonlarında hasar tespiti ve karar destek süreçlerini otomatikleştiren görüntü tabanlı YZ platformudur.

## Teknoloji Yığını
- **Frontend:** React, Axios, React Router, TailwindCSS, Framer Motion
- **Backend:** FastAPI, PyTorch, Ultralytics YOLO, SAM (Segment Anything), MongoDB
- **MLOps:** RunPod GPU API, otomatik eğitim scriptleri
- **Dil:** Türkçe UI

## Çekirdek Mimari
```
/app/
├── backend/
│   ├── server.py              # Ana FastAPI uygulaması
│   ├── image_quality.py       # Görüntü kalite kontrol modülü
│   ├── repair_engine.py       # Onarım tipi öneri motoru
│   ├── anomaly_detector.py    # Anomali/tekrar görsel tespit modülü
│   ├── sam_integration.py     # SAM piksel düzeyinde maske üretimi
│   ├── before_after.py        # Before/After yeni hasar karşılaştırma
│   ├── training_api.py        # Eğitim ve etiketleme API'si
│   └── model_manager.py       # Model yönetimi
├── frontend/src/pages/
│   ├── UploadPage.js          # Görsel yükleme
│   ├── ResultPage.js          # Analiz sonuçları
│   ├── ComparePage.js         # Before/After karşılaştırma
│   ├── HistoryPage.js         # Geçmiş analizler
│   └── TrainingPage.js        # MLOps eğitim merkezi
├── models/                     # Eğitilmiş .pt model dosyaları
├── datasets/
│   ├── unified/               # Birleşik eğitim veri seti (5370 görsel)
│   └── scripts/
│       ├── download_datasets.py
│       └── merge_vehide.py    # VehiDE veri seti birleştirme
├── training/
│   ├── train_script.sh        # v8: YOLOv8m + AdamW + early stopping
│   ├── train_carparts_seg.sh  # Parça segmentasyon eğitimi
│   └── dataset.tar.gz         # Paketlenmiş eğitim verisi
```

## Tamamlanan Özellikler

### Faz 1 — Çekirdek
- [x] YOLOv8 hasar tespiti (6 sınıf) + YOLOv8-Seg parça segmentasyonu (23 sınıf)
- [x] IoU tabanlı hasar–parça eşleme, şiddet indeksi, risk seviyesi
- [x] PDF rapor, JSON API, MongoDB persistence, web arayüzü

### Faz 2 — MLOps
- [x] RunPod GPU eğitim pipeline'ı, model versiyonlama, etiketleme aracı
- [x] Özel hasar modeli (50 epoch) ve parça segmentasyon modeli (100 epoch) eğitimi

### Faz 3 — Ar-Ge Modüller (14 Nisan 2026)
- [x] Görüntü kalite kontrolü (bulanıklık, pozlama, çözünürlük, yansıma, kontrast)
- [x] Onarım tipi önerisi (6×5 hasar-şiddet matrisi, panel maliyet çarpanı)
- [x] Manuel inceleme kuyruğu (düşük güven otomatik işaretleme)
- [x] Gelişmiş çok değişkenli şiddet skoru
- [x] Anomali/tekrar tespiti (perceptual hashing, piksel dağılım analizi)

### Faz 4 — SAM + Before/After (14 Nisan 2026)
- [x] SAM ViT-B entegrasyonu (panel bazlı kalibrasyon, hasar tipi düzeltme faktörleri)
- [x] Before/After karşılaştırma (ORB + homography + fark haritası)
- [x] Karşılaştırma sayfası (frontend)

### Faz 5 — Model İyileştirme Altyapısı (15 Nisan 2026)
- [x] Eğitim scripti v8: YOLOv8n → YOLOv8m (25M param), AdamW, early stopping (patience=25)
- [x] Agresif augmentation: mosaic, mixup, HSV jitter, scale, cosine LR
- [x] GPU belleğine göre otomatik batch size ayarlama
- [x] VehiDE veri seti birleştirme scripti (Kaggle -> YOLO format dönüşümü)
- [x] Varsayılan eğitim konfigürasyonu güncelleme (frontend + backend)

## Bekleyen / Gelecek Görevler

### P0 — Aktif (RunPod'da eğitim başlatılacak)
- [ ] YOLOv8m ile yeni model eğitimi (mevcut veri setiyle)
- [ ] VehiDE veri setini Kaggle'dan indirip birleştirme (11K+ görsel)
- [ ] Birleşik veri setiyle ikinci eğitim turu

### P1 — Sonraki Adımlar
- [ ] Ar-Ge dokümanlarındaki tutarsızlıklar (Makale 5 tarihi: 2019→2022)
- [ ] YOLO11m-seg'e geçiş değerlendirmesi (makaledeki model)
- [ ] Humans in the Loop veri seti ile parça modeli iyileştirme

### P2 — Orta Öncelik
- [ ] server.py modüler refactoring (APIRouter)
- [ ] Araç içi hasar analizi
- [ ] Gelişmiş anomali (EXIF, ELA)
- [ ] Tahmini onarım maliyeti

### P3 — Düşük Öncelik
- [ ] Dark mode
- [ ] Kapsamlı test suite
- [ ] TrainingPage.js component refactoring
