# AutoDamageIQ - Ürün Gereksinimleri Dokümanı (PRD)

## Proje Tanımı
Araç hasar tespiti ve analizi web uygulaması. YOLO modelleri ile hasar tespiti, parça segmentasyonu, SAM ile piksel hassasiyetinde alan hesaplama, VLM (GPT-4o) ile akıllı parça eşleme, MLOps pipeline (RunPod GPU eğitimi).

## Temel Gereksinimler
1. Web UI: Görsel yükleme, analiz sonuçları, callout çizgileri
2. Hasar tespiti: 6 sınıf (çatlak, göçük, cam kırığı, lamba kırığı, çizik, patlak lastik)
3. Parça segmentasyonu: 23 araç parçası
4. VLM Fallback: YOLO belirsiz kaldığında GPT-4o ile parça tespiti
5. Şiddet hesaplama: Çok değişkenli (alan, güven, tip)
6. Before/After karşılaştırma
7. PDF rapor indirme
8. Geçmiş analizler
9. MLOps: RunPod ile eğitim, model yönetimi

## Tech Stack
- **Frontend**: React, TailwindCSS, Framer Motion, Shadcn/UI
- **Backend**: FastAPI, PyTorch, Ultralytics YOLO, SAM ViT-B, OpenCV
- **Database**: MongoDB
- **Entegrasyonlar**: RunPod (GPU), OpenAI GPT-4o (Emergent LLM Key)

## Tamamlanan Özellikler

### Core
- [x] Görsel yükleme ve YOLO ile hasar tespiti
- [x] YOLOv8n-seg ile parça segmentasyonu
- [x] YOLOv8m ile geliştirilmiş hasar modeli (birleşik veri seti 13K+ görsel)
- [x] Hasar-parça eşleme (IoU tabanlı)
- [x] Çok değişkenli şiddet hesaplama
- [x] Onarım önerisi motoru
- [x] Before/After karşılaştırma (ORB matching)
- [x] PDF rapor indirme
- [x] Geçmiş analizler ve detay görüntüleme
- [x] Manuel inceleme kuyruğu

### Kalite & Anomali
- [x] Görüntü kalite değerlendirmesi (bulanıklık, parlaklık)
- [x] Anomali tespiti (pHash ile duplike algılama)

### MLOps
- [x] RunPod ile eğitim başlatma ve takip
- [x] Model yönetimi (listeleme, aktif yapma, silme)
- [x] VLM labeler (GPT-4o ile otomatik etiketleme)
- [x] Veri seti birleştirme (CarDD + VehiDE + HITL)

### Production Fix & VLM (Şubat 2026)
- [x] Graceful ML imports (production-safe, model dosyası yoksa çökmez)
- [x] Hardcoded RunPod API key temizliği
- [x] Health endpoint: model ve VLM durumu raporlama
- [x] VLM Fallback: Belirsiz parça eşleşmelerinde GPT-4o devreye girer
- [x] Frontend VLM badge gösterimi (mor "VLM" etiketi)
- [x] DRY: Paylaşılan sabitler modülü (constants.py)
- [x] datetime.utcnow() → datetime.now(timezone.utc)

## Bekleyen Görevler

### P1
- [ ] Annotation Tool: Yanlış tahminleri düzeltmek için bounding box editörü
- [ ] server.py refactoring (~1200 satır → APIRouter modülleri)

### P2
- [ ] Gelişmiş anomali tespiti (EXIF, ELA, visual-text consistency)

### P3
- [ ] İç mekan hasar analizi
- [ ] Domain adaptation panel eşleştirme
- [ ] Dark mode

## Mimari

```
/app/
├── backend/
│   ├── server.py              # Ana FastAPI uygulama
│   ├── constants.py           # Paylaşılan sabitler (DAMAGE_TR, PARTS_TR)
│   ├── vlm_parts_fallback.py  # GPT-4o VLM parça tespiti
│   ├── training_api.py        # RunPod eğitim API
│   ├── model_manager.py       # Model yönetimi
│   ├── sam_integration.py     # SAM ViT-B entegrasyonu
│   ├── before_after.py        # ORB karşılaştırma
│   ├── anomaly_detector.py    # pHash anomali
│   ├── image_quality.py       # Kalite değerlendirme
│   ├── repair_engine.py       # Onarım önerisi
│   └── vlm_labeler.py         # GPT-4o otomatik etiketleme
├── frontend/
│   └── src/pages/
│       ├── UploadPage.js
│       ├── ResultPage.js
│       ├── ComparePage.js
│       └── TrainingPage.js
└── models/                    # .pt model dosyaları (gitignored)
```
