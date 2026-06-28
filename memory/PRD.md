# AutoDamageIQ - Ürün Gereksinimleri Dokümanı (PRD)

## Proje Tanımı
Araç hasar tespiti ve analizi web uygulaması. YOLO modelleri ile hasar tespiti, parça segmentasyonu, SAM ile piksel hassasiyetinde alan hesaplama, VLM (GPT-4o) ile akıllı parça eşleme, RunPod Serverless GPU inference, MLOps pipeline.

## Temel Gereksinimler
1. Web UI: Görsel yükleme, analiz sonuçları, callout çizgileri
2. Hasar tespiti: 6 sınıf (çatlak, göçük, cam kırığı, lamba kırığı, çizik, patlak lastik)
3. Parça segmentasyonu: 23 araç parçası
4. VLM Fallback: YOLO belirsiz kaldığında GPT-4o ile parça tespiti
5. RunPod Serverless: Production'da GPU inference
6. Şiddet hesaplama: Çok değişkenli (alan, güven, tip)
7. Before/After karşılaştırma
8. PDF rapor indirme
9. Geçmiş analizler
10. MLOps: RunPod ile eğitim, model yönetimi

## Tech Stack
- **Frontend**: React, TailwindCSS, Framer Motion, Shadcn/UI
- **Backend**: FastAPI, PyTorch (lokal), OpenCV
- **Inference**: Lokal YOLO (preview) / RunPod Serverless (production)
- **Database**: MongoDB
- **Entegrasyonlar**: RunPod Serverless (GPU), OpenAI GPT-4o (Emergent LLM Key)

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
- [x] Graceful ML imports (production-safe)
- [x] Hardcoded secret temizliği
- [x] VLM Fallback (GPT-4o ile belirsiz parça tespiti)
- [x] Frontend VLM badge gösterimi
- [x] DRY: constants.py
- [x] Git history temizliği (.pt/.pth dosyaları)

### RunPod Serverless (Şubat 2026)
- [x] RunPod serverless handler (handler.py)
- [x] Dockerfile (ultralytics base + models)
- [x] Backend inference client (runpod_inference.py)
- [x] Otomatik inference modu seçimi (lokal/RunPod)
- [x] Deploy script (deploy_runpod_serverless.py)
- [ ] Docker image build & push (kullanıcı yapacak)
- [ ] RunPod endpoint oluşturma (bakiye gerekli)

## Bekleyen Görevler

### P0
- [ ] RunPod bakiye ekleme + endpoint oluşturma + Docker push

### P1
- [ ] Annotation Tool: Bounding box editörü
- [ ] server.py refactoring → APIRouter modülleri

### P2
- [ ] Gelişmiş anomali tespiti (EXIF, ELA)

### P3
- [ ] İç mekan hasar analizi
- [ ] Dark mode

## Mimari
```
/app/
├── backend/
│   ├── server.py              # Ana FastAPI (lokal + RunPod inference)
│   ├── constants.py           # Paylaşılan sabitler
│   ├── vlm_parts_fallback.py  # GPT-4o VLM parça tespiti
│   ├── runpod_inference.py    # RunPod Serverless API client
│   ├── training_api.py        # RunPod eğitim API
│   ├── model_manager.py       # Model yönetimi
│   ├── sam_integration.py     # SAM ViT-B entegrasyonu
│   ├── before_after.py        # ORB karşılaştırma
│   ├── anomaly_detector.py    # pHash anomali
│   ├── image_quality.py       # Kalite değerlendirme
│   └── repair_engine.py       # Onarım önerisi
├── frontend/src/pages/
├── runpod_serverless/
│   ├── handler.py             # RunPod worker
│   ├── Dockerfile             # GPU inference image
│   └── README.md              # Kurulum rehberi
└── deploy_runpod_serverless.py
```
