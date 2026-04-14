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
├── frontend/src/
│   ├── pages/
│   │   ├── UploadPage.js      # Görsel yükleme
│   │   ├── ResultPage.js      # Analiz sonuçları (kalite, onarım, SAM, anomali)
│   │   ├── ComparePage.js     # Before/After karşılaştırma sayfası
│   │   ├── HistoryPage.js     # Geçmiş analizler
│   │   └── TrainingPage.js    # MLOps eğitim merkezi
├── models/
│   ├── autodamage_best.pt     # Hasar tespiti modeli
│   ├── carparts_seg_best.pt   # Parça segmentasyon modeli
│   └── sam_vit_b_01ec64.pth   # SAM ViT-B checkpoint
├── datasets/                   # Birleşik/özel veri setleri
└── training/                   # RunPod eğitim scriptleri
```

## Tamamlanan Özellikler

### Faz 1 — Çekirdek (Tamamlandı)
- [x] YOLOv8 ile 6 sınıflı hasar tespiti (crack, dent, glass_shatter, lamp_broken, scratch, tire_flat)
- [x] YOLOv8-Seg ile 23 sınıflı araç parçası segmentasyonu
- [x] IoU tabanlı hasar–parça uzamsal eşleme
- [x] Şiddet indeksi (1-5) ve risk seviyesi
- [x] PDF rapor üretimi
- [x] JSON API ve MongoDB persistence
- [x] Web arayüzü (yükleme, sonuç, geçmiş, callout lines)

### Faz 2 — MLOps (Tamamlandı)
- [x] RunPod GPU üzerinde otomatik eğitim pipeline'ı
- [x] Veri seti yönetimi (birleştirme, indirme)
- [x] Model versiyonlama ve aktif model değiştirme
- [x] Etiketleme aracı (canvas bounding box + YOLO format kayıt)
- [x] Özel hasar modeli eğitimi (50 epoch) ve entegrasyonu
- [x] Özel parça segmentasyon modeli eğitimi (100 epoch) ve entegrasyonu

### Faz 3 — Ar-Ge Destekleyici Modüller (Tamamlandı — 14 Nisan 2026)
- [x] **Görüntü kalite kontrolü:** Bulanıklık, pozlama, çözünürlük, yansıma, kontrast metrikleri
- [x] **Onarım tipi önerisi:** Kural tabanlı hibrit karar motoru
- [x] **Manuel inceleme kuyruğu:** Düşük güvenli analizlerin otomatik işaretlenmesi
- [x] **Gelişmiş çok değişkenli şiddet skoru:** Alan oranı + tür ağırlığı + güven faktörü
- [x] **Anomali/tekrar görsel tespiti:** Perceptual hashing ile duplicate detection

### Faz 4 — SAM + Before/After (Tamamlandı — 14 Nisan 2026)
- [x] **SAM Entegrasyonu:** SAM ViT-B modeli ile YOLO bounding box'larından piksel düzeyinde hassas maske üretimi; alan hesaplaması (piksel, tahmini cm², yüzde); boyut bandı sınıflandırması
- [x] **Before/After Karşılaştırma:** ORB feature matching + homography ile görsel hizalama; fark haritası ve değişim bölgeleri tespiti; yeni hasar ayrıştırma ve kanıt gücü değerlendirmesi
- [x] **Karşılaştırma Sayfası (Frontend):** Görsel yükleme veya mevcut analizden seçme; sonuç kartları, yeni hasar listesi, teknik detaylar

## API Endpoints
| Method | Endpoint | Açıklama |
|--------|----------|----------|
| POST | /api/analyze | Görsel analizi (hasar+parça+kalite+SAM+onarım+anomali) |
| GET | /api/analyses | Geçmiş analizler |
| GET | /api/analyses/{id} | Tek analiz detayı |
| DELETE | /api/analyses/{id} | Analiz sil |
| GET | /api/analyses/{id}/pdf | PDF rapor |
| POST | /api/quality-check | Sadece kalite kontrolü |
| GET | /api/review-queue | Manuel inceleme kuyruğu |
| POST | /api/analyses/{id}/review | İncelendi işaretle |
| GET | /api/sam/status | SAM model durumu |
| POST | /api/compare | Analiz ID'leriyle karşılaştırma |
| POST | /api/compare/upload | Görsel yükleyerek karşılaştırma |
| GET | /api/comparisons | Geçmiş karşılaştırmalar |
| POST | /api/training/start | RunPod eğitim başlat |
| GET | /api/training/status/{job_id} | Eğitim durumu |
| GET | /api/models | Tüm modeller |
| POST | /api/models/{id}/activate | Model aktif et |

## Bekleyen / Gelecek Görevler

### P1 — Sonraki Adımlar
- [ ] Ar-Ge dokümanlarındaki tutarsızlıkların düzeltilmesi (hasar sınıf isimleri, literatür taraması)

### P2 — Orta Öncelik
- [ ] server.py modüler refactoring (APIRouter yapısı)
- [ ] Araç içi hasar analizi (ayrı model/veri seti gerekli)
- [ ] Gelişmiş anomali (EXIF analizi, ELA, görsel-metin tutarlılığı)
- [ ] Tahmini onarım maliyeti hesaplama

### P3 — Düşük Öncelik
- [ ] Dark mode
- [ ] Kapsamlı test suite
- [ ] Görüş bağımsız panel temsili ve alan uyarlama
