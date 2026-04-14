# AutoDamageIQ — PRD (Product Requirements Document)

## Proje Tanımı
AutoDamageIQ, araç dış ve iç görüntülerini analiz ederek filo yönetimi ve kiralama operasyonlarında hasar tespiti ve karar destek süreçlerini otomatikleştiren görüntü tabanlı YZ platformudur.

## Teknoloji Yığını
- **Frontend:** React, Axios, React Router, TailwindCSS, Framer Motion
- **Backend:** FastAPI, PyTorch, Ultralytics YOLO, MongoDB
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
│   ├── training_api.py        # Eğitim ve etiketleme API'si
│   ├── model_manager.py       # Model yönetimi
│   └── sam_integration.py     # SAM entegrasyonu (placeholder)
├── frontend/src/
│   ├── pages/
│   │   ├── UploadPage.js      # Görsel yükleme
│   │   ├── ResultPage.js      # Analiz sonuçları (kalite, onarım, anomali)
│   │   ├── HistoryPage.js     # Geçmiş analizler
│   │   └── TrainingPage.js    # MLOps eğitim merkezi
├── models/                     # Eğitilmiş .pt model dosyaları
├── datasets/                   # Birleşik/özel veri setleri
└── training/                   # RunPod eğitim scriptleri
```

## Tamamlanan Özellikler

### Faz 1 — Çekirdek (Tamamlandı)
- [x] YOLOv8 ile 6 sınıflı hasar tespiti (crack, dent, glass_shatter, lamp_broken, scratch, tire_flat)
- [x] YOLOv8-Seg ile 23 sınıflı araç parçası segmentasyonu
- [x] IoU tabanlı hasar–parça uzamsal eşleme
- [x] Şiddet indeksi (1-5) ve risk seviyesi (Düşük/Orta/Yüksek)
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
- [x] **Görüntü kalite kontrolü:** Bulanıklık (Laplacian varyans), pozlama, çözünürlük, yansıma, kontrast metrikleri; kalite skoru (0-100), uyarı üretimi
- [x] **Onarım tipi önerisi:** Kural tabanlı hibrit karar motoru; hasar_tipi × şiddet × panel → lokal boya/düzeltme/değişim/cam değişimi/far değişimi/lastik değişimi; maliyet seviyesi
- [x] **Manuel inceleme kuyruğu:** Düşük güvenli veya yüksek riskli analizlerin otomatik işaretlenmesi; /api/review-queue endpoint'i; incelendi olarak işaretleme
- [x] **Gelişmiş çok değişkenli şiddet skoru:** Alan oranı + tür ağırlığı + güven faktörü birleşik skor; şiddet etiketi (Düşük/Orta/Yüksek)
- [x] **Anomali/tekrar görsel tespiti:** Perceptual hashing ile aynı fotoğraf tespiti; birleşik şüphe skoru; sinyal bazlı gerekçe raporlama

## API Endpoints
| Method | Endpoint | Açıklama |
|--------|----------|----------|
| POST | /api/analyze | Görsel yükle ve analiz et (kalite+hasar+onarım+anomali) |
| GET | /api/analyses | Geçmiş analizleri listele |
| GET | /api/analyses/{id} | Tek analiz detayı |
| DELETE | /api/analyses/{id} | Analiz sil |
| GET | /api/analyses/{id}/pdf | PDF rapor indir |
| POST | /api/quality-check | Sadece kalite kontrolü (analiz yapmadan) |
| GET | /api/review-queue | Manuel inceleme bekleyen analizler |
| POST | /api/analyses/{id}/review | İncelendi olarak işaretle |
| POST | /api/training/start | RunPod eğitim başlat |
| GET | /api/training/status/{job_id} | Eğitim durumu |
| GET | /api/models | Tüm modelleri listele |
| POST | /api/models/{id}/activate | Model aktif et |

## Bekleyen / Gelecek Görevler

### P1 — Sonraki Adımlar
- [ ] SAM (Segment Anything Model) entegrasyonu — piksel düzeyinde maske + alan hesaplaması
- [ ] Before/after yeni hasar karşılaştırma analizi
- [ ] Ar-Ge dokümanlarındaki Türkçe karakter düzeltmeleri (ö, ü, ş, ç, ğ, ı)

### P2 — Orta Öncelik
- [ ] server.py refactoring (modüler APIRouter yapısı)
- [ ] Araç içi hasar analizi (ayrı model/veri seti gerekli)
- [ ] Gelişmiş anomali (EXIF analizi, ELA, görsel-metin tutarlılığı)
- [ ] Tahmini onarım maliyeti hesaplama

### P3 — Düşük Öncelik
- [ ] İki analiz karşılaştırma modu
- [ ] Dark mode
- [ ] Kapsamlı test suite
- [ ] Görüş bağımsız panel temsili ve alan uyarlama

## Bilinen Kısıtlamalar
- SAM entegrasyonu placeholder — checkpoint indirilmeli
- .pt model dosyaları git'e commit edilmemeli (.gitignore konfigüre edildi)
- Risk seviyesi değerleri Türkçe özel karakter kullanmıyor (Dusuk/Orta/Yuksek)
