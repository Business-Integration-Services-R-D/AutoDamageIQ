# AutoDamageIQ - RunPod Serverless Kurulum Rehberi

## Ön Koşullar
1. RunPod hesabınıza bakiye ekleyin: https://www.runpod.io/console/user/billing
2. Docker yüklü ve login durumunda olmalı (lokalde)

## Hızlı Kurulum (3 Adım)

### 1. Docker Image Build & Push
```bash
# Proje klasöründe çalıştırın
docker build -f runpod_serverless/Dockerfile -t DOCKERHUB_USER/autodamageiq-inference:latest .
docker push DOCKERHUB_USER/autodamageiq-inference:latest
```

### 2. Deploy Script
```bash
python deploy_runpod_serverless.py --docker-user DOCKERHUB_USER
```
Bu script otomatik olarak:
- RunPod template oluşturur
- Serverless endpoint oluşturur  
- `RUNPOD_ENDPOINT_ID`'yi backend/.env'e yazar

### 3. Production'a Deploy
- "Save to Github" yapın
- Yeniden deploy edin

## Alternatif: Manuel Kurulum
1. https://www.runpod.io/console/serverless adresine gidin
2. "New Endpoint" tıklayın
3. Docker Image: `DOCKERHUB_USER/autodamageiq-inference:latest`
4. GPU: RTX 4070 Ti (12GB) yeterli
5. Min Workers: 0, Max Workers: 1
6. Idle Timeout: 5 saniye
7. Endpoint ID'yi kopyalayın
8. `backend/.env`'e ekleyin: `RUNPOD_ENDPOINT_ID=endpoint_id_buraya`

## Nasıl Çalışır?
```
Kullanıcı → Upload Görsel → Backend (Hafif)
                                ↓
                    Lokal model var mı?
                   /                \
                Evet                Hayır
                 ↓                    ↓
           Lokal YOLO           RunPod Serverless
           (Preview)            (Production GPU)
                 ↓                    ↓
              Sonuç ← ← ← ← ← ← Sonuç
                 ↓
            VLM Fallback (belirsiz parçalar)
                 ↓
           MongoDB'ye Kaydet
```

## Dosya Yapısı
```
/app/
├── runpod_serverless/
│   ├── handler.py          # RunPod worker handler
│   └── Dockerfile          # Docker image tanımı
├── backend/
│   ├── runpod_inference.py  # RunPod API istemcisi
│   └── .env                 # RUNPOD_ENDPOINT_ID buraya
└── deploy_runpod_serverless.py  # Tek komut deploy script
```
