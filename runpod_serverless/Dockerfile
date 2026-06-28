FROM ultralytics/ultralytics:latest-python

WORKDIR /

# Gerekli kütüphaneler
RUN pip install --no-cache-dir runpod requests

# Models dizini oluştur (cold start'ta indirilecek)
RUN mkdir -p /models

# Handler kopyala
COPY runpod_serverless/handler.py /handler.py

# Başlat
CMD ["python", "-u", "/handler.py"]
