FROM ultralytics/ultralytics:latest-python

WORKDIR /

# RunPod SDK
RUN pip install --no-cache-dir runpod

# Model dosyalarını kopyala
COPY models/autodamage_best.pt /models/autodamage_best.pt
COPY models/carparts_seg_best.pt /models/carparts_seg_best.pt

# Handler kopyala
COPY runpod_serverless/handler.py /handler.py

# Başlat
CMD ["python", "-u", "/handler.py"]
