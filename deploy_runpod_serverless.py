"""
AutoDamageIQ - RunPod Serverless Deployment Script
====================================================
Bu script:
1. Docker image'ı build edip push eder (Docker gerekli)
2. RunPod serverless template + endpoint oluşturur
3. RUNPOD_ENDPOINT_ID'yi .env'e yazar

Kullanım:
  python deploy_runpod_serverless.py --docker-user YOUR_DOCKERHUB_USER
  
Gereksinimler:
  - Docker yüklü ve login durumunda
  - RunPod API key (.env'de)
  - Model dosyaları /app/models/ altında
"""

import os
import sys
import json
import subprocess
import requests
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / "backend" / ".env")

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
RUNPOD_GQL = "https://api.runpod.io/graphql"
APP_DIR = Path(__file__).parent


def gql(query: str):
    r = requests.post(RUNPOD_GQL, params={"api_key": RUNPOD_API_KEY},
                      headers={"Content-Type": "application/json"},
                      json={"query": query})
    return r.json()


def step1_build_docker(docker_user: str, tag: str = "latest"):
    """Docker image build & push"""
    image_name = f"{docker_user}/autodamageiq-inference:{tag}"
    print(f"\n[1/4] Docker image build: {image_name}")
    
    # Model dosyalarını kontrol et
    damage_model = APP_DIR / "models" / "autodamage_best.pt"
    parts_model = APP_DIR / "models" / "carparts_seg_best.pt"
    
    if not damage_model.exists():
        print(f"HATA: {damage_model} bulunamadi!")
        sys.exit(1)
    
    # Build
    cmd = f"docker build -f runpod_serverless/Dockerfile -t {image_name} ."
    print(f"  Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=str(APP_DIR))
    if result.returncode != 0:
        print("Docker build BASARISIZ!")
        sys.exit(1)
    
    # Push
    print(f"\n[2/4] Docker push: {image_name}")
    result = subprocess.run(f"docker push {image_name}", shell=True)
    if result.returncode != 0:
        print("Docker push BASARISIZ! 'docker login' yaptiginizdan emin olun.")
        sys.exit(1)
    
    return image_name


def step2_create_template(image_name: str) -> str:
    """RunPod serverless template olustur"""
    print(f"\n[3/4] RunPod template olusturuluyor...")
    
    mutation = '''
    mutation {
      saveTemplate(input: {
        name: "autodamageiq-inference-v2"
        imageName: "%s"
        dockerArgs: ""
        containerDiskInGb: 20
        volumeInGb: 0
        isServerless: true
        env: []
      }) {
        id
        name
      }
    }
    ''' % image_name
    
    result = gql(mutation)
    if "errors" in result:
        print(f"Template hatasi: {result['errors']}")
        sys.exit(1)
    
    template_id = result["data"]["saveTemplate"]["id"]
    print(f"  Template ID: {template_id}")
    return template_id


def step3_create_endpoint(template_id: str) -> str:
    """RunPod serverless endpoint olustur"""
    print(f"\n[4/4] RunPod endpoint olusturuluyor...")
    
    mutation = '''
    mutation {
      saveEndpoint(input: {
        name: "autodamageiq-yolo"
        templateId: "%s"
        gpuIds: "NVIDIA GeForce RTX 4070 Ti"
        workersMin: 0
        workersMax: 1
        idleTimeout: 5
        scalerType: "QUEUE_DELAY"
        scalerValue: 4
      }) {
        id
        name
      }
    }
    ''' % template_id
    
    result = gql(mutation)
    if "errors" in result:
        print(f"Endpoint hatasi: {result['errors']}")
        # Bakiye sorunu olabilir
        if "balance" in str(result["errors"]).lower():
            print("\n⚠️  RunPod hesabiniza bakiye eklemeniz gerekiyor!")
            print("   https://www.runpod.io/console/serverless")
        sys.exit(1)
    
    endpoint_id = result["data"]["saveEndpoint"]["id"]
    print(f"  Endpoint ID: {endpoint_id}")
    
    # .env'e yaz
    env_path = APP_DIR / "backend" / ".env"
    env_content = env_path.read_text()
    if "RUNPOD_ENDPOINT_ID" not in env_content:
        with open(env_path, "a") as f:
            f.write(f"\nRUNPOD_ENDPOINT_ID={endpoint_id}\n")
        print(f"  RUNPOD_ENDPOINT_ID={endpoint_id} -> backend/.env'e eklendi")
    else:
        # Güncelle
        lines = env_content.split("\n")
        updated = []
        for line in lines:
            if line.startswith("RUNPOD_ENDPOINT_ID"):
                updated.append(f"RUNPOD_ENDPOINT_ID={endpoint_id}")
            else:
                updated.append(line)
        env_path.write_text("\n".join(updated))
        print(f"  RUNPOD_ENDPOINT_ID güncellendi: {endpoint_id}")
    
    return endpoint_id


def main():
    parser = argparse.ArgumentParser(description="AutoDamageIQ RunPod Serverless Deploy")
    parser.add_argument("--docker-user", required=True, help="Docker Hub kullanici adi")
    parser.add_argument("--tag", default="latest", help="Docker image tag")
    parser.add_argument("--skip-docker", action="store_true", help="Docker build/push atla")
    parser.add_argument("--image", help="Mevcut Docker image kullan (skip-docker ile)")
    args = parser.parse_args()
    
    if not RUNPOD_API_KEY:
        print("HATA: RUNPOD_API_KEY .env'de tanimli degil!")
        sys.exit(1)
    
    print("=" * 60)
    print("AutoDamageIQ - RunPod Serverless Deployment")
    print("=" * 60)
    
    if args.skip_docker and args.image:
        image_name = args.image
    else:
        image_name = step1_build_docker(args.docker_user, args.tag)
    
    template_id = step2_create_template(image_name)
    endpoint_id = step3_create_endpoint(template_id)
    
    print("\n" + "=" * 60)
    print("✅ DEPLOY TAMAMLANDI!")
    print(f"   Endpoint ID: {endpoint_id}")
    print(f"   API URL: https://api.runpod.ai/v2/{endpoint_id}/runsync")
    print(f"   Health: https://api.runpod.ai/v2/{endpoint_id}/health")
    print("=" * 60)
    print("\nBackend'i yeniden baslatin: sudo supervisorctl restart backend")


if __name__ == "__main__":
    main()
