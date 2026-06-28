"""
RunPod Template Güncelleme Script
===================================
GitHub Actions image build'den sonra çalıştırılır.
Template'i yeni Docker image ile günceller.

Kullanım:
  python update_runpod_template.py ghcr.io/business-integration-services-r-d/autodamageiq/inference:latest
"""

import sys
import os
import requests
import json
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent / "backend" / ".env")

API_KEY = os.environ.get("RUNPOD_API_KEY")
TEMPLATE_ID = "wet74hz4jo"
EMERGENT_KEY = os.environ.get("EMERGENT_LLM_KEY", "")

def update_template(image_name: str):
    mutation = '''
    mutation {
      saveTemplate(input: {
        id: "%s"
        name: "autodamageiq-inference"
        imageName: "%s"
        dockerArgs: ""
        containerDiskInGb: 20
        volumeInGb: 0
        isServerless: true
        env: [
          { key: "EMERGENT_LLM_KEY", value: "%s" }
        ]
      }) {
        id
        name
        imageName
      }
    }
    ''' % (TEMPLATE_ID, image_name, EMERGENT_KEY)

    r = requests.post('https://api.runpod.io/graphql',
        params={'api_key': API_KEY},
        headers={'Content-Type': 'application/json'},
        json={'query': mutation}
    )
    result = r.json()
    
    if 'errors' in result:
        print(f"HATA: {result['errors']}")
        sys.exit(1)
    
    data = result['data']['saveTemplate']
    print(f"✅ Template güncellendi!")
    print(f"   ID: {data['id']}")
    print(f"   Image: {data['imageName']}")
    print(f"\n   RunPod endpoint artık yeni image'ı kullanacak.")
    print(f"   Yeni worker başladığında otomatik güncel image çekilir.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanım: python update_runpod_template.py IMAGE_URL")
        print("Örnek: python update_runpod_template.py ghcr.io/org/repo/inference:latest")
        sys.exit(1)
    
    update_template(sys.argv[1])
