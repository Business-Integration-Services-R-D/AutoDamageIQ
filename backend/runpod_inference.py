"""
AutoDamageIQ - RunPod Serverless Inference Client
===================================================
Production'da lokal model yerine RunPod Serverless endpoint'i çağırır.
YOLO hasar + parça tespiti RunPod GPU üzerinde yapılır.
"""

import os
import base64
import logging
import asyncio
import requests
from typing import Dict, Any, Optional

logger = logging.getLogger("autodamageid.runpod_inference")

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
RUNPOD_ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID")


def is_runpod_available() -> bool:
    """RunPod serverless kullanılabilir mi?"""
    return bool(RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID)


def get_runpod_status() -> Dict[str, Any]:
    """RunPod endpoint durumunu kontrol et"""
    if not is_runpod_available():
        return {"available": False, "reason": "RUNPOD_API_KEY or RUNPOD_ENDPOINT_ID not set"}
    
    try:
        url = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/health"
        headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}
        r = requests.get(url, headers=headers, timeout=10)
        data = r.json()
        return {
            "available": True,
            "endpoint_id": RUNPOD_ENDPOINT_ID,
            "workers": data.get("workers", {}),
            "jobs": data.get("jobs", {})
        }
    except Exception as e:
        return {"available": False, "reason": str(e)}


async def run_inference_async(image_base64: str, timeout: int = 90) -> Dict[str, Any]:
    """
    RunPod serverless endpoint'inde hasar analizi çalıştır.
    
    Args:
        image_base64: JPEG/PNG görsel base64 string
        timeout: Maksimum bekleme süresi (saniye)
    
    Returns:
        Analiz sonuçları (damages, parts, summary)
    """
    if not is_runpod_available():
        raise RuntimeError("RunPod serverless not configured")
    
    url = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/runsync"
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "input": {
            "image_base64": image_base64
        }
    }
    
    # runsync çağrısı (senkron - cevap gelene kadar bekle)
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, lambda: requests.post(
        url, json=payload, headers=headers, timeout=timeout
    ))
    
    if response.status_code != 200:
        error_detail = response.text[:500]
        logger.error(f"RunPod API error {response.status_code}: {error_detail}")
        raise RuntimeError(f"RunPod inference failed: {response.status_code}")
    
    result = response.json()
    status = result.get("status")
    
    if status == "COMPLETED":
        output = result.get("output", {})
        if "error" in output:
            raise RuntimeError(f"RunPod worker error: {output['error']}")
        return output
    elif status == "FAILED":
        raise RuntimeError(f"RunPod job failed: {result.get('error', 'Unknown')}")
    elif status == "IN_QUEUE" or status == "IN_PROGRESS":
        # runsync zaman aşımına uğradı, async polling yap
        job_id = result.get("id")
        return await _poll_job(job_id, timeout)
    else:
        raise RuntimeError(f"Unexpected RunPod status: {status}")


async def _poll_job(job_id: str, timeout: int = 90) -> Dict[str, Any]:
    """Async job sonucunu poll et"""
    url = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/status/{job_id}"
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}
    
    loop = asyncio.get_event_loop()
    elapsed = 0
    interval = 2
    
    while elapsed < timeout:
        await asyncio.sleep(interval)
        elapsed += interval
        
        r = await loop.run_in_executor(None, lambda: requests.get(url, headers=headers, timeout=15))
        data = r.json()
        status = data.get("status")
        
        if status == "COMPLETED":
            output = data.get("output", {})
            if "error" in output:
                raise RuntimeError(f"RunPod worker error: {output['error']}")
            return output
        elif status == "FAILED":
            raise RuntimeError(f"RunPod job failed: {data.get('error', 'Unknown')}")
        
        logger.debug(f"RunPod job {job_id}: {status} ({elapsed}s)")
    
    raise TimeoutError(f"RunPod inference timed out after {timeout}s")
