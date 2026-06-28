"""
AutoDamageIQ - VLM Fallback & Analysis Backend Tests
Tests:
- /api/health returns ml_available, damage_model, parts_model, vlm_available
- /api/analyze returns damages with 'part_source' field
- VLM fallback enhances unmatched damages (part=null)
- vlm_enhanced count is reported
- analyses listing + detail retrieval (no MongoDB ObjectId leak)
- PDF download
- Quality check endpoint independent
"""
import os
import time
import pytest
import requests

BASE_URL = os.environ.get("REACT_APP_BACKEND_URL", "https://damage-vision.preview.emergentagent.com").rstrip("/")
ASSETS = "/app/assets"


@pytest.fixture(scope="module")
def session():
    s = requests.Session()
    return s


# -------------------------- Health --------------------------
class TestHealth:
    def test_health_fields(self, session):
        r = session.get(f"{BASE_URL}/api/health", timeout=15)
        assert r.status_code == 200, r.text
        data = r.json()
        for key in ["status", "ml_available", "damage_model", "parts_model", "vlm_available"]:
            assert key in data, f"Missing key in health: {key}"
        assert data["status"] == "healthy"
        assert isinstance(data["ml_available"], bool)
        assert isinstance(data["vlm_available"], bool)


# -------------------------- Analyze (YOLO + VLM Fallback) ----
def _post_image(session, image_path, timeout=120):
    with open(image_path, "rb") as f:
        files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
        return session.post(f"{BASE_URL}/api/analyze", files=files, timeout=timeout)


@pytest.fixture(scope="module")
def crash6_result(session):
    """crash_6.jpg expected to have unmatched damages -> triggers VLM"""
    path = f"{ASSETS}/crash_6.jpg"
    assert os.path.exists(path)
    r = _post_image(session, path, timeout=180)
    assert r.status_code == 200, f"Analyze failed: {r.status_code} {r.text[:300]}"
    data = r.json()
    return data


@pytest.fixture(scope="module")
def crash1_result(session):
    path = f"{ASSETS}/crash_1.jpg"
    r = _post_image(session, path, timeout=180)
    assert r.status_code == 200, f"Analyze failed: {r.status_code} {r.text[:300]}"
    return r.json()


class TestAnalyze:
    def test_response_structure(self, crash1_result):
        for k in ["id", "created_at", "image_base64", "results"]:
            assert k in crash1_result, f"Missing key: {k}"
        results = crash1_result["results"]
        for k in ["damages", "parts", "summary", "vlm_enhanced"]:
            assert k in results, f"Missing key in results: {k}"
        assert isinstance(results["vlm_enhanced"], int)
        assert results["vlm_enhanced"] >= 0

    def test_damages_have_part_source(self, crash1_result):
        damages = crash1_result["results"]["damages"]
        if not damages:
            pytest.skip("No damages detected on crash_1.jpg")
        for d in damages:
            assert "part_source" in d, f"damage missing part_source: {d.get('id')}"
            assert d["part_source"] in ("yolo", "vlm", "none"), f"Invalid part_source: {d['part_source']}"

    def test_vlm_fallback_on_crash6(self, crash6_result):
        """crash_6 is known to have unmatched damages -> VLM enhancement expected."""
        damages = crash6_result["results"]["damages"]
        results = crash6_result["results"]
        # part_source field present on all
        sources = [d.get("part_source") for d in damages]
        assert all(s in ("yolo", "vlm", "none") for s in sources), f"Bad sources: {sources}"
        # vlm_enhanced reflects count of vlm part_source
        vlm_count_actual = sum(1 for s in sources if s == "vlm")
        assert results["vlm_enhanced"] == vlm_count_actual, (
            f"vlm_enhanced={results['vlm_enhanced']} != actual={vlm_count_actual}"
        )
        print(f"crash_6 damages={len(damages)} part_sources={sources} vlm_enhanced={results['vlm_enhanced']}")

    def test_vlm_enhanced_damages_have_metadata(self, crash6_result):
        damages = crash6_result["results"]["damages"]
        vlm_damages = [d for d in damages if d.get("part_source") == "vlm"]
        if not vlm_damages:
            pytest.skip("No VLM-enhanced damages on crash_6 this run")
        for d in vlm_damages:
            assert d.get("part") is not None, "VLM-enhanced damage must have part set"
            assert d.get("part_tr") is not None, "part_tr should be set"
            assert "vlm_confidence" in d
            assert d["vlm_confidence"] >= 0.5


# -------------------------- Analyses listing / detail --------
class TestAnalysesPersistence:
    def test_list_analyses(self, session, crash1_result):
        r = session.get(f"{BASE_URL}/api/analyses?limit=10", timeout=30)
        assert r.status_code == 200, r.text
        data = r.json()
        assert isinstance(data, list)
        ids = [a.get("id") for a in data]
        assert crash1_result["id"] in ids, "Created analysis not found in list"

    def test_get_analysis_by_id(self, session, crash6_result):
        aid = crash6_result["id"]
        r = session.get(f"{BASE_URL}/api/analyses/{aid}", timeout=30)
        assert r.status_code == 200, r.text
        data = r.json()
        # No MongoDB _id leak
        assert "_id" not in data, "MongoDB _id leaked in response"
        assert data.get("id") == aid
        # VLM data preserved
        assert "vlm_enhanced" in data["results"]
        for d in data["results"]["damages"]:
            assert "part_source" in d

    def test_pdf_download(self, session, crash1_result):
        aid = crash1_result["id"]
        r = session.get(f"{BASE_URL}/api/analyses/{aid}/pdf", timeout=60)
        assert r.status_code == 200, f"PDF failed: {r.status_code} {r.text[:200]}"
        ct = r.headers.get("content-type", "")
        assert "pdf" in ct.lower(), f"Unexpected content-type: {ct}"
        assert r.content[:4] == b"%PDF", "Response is not a PDF"


# -------------------------- Quality Check --------------------
class TestQualityCheck:
    def test_quality_check_independent(self, session):
        path = f"{ASSETS}/crash_2.jpg"
        with open(path, "rb") as f:
            files = {"file": ("crash_2.jpg", f, "image/jpeg")}
            r = session.post(f"{BASE_URL}/api/quality-check", files=files, timeout=30)
        assert r.status_code == 200, r.text
        data = r.json()
        # Common keys expected from assess_image_quality
        assert isinstance(data, dict)
        assert any(k in data for k in ("warnings", "score", "quality_score", "is_acceptable", "blur_score"))
