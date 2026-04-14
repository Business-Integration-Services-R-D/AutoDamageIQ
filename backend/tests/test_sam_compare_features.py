"""
AutoDamageIQ Backend API Tests - SAM & Before/After Features
=============================================================
Tests for features 6 (SAM integration) and 7 (Before/After comparison):
1. SAM status endpoint
2. SAM mask data in analyze results
3. Before/After comparison via upload
4. Before/After comparison via existing analysis IDs
5. Comparisons list endpoint
"""

import pytest
import requests
import os
import time

BASE_URL = os.environ.get('REACT_APP_BACKEND_URL', '').rstrip('/')
TEST_IMAGE_PATH_1 = "/app/assets/crash_1.jpg"
TEST_IMAGE_PATH_2 = "/app/assets/crash_2.jpg"
TEST_IMAGE_PATH_3 = "/app/assets/crash_3.jpg"


class TestSAMStatusEndpoint:
    """SAM status endpoint tests"""
    
    def test_sam_status_returns_info(self):
        """Test /api/sam/status returns SAM model status"""
        response = requests.get(f"{BASE_URL}/api/sam/status")
        assert response.status_code == 200
        data = response.json()
        
        # Verify SAM status structure
        assert "library_installed" in data
        assert "checkpoint_exists" in data
        assert "checkpoint_path" in data
        assert "model_type" in data
        assert "model_loaded" in data
        assert "device" in data
        
        print(f"✓ SAM status endpoint works")
        print(f"  Library installed: {data['library_installed']}")
        print(f"  Checkpoint exists: {data['checkpoint_exists']}")
        print(f"  Model loaded: {data['model_loaded']}")
        print(f"  Device: {data['device']}")
        
        return data


class TestSAMInAnalyze:
    """SAM integration in analyze endpoint tests"""
    
    def test_analyze_returns_sam_available_flag(self):
        """Test /api/analyze returns sam_available flag"""
        with open(TEST_IMAGE_PATH_1, 'rb') as f:
            files = {'file': ('crash_1.jpg', f, 'image/jpeg')}
            response = requests.post(f"{BASE_URL}/api/analyze", files=files, timeout=180)
        
        assert response.status_code == 200
        data = response.json()
        results = data["results"]
        
        # Verify SAM availability flag
        assert "sam_available" in results
        assert "sam_used" in results
        
        print(f"✓ Analyze returns SAM flags - available: {results['sam_available']}, used: {results['sam_used']}")
        return data
    
    def test_analyze_returns_sam_data_for_damages(self):
        """Test /api/analyze returns sam_data for each damage when SAM is available"""
        with open(TEST_IMAGE_PATH_3, 'rb') as f:
            files = {'file': ('crash_3.jpg', f, 'image/jpeg')}
            response = requests.post(f"{BASE_URL}/api/analyze", files=files, timeout=180)
        
        assert response.status_code == 200
        data = response.json()
        results = data["results"]
        
        # Check if SAM was used
        if results.get("sam_used"):
            # Verify each damage has sam_data
            for damage in results["damages"]:
                assert "sam_data" in damage, f"Damage {damage['id']} missing sam_data"
                sam_data = damage["sam_data"]
                assert "available" in sam_data
                
                if sam_data["available"]:
                    assert "mask_score" in sam_data
                    assert "measurements" in sam_data
                    measurements = sam_data["measurements"]
                    assert "area_pixels" in measurements
                    assert "area_cm2" in measurements
                    assert "area_percentage" in measurements
                    assert "size_band" in measurements
                    
                    # Verify size_band is one of expected values
                    valid_size_bands = ["Cok Kucuk", "Kucuk", "Orta", "Buyuk", "Cok Buyuk"]
                    assert measurements["size_band"] in valid_size_bands
                    
            print(f"✓ Analyze returns SAM data for {len(results['damages'])} damages")
            for i, dmg in enumerate(results["damages"]):
                if dmg["sam_data"]["available"]:
                    m = dmg["sam_data"]["measurements"]
                    print(f"  Damage {i+1}: {dmg['type_tr']} - Area: {m['area_cm2']}cm², Size: {m['size_band']}")
        else:
            print("✓ SAM not used in this analysis (may not be loaded yet)")
        
        return data["id"]


class TestCompareUploadEndpoint:
    """Before/After comparison via upload endpoint tests"""
    
    def test_compare_upload_returns_comparison(self):
        """Test /api/compare/upload returns comparison result"""
        with open(TEST_IMAGE_PATH_1, 'rb') as before_f, open(TEST_IMAGE_PATH_2, 'rb') as after_f:
            files = {
                'before_file': ('before.jpg', before_f, 'image/jpeg'),
                'after_file': ('after.jpg', after_f, 'image/jpeg')
            }
            response = requests.post(f"{BASE_URL}/api/compare/upload", files=files, timeout=180)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify comparison result structure
        assert "has_new_damage" in data
        assert "verdict" in data
        assert "verdict_confidence" in data
        assert "new_damages" in data
        assert "new_damage_count" in data
        assert "alignment" in data
        assert "difference" in data
        assert "summary" in data
        
        # Verify before/after analysis summaries
        assert "before_analysis" in data
        assert "after_analysis" in data
        assert "damage_count" in data["before_analysis"]
        assert "damage_count" in data["after_analysis"]
        
        # Verify verdict confidence is valid
        assert data["verdict_confidence"] in ["Yuksek", "Orta", "Dusuk"]
        
        print(f"✓ Compare upload endpoint works")
        print(f"  Verdict: {data['verdict']}")
        print(f"  Confidence: {data['verdict_confidence']}")
        print(f"  Has new damage: {data['has_new_damage']}")
        print(f"  New damage count: {data['new_damage_count']}")
        print(f"  Before damages: {data['before_analysis']['damage_count']}")
        print(f"  After damages: {data['after_analysis']['damage_count']}")
        
        return data
    
    def test_compare_upload_returns_alignment_info(self):
        """Test /api/compare/upload returns alignment information"""
        with open(TEST_IMAGE_PATH_1, 'rb') as before_f, open(TEST_IMAGE_PATH_3, 'rb') as after_f:
            files = {
                'before_file': ('before.jpg', before_f, 'image/jpeg'),
                'after_file': ('after.jpg', after_f, 'image/jpeg')
            }
            response = requests.post(f"{BASE_URL}/api/compare/upload", files=files, timeout=180)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify alignment info
        alignment = data["alignment"]
        assert "used" in alignment
        assert "success" in alignment or "reason" in alignment
        
        # Verify difference stats
        difference = data["difference"]
        assert "change_percentage" in difference
        assert "changed_pixels" in difference
        
        print(f"✓ Compare upload returns alignment info")
        print(f"  Alignment used: {alignment.get('used', False)}")
        print(f"  Change percentage: {difference['change_percentage']}%")
        
        return data


class TestCompareByIDEndpoint:
    """Before/After comparison via existing analysis IDs tests"""
    
    @pytest.fixture(scope="class")
    def analysis_ids(self):
        """Create two analyses to use for comparison"""
        ids = []
        
        # Create first analysis
        with open(TEST_IMAGE_PATH_1, 'rb') as f:
            files = {'file': ('before.jpg', f, 'image/jpeg')}
            response = requests.post(f"{BASE_URL}/api/analyze", files=files, timeout=180)
        assert response.status_code == 200
        ids.append(response.json()["id"])
        
        # Create second analysis
        with open(TEST_IMAGE_PATH_2, 'rb') as f:
            files = {'file': ('after.jpg', f, 'image/jpeg')}
            response = requests.post(f"{BASE_URL}/api/analyze", files=files, timeout=180)
        assert response.status_code == 200
        ids.append(response.json()["id"])
        
        return ids
    
    def test_compare_by_id_returns_comparison(self, analysis_ids):
        """Test /api/compare returns comparison result for existing analyses"""
        before_id, after_id = analysis_ids
        
        response = requests.post(
            f"{BASE_URL}/api/compare",
            json={"before_id": before_id, "after_id": after_id},
            timeout=180
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify comparison result structure
        assert "id" in data  # Comparison ID
        assert "before_id" in data
        assert "after_id" in data
        assert "has_new_damage" in data
        assert "verdict" in data
        assert "verdict_confidence" in data
        assert "new_damages" in data
        assert "alignment" in data
        assert "difference" in data
        assert "summary" in data
        
        # Verify IDs match
        assert data["before_id"] == before_id
        assert data["after_id"] == after_id
        
        print(f"✓ Compare by ID endpoint works")
        print(f"  Comparison ID: {data['id'][:8]}")
        print(f"  Verdict: {data['verdict']}")
        print(f"  Has new damage: {data['has_new_damage']}")
        
        return data
    
    def test_compare_by_id_not_found_before(self):
        """Test /api/compare returns 404 for non-existent before analysis"""
        response = requests.post(
            f"{BASE_URL}/api/compare",
            json={"before_id": "non-existent-id", "after_id": "some-id"},
            timeout=30
        )
        
        assert response.status_code == 404
        print("✓ Compare returns 404 for non-existent before analysis")
    
    def test_compare_by_id_not_found_after(self, analysis_ids):
        """Test /api/compare returns 404 for non-existent after analysis"""
        before_id = analysis_ids[0]
        
        response = requests.post(
            f"{BASE_URL}/api/compare",
            json={"before_id": before_id, "after_id": "non-existent-id"},
            timeout=30
        )
        
        assert response.status_code == 404
        print("✓ Compare returns 404 for non-existent after analysis")


class TestComparisonsListEndpoint:
    """Comparisons list endpoint tests"""
    
    def test_comparisons_list_returns_array(self):
        """Test /api/comparisons returns list of past comparisons"""
        response = requests.get(f"{BASE_URL}/api/comparisons")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        
        # If there are comparisons, verify structure
        if data:
            item = data[0]
            assert "id" in item
            assert "before_id" in item
            assert "after_id" in item
            assert "created_at" in item
            assert "has_new_damage" in item
            assert "verdict" in item
            assert "new_damage_count" in item
            
            print(f"✓ Comparisons list returns {len(data)} items")
            print(f"  Latest: {item['verdict']} (new damages: {item['new_damage_count']})")
        else:
            print("✓ Comparisons list is empty (no comparisons yet)")
    
    def test_comparisons_list_with_limit(self):
        """Test /api/comparisons respects limit parameter"""
        response = requests.get(f"{BASE_URL}/api/comparisons?limit=5")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) <= 5
        
        print(f"✓ Comparisons list respects limit - returned {len(data)} items")


class TestExistingFeaturesStillWork:
    """Verify existing features still work after SAM/Compare additions"""
    
    def test_quality_check_still_works(self):
        """Test /api/quality-check still works"""
        with open(TEST_IMAGE_PATH_1, 'rb') as f:
            files = {'file': ('test.jpg', f, 'image/jpeg')}
            response = requests.post(f"{BASE_URL}/api/quality-check", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert "quality_score" in data
        assert "quality_level" in data
        
        print(f"✓ Quality check still works - Level: {data['quality_level']}")
    
    def test_review_queue_still_works(self):
        """Test /api/review-queue still works"""
        response = requests.get(f"{BASE_URL}/api/review-queue")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        
        print(f"✓ Review queue still works - {len(data)} items")
    
    def test_analyze_still_returns_quality_anomaly_repair(self):
        """Test /api/analyze still returns quality, anomaly, and repair data"""
        with open(TEST_IMAGE_PATH_1, 'rb') as f:
            files = {'file': ('test.jpg', f, 'image/jpeg')}
            response = requests.post(f"{BASE_URL}/api/analyze", files=files, timeout=180)
        
        assert response.status_code == 200
        data = response.json()
        results = data["results"]
        
        # Verify quality data
        assert "quality" in results
        assert "quality_score" in results["quality"]
        
        # Verify anomaly data
        assert "anomaly" in results
        assert "anomaly_score" in results["anomaly"]
        
        # Verify repair data in damages
        if results["damages"]:
            for damage in results["damages"]:
                assert "repair" in damage
                assert "severity_details" in damage
        
        print("✓ Analyze still returns quality, anomaly, and repair data")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
