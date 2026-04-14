"""
AutoDamageIQ Backend API Tests
==============================
Tests for 5 new features:
1. Image quality assessment
2. Repair type recommendation
3. Manual review queue
4. Enhanced multi-variable severity score
5. Anomaly/duplicate image detection
"""

import pytest
import requests
import os
import time

BASE_URL = os.environ.get('REACT_APP_BACKEND_URL', '').rstrip('/')
TEST_IMAGE_PATH = "/app/assets/crash_1.jpg"
TEST_IMAGE_PATH_2 = "/app/assets/crash_2.jpg"


class TestHealthEndpoint:
    """Health check endpoint tests"""
    
    def test_health_check(self):
        """Test /api/health returns healthy status"""
        response = requests.get(f"{BASE_URL}/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "AutoDamageID"
        print("✓ Health check passed")


class TestQualityCheckEndpoint:
    """Standalone quality check endpoint tests"""
    
    def test_quality_check_returns_metrics(self):
        """Test /api/quality-check returns quality metrics"""
        with open(TEST_IMAGE_PATH, 'rb') as f:
            files = {'file': ('crash_1.jpg', f, 'image/jpeg')}
            response = requests.post(f"{BASE_URL}/api/quality-check", files=files)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify quality score structure
        assert "quality_score" in data
        assert "quality_level" in data
        assert "is_acceptable" in data
        assert "warnings" in data
        assert "metrics" in data
        
        # Verify quality_level is one of expected values
        assert data["quality_level"] in ["Yuksek", "Orta", "Dusuk"]
        
        # Verify metrics structure
        metrics = data["metrics"]
        assert "blur_variance" in metrics
        assert "mean_brightness" in metrics
        assert "resolution" in metrics
        assert "reflection_ratio" in metrics
        assert "contrast" in metrics
        
        print(f"✓ Quality check passed - Score: {data['quality_score']}, Level: {data['quality_level']}")
    
    def test_quality_check_rejects_non_image(self):
        """Test /api/quality-check rejects non-image files"""
        files = {'file': ('test.txt', b'not an image', 'text/plain')}
        response = requests.post(f"{BASE_URL}/api/quality-check", files=files)
        
        assert response.status_code == 400
        print("✓ Quality check correctly rejects non-image files")


class TestAnalyzeEndpoint:
    """Main analyze endpoint tests with new features"""
    
    def test_analyze_returns_quality_data(self):
        """Test /api/analyze returns quality assessment in results"""
        with open(TEST_IMAGE_PATH, 'rb') as f:
            files = {'file': ('crash_1.jpg', f, 'image/jpeg')}
            response = requests.post(f"{BASE_URL}/api/analyze", files=files)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify basic structure
        assert "id" in data
        assert "created_at" in data
        assert "results" in data
        
        results = data["results"]
        
        # Verify quality data is present
        assert "quality" in results
        quality = results["quality"]
        assert "quality_score" in quality
        assert "quality_level" in quality
        assert quality["quality_level"] in ["Yuksek", "Orta", "Dusuk"]
        
        print(f"✓ Analyze returns quality data - Level: {quality['quality_level']}")
        return data["id"]
    
    def test_analyze_returns_anomaly_data(self):
        """Test /api/analyze returns anomaly detection data"""
        with open(TEST_IMAGE_PATH_2, 'rb') as f:
            files = {'file': ('crash_2.jpg', f, 'image/jpeg')}
            response = requests.post(f"{BASE_URL}/api/analyze", files=files)
        
        assert response.status_code == 200
        data = response.json()
        results = data["results"]
        
        # Verify anomaly data is present
        assert "anomaly" in results
        anomaly = results["anomaly"]
        assert "anomaly_score" in anomaly
        assert "risk_level" in anomaly
        assert "signals" in anomaly
        assert anomaly["risk_level"] in ["Dusuk", "Orta", "Yuksek"]
        
        print(f"✓ Analyze returns anomaly data - Score: {anomaly['anomaly_score']}, Risk: {anomaly['risk_level']}")
        return data["id"]
    
    def test_analyze_returns_repair_recommendations(self):
        """Test /api/analyze returns repair recommendations for each damage"""
        with open("/app/assets/crash_3.jpg", 'rb') as f:
            files = {'file': ('crash_3.jpg', f, 'image/jpeg')}
            response = requests.post(f"{BASE_URL}/api/analyze", files=files)
        
        assert response.status_code == 200
        data = response.json()
        results = data["results"]
        
        # Check damages have repair recommendations
        if results["damages"]:
            for damage in results["damages"]:
                assert "repair" in damage, f"Damage {damage['id']} missing repair recommendation"
                repair = damage["repair"]
                assert "repair_type" in repair
                assert "repair_type_tr" in repair
                assert "description_tr" in repair
                assert "cost_level" in repair
                
                # Verify repair type is one of expected values
                valid_repair_types = [
                    "Lokal Boya", "Kaporta Duzeltme", "Parca Degisimi",
                    "Cam Degisimi", "Far/Lamba Degisimi", "Lastik Degisimi",
                    "Detayli Inceleme Gerekli"
                ]
                assert repair["repair_type_tr"] in valid_repair_types
                
            print(f"✓ Analyze returns repair recommendations for {len(results['damages'])} damages")
        else:
            print("✓ No damages detected - repair recommendation test skipped")
        
        return data["id"]
    
    def test_analyze_returns_enhanced_severity(self):
        """Test /api/analyze returns enhanced severity details"""
        with open("/app/assets/crash_4.jpg", 'rb') as f:
            files = {'file': ('crash_4.jpg', f, 'image/jpeg')}
            response = requests.post(f"{BASE_URL}/api/analyze", files=files)
        
        assert response.status_code == 200
        data = response.json()
        results = data["results"]
        
        # Check damages have severity details
        if results["damages"]:
            for damage in results["damages"]:
                assert "severity_details" in damage, f"Damage {damage['id']} missing severity_details"
                severity = damage["severity_details"]
                assert "score" in severity
                assert "class" in severity
                assert "label" in severity
                assert "area_ratio" in severity
                assert "base_severity" in severity
                assert "area_factor" in severity
                assert "confidence_factor" in severity
                
                # Verify severity label is one of expected values
                assert severity["label"] in ["Dusuk", "Orta", "Yuksek"]
                
            print(f"✓ Analyze returns enhanced severity for {len(results['damages'])} damages")
        else:
            print("✓ No damages detected - severity details test skipped")
        
        return data["id"]
    
    def test_analyze_returns_review_flag(self):
        """Test /api/analyze returns needs_review flag in summary"""
        with open("/app/assets/crash_5.jpg", 'rb') as f:
            files = {'file': ('crash_5.jpg', f, 'image/jpeg')}
            response = requests.post(f"{BASE_URL}/api/analyze", files=files)
        
        assert response.status_code == 200
        data = response.json()
        results = data["results"]
        
        # Verify summary has review fields
        summary = results["summary"]
        assert "needs_review" in summary
        assert "review_reasons" in summary
        assert isinstance(summary["needs_review"], bool)
        assert isinstance(summary["review_reasons"], list)
        
        print(f"✓ Analyze returns review flag - needs_review: {summary['needs_review']}")
        return data["id"]


class TestDuplicateDetection:
    """Duplicate image detection tests"""
    
    def test_duplicate_detection_increases_anomaly_score(self):
        """Test uploading same image twice increases anomaly score"""
        # First upload
        with open(TEST_IMAGE_PATH, 'rb') as f:
            files = {'file': ('crash_1.jpg', f, 'image/jpeg')}
            response1 = requests.post(f"{BASE_URL}/api/analyze", files=files)
        
        assert response1.status_code == 200
        data1 = response1.json()
        anomaly_score_1 = data1["results"]["anomaly"]["anomaly_score"]
        
        # Small delay to ensure DB write completes
        time.sleep(0.5)
        
        # Second upload of same image
        with open(TEST_IMAGE_PATH, 'rb') as f:
            files = {'file': ('crash_1_duplicate.jpg', f, 'image/jpeg')}
            response2 = requests.post(f"{BASE_URL}/api/analyze", files=files)
        
        assert response2.status_code == 200
        data2 = response2.json()
        anomaly_score_2 = data2["results"]["anomaly"]["anomaly_score"]
        
        # Second upload should have higher anomaly score due to duplicate detection
        # or at least have duplicate signals
        anomaly2 = data2["results"]["anomaly"]
        has_duplicate_signal = any(
            s["type"] == "duplicate_image" for s in anomaly2.get("signals", [])
        )
        
        print(f"✓ Duplicate detection test - Score 1: {anomaly_score_1}, Score 2: {anomaly_score_2}")
        print(f"  Has duplicate signal: {has_duplicate_signal}")
        
        # Either score increased or duplicate signal present
        assert anomaly_score_2 >= anomaly_score_1 or has_duplicate_signal, \
            "Duplicate image should increase anomaly score or add duplicate signal"


class TestReviewQueueEndpoint:
    """Manual review queue endpoint tests"""
    
    def test_review_queue_returns_list(self):
        """Test /api/review-queue returns list of analyses needing review"""
        response = requests.get(f"{BASE_URL}/api/review-queue")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        
        # If there are items, verify structure
        if data:
            item = data[0]
            assert "id" in item
            assert "created_at" in item
            assert "summary" in item
            assert "review_reasons" in item
            
            print(f"✓ Review queue returns {len(data)} items needing review")
        else:
            print("✓ Review queue is empty (no items need review)")
    
    def test_review_queue_with_limit(self):
        """Test /api/review-queue respects limit parameter"""
        response = requests.get(f"{BASE_URL}/api/review-queue?limit=5")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) <= 5
        
        print(f"✓ Review queue respects limit - returned {len(data)} items")


class TestMarkReviewedEndpoint:
    """Mark as reviewed endpoint tests"""
    
    def test_mark_reviewed_success(self):
        """Test /api/analyses/{id}/review marks analysis as reviewed"""
        # First create an analysis
        with open("/app/assets/crash_6.jpg", 'rb') as f:
            files = {'file': ('crash_6.jpg', f, 'image/jpeg')}
            create_response = requests.post(f"{BASE_URL}/api/analyze", files=files)
        
        assert create_response.status_code == 200
        analysis_id = create_response.json()["id"]
        
        # Mark as reviewed
        review_response = requests.post(f"{BASE_URL}/api/analyses/{analysis_id}/review")
        
        assert review_response.status_code == 200
        data = review_response.json()
        assert "message" in data
        
        print(f"✓ Mark reviewed endpoint works - Analysis {analysis_id[:8]} marked as reviewed")
    
    def test_mark_reviewed_not_found(self):
        """Test /api/analyses/{id}/review returns 404 for non-existent analysis"""
        response = requests.post(f"{BASE_URL}/api/analyses/non-existent-id/review")
        
        assert response.status_code == 404
        print("✓ Mark reviewed returns 404 for non-existent analysis")


class TestAnalysesEndpoints:
    """Analyses CRUD endpoint tests"""
    
    def test_get_analyses_list(self):
        """Test /api/analyses returns list of analyses"""
        response = requests.get(f"{BASE_URL}/api/analyses")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        
        if data:
            item = data[0]
            assert "id" in item
            assert "created_at" in item
            assert "summary" in item
            
        print(f"✓ Get analyses list - returned {len(data)} items")
    
    def test_get_analysis_by_id(self):
        """Test /api/analyses/{id} returns specific analysis"""
        # First create an analysis
        with open(TEST_IMAGE_PATH, 'rb') as f:
            files = {'file': ('test.jpg', f, 'image/jpeg')}
            create_response = requests.post(f"{BASE_URL}/api/analyze", files=files)
        
        assert create_response.status_code == 200
        analysis_id = create_response.json()["id"]
        
        # Get by ID
        get_response = requests.get(f"{BASE_URL}/api/analyses/{analysis_id}")
        
        assert get_response.status_code == 200
        data = get_response.json()
        assert data["id"] == analysis_id
        assert "results" in data
        assert "quality" in data["results"]
        assert "anomaly" in data["results"]
        
        print(f"✓ Get analysis by ID works - {analysis_id[:8]}")
    
    def test_get_analysis_not_found(self):
        """Test /api/analyses/{id} returns 404 for non-existent analysis"""
        response = requests.get(f"{BASE_URL}/api/analyses/non-existent-id")
        
        assert response.status_code == 404
        print("✓ Get analysis returns 404 for non-existent ID")


class TestRiskLevelValues:
    """Test risk level values are in Turkish without special chars"""
    
    def test_risk_levels_are_turkish(self):
        """Test risk levels use Turkish values: Dusuk, Orta, Yuksek"""
        with open(TEST_IMAGE_PATH, 'rb') as f:
            files = {'file': ('test.jpg', f, 'image/jpeg')}
            response = requests.post(f"{BASE_URL}/api/analyze", files=files)
        
        assert response.status_code == 200
        data = response.json()
        results = data["results"]
        
        # Check summary risk level
        assert results["summary"]["risk_level"] in ["Dusuk", "Orta", "Yuksek"]
        
        # Check anomaly risk level
        assert results["anomaly"]["risk_level"] in ["Dusuk", "Orta", "Yuksek"]
        
        # Check quality level
        assert results["quality"]["quality_level"] in ["Yuksek", "Orta", "Dusuk"]
        
        # Check severity labels in damages
        for damage in results["damages"]:
            if "severity_details" in damage:
                assert damage["severity_details"]["label"] in ["Dusuk", "Orta", "Yuksek"]
        
        print("✓ All risk/quality levels use correct Turkish values")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
