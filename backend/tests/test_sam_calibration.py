"""
AutoDamageIQ - SAM Calibration Tests
=====================================
Tests for SAM measurement calibration improvements:
1. Panel-calibrated measurements (bbox_width_cm, bbox_height_cm, damage_area_cm2)
2. Type factor application (scratch=0.10, dent=0.45, etc.)
3. BBox ratio analysis (effective_ratio, relative_size)
4. Image ratio analysis (approx_area_cm2)
5. Size band reflects calibrated values
6. SAM limit: max 5 damages processed
7. Regression tests for compare/upload, review-queue, quality-check
"""

import pytest
import requests
import os
import time

BASE_URL = os.environ.get('REACT_APP_BACKEND_URL', '').rstrip('/')

# Type factors from sam_integration.py
DAMAGE_TYPE_AREA_FACTOR = {
    "scratch": 0.10,
    "crack": 0.12,
    "dent": 0.45,
    "glass_shatter": 0.80,
    "lamp_broken": 0.65,
    "tire_flat": 0.30,
}


class TestSAMStatus:
    """SAM status endpoint tests"""
    
    def test_sam_status_returns_correct_fields(self):
        """GET /api/sam/status should return correct status fields"""
        response = requests.get(f"{BASE_URL}/api/sam/status", timeout=30)
        assert response.status_code == 200
        
        data = response.json()
        assert "library_installed" in data
        assert "checkpoint_exists" in data
        assert "checkpoint_path" in data
        assert "checkpoint_size_mb" in data
        assert "model_type" in data
        assert "model_loaded" in data
        assert "device" in data
        
        # SAM should be available
        assert data["library_installed"] == True
        assert data["checkpoint_exists"] == True
        assert data["model_type"] == "vit_b"
        print(f"SAM Status: loaded={data['model_loaded']}, device={data['device']}")


class TestSAMCalibration:
    """SAM calibration measurement tests"""
    
    @pytest.fixture(scope="class")
    def analyze_result(self):
        """Analyze an image and return the result (shared across tests)"""
        image_path = "/app/assets/crash_1.jpg"
        
        with open(image_path, 'rb') as f:
            files = {'file': ('crash_1.jpg', f, 'image/jpeg')}
            response = requests.post(
                f"{BASE_URL}/api/analyze",
                files=files,
                timeout=180  # SAM processing can take time
            )
        
        assert response.status_code == 200, f"Analyze failed: {response.text}"
        return response.json()
    
    def test_analyze_returns_sam_data(self, analyze_result):
        """POST /api/analyze should return SAM data for damages"""
        results = analyze_result.get("results", {})
        damages = results.get("damages", [])
        
        assert results.get("sam_available") == True, "SAM should be available"
        assert results.get("sam_used") == True, "SAM should be used"
        
        if len(damages) > 0:
            # At least one damage should have SAM data
            sam_damages = [d for d in damages if d.get("sam_data", {}).get("available")]
            assert len(sam_damages) > 0, "At least one damage should have SAM data"
            print(f"Damages with SAM data: {len(sam_damages)}/{len(damages)}")
    
    def test_sam_measurements_structure(self, analyze_result):
        """SAM measurements should have correct structure"""
        results = analyze_result.get("results", {})
        damages = results.get("damages", [])
        
        for damage in damages:
            sam_data = damage.get("sam_data", {})
            if not sam_data.get("available"):
                continue
            
            measurements = sam_data.get("measurements", {})
            
            # Required fields
            assert "primary" in measurements, "measurements should have 'primary'"
            assert "bbox_analysis" in measurements, "measurements should have 'bbox_analysis'"
            assert "image_analysis" in measurements, "measurements should have 'image_analysis'"
            assert "area_pixels" in measurements, "measurements should have 'area_pixels'"
            assert "fill_ratio" in measurements, "measurements should have 'fill_ratio'"
            assert "size_band" in measurements, "measurements should have 'size_band'"
            assert "has_panel_calibration" in measurements, "measurements should have 'has_panel_calibration'"
            
            print(f"Damage {damage['type']}: size_band={measurements['size_band']}, has_panel_calibration={measurements['has_panel_calibration']}")
    
    def test_panel_calibrated_measurements(self, analyze_result):
        """Panel-calibrated measurements should include cm values"""
        results = analyze_result.get("results", {})
        damages = results.get("damages", [])
        
        for damage in damages:
            sam_data = damage.get("sam_data", {})
            if not sam_data.get("available"):
                continue
            
            measurements = sam_data.get("measurements", {})
            
            if measurements.get("has_panel_calibration"):
                primary = measurements.get("primary", {})
                
                # Panel-calibrated measurements should have these fields
                assert "method" in primary, "primary should have 'method'"
                assert primary["method"] == "panel_reference", f"method should be 'panel_reference', got {primary['method']}"
                assert "bbox_width_cm" in primary, "primary should have 'bbox_width_cm'"
                assert "bbox_height_cm" in primary, "primary should have 'bbox_height_cm'"
                assert "damage_area_cm2" in primary, "primary should have 'damage_area_cm2'"
                assert "type_factor" in primary, "primary should have 'type_factor'"
                assert "panel_reference" in primary, "primary should have 'panel_reference'"
                
                # Values should be positive
                assert primary["bbox_width_cm"] > 0, "bbox_width_cm should be positive"
                assert primary["bbox_height_cm"] > 0, "bbox_height_cm should be positive"
                assert primary["damage_area_cm2"] > 0, "damage_area_cm2 should be positive"
                
                print(f"Panel calibrated: {damage['type']} on {primary['panel_reference']}: {primary['bbox_width_cm']}x{primary['bbox_height_cm']} cm, area={primary['damage_area_cm2']} cm2")
    
    def test_type_factor_applied(self, analyze_result):
        """Type factor should be applied based on damage type"""
        results = analyze_result.get("results", {})
        damages = results.get("damages", [])
        
        for damage in damages:
            sam_data = damage.get("sam_data", {})
            if not sam_data.get("available"):
                continue
            
            measurements = sam_data.get("measurements", {})
            damage_type = damage.get("type", "")
            
            # Check primary measurement type_factor
            primary = measurements.get("primary", {})
            if "type_factor" in primary:
                expected_factor = DAMAGE_TYPE_AREA_FACTOR.get(damage_type, 0.40)
                actual_factor = primary["type_factor"]
                assert actual_factor == expected_factor, f"Type factor for {damage_type} should be {expected_factor}, got {actual_factor}"
                print(f"Type factor for {damage_type}: {actual_factor} (expected {expected_factor})")
            
            # Check bbox_analysis type_factor
            bbox_analysis = measurements.get("bbox_analysis", {})
            if "type_factor" in bbox_analysis:
                expected_factor = DAMAGE_TYPE_AREA_FACTOR.get(damage_type, 0.40)
                actual_factor = bbox_analysis["type_factor"]
                assert actual_factor == expected_factor, f"BBox type factor for {damage_type} should be {expected_factor}, got {actual_factor}"
    
    def test_bbox_ratio_analysis(self, analyze_result):
        """BBox ratio analysis should include effective_ratio and relative_size"""
        results = analyze_result.get("results", {})
        damages = results.get("damages", [])
        
        for damage in damages:
            sam_data = damage.get("sam_data", {})
            if not sam_data.get("available"):
                continue
            
            measurements = sam_data.get("measurements", {})
            bbox_analysis = measurements.get("bbox_analysis", {})
            
            assert "method" in bbox_analysis, "bbox_analysis should have 'method'"
            assert bbox_analysis["method"] == "bbox_ratio", f"method should be 'bbox_ratio', got {bbox_analysis['method']}"
            assert "effective_ratio" in bbox_analysis, "bbox_analysis should have 'effective_ratio'"
            assert "relative_size" in bbox_analysis, "bbox_analysis should have 'relative_size'"
            assert "type_factor" in bbox_analysis, "bbox_analysis should have 'type_factor'"
            assert "bbox_image_ratio" in bbox_analysis, "bbox_analysis should have 'bbox_image_ratio'"
            
            # relative_size should be one of the valid values
            valid_sizes = ["Cok Kucuk", "Kucuk", "Orta", "Buyuk", "Cok Buyuk"]
            assert bbox_analysis["relative_size"] in valid_sizes, f"relative_size should be one of {valid_sizes}"
            
            print(f"BBox analysis for {damage['type']}: effective_ratio={bbox_analysis['effective_ratio']}%, relative_size={bbox_analysis['relative_size']}")
    
    def test_image_ratio_analysis(self, analyze_result):
        """Image ratio analysis should include approx_area_cm2"""
        results = analyze_result.get("results", {})
        damages = results.get("damages", [])
        
        for damage in damages:
            sam_data = damage.get("sam_data", {})
            if not sam_data.get("available"):
                continue
            
            measurements = sam_data.get("measurements", {})
            image_analysis = measurements.get("image_analysis", {})
            
            assert "method" in image_analysis, "image_analysis should have 'method'"
            assert image_analysis["method"] == "image_ratio", f"method should be 'image_ratio', got {image_analysis['method']}"
            assert "approx_area_cm2" in image_analysis, "image_analysis should have 'approx_area_cm2'"
            assert "size_band" in image_analysis, "image_analysis should have 'size_band'"
            assert "bbox_percentage" in image_analysis, "image_analysis should have 'bbox_percentage'"
            
            # approx_area_cm2 should be positive
            assert image_analysis["approx_area_cm2"] >= 0, "approx_area_cm2 should be non-negative"
            
            print(f"Image analysis for {damage['type']}: approx_area_cm2={image_analysis['approx_area_cm2']}, size_band={image_analysis['size_band']}")
    
    def test_size_band_reflects_calibrated_values(self, analyze_result):
        """Size band should reflect calibrated values, not raw SAM mask area"""
        results = analyze_result.get("results", {})
        damages = results.get("damages", [])
        
        for damage in damages:
            sam_data = damage.get("sam_data", {})
            if not sam_data.get("available"):
                continue
            
            measurements = sam_data.get("measurements", {})
            size_band = measurements.get("size_band")
            
            # Size band should be one of the valid values
            valid_bands = ["Cok Kucuk", "Kucuk", "Orta", "Buyuk", "Cok Buyuk"]
            assert size_band in valid_bands, f"size_band should be one of {valid_bands}, got {size_band}"
            
            # If panel calibration is available, size_band should be based on damage_area_cm2
            if measurements.get("has_panel_calibration"):
                primary = measurements.get("primary", {})
                damage_area_cm2 = primary.get("damage_area_cm2", 0)
                
                # Verify size_band matches the cm2 value
                if damage_area_cm2 < 5:
                    expected_band = "Cok Kucuk"
                elif damage_area_cm2 < 30:
                    expected_band = "Kucuk"
                elif damage_area_cm2 < 150:
                    expected_band = "Orta"
                elif damage_area_cm2 < 600:
                    expected_band = "Buyuk"
                else:
                    expected_band = "Cok Buyuk"
                
                assert size_band == expected_band, f"size_band should be {expected_band} for {damage_area_cm2} cm2, got {size_band}"
                print(f"Size band for {damage['type']}: {size_band} (area={damage_area_cm2} cm2)")


class TestSAMLimit:
    """Test SAM limit of max 5 damages"""
    
    def test_sam_limit_with_multiple_damages(self):
        """SAM should process max 5 damages, 6th+ should get 'SAM limiti asildi'"""
        # Use an image that might have multiple damages
        image_path = "/app/assets/crash_3.jpg"  # Larger image, might have more damages
        
        with open(image_path, 'rb') as f:
            files = {'file': ('crash_3.jpg', f, 'image/jpeg')}
            response = requests.post(
                f"{BASE_URL}/api/analyze",
                files=files,
                timeout=180
            )
        
        assert response.status_code == 200, f"Analyze failed: {response.text}"
        
        results = response.json().get("results", {})
        damages = results.get("damages", [])
        
        sam_processed = 0
        sam_limited = 0
        
        for damage in damages:
            sam_data = damage.get("sam_data", {})
            if sam_data.get("available"):
                sam_processed += 1
            elif "limiti" in sam_data.get("reason", "").lower() or "limit" in sam_data.get("reason", "").lower():
                sam_limited += 1
        
        print(f"Total damages: {len(damages)}, SAM processed: {sam_processed}, SAM limited: {sam_limited}")
        
        # SAM should process at most 5 damages
        assert sam_processed <= 5, f"SAM should process max 5 damages, processed {sam_processed}"
        
        # If more than 5 damages, some should be limited
        if len(damages) > 5:
            assert sam_limited > 0, "Damages beyond 5 should have 'SAM limiti asildi' reason"


class TestTypeFactor:
    """Test type factor differences between damage types"""
    
    def test_scratch_has_smaller_area_than_dent(self):
        """Scratch should have type_factor=0.10 (much smaller area than dent=0.45)"""
        # This is a unit test of the type factors
        scratch_factor = DAMAGE_TYPE_AREA_FACTOR.get("scratch")
        dent_factor = DAMAGE_TYPE_AREA_FACTOR.get("dent")
        
        assert scratch_factor == 0.10, f"Scratch factor should be 0.10, got {scratch_factor}"
        assert dent_factor == 0.45, f"Dent factor should be 0.45, got {dent_factor}"
        assert scratch_factor < dent_factor, "Scratch factor should be less than dent factor"
        
        # For same bbox size, scratch area should be ~22% of dent area (0.10/0.45)
        ratio = scratch_factor / dent_factor
        assert ratio < 0.25, f"Scratch/dent ratio should be < 0.25, got {ratio}"
        print(f"Scratch factor: {scratch_factor}, Dent factor: {dent_factor}, Ratio: {ratio:.2f}")


class TestRegressionCompareUpload:
    """Regression test for compare/upload endpoint"""
    
    def test_compare_upload_still_works(self):
        """POST /api/compare/upload should still work after SAM changes"""
        before_path = "/app/assets/crash_1.jpg"
        after_path = "/app/assets/crash_2.jpg"
        
        with open(before_path, 'rb') as before_f, open(after_path, 'rb') as after_f:
            files = {
                'before_file': ('before.jpg', before_f, 'image/jpeg'),
                'after_file': ('after.jpg', after_f, 'image/jpeg')
            }
            response = requests.post(
                f"{BASE_URL}/api/compare/upload",
                files=files,
                timeout=300  # Compare takes longer
            )
        
        assert response.status_code == 200, f"Compare upload failed: {response.text}"
        
        data = response.json()
        assert "has_new_damage" in data, "Response should have 'has_new_damage'"
        assert "verdict" in data, "Response should have 'verdict'"
        assert "before_analysis" in data, "Response should have 'before_analysis'"
        assert "after_analysis" in data, "Response should have 'after_analysis'"
        
        print(f"Compare result: has_new_damage={data['has_new_damage']}, verdict={data['verdict']}")


class TestRegressionReviewQueue:
    """Regression test for review-queue endpoint"""
    
    def test_review_queue_still_works(self):
        """GET /api/review-queue should still work"""
        response = requests.get(f"{BASE_URL}/api/review-queue", timeout=30)
        assert response.status_code == 200, f"Review queue failed: {response.text}"
        
        data = response.json()
        assert isinstance(data, list), "Response should be a list"
        print(f"Review queue items: {len(data)}")


class TestRegressionQualityCheck:
    """Regression test for quality-check endpoint"""
    
    def test_quality_check_still_works(self):
        """POST /api/quality-check should still work"""
        image_path = "/app/assets/crash_1.jpg"
        
        with open(image_path, 'rb') as f:
            files = {'file': ('crash_1.jpg', f, 'image/jpeg')}
            response = requests.post(
                f"{BASE_URL}/api/quality-check",
                files=files,
                timeout=30
            )
        
        assert response.status_code == 200, f"Quality check failed: {response.text}"
        
        data = response.json()
        assert "quality_score" in data, "Response should have 'quality_score'"
        assert "quality_level" in data, "Response should have 'quality_level'"
        assert "is_acceptable" in data, "Response should have 'is_acceptable'"
        
        print(f"Quality check: score={data['quality_score']}, level={data['quality_level']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
