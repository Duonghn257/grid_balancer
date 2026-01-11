#!/usr/bin/env python3
"""
Test script cho Inference class
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.inference import ElectricityConsumptionInference
import json

# Khởi tạo Inference class
print("=" * 80)
print("TESTING INFERENCE CLASS")
print("=" * 80)

inference = ElectricityConsumptionInference()

# Test data từ user
json_data = {
    'time': '2016-01-01T21:00:00',
    'building_code': 'Bear_education_Sharon',
    'site_id': 'Bear',
    'primaryspaceusage': 'Education',
    'sub_primaryspaceusage': 'Education',
    'industry': None,
    'subindustry': None,
    'sqm': 5261.7,
    'sqft': 56637.0,
    'lat': 37.871903400000036,
    'lng': -122.26072860000008,
    'timezone': 'US/Pacific',
    'heatingtype': None,
    'yearbuilt': 1953,
    'date_opened': None,
    'numberoffloors': 5,
    'occupants': None,
    'energystarscore': None,
    'eui': None,
    'site_eui': None,
    'source_eui': None,
    'leed_level': None,
    'rating': None,
    'air_temperature': 6.1,
    'cloud_coverage': None,
    'dew_temperature': -2.2,
    'precip_depth_1hr': 0.0,
    'precip_depth_6hr': None,
    'sea_lvl_pressure': 1020.7,
    'wind_direction': 80.0,
    'wind_speed': 2.6,
    'id': 66,
    'electricity': 98.25,
    'hotwater': 0.0,
    'chilledwater': 0.0,
    'steam': 0.0,
    'water': 0.0,
    'irrigation': 0.0,
    'solar': 0.0,
    'gas': 0.0
}

# Test 1: Predict từ JSON
print("\n" + "=" * 80)
print("TEST 1: Predict từ JSON")
print("=" * 80)
result = inference.predict_from_json(json_data)
print(json.dumps(result, indent=2))

# Test 2: Predict với threshold
print("\n" + "=" * 80)
print("TEST 2: Predict với threshold")
print("=" * 80)
result = inference.predict_with_threshold(json_data, threshold=100.0)
print(json.dumps(result, indent=2))

# Test 3: Validate input
print("\n" + "=" * 80)
print("TEST 3: Validate input")
print("=" * 80)
validation = inference.validate_input(json_data)
print(json.dumps(validation, indent=2))

# Test 4: Feature importance
print("\n" + "=" * 80)
print("TEST 4: Feature Importance (Top 10)")
print("=" * 80)
importance = inference.get_feature_importance(top_n=10)
print(importance.to_string())

# Test 5: Confidence interval
print("\n" + "=" * 80)
print("TEST 5: Predict với confidence interval")
print("=" * 80)
result = inference.predict_with_confidence_interval(json_data, confidence=0.95)
print(json.dumps(result, indent=2))

# Test 6: Prediction explanation
print("\n" + "=" * 80)
print("TEST 6: Prediction explanation")
print("=" * 80)
explanation = inference.get_prediction_explanation(json_data)
print(json.dumps(explanation, indent=2))

# Test 7: Predict future (24 hours)
print("\n" + "=" * 80)
print("TEST 7: Predict future (24 hours)")
print("=" * 80)
future_predictions = inference.predict_future(
    building_id='Bear_education_Sharon',
    start_time='2016-01-01T00:00:00',
    hours=24
)
print(f"✅ Đã dự đoán {len(future_predictions)} giờ")
print(f"\nSample (5 giờ đầu):")
print(future_predictions.head().to_string())

print("\n" + "=" * 80)
print("✅ TẤT CẢ TESTS HOÀN THÀNH!")
print("=" * 80)
