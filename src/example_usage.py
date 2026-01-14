#!/usr/bin/env python3
"""
Example usage of the ElectricityConsumptionInference class
"""

import json
from inference import ElectricityConsumptionInference

# Initialize the inference class
print("=" * 80)
print("ELECTRICITY CONSUMPTION INFERENCE - EXAMPLE USAGE")
print("=" * 80)

inference = ElectricityConsumptionInference()

# Example 1: Single prediction from JSON
print("\n" + "=" * 80)
print("EXAMPLE 1: Single Prediction")
print("=" * 80)

json_data = {
    'time': '2016-01-01T21:00:00',
    'building_id': 'Bear_education_Sharon',
    'site_id': 'Bear',
    'primaryspaceusage': 'Education',
    'sub_primaryspaceusage': 'Education',
    'sqm': 5261.7,
    'yearbuilt': 1953,
    'numberoffloors': 5,
    'occupants': 100,
    'timezone': 'US/Pacific',
    'airTemperature': 6.1,
    'cloudCoverage': 50.0,
    'dewTemperature': -2.2,
    'windSpeed': 2.6,
    'seaLvlPressure': 1020.7,
    'precipDepth1HR': 0.0,
}

result = inference.predict_from_json(json_data) ## Predict : 12h -> output: 98.25 kWh

print(json.dumps(result, indent=2))

# Example 2: Predict with threshold
print("\n" + "=" * 80)
print("EXAMPLE 2: Prediction with Threshold")
print("=" * 80)

result = inference.predict_with_threshold(json_data, threshold=100.0)
print(f"Prediction: {result['prediction']:.2f} kWh")
print(f"Classification: {result['classification']}")
print(f"Above threshold: {result['is_above_threshold']}")

# Example 3: Validate input
print("\n" + "=" * 80)
print("EXAMPLE 3: Input Validation")
print("=" * 80)

validation = inference.validate_input(json_data)
print(f"Valid: {validation['valid']}")
if validation['warnings']:
    print(f"Warnings: {validation['warnings']}")

# Example 4: Feature importance
print("\n" + "=" * 80)
print("EXAMPLE 4: Feature Importance (Top 10)")
print("=" * 80)

importance = inference.get_feature_importance(top_n=10)
print(importance.to_string())

# Example 5: Predict future hours
print("\n" + "=" * 80)
print("EXAMPLE 5: Predict Future (24 hours)")
print("=" * 80)

future_predictions = inference.predict_future(
    building_id='Bear_education_Sharon',
    start_time='2016-01-01T00:00:00',
    hours=24
)
print(f"✅ Predicted {len(future_predictions)} hours")
print(f"\nFirst 5 hours:")
print(future_predictions.head().to_string())

# Example 6: City-level prediction
print("\n" + "=" * 80)
print("EXAMPLE 6: City-Level Prediction")
print("=" * 80)

city_summary = inference.get_city_summary(
    site_id='Bear',
    timestamp='2016-01-01T12:00:00',
    weather_data={
        'airTemperature': 15.0,
        'cloudCoverage': 30.0,
        'windSpeed': 5.0
    }
)
print(json.dumps(city_summary, indent=2))

print("\n" + "=" * 80)
print("✅ ALL EXAMPLES COMPLETED!")
print("=" * 80)
