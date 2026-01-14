# DiCE Explainer v2 - Usage Guide

## Overview

The `DiceExplainer` v2 provides counterfactual explanations to suggest feature adjustments that would reduce electricity consumption below a specified threshold. This version is adapted for the new XGBoost model architecture with multi-horizon predictions.

## Key Features

- **Counterfactual Generation**: Generate diverse counterfactual examples showing how to reduce consumption
- **Actionable Features**: Only suggests changes to features that can actually be adjusted (Chilledwater, Hotwater)
- **Threshold-based Recommendations**: Focus on reducing consumption below a target threshold
- **Multi-Horizon Support**: Works with the new 24-hour forecast model architecture
- **Detailed Change Analysis**: Shows exactly what needs to change and by how much

## Installation

Ensure you have the required dependencies:

```bash
pip install dice-ml pandas numpy scikit-learn xgboost joblib
```

## Quick Start

```python
from src_v2.dice_explainer import DiceExplainer

# Initialize
explainer = DiceExplainer(
    model_dir="models_1578_csv",
    encode_path="models_1578_csv/categorical_encoder.pkl",
    processed_data_path="data_1578_csv/train_encode.csv"
)

# Prepare building data (same format as inference/model.py)
json_data = {
    'time': '2017-08-31T06:00:00',
    'building_code': 'Fox_education_Wendell',
    'site_id': 'Fox',
    'sqm': 20402.2,
    'primaryspaceusage': 'Education',
    'air_temperature': 16.12,
    'dew_temperature': 13.3,
    'wind_speed': 2.13,
    'Chilledwater': 4151.77,
    'Hotwater': 10306.87,
    # ... other fields
}

# Get current prediction
result = explainer.forecaster(json_data)
current_pred = result[0]['electric']  # First hour prediction

# Generate recommendations to reduce consumption below threshold
threshold = current_pred * 0.8  # 80% of current = 20% reduction
recommendations = explainer.generate_recommendations(
    json_data=json_data,
    threshold=threshold,
    total_cfs=5,
    method='random'  # or 'genetic' for better quality
)

# View recommendations
if recommendations['success']:
    for rec in recommendations['recommendations']:
        print(f"Predicted: {rec['predicted_consumption']:.2f} kWh")
        print(f"Reduction: {rec['reduction']:.2f} kWh")
        for change in rec['changes']:
            print(f"  • {change['action']}")
```

## Main Methods

### `generate_recommendations(json_data, threshold, total_cfs=5, method='random')`

Generate counterfactual recommendations to reduce consumption below threshold (for first hour t+1).

**Parameters:**
- `json_data`: Dictionary with building and weather data (same format as `inference/model.py`)
- `threshold`: Target consumption threshold (kWh) for first hour (t+1)
- `total_cfs`: Number of counterfactual examples to generate
- `method`: DiCE method ('random' for speed, 'genetic' for better results)

**Returns:**
```python
{
    'success': True,
    'current_prediction': 150.5,
    'threshold': 120.0,
    'below_threshold': False,
    'needs_reduction': 30.5,
    'total_counterfactuals': 5,
    'recommendations': [
        {
            'counterfactual_id': 1,
            'predicted_consumption': 115.2,
            'reduction': 35.3,
            'reduction_pct': 23.5,
            'below_threshold': True,
            'changes': [
                {
                    'feature': 'Chilledwater',
                    'original': 4151.77,
                    'new': 2906.24,
                    'change': -1245.53,
                    'change_pct': -30.0,
                    'action': 'Chilledwater: 4151.77 → 2906.24 (-30.0%)',
                    'description': 'Chilled water consumption (can reduce HVAC usage)'
                },
                # ... more changes
            ]
        },
        # ... more recommendations
    ]
}
```

### `generate_recommendations_for_hour(json_data, threshold, hour_offset, total_cfs=5, method='random')`

Generate counterfactual recommendations for a specific hour ahead.

**Parameters:**
- `json_data`: Dictionary with building and weather data
- `threshold`: Target consumption threshold (kWh) for the specified hour
- `hour_offset`: Hours ahead to predict (1-24, where 1 = next hour, 2 = hour after next, etc.)
- `total_cfs`: Number of counterfactual examples to generate
- `method`: DiCE method ('random' or 'genetic')

**Returns:**
Dictionary with recommendations for the specific hour, including:
- `hour_offset`: The hour offset used
- `time`: ISO format timestamp for that hour
- `current_prediction`: Current predicted consumption
- `recommendations`: List of counterfactual recommendations

**Example:**
```python
# Get recommendations for hour 5 ahead (t+5)
result = explainer.generate_recommendations_for_hour(
    json_data=building_data,
    threshold=100.0,
    hour_offset=5,
    total_cfs=5
)
```

### `monitor_24_hours(json_data, threshold, total_cfs=3, method='random', only_problematic_hours=True)`

Monitor all 24 hours ahead and generate recommendations for hours that exceed threshold.

**Parameters:**
- `json_data`: Dictionary with building and weather data
- `threshold`: Target consumption threshold (kWh) for each hour
- `total_cfs`: Number of counterfactual examples to generate per hour
- `method`: DiCE method ('random' or 'genetic')
- `only_problematic_hours`: If True, only generate recommendations for hours exceeding threshold

**Returns:**
Dictionary with monitoring results:
- `hours_monitored`: Total hours monitored (24)
- `hours_exceeding_threshold`: Number of hours exceeding threshold
- `hours_with_recommendations`: Number of hours with successful recommendations
- `hourly_results`: List of results for each hour (1-24)

**Example:**
```python
# Monitor all 24 hours
monitoring = explainer.monitor_24_hours(
    json_data=building_data,
    threshold=100.0,
    total_cfs=3,
    only_problematic_hours=True
)

# Check which hours exceed threshold
for hour_info in monitoring['hourly_results']:
    if hour_info['exceeds_threshold']:
        print(f"Hour t+{hour_info['hour_offset']}: {hour_info['current_prediction']:.2f} kWh")
```

### `get_simple_recommendations(json_data, threshold, top_n=3)`

Get simplified recommendations (faster, less detailed) for first hour.

**Parameters:**
- `json_data`: Dictionary with building and weather data
- `threshold`: Target consumption threshold
- `top_n`: Number of top recommendations to return

**Returns:**
Simplified recommendations dictionary with top N recommendations and key changes.

## Actionable Features

### ✅ Can Be Adjusted:
- **`Chilledwater`**: Chilled water consumption (can reduce HVAC usage)
  - Direction: decrease
  - Range: 10-50% reduction
- **`Hotwater`**: Hot water consumption (can reduce usage)
  - Direction: decrease
  - Range: 10-50% reduction

### ❌ Cannot Be Adjusted:
- **`hour`**, **`dayofweek`**, **`is_weekend`**, **`month`**: Time features (fixed by timestamp)
- **`airTemperature`**, **`dewTemperature`**, **`windSpeed`**: Weather data (cannot control)
- **`sqm`**: Building area (fixed physical property)
- **`primaryspaceusage`**, **`site_id`**, **`building_id`**: Fixed building properties

## Model Architecture

The new dice_explainer works with:
- **Multi-horizon models**: 24 separate XGBoost models (one for each hour)
- **Feature set**: Uses `X_FEATURE_INPUT` from `inference/preprocess.py`
  - Time features: `hour`, `dayofweek`, `is_weekend`, `month`
  - Weather: `airTemperature`, `dewTemperature`, `windSpeed`
  - Building: `sqm`, `primaryspaceusage`, `site_id`, `building_id`
  - Energy: `Chilledwater`, `Hotwater`
- **Encoder**: Uses `LabelEncoder` from `inference/encoder.py` for categorical features

## Example Use Cases

### 1. Building Exceeds Energy Budget

```python
# Building consuming 150 kWh, need to reduce to 120 kWh
result = explainer.generate_recommendations(
    json_data=building_data,
    threshold=120.0
)

# Get top recommendation
top_rec = result['recommendations'][0]
print(f"To reduce to {top_rec['predicted_consumption']:.2f} kWh:")
for change in top_rec['changes']:
    print(f"  • {change['action']}")
```

### 2. Quick Recommendations

```python
# Get simplified recommendations
simple = explainer.get_simple_recommendations(
    json_data=building_data,
    threshold=100.0,
    top_n=3
)

for rec in simple['top_recommendations']:
    print(f"Option: Reduce by {rec['reduction_pct']:.1f}%")
    for change in rec['key_changes']:
        print(f"  • {change['action']}")
```

### 3. Recommendations for Specific Hour

```python
# Get recommendations for hour 12 ahead (noon tomorrow)
result = explainer.generate_recommendations_for_hour(
    json_data=building_data,
    threshold=120.0,
    hour_offset=12,  # 12 hours ahead
    total_cfs=5
)

if result['success']:
    print(f"Hour t+12 ({result['time']}): {result['current_prediction']:.2f} kWh")
    for rec in result['recommendations'][:3]:
        print(f"  Option: {rec['predicted_consumption']:.2f} kWh (reduction: {rec['reduction']:.2f} kWh)")
```

### 4. Monitor All 24 Hours

```python
# Monitor all 24 hours and get recommendations for problematic hours
monitoring = explainer.monitor_24_hours(
    json_data=building_data,
    threshold=100.0,
    total_cfs=3,
    only_problematic_hours=True
)

print(f"Hours exceeding threshold: {monitoring['hours_exceeding_threshold']}")

# Get recommendations for the worst hour
worst_hour = max(
    [h for h in monitoring['hourly_results'] if h['exceeds_threshold']],
    key=lambda x: x['current_prediction']
)
print(f"Worst hour: t+{worst_hour['hour_offset']} ({worst_hour['time']})")
if worst_hour['recommendations']:
    top_rec = worst_hour['recommendations']['recommendations'][0]
    print(f"Top recommendation: {top_rec['predicted_consumption']:.2f} kWh")
```

### 5. Integration with Dashboard

```python
# Check if building needs optimization
current = explainer.forecaster(building_data)[0]['electric']
threshold = 100.0

if current > threshold:
    # Generate recommendations
    recommendations = explainer.generate_recommendations(
        json_data=building_data,
        threshold=threshold,
        total_cfs=3
    )
    
    # Display in dashboard
    for rec in recommendations['recommendations']:
        if rec['below_threshold']:
            display_recommendation(rec)
```

## Configuration

### DiCE Methods

- **`method='random'`**: Faster, good for testing and quick recommendations
- **`method='genetic'`**: Slower but produces better, more diverse counterfactuals

### Permitted Ranges

The explainer automatically sets permitted ranges for actionable features:
- **Chilledwater**: Can reduce by 10-50%
- **Hotwater**: Can reduce by 10-50%

These ranges ensure realistic recommendations that don't suggest impossible changes.

## Differences from v1

1. **Model Architecture**: Works with 24 separate models instead of a single wrapped model
2. **Feature Set**: Uses the new feature set from `X_FEATURE_INPUT`
3. **Encoder**: Uses the new `LabelEncoder` class
4. **Target**: Focuses on first hour prediction (t+1) for DiCE analysis
5. **Actionable Features**: Only Chilledwater and Hotwater are adjustable (simpler than v1)

## Troubleshooting

### Error: "No actionable features available"

This means all features in your input are fixed (time, weather, building properties). Ensure you're providing `Chilledwater` and/or `Hotwater` values in your input.

### Error: "Processed data not found"

Ensure `data_1578_csv/train_encode.csv` exists. This file is created by `notebooks/step2_encode_data.ipynb`.

### Error: "Model files not found"

Ensure all model files (`model_hour_1.pkl` through `model_hour_24.pkl`) exist in the model directory. These are created by `notebooks/step3_training.ipynb`.

## See Also

- `inference/model.py` - Main inference interface
- `notebooks/step2_encode_data.ipynb` - Data encoding process
- `notebooks/step3_training.ipynb` - Model training process
- `src_v2/dice_example.py` - Complete usage example
