# DiCE Explainer for Electricity Consumption Reduction

## Overview

The `DiceExplainer` class provides counterfactual explanations to suggest feature adjustments that would reduce electricity consumption below a specified threshold.

## Features

- **Counterfactual Generation**: Generate diverse counterfactual examples showing how to reduce consumption
- **Actionable Features**: Only suggests changes to features that can actually be adjusted
- **Threshold-based Recommendations**: Focus on reducing consumption below a target threshold
- **Detailed Change Analysis**: Shows exactly what needs to change and by how much

## Quick Start

```python
from src.dice_explainer import DiceExplainer

# Initialize
explainer = DiceExplainer()

# Prepare building data
json_data = {
    'time': '2016-01-01T21:00:00',
    'building_id': 'Bear_education_Sharon',
    'sqm': 5261.7,
    'occupants': 200,
    'airTemperature': 25.0,
    # ... other features
}

# Generate recommendations to reduce consumption below 100 kWh
result = explainer.generate_recommendations(
    json_data=json_data,
    threshold=100.0,
    total_cfs=5
)

# View recommendations
for rec in result['recommendations']:
    print(f"Predicted: {rec['predicted_consumption']:.2f} kWh")
    print(f"Reduction: {rec['reduction']:.2f} kWh")
    for change in rec['changes']:
        print(f"  • {change['action']}")
```

## Main Methods

### `generate_recommendations(json_data, threshold, total_cfs=5, method='random')`

Generate counterfactual recommendations to reduce consumption below threshold.

**Parameters:**
- `json_data`: Dictionary with building and weather data
- `threshold`: Target consumption threshold (kWh)
- `total_cfs`: Number of counterfactual examples to generate
- `method`: DiCE method ('random' for speed, 'genetic' for better results)

**Returns:**
```python
{
    'success': True,
    'current_prediction': 150.5,
    'threshold': 100.0,
    'below_threshold': False,
    'needs_reduction': 50.5,
    'total_counterfactuals': 5,
    'recommendations': [
        {
            'predicted_consumption': 95.2,
            'reduction': 55.3,
            'reduction_pct': 36.7,
            'below_threshold': True,
            'changes': [
                {
                    'feature': 'sqm',
                    'description': 'Building area (square meters)',
                    'original_value': 5261.7,
                    'suggested_value': 4200.0,
                    'change': -1061.7,
                    'change_pct': -20.2,
                    'action': 'Reduce building area by 1062 sqm (20.2%)'
                },
                # ... more changes
            ]
        }
    ]
}
```

### `get_simple_recommendations(json_data, threshold, top_n=3)`

Get simplified recommendations (faster, less detailed).

**Returns:**
```python
{
    'success': True,
    'current_prediction': 150.5,
    'threshold': 100.0,
    'needs_reduction': 50.5,
    'top_recommendations': [
        {
            'predicted_consumption': 95.2,
            'reduction': 55.3,
            'reduction_pct': 36.7,
            'key_changes': [
                {
                    'feature': 'sqm',
                    'action': 'Reduce building area by 1062 sqm (20.2%)',
                    'impact': '20.2%'
                }
            ]
        }
    ]
}
```

### `get_actionable_features()`

Get list of features that can be adjusted.

**Returns:**
```python
['sqm', 'occupants', 'airTemperature', 'hour', 'is_weekend']
```

## Actionable Features

The explainer only suggests changes to features that can actually be adjusted:

### ✅ Can Be Adjusted:
- **`sqm`**: Building area (can reduce space usage)
- **`occupants`**: Number of occupants (can reduce occupancy)
- **`airTemperature`**: Air temperature (can adjust HVAC settings)
- **`hour`**: Operating hour (can adjust schedule)
- **`is_weekend`**: Weekend flag (can adjust schedule)

### ❌ Cannot Be Adjusted:
- **`yearbuilt`**: Year built (fixed)
- **`numberoffloors`**: Number of floors (fixed building structure)
- **`primaryspaceusage`**: Primary space usage (fixed)
- **`site_id`**: Site ID (fixed location)
- **`electricity_lag1`**: Lag features (depend on past data)

## Example Use Cases

### 1. Building Exceeds Energy Budget

```python
# Building consuming 150 kWh, need to reduce to 100 kWh
result = explainer.generate_recommendations(
    json_data=building_data,
    threshold=100.0
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

### 3. Integration with Dashboard

```python
# Check if building needs optimization
current = explainer.inference.predict(building_data)
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

- **`sqm`**: Can reduce by up to 30%, minimum 100 sqm
- **`occupants`**: Can reduce by up to 50%, minimum 1
- **`airTemperature`**: Can adjust by ±5°C (HVAC control)
- **`hour`**: Can change to any hour (0-23)

## Integration with Inference Class

The `DiceExplainer` uses the `ElectricityConsumptionInference` class internally:

```python
from src.inference import ElectricityConsumptionInference
from src.dice_explainer import DiceExplainer

# Option 1: Let DiceExplainer create inference instance
explainer = DiceExplainer()

# Option 2: Use existing inference instance
inference = ElectricityConsumptionInference()
explainer = DiceExplainer(inference=inference)
```

## Performance Notes

- **Setup Time**: Initial setup takes ~10-30 seconds (loading data and creating DiCE objects)
- **Recommendation Generation**: 
  - `method='random'`: ~1-5 seconds per recommendation set
  - `method='genetic'`: ~10-30 seconds per recommendation set
- **Memory**: Requires ~500MB-1GB RAM for DiCE operations

## Error Handling

The explainer handles various error cases:

- **Already below threshold**: Returns message that no changes needed
- **No valid counterfactuals found**: Returns error message
- **Invalid input data**: Validates and provides warnings

## See Also

- `src/inference.py`: Base inference class
- `src/dice_example.py`: Example usage script
- `docs/DICE_INTEGRATION.md`: Detailed DiCE integration guide
