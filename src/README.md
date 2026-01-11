# Electricity Consumption Inference Class

This module provides a comprehensive inference class for predicting electricity consumption using the trained XGBoost model.

## Overview

The `ElectricityConsumptionInference` class loads the trained XGBoost model and provides various methods to:
- Predict electricity consumption for single or multiple buildings
- Handle data preprocessing automatically
- Provide utility functions for dashboard applications
- Support city-level and building-level predictions

## Installation

The inference class requires the following files to be present:
- `output/models/xgboost_wrapped_dice.pkl` - Trained wrapped model
- `output/models/label_encoders_dice.pkl` - Label encoders for categorical features
- `output/models/model_info_dice.json` - Model metadata
- `output/features_info.json` - Feature information
- `output/processed_data.parquet` - Historical data (optional, for lag features)

## Quick Start

```python
from src.inference import ElectricityConsumptionInference

# Initialize the inference class
inference = ElectricityConsumptionInference()

# Prepare input data
json_data = {
    'time': '2016-01-01T21:00:00',
    'building_id': 'Bear_education_Sharon',
    'site_id': 'Bear',
    'primaryspaceusage': 'Education',
    'sqm': 5261.7,
    'yearbuilt': 1953,
    'numberoffloors': 5,
    'airTemperature': 6.1,
    'windSpeed': 2.6,
    # ... other features
}

# Make prediction
result = inference.predict_from_json(json_data)
print(f"Predicted consumption: {result['prediction']:.2f} kWh")
```

## Main Methods

### 1. Single Prediction

#### `predict(data, include_lag=True)`
Predict electricity consumption for a single data point.

**Parameters:**
- `data`: Dictionary or DataFrame with building and weather data
- `include_lag`: Whether to include lag features (requires historical data)

**Returns:**
- `float`: Predicted electricity consumption in kWh

**Example:**
```python
prediction = inference.predict(json_data)
```

#### `predict_from_json(json_data)`
Predict from JSON data and return detailed result.

**Returns:**
```python
{
    'success': True,
    'prediction': 98.25,
    'unit': 'kWh',
    'timestamp': '2016-01-01T21:00:00',
    'building_id': 'Bear_education_Sharon',
    'model_info': {...}
}
```

### 2. Batch Prediction

#### `predict_batch(data_list, include_lag=True)`
Predict electricity consumption for multiple data points.

**Parameters:**
- `data_list`: List of dictionaries with building and weather data
- `include_lag`: Whether to include lag features

**Returns:**
- `np.ndarray`: Array of predictions

**Example:**
```python
predictions = inference.predict_batch([data1, data2, data3])
```

### 3. Prediction with Threshold

#### `predict_with_threshold(json_data, threshold)`
Predict and classify based on threshold.

**Returns:**
```python
{
    'success': True,
    'prediction': 98.25,
    'threshold': 100.0,
    'classification': 'low',  # or 'high'
    'is_above_threshold': False
}
```

### 4. Input Validation

#### `validate_input(json_data)`
Validate input data for prediction.

**Returns:**
```python
{
    'valid': True,
    'errors': [],
    'warnings': ['Missing important field: ...']
}
```

### 5. Feature Importance

#### `get_feature_importance(top_n=20)`
Get feature importance from the model.

**Returns:**
- `pd.DataFrame`: DataFrame with feature names and importance scores

**Example:**
```python
importance = inference.get_feature_importance(top_n=10)
print(importance)
```

### 6. Confidence Interval

#### `predict_with_confidence_interval(json_data, confidence=0.95)`
Predict with confidence interval.

**Returns:**
```python
{
    'success': True,
    'prediction': 98.25,
    'confidence_interval': {
        'lower': 68.0,
        'upper': 128.5,
        'confidence': 0.95,
        'margin': 30.3
    }
}
```

### 7. Prediction Explanation

#### `get_prediction_explanation(json_data)`
Get explanation for prediction (feature contributions).

**Returns:**
```python
{
    'success': True,
    'prediction': 98.25,
    'explanation': {
        'prediction': 98.25,
        'top_features': [...],
        'input_features': {...},
        'model_performance': {...}
    }
}
```

### 8. Future Predictions

#### `predict_future(building_id, start_time, hours=24, weather_data=None, building_data=None)`
Predict electricity consumption for future hours.

**Parameters:**
- `building_id`: Building ID
- `start_time`: Start timestamp (string or datetime)
- `hours`: Number of hours to predict
- `weather_data`: Optional list of weather data for each hour
- `building_data`: Optional building metadata

**Returns:**
- `pd.DataFrame`: DataFrame with predictions for each hour

**Example:**
```python
future_predictions = inference.predict_future(
    building_id='Bear_education_Sharon',
    start_time='2016-01-01T00:00:00',
    hours=24
)
```

### 9. City-Level Predictions

#### `predict_by_city(site_id, timestamp, weather_data=None)`
Predict electricity consumption for all buildings in a city/site.

**Parameters:**
- `site_id`: Site/City ID
- `timestamp`: Timestamp for prediction
- `weather_data`: Optional weather data for the site

**Returns:**
- `pd.DataFrame`: DataFrame with predictions for all buildings

**Example:**
```python
city_predictions = inference.predict_by_city(
    site_id='Bear',
    timestamp='2016-01-01T12:00:00',
    weather_data={
        'airTemperature': 15.0,
        'cloudCoverage': 30.0,
        'windSpeed': 5.0
    }
)
```

#### `get_city_summary(site_id, timestamp, weather_data=None)`
Get summary statistics for a city/site.

**Returns:**
```python
{
    'site_id': 'Bear',
    'timestamp': '2016-01-01T12:00:00',
    'total_buildings': 50,
    'total_consumption': 5000.0,
    'average_consumption': 100.0,
    'median_consumption': 95.0,
    'min_consumption': 20.0,
    'max_consumption': 300.0,
    'std_consumption': 50.0,
    'by_usage_type': {...}
}
```

## Input Data Format

The input data should be a dictionary with the following fields:

### Required Fields
- `time` or `timestamp`: Timestamp for prediction (ISO format string or datetime)
- `building_id` or `building_code`: Building identifier

### Important Fields (recommended)
- `sqm`: Building area in square meters
- `airTemperature`: Air temperature in Celsius
- `primaryspaceusage`: Primary space usage type
- `site_id`: Site/City ID

### Continuous Features
- `sqm`: Building area (m²)
- `yearbuilt`: Year building was built
- `numberoffloors`: Number of floors
- `occupants`: Number of occupants
- `airTemperature`: Air temperature (°C)
- `cloudCoverage`: Cloud coverage (%)
- `dewTemperature`: Dew temperature (°C)
- `windSpeed`: Wind speed (m/s)
- `seaLvlPressure`: Sea level pressure (hPa)
- `precipDepth1HR`: Precipitation depth 1 hour (mm)

### Categorical Features
- `primaryspaceusage`: Primary space usage (e.g., 'Education', 'Office')
- `sub_primaryspaceusage`: Sub-primary space usage
- `site_id`: Site ID
- `timezone`: Timezone (e.g., 'US/Pacific')
- `season`: Season (automatically calculated from timestamp)

### Alternative Field Names
The class supports alternative field names:
- `air_temperature` → `airTemperature`
- `cloud_coverage` → `cloudCoverage`
- `dew_temperature` → `dewTemperature`
- `wind_speed` → `windSpeed`
- `sea_lvl_pressure` → `seaLvlPressure`
- `precip_depth_1hr` → `precipDepth1HR`
- `building_code` → `building_id`

## Time Features

Time features are automatically created from the timestamp:
- `hour`: Hour of day (0-23)
- `day_of_week`: Day of week (0-6)
- `month`: Month (1-12)
- `year`: Year
- `is_weekend`: Whether it's weekend (0 or 1)
- `season`: Season (Winter, Spring, Summer, Fall)
- Cyclical encodings: `hour_sin`, `hour_cos`, `day_of_week_sin`, `day_of_week_cos`, `month_sin`, `month_cos`

## Lag Features

If historical data is available, lag features are automatically calculated:
- `electricity_lag1`: Consumption 1 hour ago
- `electricity_lag24`: Consumption 24 hours ago (same hour yesterday)
- `electricity_lag168`: Consumption 168 hours ago (same hour last week)
- `electricity_rolling_mean_24h`: Rolling mean of last 24 hours
- `electricity_rolling_std_24h`: Rolling standard deviation of last 24 hours
- `electricity_rolling_mean_7d`: Rolling mean of last 7 days

## Dashboard Integration

The inference class is designed for dashboard applications. Key features:

1. **City-Level Aggregation**: Use `get_city_summary()` to get city-wide statistics
2. **Batch Processing**: Use `predict_batch()` for multiple buildings
3. **Future Forecasting**: Use `predict_future()` for time-series predictions
4. **Validation**: Use `validate_input()` to check data quality
5. **Explanations**: Use `get_prediction_explanation()` for interpretability

## Example: Dashboard Usage

```python
from src.inference import ElectricityConsumptionInference

inference = ElectricityConsumptionInference()

# Get city summary for dashboard
city_summary = inference.get_city_summary(
    site_id='Bear',
    timestamp='2016-01-01T12:00:00',
    weather_data={
        'airTemperature': 15.0,
        'cloudCoverage': 30.0,
        'windSpeed': 5.0
    }
)

# Display in dashboard
print(f"Total Consumption: {city_summary['total_consumption']:.2f} kWh")
print(f"Average per Building: {city_summary['average_consumption']:.2f} kWh")
print(f"Buildings: {city_summary['total_buildings']}")

# Get predictions by usage type
for usage_type, stats in city_summary['by_usage_type'].items():
    print(f"{usage_type}: {stats['total']:.2f} kWh ({stats['count']} buildings)")
```

## Error Handling

The class handles missing values gracefully:
- Missing continuous features are set to 0.0
- Missing categorical features are set to 'Unknown'
- Unknown categorical values are mapped to the first class in the encoder
- Missing historical data results in lag features set to 0.0

## Model Information

The model information is available via:
- `inference.model_info`: Model metadata and performance metrics
- `inference.features_info`: Feature information
- `inference.model_info['performance']`: Test performance metrics

## Notes

1. **Lag Features**: Lag features require historical data. If `processed_data.parquet` is not available, lag features will be set to 0.0 or mean values.

2. **Performance**: The model has the following test performance:
   - R²: ~0.984
   - RMSE: ~30.3 kWh
   - MAE: ~7.6 kWh

3. **Predictions**: All predictions are non-negative (clamped to 0.0 minimum).

4. **Thread Safety**: The class is not thread-safe. For concurrent predictions, create separate instances.

## See Also

- `scripts/test_inference.py`: Test script with examples
- `src/example_usage.py`: Usage examples
- `docs/DATA_EXPLAINATION.md`: Data documentation
