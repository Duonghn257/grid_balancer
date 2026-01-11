# 🔮 Inference Class Guide

Hướng dẫn sử dụng `ElectricityConsumptionInference` class để dự đoán lượng điện tiêu thụ.

## 📋 Tổng quan

`ElectricityConsumptionInference` là class chính để:
- ✅ Dự đoán lượng điện tiêu thụ từ JSON data
- ✅ Dự đoán cho một building cụ thể
- ✅ Batch prediction
- ✅ Threshold checking
- ✅ Confidence intervals
- ✅ Feature importance
- ✅ Data validation
- ✅ Future predictions

## 🚀 Quick Start

### 1. Khởi tạo

```python
from scripts.inference import ElectricityConsumptionInference

# Khởi tạo với default paths
inference = ElectricityConsumptionInference()

# Hoặc chỉ định custom paths
inference = ElectricityConsumptionInference(
    model_path="output/models/xgboost_wrapped_dice.pkl",
    model_info_path="output/models/model_info_dice.json",
    features_info_path="output/features_info.json"
)
```

### 2. Predict từ JSON

```python
json_data = {
    'time': '2016-01-01T21:00:00',
    'building_code': 'Bear_education_Sharon',
    'site_id': 'Bear',
    'primaryspaceusage': 'Education',
    'sqm': 5261.7,
    'yearbuilt': 1953,
    'numberoffloors': 5,
    'timezone': 'US/Pacific',
    'air_temperature': 6.1,
    'wind_speed': 2.6,
    # ... các features khác
}

result = inference.predict_from_json(json_data)
print(f"Predicted consumption: {result['predicted_consumption']:.2f} kWh")
```

## 📚 API Reference

### Core Functions

#### `predict_from_json(json_data)`
Dự đoán từ JSON data.

**Input:**
```python
json_data = {
    'time': '2016-01-01T21:00:00',  # Required
    'building_code': 'Bear_education_Sharon',  # Required
    'sqm': 5261.7,
    'air_temperature': 6.1,
    # ... các features khác
}
```

**Output:**
```python
{
    'predicted_consumption': 98.25,  # kWh
    'building_id': 'Bear_education_Sharon',
    'timestamp': '2016-01-01T21:00:00',
    'features_used': 33,
    'prediction_date': '2024-01-01T12:00:00'
}
```

#### `predict_building(building_id, timestamp, df_data=None)`
Dự đoán cho một building cụ thể.

```python
result = inference.predict_building(
    building_id='Bear_education_Sharon',
    timestamp='2016-01-01T21:00:00',
    df_data=df  # Optional: DataFrame chứa building info
)
```

#### `predict_with_threshold(json_data, threshold)`
Dự đoán và kiểm tra threshold.

```python
result = inference.predict_with_threshold(
    json_data,
    threshold=100.0  # kWh
)

# Result includes:
# - predicted_consumption
# - exceeds_threshold: True/False
# - difference: Số kWh vượt quá
# - recommendation: Gợi ý
```

#### `predict_batch(json_list)`
Dự đoán cho nhiều records cùng lúc.

```python
json_list = [json_data1, json_data2, ...]
results_df = inference.predict_batch(json_list)
```

### Utility Functions

#### `predict_future(building_id, start_time, hours=24)`
Dự đoán lượng điện trong tương lai (nhiều giờ).

```python
future_df = inference.predict_future(
    building_id='Bear_education_Sharon',
    start_time='2016-01-01T00:00:00',
    hours=24  # Dự đoán 24 giờ
)
```

#### `predict_with_confidence_interval(json_data, confidence=0.95)`
Dự đoán với confidence interval.

```python
result = inference.predict_with_confidence_interval(
    json_data,
    confidence=0.95  # 95% confidence
)

# Result includes:
# - predicted_consumption
# - lower_bound
# - upper_bound
# - margin
```

#### `get_feature_importance(top_n=20)`
Lấy feature importance.

```python
importance_df = inference.get_feature_importance(top_n=10)
print(importance_df)
```

#### `validate_input(json_data)`
Validate input JSON data.

```python
validation = inference.validate_input(json_data)
# Returns: {'valid': True/False, 'errors': [...], 'warnings': [...]}
```

#### `compare_buildings(building_ids, timestamp)`
So sánh dự đoán giữa nhiều buildings.

```python
results_df = inference.compare_buildings(
    building_ids=['Building1', 'Building2', 'Building3'],
    timestamp='2016-01-01T21:00:00'
)
```

#### `get_prediction_explanation(json_data)`
Giải thích prediction dựa trên feature importance.

```python
explanation = inference.get_prediction_explanation(json_data)
```

#### `get_model_info()`
Lấy thông tin về model.

```python
info = inference.get_model_info()
```

## 📊 JSON Input Format

### Required Fields:
- `time` hoặc `timestamp`: Thời điểm cần dự đoán (ISO format)
- `building_code` hoặc `building_id`: ID của building

### Important Fields:
- `sqm`: Diện tích (m²)
- `air_temperature`: Nhiệt độ không khí (°C)
- `primaryspaceusage`: Loại sử dụng chính

### Optional Fields:
- `yearbuilt`: Năm xây dựng
- `numberoffloors`: Số tầng
- `occupants`: Số người
- `site_id`: Mã site
- `timezone`: Múi giờ
- Weather features: `cloud_coverage`, `wind_speed`, `dew_temperature`, etc.

### Example:
```json
{
    "time": "2016-01-01T21:00:00",
    "building_code": "Bear_education_Sharon",
    "site_id": "Bear",
    "primaryspaceusage": "Education",
    "sqm": 5261.7,
    "yearbuilt": 1953,
    "numberoffloors": 5,
    "timezone": "US/Pacific",
    "air_temperature": 6.1,
    "wind_speed": 2.6
}
```

## 🔍 Feature Mapping

Class tự động map các fields từ JSON sang features của model:

| JSON Field | Model Feature |
|------------|---------------|
| `air_temperature` | `airTemperature` |
| `cloud_coverage` | `cloudCoverage` |
| `dew_temperature` | `dewTemperature` |
| `precip_depth_1hr` | `precipDepth1HR` |
| `sea_lvl_pressure` | `seaLvlPressure` |
| `wind_speed` | `windSpeed` |
| `building_code` | `building_id` |
| `time` | `timestamp` |

## ⚠️ Xử lý Missing Values

Class tự động xử lý missing values:

- **Continuous features**: Sử dụng giá trị mặc định hợp lý
  - `occupants`: 100.0
  - `yearbuilt`: 1980.0
  - `numberoffloors`: 3.0
  - Weather features: 0.0

- **Lag features**: 0.0 (không có dữ liệu quá khứ)

- **Rolling features**: Ước tính dựa trên `sqm` hoặc giá trị trung bình

- **Categorical features**: 'Unknown'

## 📝 Examples

### Example 1: Basic Prediction
```python
from scripts.inference import ElectricityConsumptionInference

inference = ElectricityConsumptionInference()

json_data = {
    'time': '2016-01-01T21:00:00',
    'building_code': 'Bear_education_Sharon',
    'sqm': 5261.7,
    'air_temperature': 6.1
}

result = inference.predict_from_json(json_data)
print(f"Prediction: {result['predicted_consumption']:.2f} kWh")
```

### Example 2: Threshold Checking
```python
result = inference.predict_with_threshold(json_data, threshold=100.0)

if result['exceeds_threshold']:
    print(f"⚠️ Vượt quá threshold!")
    print(f"   Cần giảm: {result['difference']:.2f} kWh")
    print(f"   Recommendation: {result['recommendation']}")
```

### Example 3: Future Prediction
```python
future_df = inference.predict_future(
    building_id='Bear_education_Sharon',
    start_time='2016-01-01T00:00:00',
    hours=24
)

# Plot hoặc analyze
print(future_df[['timestamp', 'predicted_consumption']])
```

### Example 4: Batch Prediction
```python
json_list = [
    {'time': '2016-01-01T00:00:00', 'building_code': 'Building1', ...},
    {'time': '2016-01-01T01:00:00', 'building_code': 'Building2', ...},
    ...
]

results_df = inference.predict_batch(json_list)
results_df.to_csv('predictions.csv', index=False)
```

## 🧪 Testing

Chạy test script:

```bash
python scripts/test_inference.py
```

## 📚 Tài liệu tham khảo

- [PIPELINE_XGBOOST_SUMMARY.md](./PIPELINE_XGBOOST_SUMMARY.md) - Pipeline tổng quan
- [DICE_INTEGRATION.md](./DICE_INTEGRATION.md) - DiCE integration
- [WHY_WRAPPED_MODEL.md](./WHY_WRAPPED_MODEL.md) - Giải thích wrapped model

---

**Happy Predicting! 🚀**
