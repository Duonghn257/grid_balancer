# Predict Future với Current Consumption + DICE Monitoring

## Tổng quan

Tính năng này cho phép bạn:
1. **Input**: Truyền `electricity_consumption` tại thời điểm hiện tại (ví dụ: 21:00 = 50.0 kWh)
2. **Dự đoán đệ quy**: Model sẽ dùng prediction tại 22:00 để predict 23:00, rồi dùng 23:00 để predict 00:00, cứ như vậy
3. **DICE Monitoring**: Tự động kiểm tra threshold và đưa ra recommendations khi vượt ngưỡng

## Cách sử dụng

### Ví dụ cơ bản

```python
from src.dice_explainer import DiceExplainer

# Khởi tạo
explainer = DiceExplainer()

# Input data tại thời điểm 21:00
json_data = {
    'time': '2016-01-01T21:00:00',
    'building_id': 'Bear_education_Sharon',
    'site_id': 'Bear',
    'primaryspaceusage': 'Education',
    'sqm': 5261.7,
    'yearbuilt': 1953,
    'numberoffloors': 5,
    'occupants': 200,
    'timezone': 'US/Pacific',
    'airTemperature': 25.0,
    'cloudCoverage': 30.0,
    'dewTemperature': 18.0,
    'windSpeed': 2.6,
    'seaLvlPressure': 1020.7,
    'precipDepth1HR': 0.0
}

# Điện tiêu thụ hiện tại tại 21:00
current_consumption = 50.0  # kWh

# Dự đoán 24 giờ tới với threshold = 50.0 kWh
result = explainer.predict_future_with_monitoring(
    json_data=json_data,
    current_electricity_consumption=current_consumption,
    hours=24,
    threshold=50.0
)

# Xem kết quả
print(f"Total hours predicted: {result['summary']['total_hours']}")
print(f"Hours above threshold: {result['summary']['hours_above_threshold']}")
print(f"Max consumption: {result['summary']['max_consumption']:.2f} kWh")

# Xem alerts
for alert in result['alerts']:
    print(f"\n⚠️  Alert at {alert['timestamp']}:")
    print(f"   Predicted: {alert['predicted_consumption']:.2f} kWh")
    print(f"   Threshold: {alert['threshold']:.2f} kWh")
    print(f"   Exceeded by: {alert['exceeded_by']:.2f} kWh")
    
    # Xem recommendations
    if alert['recommendations']:
        print(f"   📋 Recommendations:")
        for rec in alert['recommendations']:
            print(f"      - Reduce to {rec['predicted_consumption']:.2f} kWh")
            print(f"        Reduction: {rec['reduction']:.2f} kWh ({rec['reduction_pct']:.1f}%)")
```

## Cách hoạt động

### 1. Dự đoán đệ quy

- **Bước 1**: Dùng `current_electricity_consumption` (50.0 kWh tại 21:00) để tính lag features
- **Bước 2**: Predict consumption tại 22:00
- **Bước 3**: Dùng prediction tại 22:00 để tính lag features và predict 23:00
- **Bước 4**: Tiếp tục như vậy cho đến hết 24 giờ

### 2. Lag Features được sử dụng

- `electricity_lag24`: Consumption 24 giờ trước (nếu có)
- `electricity_rolling_mean_4h`: Trung bình 4 giờ gần nhất
- `electricity_rolling_mean_24h`: Trung bình 24 giờ gần nhất

### 3. Threshold Monitoring

- Tại mỗi time step, hệ thống kiểm tra xem `predicted_consumption > threshold`
- Nếu vượt threshold, hệ thống sẽ:
  1. Tạo alert với thông tin chi tiết
  2. Gọi DICE để generate recommendations
  3. Trả về các recommendations để giảm consumption xuống dưới threshold

## API Reference

### `predict_future_with_monitoring()`

**Parameters:**
- `json_data` (Dict): Building và weather data tại start time
- `current_electricity_consumption` (float): Điện tiêu thụ hiện tại (kWh)
- `hours` (int): Số giờ cần dự đoán (mặc định: 24)
- `threshold` (float): Ngưỡng cảnh báo (kWh, mặc định: 50.0)
- `weather_data` (Optional[List[Dict]]): Weather data cho từng giờ (optional)

**Returns:**
```python
{
    'success': True,
    'predictions': DataFrame,  # Predictions cho từng giờ
    'alerts': List[Dict],      # Các alerts khi vượt threshold
    'total_alerts': int,        # Tổng số alerts
    'threshold': float,         # Threshold đã sử dụng
    'summary': {
        'total_hours': int,
        'hours_above_threshold': int,
        'max_consumption': float,
        'min_consumption': float,
        'mean_consumption': float,
        'first_alert_hour': int or None,
        'last_alert_hour': int or None
    }
}
```

### `predict_future_with_current_consumption()` (inference.py)

Method này có thể được gọi trực tiếp nếu bạn chỉ cần predictions mà không cần DICE monitoring:

```python
from src.inference import ElectricityConsumptionInference

inference = ElectricityConsumptionInference()

predictions_df = inference.predict_future_with_current_consumption(
    building_id='Bear_education_Sharon',
    start_time='2016-01-01T21:00:00',
    current_electricity_consumption=50.0,
    hours=24
)
```

## Lưu ý

1. **Model không sử dụng `electricity_consumption` trực tiếp như feature**: Model sử dụng lag features (lag24, rolling means) được tính từ consumption history
2. **Dự đoán đệ quy có thể tích lũy sai số**: Mỗi prediction phụ thuộc vào predictions trước đó, nên sai số có thể tích lũy theo thời gian
3. **DICE recommendations**: Chỉ được generate khi threshold bị vượt, và có thể mất thời gian nếu có nhiều alerts

## Ví dụ nâng cao

### Với weather data cho từng giờ

```python
# Tạo weather data cho 24 giờ
weather_data = []
for i in range(24):
    weather_data.append({
        'airTemperature': 25.0 + i * 0.1,  # Nhiệt độ tăng dần
        'cloudCoverage': 30.0,
        'dewTemperature': 18.0,
        'windSpeed': 2.6,
        'seaLvlPressure': 1020.7,
        'precipDepth1HR': 0.0
    })

result = explainer.predict_future_with_monitoring(
    json_data=json_data,
    current_electricity_consumption=50.0,
    hours=24,
    threshold=50.0,
    weather_data=weather_data
)
```

### Export kết quả

```python
# Export predictions to CSV
result['predictions'].to_csv('predictions.csv', index=False)

# Export alerts to JSON
import json
with open('alerts.json', 'w') as f:
    json.dump(result['alerts'], f, indent=2, default=str)
```

## Troubleshooting

### Lỗi: "building_id is required"
- Đảm bảo `json_data` có `building_id` hoặc `building_code`

### Lỗi: "time is required"
- Đảm bảo `json_data` có `time` hoặc `timestamp`

### Predictions quá cao/thấp
- Kiểm tra `current_electricity_consumption` có đúng không
- Kiểm tra weather data có hợp lý không
- Kiểm tra building metadata (sqm, occupants, etc.)

### DICE recommendations không được generate
- Kiểm tra xem có alerts không (predictions có vượt threshold không)
- Kiểm tra log để xem có lỗi khi generate recommendations không
