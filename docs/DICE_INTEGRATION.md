# 🔍 DiCE Integration Guide - Diverse Counterfactual Explanations

## 📋 Tổng quan

DiCE (Diverse Counterfactual Explanations) là một thư viện để tạo các counterfactual explanations - tức là gợi ý các cách thay đổi features để đạt được kết quả mong muốn.

**Use case trong bài toán này:**
- Khi lượng điện tiêu thụ dự đoán **vượt quá ngưỡng threshold**
- DiCE sẽ gợi ý các cách **điều chỉnh features** để giảm lượng điện tiêu thụ xuống dưới ngưỡng

## 🎯 Tại sao chọn XGBoost?

### So sánh tương thích với DiCE:

| Model | DiCE Backend | Tương thích | Ghi chú |
|-------|--------------|-------------|---------|
| **XGBoost** | `sklearn` | ✅ **Tốt nhất** | Cần wrapper class để encode categorical |
| **LightGBM** | `sklearn` | ✅ Tốt | Tương tự XGBoost |
| **Random Forest** | `sklearn` | ✅ Tốt | Native sklearn, không cần wrapper |
| **Linear Regression** | `sklearn` | ✅ Tốt | Đơn giản nhất nhưng accuracy thấp hơn |

### Lý do chọn XGBoost:

1. **Performance tốt**: Test R² = 0.9843 (tốt nhất trong các models)
2. **Tương thích DiCE**: Hỗ trợ backend='sklearn'
3. **Wrapper class**: Đã tạo sẵn `XGBoostWrapper` để tự động encode categorical features
4. **Feature importance**: Có thể giải thích được các features quan trọng

## 🔧 Cấu trúc Wrapper Class

```python
class XGBoostWrapper:
    """
    Wrapper class để tự động encode categorical features trước khi predict
    Tương thích với DiCE (Diverse Counterfactual Explanations)
    """
    def __init__(self, model, label_encoders, categorical_features):
        self.model = model
        self.label_encoders = label_encoders
        self.categorical_features = categorical_features
    
    def predict(self, X):
        # Tự động encode categorical features
        # Xử lý unknown values
        # Predict với XGBoost model
        return predictions
```

## 📊 Pipeline với XGBoost cho DiCE

### Bước 1: Training Model
```bash
python scripts/06_train_xgboost_for_dice.py
```

**Output:**
- `output/models/xgboost_dice.pkl` - XGBoost model gốc
- `output/models/xgboost_wrapped_dice.pkl` - Wrapped model cho DiCE
- `output/models/label_encoders_dice.pkl` - Label encoders
- `output/models/model_info_dice.json` - Thông tin model

### Bước 2: Prediction
```bash
python scripts/07_predict_with_xgboost.py
```

**Output:**
- `output/predictions_xgboost.csv` - Kết quả dự đoán

### Bước 3: DiCE Counterfactual Explanations (Sẽ triển khai)

```python
import dice_ml
from dice_ml import Dice

# Load wrapped model
with open('output/models/xgboost_wrapped_dice.pkl', 'rb') as f:
    model = pickle.load(f)

# Load data
df = pd.read_parquet("./output/processed_data.parquet")

# Tạo DiCE Data object
dice_data = dice_ml.Data(
    dataframe=df,
    continuous_features=['sqm', 'occupants', 'airTemperature', ...],
    outcome_name='electricity_consumption'
)

# Tạo DiCE Model object
dice_model = dice_ml.Model(
    model=model,
    backend='sklearn',
    model_type='regressor'
)

# Tạo DiCE Explainer
explainer = Dice(dice_data, dice_model, method='random')

# Tạo counterfactuals
counterfactuals = explainer.generate_counterfactuals(
    query_instance,
    total_CFs=5,
    desired_range=[0, THRESHOLD],  # Mục tiêu: <= threshold
    permitted_range=permitted_range
)
```

## 🎯 Features có thể điều chỉnh (Adjustable Features)

### ⭐ Có thể điều chỉnh:
- `sqm` - Diện tích (có thể giảm)
- `occupants` - Số người (có thể giảm)
- `airTemperature` - Nhiệt độ (có thể điều chỉnh HVAC)
- `hour` - Giờ sử dụng (có thể thay đổi lịch)
- `day_of_week` - Ngày trong tuần
- `month` - Tháng (không thể điều chỉnh trực tiếp)
- `cloudCoverage`, `windSpeed` - Thời tiết (không thể điều chỉnh)

### ❌ Không thể điều chỉnh:
- `yearbuilt` - Năm xây dựng (cố định)
- `numberoffloors` - Số tầng (cố định)
- `primaryspaceusage` - Loại sử dụng (cố định)
- `site_id`, `timezone` - Địa lý (cố định)
- `electricity_lag1`, `electricity_lag24` - Lag features (phụ thuộc dữ liệu quá khứ)
- `electricity_rolling_mean_24h` - Rolling statistics (phụ thuộc dữ liệu quá khứ)

## 📝 Workflow DiCE

```
1. Load model và dữ liệu
   ↓
2. Dự đoán lượng điện tiêu thụ
   ↓
3. Kiểm tra: prediction > THRESHOLD?
   ↓
4. Nếu có: Tạo counterfactual explanations
   ↓
5. DiCE gợi ý các cách điều chỉnh features
   ↓
6. Hiển thị các phương án và phân tích
```

## 🔍 Ví dụ sử dụng

### Scenario: Building có điện tiêu thụ cao

```python
# 1. Dự đoán
prediction = model.predict(building_features)  # 500 kWh

# 2. Kiểm tra threshold
THRESHOLD = 300  # kWh
if prediction > THRESHOLD:
    # 3. Tạo counterfactuals
    counterfactuals = explainer.generate_counterfactuals(
        building_features,
        total_CFs=5,
        desired_range=[0, THRESHOLD]
    )
    
    # 4. Kết quả: DiCE gợi ý
    # - Giảm occupants từ 200 → 150
    # - Giảm sqm từ 5000 → 4500
    # - Điều chỉnh airTemperature từ 25°C → 23°C
    # → Dự đoán mới: 280 kWh ✅
```

## ⚙️ Cấu hình DiCE

### Methods:
- **`method='random'`**: Nhanh, phù hợp cho testing
- **`method='genetic'`**: Chậm hơn nhưng kết quả tốt hơn, phù hợp cho production

### Parameters:
```python
counterfactuals = explainer.generate_counterfactuals(
    query_instance,
    total_CFs=5,                    # Số lượng counterfactuals
    desired_range=[0, THRESHOLD],   # Khoảng giá trị mong muốn
    permitted_range={                # Giới hạn thay đổi
        'sqm': [min_sqm, max_sqm],
        'occupants': [min_occ, max_occ],
        ...
    },
    proximity_weight=0.5,           # Trọng số cho proximity
    diversity_weight=1.0,            # Trọng số cho diversity
    sparsity_weight=0.1              # Trọng số cho sparsity
)
```

## 📚 Tài liệu tham khảo

- [DiCE Documentation](https://github.com/interpretml/DiCE)
- [DiCE Paper](https://arxiv.org/abs/1905.07697)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

## 🚀 Next Steps

1. ✅ Training XGBoost model với wrapper class
2. ✅ Prediction pipeline
3. ⏳ Triển khai DiCE integration script
4. ⏳ Tạo visualization cho counterfactuals
5. ⏳ Tạo API endpoint cho DiCE recommendations

---

**Lưu ý**: DiCE yêu cầu các features có thể điều chỉnh phải là continuous hoặc categorical đã được encode. Wrapper class đã xử lý việc này tự động.
