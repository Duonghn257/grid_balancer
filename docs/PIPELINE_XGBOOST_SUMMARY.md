# 🚀 Pipeline Dự đoán Điện Tiêu thụ với XGBoost (Cho DiCE)

## 📋 Tổng quan

Pipeline hoàn chỉnh để dự đoán lượng điện tiêu thụ sử dụng **XGBoost** làm mô hình duy nhất, được tối ưu hóa cho việc tích hợp **DiCE (Diverse Counterfactual Explanations)**.

## 🎯 Tại sao XGBoost?

1. **Performance tốt nhất**: Test R² = 0.9843, RMSE = 30.30 kWh
2. **Tương thích DiCE**: Hỗ trợ backend='sklearn' với wrapper class
3. **Feature importance**: Có thể giải thích được
4. **Production-ready**: Nhanh, ổn định, dễ deploy

## 🔄 Pipeline Flow

```
┌─────────────────────┐
│ 1. EDA Analysis     │ → Phân tích dataset
└──────────┬──────────┘
           │
┌──────────▼──────────────┐
│ 2. Data Preprocessing   │ → Merge, feature engineering
└──────────┬──────────────┘
           │
┌──────────▼──────────────────────┐
│ 3. Train XGBoost for DiCE       │ → Train với wrapper class
└──────────┬──────────────────────┘
           │
┌──────────▼──────────────────┐
│ 4. Predict with XGBoost     │ → Dự đoán lượng điện
└──────────┬──────────────────┘
           │
┌──────────▼──────────────────────┐
│ 5. DiCE Counterfactuals (TODO)  │ → Gợi ý điều chỉnh features
└─────────────────────────────────┘
```

## 📁 Cấu trúc Files

### Scripts chính:
- `scripts/01_eda_analysis.py` - EDA
- `scripts/02_data_preprocessing.py` - Preprocessing
- `scripts/06_train_xgboost_for_dice.py` - **Train XGBoost cho DiCE**
- `scripts/07_predict_with_xgboost.py` - **Prediction với XGBoost**

### Output:
- `output/models/xgboost_dice.pkl` - XGBoost model
- `output/models/xgboost_wrapped_dice.pkl` - Wrapped model (cho DiCE)
- `output/models/label_encoders_dice.pkl` - Label encoders
- `output/models/model_info_dice.json` - Model info
- `output/predictions_xgboost.csv` - Kết quả dự đoán

## 🚀 Quick Start

### 1. EDA Analysis
```bash
python scripts/01_eda_analysis.py
```

### 2. Data Preprocessing
```bash
python scripts/02_data_preprocessing.py
```

### 3. Train XGBoost Model
```bash
python scripts/06_train_xgboost_for_dice.py
```

**Kết quả:**
- Model được train và lưu
- Wrapper class được tạo sẵn cho DiCE
- Test R² ~ 0.98, RMSE ~ 30 kWh

### 4. Prediction
```bash
python scripts/07_predict_with_xgboost.py
```

**Kết quả:**
- `output/predictions_xgboost.csv` - Kết quả dự đoán
- Metrics: RMSE, MAE, R²

## 📊 Model Performance

### XGBoost Model:
- **Train R²**: 0.9939
- **Test R²**: 0.9843
- **Train RMSE**: 18.32 kWh
- **Test RMSE**: 30.30 kWh
- **Test MAE**: 7.61 kWh
- **Test MAPE**: 16.88%

### So sánh với các models khác:

| Model | Test R² | Test RMSE | DiCE Compatible |
|-------|---------|-----------|-----------------|
| **XGBoost** | **0.9843** | **30.30** | ✅ Yes |
| LightGBM | 0.9683 | 34.49 | ✅ Yes |
| Random Forest | 0.9702 | 33.45 | ✅ Yes |
| Linear Regression | 0.9786 | 35.38 | ✅ Yes |

## 🔧 XGBoost Hyperparameters

```python
xgb_model = xgb.XGBRegressor(
    n_estimators=200,        # Số cây
    max_depth=8,             # Độ sâu tối đa
    learning_rate=0.05,      # Tốc độ học
    subsample=0.8,           # Tỷ lệ sample
    colsample_bytree=0.8,    # Tỷ lệ features
    min_child_weight=3,       # Trọng số tối thiểu
    random_state=42,
    n_jobs=-1,
    objective='reg:squarederror',
    eval_metric='rmse'
)
```

## 🎯 Features được sử dụng

### Continuous Features (28):
- `sqm`, `yearbuilt`, `numberoffloors`, `occupants`
- `airTemperature`, `cloudCoverage`, `dewTemperature`, `windSpeed`, `seaLvlPressure`, `precipDepth1HR`
- `hour`, `day_of_week`, `month`, `year`, `is_weekend`
- `hour_sin`, `hour_cos`, `day_of_week_sin`, `day_of_week_cos`, `month_sin`, `month_cos`
- `electricity_lag1`, `electricity_lag24`, `electricity_lag168`
- `electricity_rolling_mean_24h`, `electricity_rolling_std_24h`, `electricity_rolling_mean_7d`

### Categorical Features (5):
- `primaryspaceusage`, `sub_primaryspaceusage`, `site_id`, `timezone`, `season`

## 🔍 DiCE Integration

### Wrapper Class:
Model đã được wrap trong `XGBoostWrapper` để:
- Tự động encode categorical features
- Xử lý unknown values
- Tương thích với DiCE backend='sklearn'

### Sử dụng với DiCE:
```python
# Load wrapped model
with open('output/models/xgboost_wrapped_dice.pkl', 'rb') as f:
    model = pickle.load(f)

# Sử dụng với DiCE
dice_model = dice_ml.Model(
    model=model,
    backend='sklearn',
    model_type='regressor'
)
```

## 📈 Use Cases

### 1. Dự đoán lượng điện tiêu thụ
```python
prediction = model.predict(building_features)
# Output: 250.5 kWh
```

### 2. Kiểm tra threshold
```python
THRESHOLD = 300  # kWh
if prediction > THRESHOLD:
    # Cần điều chỉnh features
    # → Sử dụng DiCE để gợi ý
```

### 3. DiCE Counterfactuals (Sẽ triển khai)
```python
counterfactuals = explainer.generate_counterfactuals(
    building_features,
    desired_range=[0, THRESHOLD]
)
# → Gợi ý: Giảm occupants, sqm, điều chỉnh temperature
```

## ⚙️ Tùy chỉnh

### Thay đổi số lượng buildings để train:
Trong `scripts/06_train_xgboost_for_dice.py`:
```python
sample_size = min(200, df['building_id'].nunique())  # Mặc định 200
# Để train toàn bộ:
sample_size = df['building_id'].nunique()
```

### Thay đổi hyperparameters:
Trong `scripts/06_train_xgboost_for_dice.py`, điều chỉnh các tham số của `XGBRegressor`.

## 📚 Tài liệu liên quan

- [DICE_INTEGRATION.md](./DICE_INTEGRATION.md) - Hướng dẫn chi tiết về DiCE
- [README_PIPELINE.md](./README_PIPELINE.md) - Pipeline tổng quát
- [DATA_EXPLAINATION.md](./DATA_EXPLAINATION.md) - Giải thích dataset

## 🎯 Next Steps

1. ✅ Training XGBoost với wrapper class
2. ✅ Prediction pipeline
3. ⏳ DiCE integration script
4. ⏳ Visualization cho counterfactuals
5. ⏳ API endpoint

---

**Pipeline này đã sẵn sàng cho việc tích hợp DiCE!** 🚀
