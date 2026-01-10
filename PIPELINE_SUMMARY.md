# 📊 Tóm tắt Pipeline Dự đoán Điện Tiêu thụ

## 🎯 Mục tiêu

Xây dựng pipeline để dự đoán lượng điện tiêu thụ của các tòa nhà sử dụng các mô hình hồi quy (XGBoost, LightGBM, Random Forest, Linear Regression).

## 🔄 Quy trình Pipeline

```
┌─────────────────┐
│  1. EDA Analysis  │  → Phân tích dataset, missing values, distributions
└────────┬────────┘
         │
┌────────▼──────────────┐
│ 2. Data Preprocessing │  → Merge data, feature engineering, xử lý missing
└────────┬──────────────┘
         │
┌────────▼──────────┐
│ 3. Train Models   │  → Train XGBoost, LightGBM, RF, Linear Regression
└────────┬──────────┘
         │
┌────────▼──────────────┐
│ 4. Evaluate Models    │  → So sánh models, tạo visualizations
└────────┬──────────────┘
         │
┌────────▼──────────┐
│ 5. Predict        │  → Dự đoán cho dữ liệu mới
└───────────────────┘
```

## 📁 Cấu trúc Files

### Scripts (trong `scripts/`)
- `01_eda_analysis.py` - Phân tích EDA
- `02_data_preprocessing.py` - Xử lý dữ liệu và feature engineering
- `03_train_models.py` - Training các models
- `04_evaluate_models.py` - Đánh giá và visualization
- `05_predict.py` - Prediction
- `run_full_pipeline.py` - Chạy toàn bộ pipeline

### Output (trong `output/`)
- `processed_data.parquet` - Dữ liệu đã xử lý
- `models/` - Các models đã train
  - `xgboost.pkl`
  - `lightgbm.pkl`
  - `randomforest.pkl`
  - `linearregression.pkl`
  - `scaler.pkl`
  - `label_encoders.pkl`
  - `model_info.json`
  - `results_comparison.csv`
- `visualizations/` - Các biểu đồ đánh giá
- `predictions.csv` - Kết quả dự đoán

### Analysis (trong `analysis/`)
- Các biểu đồ từ EDA analysis

## 🔑 Features Quan trọng

### ⭐⭐⭐ Rất quan trọng:
- `sqm` - Diện tích
- `occupants` - Số người
- `primaryspaceusage` - Loại sử dụng
- `airTemperature` - Nhiệt độ
- `electricity_lag1` - Lag 1 giờ (correlation cao nhất ~0.98)

### ⭐⭐ Quan trọng:
- `yearbuilt` - Năm xây dựng
- `numberoffloors` - Số tầng
- `hour` - Giờ trong ngày
- `day_of_week` - Ngày trong tuần
- `month` - Tháng
- `electricity_lag24` - Lag 24 giờ
- `electricity_rolling_mean_24h` - Trung bình 24h

## 📈 Models được sử dụng

1. **XGBoost** - Gradient Boosting, thường cho kết quả tốt nhất
2. **LightGBM** - Gradient Boosting nhanh hơn
3. **Random Forest** - Ensemble method
4. **Linear Regression** - Baseline model

## 🎯 Metrics đánh giá

- **RMSE** - Root Mean Squared Error
- **MAE** - Mean Absolute Error  
- **R²** - R-squared (hệ số xác định)
- **MAPE** - Mean Absolute Percentage Error

## ⚡ Cách sử dụng nhanh

### Option 1: Chạy toàn bộ pipeline
```bash
python scripts/run_full_pipeline.py
```

### Option 2: Chạy từng bước
```bash
# Bước 1: EDA
python scripts/01_eda_analysis.py

# Bước 2: Preprocessing
python scripts/02_data_preprocessing.py

# Bước 3: Training
python scripts/03_train_models.py

# Bước 4: Evaluation
python scripts/04_evaluate_models.py

# Bước 5: Prediction
python scripts/05_predict.py
```

## 📊 Kết quả mong đợi

Sau khi chạy pipeline, bạn sẽ có:

1. **EDA Analysis**: Hiểu rõ về dataset, missing values, distributions
2. **Processed Data**: Dữ liệu sạch, đã feature engineering
3. **Trained Models**: 4 models đã train, model tốt nhất được chọn tự động
4. **Evaluations**: So sánh models, feature importance, visualizations
5. **Predictions**: Kết quả dự đoán cho dữ liệu mới

## ⚠️ Lưu ý

1. **Thời gian**: Training có thể mất vài phút đến vài giờ tùy số lượng buildings
2. **Memory**: Dataset lớn (~25M records), cần đủ RAM
3. **Missing values**: Một số features có nhiều missing (occupants ~86%, yearbuilt ~50%)
4. **Time series**: Chia train/test theo thời gian, không random

## 🔧 Tùy chỉnh

### Thay đổi số buildings để train
Trong `scripts/03_train_models.py`:
```python
sample_size = min(200, df['building_id'].nunique())  # Mặc định 200
```

### Thay đổi hyperparameters
Trong `scripts/03_train_models.py`, điều chỉnh các tham số của từng model.

### Thêm features mới
1. Thêm trong `scripts/02_data_preprocessing.py`
2. Cập nhật `features_info.json` nếu cần
3. Retrain models

## 📚 Tài liệu

- [QUICK_START.md](./QUICK_START.md) - Hướng dẫn nhanh
- [README_PIPELINE.md](./README_PIPELINE.md) - Hướng dẫn chi tiết
- [DATA_EXPLAINATION.md](./DATA_EXPLAINATION.md) - Giải thích dataset

---

**Happy Coding! 🚀**

