# 🚀 Pipeline Dự đoán Lượng Điện Tiêu thụ

Hướng dẫn sử dụng pipeline để dự đoán lượng điện tiêu thụ của các tòa nhà sử dụng XGBoost và các mô hình hồi quy khác.

## 📋 Tổng quan

Pipeline này bao gồm 5 bước chính:

1. **EDA Analysis** - Phân tích khám phá dữ liệu
2. **Data Preprocessing** - Xử lý và feature engineering
3. **Train Models** - Training các mô hình hồi quy
4. **Evaluate Models** - Đánh giá và so sánh models
5. **Predict** - Dự đoán cho dữ liệu mới

## 📁 Cấu trúc Project

```
grid_balancer/
├── datasets/                    # Dữ liệu gốc
│   ├── metadata.csv
│   ├── electricity_cleaned.csv
│   └── weather.csv
├── scripts/                     # Các script pipeline
│   ├── 01_eda_analysis.py
│   ├── 02_data_preprocessing.py
│   ├── 03_train_models.py
│   ├── 04_evaluate_models.py
│   └── 05_predict.py
├── analysis/                    # Kết quả phân tích EDA
├── output/                      # Kết quả output
│   ├── processed_data.parquet
│   ├── models/                 # Models đã train
│   ├── visualizations/        # Biểu đồ đánh giá
│   └── predictions.csv        # Kết quả dự đoán
└── requirements.txt
```

## 🔧 Cài đặt

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

**Lưu ý**: File `requirements.txt` hiện tại có thể thiếu một số packages. Hãy cài đặt thêm:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm pyarrow
```

### 2. Kiểm tra cấu trúc dữ liệu

Đảm bảo các file dữ liệu nằm trong thư mục `datasets/`:
- `metadata.csv`
- `electricity_cleaned.csv`
- `weather.csv`

## 🚀 Sử dụng Pipeline

### Bước 1: Phân tích EDA

Chạy script phân tích khám phá dữ liệu:

```bash
python scripts/01_eda_analysis.py
```

**Kết quả:**
- Phân tích missing values
- Phân bố các features
- Patterns theo thời gian
- Các biểu đồ được lưu trong `analysis/`

### Bước 2: Xử lý Dữ liệu

Chạy script preprocessing và feature engineering:

```bash
python scripts/02_data_preprocessing.py
```

**Kết quả:**
- Merge các file dữ liệu
- Tạo features thời gian (hour, day_of_week, month, season, ...)
- Tạo lag features và rolling statistics
- Xử lý missing values
- Dữ liệu đã xử lý được lưu trong `output/processed_data.parquet`

### Bước 3: Training Models

Chạy script training các mô hình:

```bash
python scripts/03_train_models.py
```

**Các mô hình được train:**
- ✅ XGBoost
- ✅ LightGBM
- ✅ Random Forest
- ✅ Linear Regression (baseline)

**Kết quả:**
- Models được lưu trong `output/models/`
- So sánh kết quả các models
- Model tốt nhất được tự động chọn

**Lưu ý**: Script này train với 200 buildings mẫu để nhanh. Để train toàn bộ, sửa dòng:
```python
sample_size = min(200, df['building_id'].nunique())
```
thành:
```python
sample_size = df['building_id'].nunique()
```

### Bước 4: Đánh giá Models

Chạy script đánh giá và visualization:

```bash
python scripts/04_evaluate_models.py
```

**Kết quả:**
- Scatter plots: Actual vs Predicted
- Time series predictions
- Feature importance plots
- Residual plots
- Metrics comparison
- Tất cả biểu đồ được lưu trong `output/visualizations/`

### Bước 5: Prediction

Chạy script dự đoán cho dữ liệu mới:

```bash
python scripts/05_predict.py
```

**Kết quả:**
- File `output/predictions.csv` chứa kết quả dự đoán
- File `output/predictions_summary.json` chứa summary

## 📊 Features được sử dụng

### Continuous Features:
- `sqm`: Diện tích tòa nhà
- `yearbuilt`: Năm xây dựng
- `numberoffloors`: Số tầng
- `occupants`: Số người sử dụng
- `airTemperature`: Nhiệt độ không khí
- `cloudCoverage`: Độ che phủ mây
- `windSpeed`: Tốc độ gió
- `dewTemperature`: Nhiệt độ điểm sương
- `seaLvlPressure`: Áp suất mực nước biển

### Time Features:
- `hour`: Giờ trong ngày (0-23)
- `day_of_week`: Ngày trong tuần (0-6)
- `month`: Tháng (1-12)
- `is_weekend`: Cuối tuần (0/1)
- `season`: Mùa (Spring/Summer/Fall/Winter)
- Cyclical encoding: `hour_sin`, `hour_cos`, `day_of_week_sin`, `day_of_week_cos`, `month_sin`, `month_cos`

### Lag Features:
- `electricity_lag1`: Điện tiêu thụ 1 giờ trước
- `electricity_lag24`: Điện tiêu thụ 24 giờ trước (cùng giờ hôm trước)
- `electricity_lag168`: Điện tiêu thụ 168 giờ trước (cùng giờ tuần trước)
- `electricity_rolling_mean_24h`: Trung bình 24 giờ
- `electricity_rolling_std_24h`: Độ lệch chuẩn 24 giờ
- `electricity_rolling_mean_7d`: Trung bình 7 ngày

### Categorical Features:
- `primaryspaceusage`: Loại sử dụng chính
- `sub_primaryspaceusage`: Phân loại chi tiết
- `site_id`: Mã site
- `timezone`: Múi giờ

## 📈 Metrics đánh giá

Các metrics được sử dụng:
- **RMSE** (Root Mean Squared Error): Căn bậc hai của trung bình bình phương sai số
- **MAE** (Mean Absolute Error): Trung bình giá trị tuyệt đối sai số
- **R²** (R-squared): Hệ số xác định
- **MAPE** (Mean Absolute Percentage Error): Trung bình phần trăm sai số tuyệt đối

## 🔍 Tùy chỉnh

### Thay đổi số lượng buildings để train

Trong `scripts/03_train_models.py`, sửa:
```python
sample_size = min(200, df['building_id'].nunique())
```

### Thay đổi hyperparameters

Trong `scripts/03_train_models.py`, có thể điều chỉnh các tham số của từng model:

**XGBoost:**
```python
xgb_model = xgb.XGBRegressor(
    n_estimators=200,      # Số cây
    max_depth=8,           # Độ sâu tối đa
    learning_rate=0.05,    # Tốc độ học
    subsample=0.8,        # Tỷ lệ sample
    colsample_bytree=0.8,  # Tỷ lệ features
    ...
)
```

**LightGBM:**
```python
lgb_model = lgb.LGBMRegressor(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.05,
    ...
)
```

### Thêm features mới

1. Thêm feature engineering trong `scripts/02_data_preprocessing.py`
2. Cập nhật `features_info.json` nếu cần
3. Retrain models

## ⚠️ Lưu ý

1. **Memory**: Dataset lớn (~25M records), cần đủ RAM
2. **Thời gian**: Training có thể mất vài phút đến vài giờ tùy số lượng buildings
3. **Missing values**: Một số features có nhiều missing values (occupants, yearbuilt, ...)
4. **Time series split**: Chia train/test theo thời gian, không random để tránh data leakage

## 🐛 Troubleshooting

### Lỗi: Module not found
```bash
pip install <module_name>
```

### Lỗi: Out of memory
- Giảm số lượng buildings trong training
- Sử dụng sample nhỏ hơn

### Lỗi: File not found
- Kiểm tra đường dẫn đến datasets
- Đảm bảo đã chạy các script theo thứ tự

## 📚 Tài liệu tham khảo

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [DATA_EXPLAINATION.md](./DATA_EXPLAINATION.md) - Giải thích chi tiết về dataset

## 📝 License

Project này sử dụng dataset từ Building Data Genome Project 2.

---

**Tác giả**: Pipeline được tạo để hỗ trợ bài toán dự đoán năng lượng điện tiêu thụ

