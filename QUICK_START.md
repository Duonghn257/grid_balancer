# 🚀 Quick Start Guide

Hướng dẫn nhanh để chạy pipeline dự đoán lượng điện tiêu thụ.

## ⚡ Chạy nhanh (Tất cả các bước)

```bash
# Chạy toàn bộ pipeline
python scripts/run_full_pipeline.py
```

## 📝 Chạy từng bước

### 1. Phân tích EDA
```bash
python scripts/01_eda_analysis.py
```

### 2. Xử lý dữ liệu
```bash
python scripts/02_data_preprocessing.py
```

### 3. Training models
```bash
python scripts/03_train_models.py
```

### 4. Đánh giá models
```bash
python scripts/04_evaluate_models.py
```

### 5. Prediction
```bash
python scripts/05_predict.py
```

## 📦 Cài đặt

```bash
pip install -r requirements.txt
```

## 📊 Kết quả

Sau khi chạy xong, bạn sẽ có:

- **EDA**: `analysis/` - Các biểu đồ phân tích
- **Processed Data**: `output/processed_data.parquet` - Dữ liệu đã xử lý
- **Models**: `output/models/` - Các models đã train
- **Visualizations**: `output/visualizations/` - Biểu đồ đánh giá
- **Predictions**: `output/predictions.csv` - Kết quả dự đoán

## ⚙️ Tùy chỉnh

### Thay đổi số lượng buildings để train

Mở `scripts/03_train_models.py`, tìm dòng:
```python
sample_size = min(200, df['building_id'].nunique())
```

Sửa thành số bạn muốn, hoặc để train toàn bộ:
```python
sample_size = df['building_id'].nunique()
```

## 📚 Chi tiết

Xem [README_PIPELINE.md](./README_PIPELINE.md) để biết thêm chi tiết.

