# Hướng dẫn Retrain Model với Reduced Lag Features

## 📋 Tóm tắt thay đổi

### 1. **Filter Data: Chỉ lấy từ 2017-10-01 trở đi**
- File: `scripts/02_data_preprocessing.py`
- Thêm filter: `df_final = df_final[df_final['timestamp'] >= pd.Timestamp('2017-10-01')]`

### 2. **Reduced Lag Features: Chỉ giữ electricity_lag1**
- File: `scripts/02_data_preprocessing.py`
- **Giữ lại**: `electricity_lag1` (87% importance)
- **Bỏ**: 
  - `electricity_lag24` (7% importance)
  - `electricity_lag168` (0.3% importance)
  - `electricity_rolling_mean_24h` (2% importance)
  - `electricity_rolling_std_24h` (0.07% importance)
  - `electricity_rolling_mean_7d` (0.1% importance)

### 3. **Cập nhật Inference Code**
- File: `src/inference.py`
- Chỉ tính `electricity_lag1` trong `_get_lag_features()`
- Cập nhật `predict_future()` để chỉ dùng `electricity_lag1`

### 4. **Cập nhật DiCE Explainer**
- File: `src/dice_explainer.py`
- Scale `electricity_lag1` theo reduction ratio của `occupants`

---

## 🚀 Cách chạy Retrain

### Option 1: Chạy script tự động (Khuyến nghị)

```bash
python scripts/retrain_with_reduced_lag.py
```

Script này sẽ:
1. Chạy preprocessing với filter date và reduced lag features
2. Train XGBoost model mới
3. Lưu model và features info mới

### Option 2: Chạy từng bước thủ công

```bash
# Bước 1: Preprocess data
python scripts/02_data_preprocessing.py

# Bước 2: Train model
python scripts/06_train_xgboost_for_dice.py
```

---

## 📊 Kết quả mong đợi

### Trước retrain:
- **Lag features importance**: 97%
- **Occupants importance**: 0.07%
- **Model không nhạy cảm với thay đổi của occupants**

### Sau retrain:
- **Lag features importance**: ~50-70% (chỉ có electricity_lag1)
- **Occupants importance**: Tăng lên (dự kiến 5-15%)
- **Model nhạy cảm hơn với thay đổi của occupants**
- **DiCE có thể tìm được recommendations thực tế**

---

## ✅ Kiểm tra sau khi retrain

### 1. Test model behavior
```bash
python src/test_model_behavior.py
```

**Kiểm tra:**
- ✅ `occupants` có importance cao hơn (>1%)
- ✅ Model nhạy cảm với thay đổi của `occupants` (thay đổi >20% khi giảm 50% occupants)
- ✅ Lag features impact giảm xuống (<50%)

### 2. Test DiCE với multiple scenarios
```bash
python src/test_dice_multiple_scenarios.py
```

**Kiểm tra:**
- ✅ Tỷ lệ tìm được recommendations thực tế >50%
- ✅ Recommendations gần threshold (80-100% của threshold)

### 3. Test với simple recommender
```bash
python src/dice_usage_example.py
```

**Kiểm tra:**
- ✅ Tìm được recommendations thực tế
- ✅ Recommendations có giá trị hợp lý (gần threshold)

---

## ⚠️ Lưu ý

1. **Backup models cũ** (nếu cần):
   ```bash
   cp output/models/xgboost_wrapped_dice.pkl output/models/xgboost_wrapped_dice_backup.pkl
   cp output/models/label_encoders_dice.pkl output/models/label_encoders_dice_backup.pkl
   ```

2. **Model accuracy có thể giảm**:
   - Model cũ: R² = 0.9843 (với nhiều lag features)
   - Model mới: R² có thể giảm xuống 0.95-0.97 (với ít lag features)
   - Đây là trade-off để model học được mối quan hệ với features khác

3. **Thời gian retrain**:
   - Preprocessing: ~5-10 phút
   - Training: ~10-30 phút (tùy số lượng buildings)

---

## 🔍 Troubleshooting

### Nếu model accuracy giảm quá nhiều (<0.90):
- Có thể cần giữ thêm `electricity_lag24` (7% importance)
- Hoặc điều chỉnh hyperparameters của XGBoost

### Nếu vẫn không tìm được recommendations thực tế:
- Kiểm tra feature importance của `occupants` (nên >1%)
- Kiểm tra model sensitivity test
- Có thể cần thêm features có thể điều chỉnh khác

---

## 📝 Files đã được cập nhật

1. ✅ `scripts/02_data_preprocessing.py` - Filter date + reduced lag features
2. ✅ `src/inference.py` - Chỉ tính electricity_lag1
3. ✅ `src/dice_explainer.py` - Scale electricity_lag1
4. ✅ `scripts/retrain_with_reduced_lag.py` - Script retrain tự động

---

**Chúc bạn retrain thành công! 🎉**
