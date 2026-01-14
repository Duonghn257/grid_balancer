# Giải pháp cho vấn đề Lag Features trong Counterfactual Prediction

## 🔍 Vấn đề đã phát hiện

Từ test `test_model_behavior.py`, chúng ta đã phát hiện:

1. **Lag features chiếm 97% importance**:
   - `electricity_lag1`: 87.28%
   - `electricity_lag24`: 7.22%
   - `electricity_rolling_mean_24h`: 2.14%
   - Tổng: ~97%

2. **Occupants chỉ có 0.07% importance**:
   - Model không học được mối quan hệ giữa `occupants` và consumption
   - Khi thay đổi `occupants`, prediction không thay đổi đáng kể

3. **Khi predict counterfactual**:
   - Với `include_lag=False`: prediction = 1.46 kWh (quá thấp)
   - Với `include_lag=True`: prediction = 87.87 kWh (không thay đổi)
   - Lag features từ instance gốc chi phối prediction

---

## 💡 Các giải pháp (theo thứ tự ưu tiên)

### ✅ Option 1: Xử lý lag features khi predict counterfactual (ĐÃ IMPLEMENT)

**Cách làm:**
- Khi predict counterfactual, dùng **mean lag features từ training data** thay vì lag features từ instance gốc
- Điều này cho một baseline trung tính hơn

**Ưu điểm:**
- ✅ Không cần retrain model
- ✅ Nhanh, dễ implement
- ✅ Có thể test ngay

**Nhược điểm:**
- ⚠️ Vẫn phụ thuộc vào model đã được train với lag features
- ⚠️ Có thể không chính xác 100%

**Code đã implement:**
```python
# Trong dice_explainer.py, line ~740
# Use mean lag features from training data
for lag_feat in lag_feature_names:
    if lag_feat in self.inference._historical_data.columns:
        mean_lag_value = float(self.inference._historical_data[lag_feat].mean())
        cf_data_with_lag[lag_feat] = mean_lag_value
```

**Test:**
```bash
python src/dice_usage_example.py
```

---

### 🔄 Option 2: Scale lag features theo reduction ratio

**Cách làm:**
- Scale lag features theo tỷ lệ giảm của `occupants`
- Ví dụ: Giảm `occupants` 50% → scale lag features xuống 50%

**Ưu điểm:**
- ✅ Phản ánh mối quan hệ giữa occupants và consumption
- ✅ Không cần retrain

**Nhược điểm:**
- ⚠️ Giả định mối quan hệ tuyến tính (có thể không đúng)
- ⚠️ Cần điều chỉnh công thức scale

**Code (đã comment trong dice_explainer.py):**
```python
# Uncomment để dùng approach này
reduction_ratio = cf_occupants / original_occupants
scaled_lag = original_lag * reduction_ratio
```

---

### 🔧 Option 3: Retrain model với ít lag features hơn

**Khi nào cần:**
- ✅ Option 1 và 2 không hoạt động tốt
- ✅ Muốn model học được mối quan hệ tốt hơn với các features khác
- ✅ Có thời gian và resources để retrain

**Cách làm:**

1. **Loại bỏ một số lag features**:
   ```python
   # Trong 02_data_preprocessing.py hoặc 06_train_xgboost_for_dice.py
   # Chỉ giữ lại lag features quan trọng nhất
   lag_features = [
       'electricity_lag1',  # Giữ lại (quan trọng nhất)
       # 'electricity_lag24',  # Có thể bỏ
       # 'electricity_lag168',  # Có thể bỏ
       # 'electricity_rolling_mean_24h',  # Có thể bỏ
       # 'electricity_rolling_std_24h',  # Có thể bỏ
       # 'electricity_rolling_mean_7d',  # Có thể bỏ
   ]
   ```

2. **Hoặc giảm weight của lag features**:
   - Sử dụng feature selection
   - Hoặc train với regularization để giảm overfitting vào lag features

3. **Retrain model**:
   ```bash
   python scripts/06_train_xgboost_for_dice.py
   ```

**Ưu điểm:**
- ✅ Model sẽ học được mối quan hệ tốt hơn với các features khác
- ✅ `occupants` sẽ có importance cao hơn
- ✅ Counterfactual prediction sẽ chính xác hơn

**Nhược điểm:**
- ❌ Tốn thời gian retrain
- ❌ Có thể giảm accuracy của model (vì lag features rất quan trọng)
- ❌ Cần test lại model performance

---

### 🎯 Option 4: Tạo model riêng cho counterfactual prediction

**Cách làm:**
- Train 2 models:
  1. **Model chính**: Với đầy đủ lag features (cho prediction thông thường)
  2. **Model counterfactual**: Không có lag features (cho counterfactual prediction)

**Ưu điểm:**
- ✅ Model counterfactual sẽ nhạy cảm hơn với thay đổi của features
- ✅ Model chính vẫn giữ accuracy cao

**Nhược điểm:**
- ❌ Cần maintain 2 models
- ❌ Tốn thời gian train 2 models

---

## 📊 So sánh các options

| Option | Cần retrain? | Độ chính xác | Độ phức tạp | Thời gian |
|--------|--------------|--------------|-------------|-----------|
| Option 1: Mean lag | ❌ Không | ⭐⭐⭐ | Thấp | Ngay lập tức |
| Option 2: Scale lag | ❌ Không | ⭐⭐ | Thấp | Ngay lập tức |
| Option 3: Retrain ít lag | ✅ Có | ⭐⭐⭐⭐ | Trung bình | Vài giờ |
| Option 4: 2 models | ✅ Có | ⭐⭐⭐⭐⭐ | Cao | Vài giờ |

---

## 🚀 Kế hoạch hành động

### Bước 1: Test Option 1 (ĐÃ IMPLEMENT)
```bash
python src/dice_usage_example.py
python src/test_dice_multiple_scenarios.py
```

**Nếu Option 1 hoạt động tốt (>50% scenarios tìm được recommendations thực tế):**
- ✅ Dùng Option 1
- Không cần retrain

**Nếu Option 1 không hoạt động tốt (<50%):**
- Chuyển sang Bước 2

### Bước 2: Test Option 2
- Uncomment code scale lag features trong `dice_explainer.py`
- Test lại

**Nếu Option 2 hoạt động tốt:**
- ✅ Dùng Option 2
- Không cần retrain

**Nếu Option 2 không hoạt động tốt:**
- Chuyển sang Bước 3

### Bước 3: Retrain model (Option 3)
- Chỉ giữ lại `electricity_lag1` (lag feature quan trọng nhất)
- Bỏ các lag features khác
- Retrain model
- Test lại

---

## 💡 Khuyến nghị

**Hiện tại:**
1. ✅ **Test Option 1 trước** (đã implement)
2. Nếu không tốt → thử Option 2
3. Nếu vẫn không tốt → cân nhắc Option 3

**Lâu dài:**
- Nếu use case chính là counterfactual prediction → nên retrain với ít lag features hơn
- Nếu use case chính là prediction thông thường → giữ model hiện tại, dùng Option 1 hoặc 2

---

## 📝 Notes

- Lag features rất quan trọng cho prediction accuracy (R² = 0.9843)
- Nhưng chúng làm counterfactual prediction khó khăn
- Cần cân bằng giữa accuracy và khả năng counterfactual prediction
