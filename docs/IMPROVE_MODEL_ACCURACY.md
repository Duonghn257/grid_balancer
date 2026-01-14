# Cải Thiện Model Accuracy

## Vấn Đề
Sau khi giảm lag features, RMSE tăng từ **24 → 48.55 kWh** (tăng gấp đôi).

## Nguyên Nhân
- Lag features có tổng importance **~97%** (đặc biệt là `electricity_lag1` với 87%)
- Khi bỏ hầu hết lag features, model mất thông tin quan trọng về pattern tiêu thụ điện
- Model không đủ thông tin để dự đoán chính xác

## Giải Pháp

### 1. Giữ Thêm `electricity_lag24` (7% importance)

**Lý do:**
- `electricity_lag1`: 87% importance - **GIỮ** (quan trọng nhất)
- `electricity_lag24`: 7% importance - **GIỮ** (cải thiện accuracy)
- Tổng: **94% importance** - vẫn cho phép model học mối quan hệ với features khác

**Lợi ích:**
- ✅ Cải thiện accuracy (RMSE có thể giảm từ 48 → 35-40 kWh)
- ✅ Vẫn cho phép model học mối quan hệ với `occupants` và các features khác
- ✅ `occupants` vẫn sẽ có importance cao hơn (dự kiến 2-5%)

**Cách thực hiện:**
- Đã cập nhật `scripts/02_data_preprocessing.py` để tạo `electricity_lag24`
- Đã cập nhật `src/inference.py` để xử lý `electricity_lag24` trong prediction
- Đã cập nhật `src/dice_explainer.py` để scale `electricity_lag24` trong counterfactuals

### 2. Tune Hyperparameters của XGBoost

**Các thay đổi:**

| Hyperparameter | Cũ | Mới | Lý do |
|----------------|-----|-----|-------|
| `n_estimators` | 200 | 500 | Tăng số trees để học tốt hơn |
| `max_depth` | 8 | 10 | Cho phép model học patterns phức tạp hơn |
| `learning_rate` | 0.05 | 0.03 | Giảm learning rate (cần nhiều trees hơn) |
| `subsample` | 0.8 | 0.85 | Tăng dữ liệu training cho mỗi tree |
| `colsample_bytree` | 0.8 | 0.85 | Tăng số features cho mỗi tree |
| `min_child_weight` | 3 | 2 | Cho phép splits nhỏ hơn |
| `gamma` | - | 0.1 | Thêm regularization |
| `reg_alpha` | - | 0.1 | L1 regularization |
| `reg_lambda` | - | 1.0 | L2 regularization |
| `early_stopping_rounds` | - | 50 | Tránh overfitting |

**Lợi ích:**
- ✅ Model học tốt hơn với nhiều trees và regularization
- ✅ Tránh overfitting với early stopping
- ✅ Cải thiện generalization

### 3. Early Stopping

**Lý do:**
- Tránh overfitting khi tăng số trees
- Tự động dừng khi validation error không cải thiện

**Cách thực hiện:**
```python
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    early_stopping_rounds=50,  # Dừng nếu không cải thiện sau 50 rounds
    verbose=50
)
```

## Cách Sử Dụng

### Option 1: Chạy toàn bộ pipeline (khuyến nghị)

```bash
python scripts/retrain_improved.py
```

Script này sẽ:
1. Chạy preprocessing với `electricity_lag24`
2. Train model với tuned hyperparameters
3. Lưu model mới nếu tốt hơn model cũ

### Option 2: Chạy từng bước

**Bước 1: Preprocessing (với lag24)**
```bash
python scripts/02_data_preprocessing.py
```

**Bước 2: Train với improved hyperparameters**
```bash
python scripts/improve_model_accuracy.py
```

## Kết Quả Mong Đợi

### Trước khi cải thiện:
- Test RMSE: **48.55 kWh**
- Test R²: **0.9394**
- Lag features: chỉ `electricity_lag1`

### Sau khi cải thiện (dự kiến):
- Test RMSE: **35-40 kWh** (giảm 18-28%)
- Test R²: **0.95-0.96** (tăng nhẹ)
- Lag features: `electricity_lag1` + `electricity_lag24`
- `occupants` importance: **2-5%** (tăng từ <1%)

## Trade-offs

### Ưu điểm:
- ✅ Accuracy tốt hơn (RMSE giảm)
- ✅ Vẫn cho phép DiCE tìm recommendations dựa trên `occupants`
- ✅ Model vẫn học được mối quan hệ với features khác

### Nhược điểm:
- ⚠️ Vẫn phụ thuộc vào lag features (94% importance)
- ⚠️ Cần có historical data để predict
- ⚠️ Counterfactual prediction vẫn cần xử lý lag features đặc biệt

## So Sánh Các Phương Án

| Phương án | RMSE | Occupants Importance | DiCE Recommendations | Độ phức tạp |
|-----------|------|---------------------|---------------------|-------------|
| **Chỉ lag1** | 48.55 | <1% | ❌ Không tìm được | Thấp |
| **Lag1 + Lag24** | 35-40 | 2-5% | ✅ Có thể tìm được | Trung bình |
| **Tất cả lag features** | 24 | <0.5% | ❌ Không tìm được | Thấp |

**Kết luận:** Phương án **Lag1 + Lag24** là cân bằng tốt nhất giữa accuracy và khả năng counterfactual.

## Bước Tiếp Theo

Sau khi retrain:

1. **Test model behavior:**
   ```bash
   python src/test_model_behavior.py
   ```
   - Kiểm tra feature importance
   - Kiểm tra model sensitivity với `occupants`
   - Kiểm tra impact của lag features

2. **Test DiCE recommendations:**
   ```bash
   python src/dice_usage_example.py
   ```
   - Kiểm tra xem DiCE có tìm được recommendations không
   - Kiểm tra tính thực tế của recommendations

3. **Test multiple scenarios:**
   ```bash
   python src/test_dice_multiple_scenarios.py
   ```
   - Test với nhiều scenarios khác nhau
   - Đánh giá success rate

## Troubleshooting

### Nếu RMSE vẫn cao (>40):
1. Kiểm tra xem `electricity_lag24` đã được tạo chưa
2. Thử tăng `n_estimators` lên 1000
3. Thử tăng `max_depth` lên 12
4. Kiểm tra data quality

### Nếu DiCE vẫn không tìm được recommendations:
1. Kiểm tra `occupants` importance (phải >1%)
2. Kiểm tra `permitted_range` cho `occupants`
3. Thử điều chỉnh `desired_range` trong DiCE
4. Xem `docs/LAG_FEATURES_SOLUTION.md` để biết thêm chi tiết
