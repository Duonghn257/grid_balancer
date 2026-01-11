# 🔍 Tại sao cần Wrapped Model?

## ❓ Wrapped Model là gì?

**Wrapped Model** là một class wrapper bọc quanh XGBoost model gốc để tự động xử lý các tác vụ trước khi predict, đặc biệt là **encode categorical features**.

## 🔴 Vấn đề với XGBoost Model gốc

### XGBoost model gốc (`xgboost_dice.pkl`):
```python
# Model này chỉ nhận được NUMERIC features đã được encode
X_train_encoded = [
    'sqm': 5000.0,
    'occupants': 200.0,
    'primaryspaceusage': 2,  # Đã encode: Education = 2
    'site_id': 5,            # Đã encode: Fox = 5
    ...
]

# Predict
prediction = xgb_model.predict(X_train_encoded)  # ✅ Hoạt động
```

### Nhưng khi dùng với DiCE:
```python
# DiCE sẽ tạo counterfactuals với CATEGORICAL values gốc (chưa encode)
X_counterfactual = [
    'sqm': 4500.0,
    'occupants': 150.0,
    'primaryspaceusage': 'Education',  # ❌ String, chưa encode!
    'site_id': 'Fox',                 # ❌ String, chưa encode!
    ...
]

# Predict trực tiếp → LỖI!
prediction = xgb_model.predict(X_counterfactual)  # ❌ Lỗi vì có string
```

## ✅ Giải pháp: Wrapped Model

### Wrapped Model (`xgboost_wrapped_dice.pkl`):
```python
class XGBoostWrapper:
    def __init__(self, model, label_encoders, categorical_features):
        self.model = model                    # XGBoost model gốc
        self.label_encoders = label_encoders  # Encoders cho categorical
        self.categorical_features = categorical_features
    
    def predict(self, X):
        # 1. Tự động encode categorical features
        for col in categorical_features:
            if col in X.columns:
                le = self.label_encoders[col]
                X[col] = le.transform(X[col].astype(str))
        
        # 2. Xử lý unknown values
        # 3. Predict với model gốc
        return self.model.predict(X)
```

### Khi dùng với DiCE:
```python
# DiCE tạo counterfactuals với categorical values gốc
X_counterfactual = [
    'primaryspaceusage': 'Education',  # String
    'site_id': 'Fox',                 # String
    ...
]

# Wrapped model tự động encode → ✅ Hoạt động!
prediction = wrapped_model.predict(X_counterfactual)  # ✅ OK
```

## 📊 So sánh

| Aspect | XGBoost Model gốc | Wrapped Model |
|--------|-------------------|---------------|
| **Input** | Chỉ nhận numeric (đã encode) | Nhận cả string và numeric |
| **Categorical** | Phải encode trước | Tự động encode |
| **Unknown values** | Không xử lý | Tự động xử lý |
| **DiCE compatible** | ❌ Không (cần encode thủ công) | ✅ Có (tự động) |
| **Dễ sử dụng** | ⚠️ Phức tạp | ✅ Đơn giản |

## 🎯 Tại sao phải lưu cả 2?

### 1. **XGBoost Model gốc** (`xgboost_dice.pkl`):
- ✅ **Dùng cho production prediction** (khi đã có dữ liệu đã encode sẵn)
- ✅ **Nhanh hơn** (không cần encode)
- ✅ **Nhẹ hơn** (không có wrapper overhead)
- ✅ **Dùng cho retrain/fine-tuning**

### 2. **Wrapped Model** (`xgboost_wrapped_dice.pkl`):
- ✅ **Dùng cho DiCE** (tự động encode categorical)
- ✅ **Dễ sử dụng** (không cần encode thủ công)
- ✅ **Xử lý unknown values** tự động
- ✅ **Dùng cho testing/development**

## 💡 Ví dụ cụ thể

### Scenario 1: Production Prediction (dùng model gốc)
```python
# Dữ liệu đã được preprocess và encode sẵn
X_production = pd.DataFrame({
    'sqm': [5000.0],
    'occupants': [200.0],
    'primaryspaceusage': [2],  # Đã encode
    'site_id': [5],            # Đã encode
    ...
})

# Dùng model gốc (nhanh hơn)
with open('output/models/xgboost_dice.pkl', 'rb') as f:
    model = pickle.load(f)
prediction = model.predict(X_production)  # ✅ Nhanh, đơn giản
```

### Scenario 2: DiCE Counterfactuals (dùng wrapped model)
```python
# DiCE tạo counterfactuals với categorical values gốc
X_counterfactual = pd.DataFrame({
    'sqm': [4500.0],
    'occupants': [150.0],
    'primaryspaceusage': ['Education'],  # String, chưa encode
    'site_id': ['Fox'],                 # String, chưa encode
    ...
})

# Dùng wrapped model (tự động encode)
with open('output/models/xgboost_wrapped_dice.pkl', 'rb') as f:
    wrapped_model = pickle.load(f)

# DiCE sử dụng wrapped model
dice_model = dice_ml.Model(
    model=wrapped_model,  # ✅ Tự động encode
    backend='sklearn',
    model_type='regressor'
)

counterfactuals = explainer.generate_counterfactuals(X_counterfactual)
# ✅ Hoạt động vì wrapped model tự động encode
```

## 🔧 Cấu trúc Wrapped Model

```python
class XGBoostWrapper:
    """
    Wrapper bọc quanh XGBoost model để:
    1. Tự động encode categorical features
    2. Xử lý unknown values
    3. Đảm bảo tương thích với DiCE
    """
    
    def predict(self, X):
        # Bước 1: Convert to DataFrame
        # Bước 2: Encode categorical features
        # Bước 3: Xử lý unknown values
        # Bước 4: Predict với model gốc
        return predictions
```

## 📝 Khi nào dùng cái nào?

### Dùng **XGBoost Model gốc** khi:
- ✅ Production prediction (dữ liệu đã encode sẵn)
- ✅ Batch prediction (nhiều records cùng lúc)
- ✅ Performance là ưu tiên (nhanh hơn)
- ✅ Không cần DiCE

### Dùng **Wrapped Model** khi:
- ✅ DiCE counterfactual explanations
- ✅ Testing/Development (dữ liệu chưa encode)
- ✅ Cần xử lý unknown values tự động
- ✅ Muốn đơn giản hóa workflow

## 🎯 Kết luận

**Wrapped Model** là cần thiết vì:
1. **DiCE yêu cầu**: DiCE tạo counterfactuals với categorical values gốc (string), không phải encoded (numeric)
2. **Tự động hóa**: Không cần encode thủ công mỗi lần predict
3. **Xử lý edge cases**: Tự động xử lý unknown values
4. **Tương thích**: Đảm bảo tương thích với DiCE backend='sklearn'

**Lưu cả 2** để:
- Model gốc: Dùng cho production (nhanh, hiệu quả)
- Wrapped model: Dùng cho DiCE và development (tiện lợi, tự động)

---

**Tóm lại**: Wrapped model là "lớp bọc thông minh" giúp XGBoost model có thể làm việc với dữ liệu chưa encode, đặc biệt quan trọng cho DiCE integration! 🚀
