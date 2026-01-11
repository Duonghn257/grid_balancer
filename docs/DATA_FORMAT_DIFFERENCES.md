# Sự Khác Biệt Giữa Dữ Liệu Đầu Vào Cho XGBoost và DICE

## Tổng Quan

Dữ liệu đầu vào cho **XGBoost** và **DICE** có những khác biệt quan trọng về định dạng, đặc biệt là cách xử lý categorical features.

## 1. Định Dạng Categorical Features

### XGBoost
- **Categorical features được ENCODE thành số nguyên (integer)**
- Ví dụ: `'Education'` → `0`, `'Office'` → `1`, `'Retail'` → `2`
- Tất cả features đều là numeric (int/float)

### DICE
- **Categorical features phải ở dạng STRING gốc**
- Ví dụ: `'Education'`, `'Office'`, `'Retail'` (giữ nguyên giá trị string)
- Categorical features là object/string type

## 2. Ví Dụ Cụ Thể

### Dữ Liệu Đầu Vào (JSON)
```python
json_data = {
    'time': '2016-01-01T21:00:00',
    'building_id': 'Bear_education_Sharon',
    'primaryspaceusage': 'Education',  # String
    'sqm': 5261.7,
    'occupants': 200
}
```

### Sau Khi Preprocess Cho XGBoost
```python
# Trong inference.py, _preprocess_input():
X = pd.DataFrame([{
    'primaryspaceusage': 0,  # ← Đã encode thành integer
    'sqm': 5261.7,
    'occupants': 200,
    # ... các features khác đều là numeric
}])
```

### Sau Khi Preprocess Cho DICE
```python
# Trong dice_explainer.py, _prepare_query_instance():
query_df = pd.DataFrame([{
    'primaryspaceusage': 'Education',  # ← Decode lại thành string
    'sqm': 5261.7,
    'occupants': 200,
    'electricity_consumption': 150.5,  # ← Cần có outcome column
    # ... các features khác
}])
```

## 3. Quy Trình Xử Lý

### XGBoost Prediction Flow
```
JSON Input
    ↓
_preprocess_input()
    ↓
Encode categoricals → Integer
    ↓
All features numeric
    ↓
XGBoost Model → Prediction
```

### DICE Counterfactual Flow
```
JSON Input
    ↓
_preprocess_input() → Encode categoricals
    ↓
_prepare_query_instance() → Decode categoricals back to strings
    ↓
Categoricals: String, Continuous: Numeric
    ↓
DiCE Data Object
    ↓
Generate Counterfactuals
```

## 4. Code Tham Khảo

### XGBoost: Encode Categoricals
```python
# File: src/inference.py, lines 405-413
# Encode categorical features
for col in self.categorical_features:
    if col in df.columns and col in self.label_encoders:
        le = self.label_encoders[col]
        # Handle unknown values
        if df[col].iloc[0] not in le.classes_:
            df[col] = le.classes_[0]  # Use first class as default
        else:
            df[col] = le.transform([df[col].iloc[0]])[0]  # → Integer
```

### DICE: Decode Categoricals
```python
# File: src/dice_explainer.py, lines 517-529
# Decode categorical features back to strings
for col in self.inference.categorical_features:
    if col in query_df.columns and col in self.inference.label_encoders:
        le = self.inference.label_encoders[col]
        try:
            encoded_val = int(query_df[col].iloc[0])
            if encoded_val in le.classes_:
                query_df[col] = le.inverse_transform([encoded_val])[0]  # → String
            else:
                query_df[col] = json_data.get(col, 'Unknown')
        except:
            query_df[col] = json_data.get(col, 'Unknown')
```

## 5. Setup Dữ Liệu Cho DICE

### Trong _setup_dice() Method
```python
# File: src/dice_explainer.py, lines 213-264
# Decode categorical features back to original strings for DiCE
for col in available_categorical:
    if col not in df_for_dice.columns or col not in label_encoders:
        continue
    
    le = label_encoders[col]
    col_series = df_for_dice[col]
    
    # Decode back to original strings
    def decode_value(x):
        # Convert integer → string
        if isinstance(x, (int, float, np.integer, np.floating)):
            int_val = int(x)
            if int_val in le.classes_:
                return le.inverse_transform([int_val])[0]  # → String
        return 'Unknown'
    
    df_for_dice[col] = col_series.apply(decode_value)

# Ensure categorical features are object type (string)
for col in dice_categorical_features:
    if col in df_for_dice.columns:
        df_for_dice[col] = df_for_dice[col].astype(str)  # ← String type
```

## 6. Tại Sao Có Sự Khác Biệt?

### XGBoost
- Model được train với categorical features đã encode thành integer
- Model chỉ hiểu numeric values
- Cần encode để match với format training data

### DICE
- Cần hiểu ý nghĩa thực tế của categorical values để generate counterfactuals
- Ví dụ: DICE cần biết `'Education'` vs `'Office'` để suggest thay đổi hợp lý
- DiCE library yêu cầu categorical features ở dạng string để xử lý constraints và ranges

## 7. Wrapper Model

Để giải quyết sự khác biệt này, code sử dụng `XGBoostWrapper`:

```python
# File: src/inference.py, lines 19-69
class XGBoostWrapper:
    """
    Wrapper class để tự động encode categorical features trước khi predict
    Tương thích với DiCE (Diverse Counterfactual Explanations)
    """
    def predict(self, X):
        # Tự động encode categoricals khi DICE gọi predict
        for col in self.categorical_features:
            if col in X_encoded.columns:
                le = self.label_encoders[col]
                X_encoded[col] = X_encoded[col].astype(str)
                X_encoded[col] = le.transform(X_encoded[col])  # Encode
        
        return self.model.predict(X_encoded)
```

**Lưu ý**: DICE có thể truyền data với categoricals là string vào model, nhưng `XGBoostWrapper` sẽ tự động encode lại thành integer trước khi predict.

## 8. Tóm Tắt

| Đặc Điểm | XGBoost | DICE |
|----------|---------|------|
| **Categorical Format** | Integer (encoded) | String (original) |
| **Data Type** | All numeric | Mixed (categorical=string, continuous=numeric) |
| **Outcome Column** | Không cần | Cần có `electricity_consumption` |
| **Preprocessing** | Encode categoricals | Decode categoricals từ encoded → string |
| **Model Input** | Numeric DataFrame | DataFrame với strings cho categoricals |

## 9. Kết Luận

- **XGBoost**: Nhận dữ liệu với categorical features đã được encode thành integer
- **DICE**: Nhận dữ liệu với categorical features ở dạng string gốc
- **XGBoostWrapper**: Đóng vai trò bridge, tự động encode khi DICE gọi predict với string categoricals

Sự khác biệt này được xử lý tự động trong code thông qua:
1. `_preprocess_input()` - Encode cho XGBoost
2. `_prepare_query_instance()` - Decode cho DICE
3. `XGBoostWrapper` - Tự động encode khi DICE gọi predict
