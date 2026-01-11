#!/usr/bin/env python3
"""
Script 6: Training XGBoost Model cho DiCE Integration
Train XGBoost model với wrapper class để tương thích với DiCE
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
import os
import json
import pickle
from datetime import datetime

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import xgboost as xgb

warnings.filterwarnings('ignore')

print("=" * 80)
print("TRAINING XGBOOST MODEL CHO DiCE INTEGRATION")
print("=" * 80)

# ============================================================================
# 1. LOAD DỮ LIỆU ĐÃ XỬ LÝ
# ============================================================================

print("\n" + "=" * 80)
print("BƯỚC 1: LOAD DỮ LIỆU")
print("=" * 80)

print("\n📂 Đang load dữ liệu đã xử lý...")
df = pd.read_parquet("./output/processed_data.parquet")

print(f"✅ Dataset shape: {df.shape}")
print(f"   - Số records: {len(df)}")
print(f"   - Số buildings: {df['building_id'].nunique()}")

# Load features info
with open('output/features_info.json', 'r') as f:
    features_info = json.load(f)

# ============================================================================
# 2. CHUẨN BỊ DỮ LIỆU CHO TRAINING
# ============================================================================

print("\n" + "=" * 80)
print("BƯỚC 2: CHUẨN BỊ DỮ LIỆU")
print("=" * 80)

# Chọn subset buildings để training (có thể điều chỉnh)
print("\n📊 Chọn subset buildings để training...")
np.random.seed(42)
sample_size = min(20000, df['building_id'].nunique())  # Có thể tăng lên để train toàn bộ
sample_buildings = np.random.choice(
    df['building_id'].unique(), 
    size=sample_size, 
    replace=False
)
df_train = df[df['building_id'].isin(sample_buildings)].copy()

print(f"✅ Đã chọn {len(sample_buildings)} buildings")
print(f"   - Dataset shape: {df_train.shape}")

# Sắp xếp theo thời gian
df_train = df_train.sort_values(['building_id', 'timestamp']).reset_index(drop=True)

# Xác định features
all_features = (
    features_info['continuous_features'] + 
    features_info['time_features'] + 
    features_info['lag_features']
)

# Loại bỏ các features không có trong dataset
all_features = [f for f in all_features if f in df_train.columns]
categorical_features = [f for f in features_info['categorical_features'] if f in df_train.columns]

print(f"\n📊 Features được sử dụng:")
print(f"   - Continuous/Time/Lag: {len(all_features)}")
print(f"   - Categorical: {len(categorical_features)}")

# Tạo X và y
X = df_train[all_features + categorical_features].copy()
y = df_train[features_info['target']].copy()

# Encode categorical features
label_encoders = {}
for col in categorical_features:
    if col not in X.columns:
        continue
    
    # Đảm bảo là Series 1D
    col_data = X[col]
    if isinstance(col_data, pd.DataFrame):
        col_data = col_data.iloc[:, 0]
    elif not isinstance(col_data, pd.Series):
        col_data = pd.Series(col_data, index=X.index)
    
    le = LabelEncoder()
    X[col] = le.fit_transform(col_data.astype(str))
    label_encoders[col] = le

print(f"✅ Đã encode {len(label_encoders)} categorical features")

# Loại bỏ duplicate columns (nếu có)
if X.columns.duplicated().any():
    X = X.loc[:, ~X.columns.duplicated()]
    print(f"✅ Đã loại bỏ duplicate columns")

# Đảm bảo tất cả các cột đều là Series 1D
for col in X.columns:
    col_data = X[col]
    if isinstance(col_data, pd.DataFrame):
        X[col] = col_data.iloc[:, 0]

# ============================================================================
# 3. CHIA TRAIN/TEST SET (THEO THỜI GIAN)
# ============================================================================

print("\n" + "=" * 80)
print("BƯỚC 3: CHIA TRAIN/TEST SET")
print("=" * 80)

# Chia theo thời gian (80% train, 20% test)
split_idx = int(len(df_train) * 0.8)

X_train = X.iloc[:split_idx]
y_train = y.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_test = y.iloc[split_idx:]

print(f"✅ Train set: {X_train.shape[0]} samples")
print(f"✅ Test set: {X_test.shape[0]} samples")
print(f"\n   Train period: {df_train.iloc[0]['timestamp']} đến {df_train.iloc[split_idx-1]['timestamp']}")
print(f"   Test period: {df_train.iloc[split_idx]['timestamp']} đến {df_train.iloc[-1]['timestamp']}")

# ============================================================================
# 4. TRAIN XGBOOST MODEL
# ============================================================================

print("\n" + "=" * 80)
print("BƯỚC 4: TRAIN XGBOOST MODEL")
print("=" * 80)

xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    random_state=42,
    n_jobs=-1,
    objective='reg:squarederror',
    eval_metric='rmse'
)

print("Đang training...")
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=50
)

# Predictions
y_pred_train = xgb_model.predict(X_train)
y_pred_test = xgb_model.predict(X_test)

# Metrics
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
test_mape = mean_absolute_percentage_error(y_test, y_pred_test)

print(f"\n✅ Model Performance:")
print(f"\nTrain Set:")
print(f"  RMSE: {train_rmse:.2f} kWh")
print(f"  MAE:  {train_mae:.2f} kWh")
print(f"  R²:   {train_r2:.4f}")

print(f"\nTest Set:")
print(f"  RMSE: {test_rmse:.2f} kWh")
print(f"  MAE:  {test_mae:.2f} kWh")
print(f"  R²:   {test_r2:.4f}")
print(f"  MAPE: {test_mape:.2%}")

# ============================================================================
# 5. TẠO WRAPPER CLASS CHO DiCE
# ============================================================================

print("\n" + "=" * 80)
print("BƯỚC 5: TẠO WRAPPER CLASS CHO DiCE")
print("=" * 80)

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
        """Predict với tự động encode categorical features"""
        # Convert to DataFrame nếu là array hoặc Series
        if isinstance(X, np.ndarray):
            # Nếu là array, cần column names
            X = pd.DataFrame(X, columns=self.model.feature_names_in_)
        elif isinstance(X, pd.Series):
            X = X.to_frame().T
        
        X_encoded = X.copy()
        
        # Encode categorical features
        for col in self.categorical_features:
            if col in X_encoded.columns:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    # Chuyển đổi về string và encode
                    X_encoded[col] = X_encoded[col].astype(str)
                    # Xử lý các giá trị chưa thấy (unknown values)
                    mask = ~X_encoded[col].isin(le.classes_)
                    unknown_count = int(np.sum(mask.values)) if isinstance(mask, pd.Series) else int(np.sum(mask))
                    if unknown_count > 0:
                        X_encoded.loc[mask, col] = le.classes_[0]
                    X_encoded[col] = le.transform(X_encoded[col])
                else:
                    # Nếu không có encoder, giữ nguyên (có thể là integer rồi)
                    if X_encoded[col].dtype == 'object':
                        X_encoded[col] = 0
        
        # Đảm bảo tất cả columns là numeric
        for col in X_encoded.columns:
            if X_encoded[col].dtype == 'object':
                X_encoded[col] = pd.to_numeric(X_encoded[col], errors='coerce').fillna(0)
        
        # Đảm bảo thứ tự columns đúng với model
        if hasattr(self.model, 'feature_names_in_'):
            X_encoded = X_encoded.reindex(columns=self.model.feature_names_in_, fill_value=0)
        
        return self.model.predict(X_encoded)

# Tạo wrapped model
xgb_model_wrapped = XGBoostWrapper(
    xgb_model,
    label_encoders,
    categorical_features
)

print("✅ Đã tạo XGBoostWrapper cho DiCE")

# Test wrapper
test_pred_wrapped = xgb_model_wrapped.predict(X_test.head(10))
test_pred_original = xgb_model.predict(X_test.head(10))
diff = np.abs(test_pred_wrapped - test_pred_original).max()
print(f"✅ Test wrapper: Max difference = {diff:.6f} (should be ~0)")

# ============================================================================
# 6. LƯU MODEL VÀ THÔNG TIN
# ============================================================================

print("\n" + "=" * 80)
print("BƯỚC 6: LƯU MODEL VÀ THÔNG TIN")
print("=" * 80)

os.makedirs('output/models', exist_ok=True)

# Lưu XGBoost model
model_path = "output/models/xgboost_dice.pkl"
with open(model_path, 'wb') as f:
    pickle.dump(xgb_model, f)
print(f"✅ Đã lưu XGBoost model vào: {model_path}")

# Lưu wrapped model
wrapped_model_path = "output/models/xgboost_wrapped_dice.pkl"
with open(wrapped_model_path, 'wb') as f:
    pickle.dump(xgb_model_wrapped, f)
print(f"✅ Đã lưu Wrapped model vào: {wrapped_model_path}")

# Lưu label encoders
encoders_path = "output/models/label_encoders_dice.pkl"
with open(encoders_path, 'wb') as f:
    pickle.dump(label_encoders, f)
print(f"✅ Đã lưu Label Encoders vào: {encoders_path}")

# Lưu thông tin về features và model
model_info = {
    'model_type': 'XGBoost',
    'features_used': all_features + categorical_features,
    'continuous_features': all_features,
    'categorical_features': categorical_features,
    'training_date': datetime.now().isoformat(),
    'train_size': len(X_train),
    'test_size': len(X_test),
    'performance': {
        'train_rmse': float(train_rmse),
        'test_rmse': float(test_rmse),
        'train_mae': float(train_mae),
        'test_mae': float(test_mae),
        'train_r2': float(train_r2),
        'test_r2': float(test_r2),
        'test_mape': float(test_mape)
    },
    'dice_compatible': True,
    'wrapper_class': 'XGBoostWrapper'
}

with open('output/models/model_info_dice.json', 'w') as f:
    json.dump(model_info, f, indent=2, default=str)

print(f"✅ Đã lưu thông tin model vào: output/models/model_info_dice.json")

print("\n" + "=" * 80)
print("HOÀN THÀNH TRAINING!")
print("=" * 80)
print(f"✅ Model đã được train và lưu")
print(f"📊 Test R²: {test_r2:.4f}, Test RMSE: {test_rmse:.2f} kWh")
print(f"✅ Model đã sẵn sàng cho DiCE integration")
print(f"📁 Models đã được lưu trong: output/models/")
