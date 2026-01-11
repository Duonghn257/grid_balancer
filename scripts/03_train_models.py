#!/usr/bin/env python3
"""
Script 3: Training Pipeline với XGBoost và các mô hình hồi quy khác
Train và so sánh nhiều models: XGBoost, Random Forest, LightGBM, Linear Regression
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
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')

print("=" * 80)
print("TRAINING PIPELINE - XGBOOST VÀ CÁC MÔ HÌN HỒI QUY")
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

print(f"\n📊 Features:")
print(f"   - Continuous: {len(features_info['continuous_features'])}")
print(f"   - Categorical: {len(features_info['categorical_features'])}")
print(f"   - Time features: {len(features_info['time_features'])}")
print(f"   - Lag features: {len(features_info['lag_features'])}")

# ============================================================================
# 2. CHUẨN BỊ DỮ LIỆU CHO TRAINING
# ============================================================================

print("\n" + "=" * 80)
print("BƯỚC 2: CHUẨN BỊ DỮ LIỆU")
print("=" * 80)

# Chọn subset buildings để training nhanh hơn (có thể bỏ dòng này để train toàn bộ)
print("\n📊 Chọn subset buildings để training...")
np.random.seed(42)
sample_size = min(200, df['building_id'].nunique())  # Train với 200 buildings
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
        print(f"⚠️  Warning: Column '{col}' not found in X, skipping...")
        continue
    
    # Đảm bảo lấy Series 1D, không phải DataFrame
    col_data = X[col]
    if isinstance(col_data, pd.DataFrame):
        # Nếu là DataFrame (có duplicate column names), lấy cột đầu tiên
        col_data = col_data.iloc[:, 0]
        print(f"⚠️  Warning: Column '{col}' is a DataFrame, using first column")
    
    # Convert to Series nếu chưa phải
    if not isinstance(col_data, pd.Series):
        col_data = pd.Series(col_data)
    
    le = LabelEncoder()
    X[col] = le.fit_transform(col_data.astype(str))
    label_encoders[col] = le

print(f"✅ Đã encode {len(label_encoders)} categorical features")

# Loại bỏ duplicate columns (nếu có)
print("\n📊 Kiểm tra và loại bỏ duplicate columns...")
if X.columns.duplicated().any():
    duplicate_cols = X.columns[X.columns.duplicated()].tolist()
    print(f"⚠️  Phát hiện duplicate columns: {duplicate_cols}")
    # Giữ lại cột đầu tiên, loại bỏ các cột duplicate
    X = X.loc[:, ~X.columns.duplicated()]
    print(f"✅ Đã loại bỏ duplicate columns. Shape mới: {X.shape}")

# Đảm bảo tất cả các cột đều là Series 1D
print("\n📊 Đảm bảo tất cả cột đều là Series 1D...")
for col in X.columns:
    col_data = X[col]
    if isinstance(col_data, pd.DataFrame):
        # Nếu là DataFrame, lấy cột đầu tiên
        X[col] = col_data.iloc[:, 0]
        print(f"⚠️  Đã sửa cột '{col}' từ DataFrame thành Series")
    elif not isinstance(col_data, pd.Series):
        # Nếu không phải Series, convert
        X[col] = pd.Series(col_data, index=X.index)
        print(f"⚠️  Đã convert cột '{col}' thành Series")

print(f"✅ X shape cuối cùng: {X.shape}")
print(f"✅ Tất cả cột đều là Series 1D")

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
# 4. TRAIN CÁC MÔ HÌN
# ============================================================================

print("\n" + "=" * 80)
print("BƯỚC 4: TRAIN CÁC MÔ HÌN")
print("=" * 80)

models = {}
results = {}

# ============================================================================
# 4.1. XGBoost
# ============================================================================

print("\n" + "-" * 80)
print("4.1. Training XGBoost...")
print("-" * 80)

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
y_pred_train_xgb = xgb_model.predict(X_train)
y_pred_test_xgb = xgb_model.predict(X_test)

# Metrics
train_rmse_xgb = np.sqrt(mean_squared_error(y_train, y_pred_train_xgb))
test_rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_test_xgb))
train_mae_xgb = mean_absolute_error(y_train, y_pred_train_xgb)
test_mae_xgb = mean_absolute_error(y_test, y_pred_test_xgb)
train_r2_xgb = r2_score(y_train, y_pred_train_xgb)
test_r2_xgb = r2_score(y_test, y_pred_test_xgb)
test_mape_xgb = mean_absolute_percentage_error(y_test, y_pred_test_xgb)

models['XGBoost'] = xgb_model
results['XGBoost'] = {
    'train_rmse': train_rmse_xgb,
    'test_rmse': test_rmse_xgb,
    'train_mae': train_mae_xgb,
    'test_mae': test_mae_xgb,
    'train_r2': train_r2_xgb,
    'test_r2': test_r2_xgb,
    'test_mape': test_mape_xgb
}

print(f"✅ XGBoost - Test RMSE: {test_rmse_xgb:.2f}, Test R²: {test_r2_xgb:.4f}")

# ============================================================================
# 4.2. LightGBM
# ============================================================================

print("\n" + "-" * 80)
print("4.2. Training LightGBM...")
print("-" * 80)

lgb_model = lgb.LGBMRegressor(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

print("Đang training...")
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    eval_names=['train', 'test'],
    eval_metric='rmse',
    callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(50)]
)

# Predictions
y_pred_train_lgb = lgb_model.predict(X_train)
y_pred_test_lgb = lgb_model.predict(X_test)

# Metrics
train_rmse_lgb = np.sqrt(mean_squared_error(y_train, y_pred_train_lgb))
test_rmse_lgb = np.sqrt(mean_squared_error(y_test, y_pred_test_lgb))
train_mae_lgb = mean_absolute_error(y_train, y_pred_train_lgb)
test_mae_lgb = mean_absolute_error(y_test, y_pred_test_lgb)
train_r2_lgb = r2_score(y_train, y_pred_train_lgb)
test_r2_lgb = r2_score(y_test, y_pred_test_lgb)
test_mape_lgb = mean_absolute_percentage_error(y_test, y_pred_test_lgb)

models['LightGBM'] = lgb_model
results['LightGBM'] = {
    'train_rmse': train_rmse_lgb,
    'test_rmse': test_rmse_lgb,
    'train_mae': train_mae_lgb,
    'test_mae': test_mae_lgb,
    'train_r2': train_r2_lgb,
    'test_r2': test_r2_lgb,
    'test_mape': test_mape_lgb
}

print(f"✅ LightGBM - Test RMSE: {test_rmse_lgb:.2f}, Test R²: {test_r2_lgb:.4f}")

# ============================================================================
# 4.3. Random Forest
# ============================================================================

print("\n" + "-" * 80)
print("4.3. Training Random Forest...")
print("-" * 80)

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("Đang training...")
rf_model.fit(X_train, y_train)

# Predictions
y_pred_train_rf = rf_model.predict(X_train)
y_pred_test_rf = rf_model.predict(X_test)

# Metrics
train_rmse_rf = np.sqrt(mean_squared_error(y_train, y_pred_train_rf))
test_rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_test_rf))
train_mae_rf = mean_absolute_error(y_train, y_pred_train_rf)
test_mae_rf = mean_absolute_error(y_test, y_pred_test_rf)
train_r2_rf = r2_score(y_train, y_pred_train_rf)
test_r2_rf = r2_score(y_test, y_pred_test_rf)
test_mape_rf = mean_absolute_percentage_error(y_test, y_pred_test_rf)

models['RandomForest'] = rf_model
results['RandomForest'] = {
    'train_rmse': train_rmse_rf,
    'test_rmse': test_rmse_rf,
    'train_mae': train_mae_rf,
    'test_mae': test_mae_rf,
    'train_r2': train_r2_rf,
    'test_r2': test_r2_rf,
    'test_mape': test_mape_rf
}

print(f"✅ Random Forest - Test RMSE: {test_rmse_rf:.2f}, Test R²: {test_r2_rf:.4f}")

# ============================================================================
# 4.4. Linear Regression (Baseline)
# ============================================================================

print("\n" + "-" * 80)
print("4.4. Training Linear Regression (Baseline)...")
print("-" * 80)

# Chuẩn hóa dữ liệu cho Linear Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_model = LinearRegression()
print("Đang training...")
lr_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_train_lr = lr_model.predict(X_train_scaled)
y_pred_test_lr = lr_model.predict(X_test_scaled)

# Metrics
train_rmse_lr = np.sqrt(mean_squared_error(y_train, y_pred_train_lr))
test_rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_test_lr))
train_mae_lr = mean_absolute_error(y_train, y_pred_train_lr)
test_mae_lr = mean_absolute_error(y_test, y_pred_test_lr)
train_r2_lr = r2_score(y_train, y_pred_train_lr)
test_r2_lr = r2_score(y_test, y_pred_test_lr)
test_mape_lr = mean_absolute_percentage_error(y_test, y_pred_test_lr)

models['LinearRegression'] = lr_model
models['Scaler'] = scaler  # Lưu scaler để dùng sau
results['LinearRegression'] = {
    'train_rmse': train_rmse_lr,
    'test_rmse': test_rmse_lr,
    'train_mae': train_mae_lr,
    'test_mae': test_mae_lr,
    'train_r2': train_r2_lr,
    'test_r2': test_r2_lr,
    'test_mape': test_mape_lr
}

print(f"✅ Linear Regression - Test RMSE: {test_rmse_lr:.2f}, Test R²: {test_r2_lr:.4f}")

# ============================================================================
# 5. SO SÁNH KẾT QUẢ
# ============================================================================

print("\n" + "=" * 80)
print("BƯỚC 5: SO SÁNH KẾT QUẢ")
print("=" * 80)

results_df = pd.DataFrame(results).T
results_df = results_df.round(4)

print("\n📊 Kết quả các mô hình:")
print("=" * 80)
print(results_df.to_string())

# Tìm model tốt nhất
best_model_name = results_df['test_rmse'].idxmin()
print(f"\n🏆 Model tốt nhất (RMSE thấp nhất): {best_model_name}")
print(f"   - Test RMSE: {results_df.loc[best_model_name, 'test_rmse']:.2f}")
print(f"   - Test R²: {results_df.loc[best_model_name, 'test_r2']:.4f}")
print(f"   - Test MAE: {results_df.loc[best_model_name, 'test_mae']:.2f}")

# ============================================================================
# 6. LƯU MODELS VÀ KẾT QUẢ
# ============================================================================

print("\n" + "=" * 80)
print("BƯỚC 6: LƯU MODELS VÀ KẾT QUẢ")
print("=" * 80)

os.makedirs('output/models', exist_ok=True)

# Lưu từng model
for model_name, model in models.items():
    if model_name != 'Scaler':  # Scaler sẽ lưu riêng
        model_path = f"output/models/{model_name.lower().replace(' ', '_')}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"✅ Đã lưu {model_name} vào: {model_path}")

# Lưu scaler
scaler_path = "output/models/scaler.pkl"
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"✅ Đã lưu Scaler vào: {scaler_path}")

# Lưu label encoders
encoders_path = "output/models/label_encoders.pkl"
with open(encoders_path, 'wb') as f:
    pickle.dump(label_encoders, f)
print(f"✅ Đã lưu Label Encoders vào: {encoders_path}")

# Lưu kết quả
results_df.to_csv('output/models/results_comparison.csv')
print(f"✅ Đã lưu kết quả so sánh vào: output/models/results_comparison.csv")

# Lưu thông tin về features và best model
model_info = {
    'best_model': best_model_name,
    'features_used': all_features + categorical_features,
    'categorical_features': categorical_features,
    'training_date': datetime.now().isoformat(),
    'train_size': len(X_train),
    'test_size': len(X_test),
    'results': results
}

with open('output/models/model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2, default=str)

print(f"✅ Đã lưu thông tin model vào: output/models/model_info.json")

print("\n" + "=" * 80)
print("HOÀN THÀNH TRAINING!")
print("=" * 80)
print(f"✅ Đã train {len(models)} mô hình")
print(f"🏆 Model tốt nhất: {best_model_name}")
print(f"📁 Models đã được lưu trong: output/models/")

