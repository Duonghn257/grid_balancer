#!/usr/bin/env python3
"""
Script 5: Prediction Script
Sử dụng model đã train để dự đoán lượng điện tiêu thụ cho dữ liệu mới
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
import os
import json
import pickle

warnings.filterwarnings('ignore')

print("=" * 80)
print("PREDICTION SCRIPT")
print("=" * 80)

# ============================================================================
# 1. LOAD MODEL VÀ THÔNG TIN
# ============================================================================

print("\n" + "=" * 80)
print("BƯỚC 1: LOAD MODEL")
print("=" * 80)

# Load model info
with open('output/models/model_info.json', 'r') as f:
    model_info = json.load(f)

# Load features info
with open('output/features_info.json', 'r') as f:
    features_info = json.load(f)

# Load label encoders
with open('output/models/label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# Load best model
best_model_name = model_info['best_model']
print(f"\n📊 Model được sử dụng: {best_model_name}")

if best_model_name == 'LinearRegression':
    model_path = 'output/models/linearregression.pkl'
    scaler_path = 'output/models/scaler.pkl'
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
else:
    model_path = f"output/models/{best_model_name.lower().replace(' ', '_')}.pkl"

with open(model_path, 'rb') as f:
    model = pickle.load(f)

print("✅ Đã load model")

# ============================================================================
# 2. LOAD DỮ LIỆU ĐỂ PREDICT
# ============================================================================

print("\n" + "=" * 80)
print("BƯỚC 2: LOAD DỮ LIỆU")
print("=" * 80)

# Có thể load từ file processed hoặc từ dữ liệu mới
# Ở đây ta sẽ sử dụng test set để demo
print("\n📂 Đang load dữ liệu test...")
df = pd.read_parquet("./output/processed_data.parquet")

# Sample một số buildings để predict
np.random.seed(42)
sample_size = min(50, df['building_id'].nunique())
sample_buildings = np.random.choice(
    df['building_id'].unique(), 
    size=sample_size, 
    replace=False
)
df_predict = df[df['building_id'].isin(sample_buildings)].copy()
df_predict = df_predict.sort_values(['building_id', 'timestamp']).reset_index(drop=True)

# Lấy test set (20% cuối)
split_idx = int(len(df_predict) * 0.8)
df_predict = df_predict.iloc[split_idx:].copy()

print(f"✅ Dataset để predict: {df_predict.shape}")
print(f"   - Số buildings: {df_predict['building_id'].nunique()}")
print(f"   - Số timestamps: {df_predict['timestamp'].nunique()}")

# ============================================================================
# 3. CHUẨN BỊ FEATURES
# ============================================================================

print("\n" + "=" * 80)
print("BƯỚC 3: CHUẨN BỊ FEATURES")
print("=" * 80)

# Xác định features
all_features = (
    features_info['continuous_features'] + 
    features_info['time_features'] + 
    features_info['lag_features']
)
all_features = [f for f in all_features if f in df_predict.columns]
categorical_features = [f for f in features_info['categorical_features'] if f in df_predict.columns]

X_predict = df_predict[all_features + categorical_features].copy()

# Encode categorical features
for col in categorical_features:
    if col in label_encoders:
        le = label_encoders[col]
        X_predict[col] = X_predict[col].astype(str)
        X_predict[col] = X_predict[col].apply(
            lambda x: x if x in le.classes_ else le.classes_[0]
        )
        X_predict[col] = le.transform(X_predict[col])

print(f"✅ Features shape: {X_predict.shape}")

# ============================================================================
# 4. PREDICT
# ============================================================================

print("\n" + "=" * 80)
print("BƯỚC 4: PREDICT")
print("=" * 80)

print("\n📊 Đang thực hiện prediction...")

if best_model_name == 'LinearRegression':
    X_predict_scaled = scaler.transform(X_predict)
    predictions = model.predict(X_predict_scaled)
else:
    predictions = model.predict(X_predict)

print(f"✅ Đã predict {len(predictions)} samples")

# ============================================================================
# 5. TẠO KẾT QUẢ
# ============================================================================

print("\n" + "=" * 80)
print("BƯỚC 5: TẠO KẾT QUẢ")
print("=" * 80)

# Tạo DataFrame kết quả
results_df = pd.DataFrame({
    'building_id': df_predict['building_id'].values,
    'timestamp': df_predict['timestamp'].values,
    'predicted_consumption': predictions
})

# Thêm thông tin building nếu có
if 'primaryspaceusage' in df_predict.columns:
    results_df = pd.merge(
        results_df,
        df_predict[['building_id', 'primaryspaceusage', 'sqm', 'site_id']].drop_duplicates(),
        on='building_id',
        how='left'
    )

# Thêm actual values nếu có (để so sánh)
if features_info['target'] in df_predict.columns:
    results_df['actual_consumption'] = df_predict[features_info['target']].values
    results_df['error'] = results_df['actual_consumption'] - results_df['predicted_consumption']
    results_df['absolute_error'] = np.abs(results_df['error'])
    results_df['percentage_error'] = (results_df['absolute_error'] / results_df['actual_consumption'] * 100).round(2)

print(f"\n📊 Kết quả prediction:")
print(f"   - Tổng số predictions: {len(results_df)}")
print(f"   - Số buildings: {results_df['building_id'].nunique()}")

if 'actual_consumption' in results_df.columns:
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    rmse = np.sqrt(mean_squared_error(results_df['actual_consumption'], results_df['predicted_consumption']))
    mae = mean_absolute_error(results_df['actual_consumption'], results_df['predicted_consumption'])
    r2 = r2_score(results_df['actual_consumption'], results_df['predicted_consumption'])
    
    print(f"\n📊 Metrics:")
    print(f"   - RMSE: {rmse:.2f} kWh")
    print(f"   - MAE: {mae:.2f} kWh")
    print(f"   - R²: {r2:.4f}")

# Hiển thị sample kết quả
print(f"\n📋 Sample kết quả (10 dòng đầu):")
print(results_df.head(10).to_string())

# ============================================================================
# 6. LƯU KẾT QUẢ
# ============================================================================

print("\n" + "=" * 80)
print("BƯỚC 6: LƯU KẾT QUẢ")
print("=" * 80)

output_path = 'output/predictions.csv'
results_df.to_csv(output_path, index=False)
print(f"✅ Đã lưu kết quả vào: {output_path}")

# Lưu summary
summary = {
    'model_used': best_model_name,
    'total_predictions': len(results_df),
    'num_buildings': results_df['building_id'].nunique(),
    'num_timestamps': results_df['timestamp'].nunique(),
    'date_range': {
        'start': str(results_df['timestamp'].min()),
        'end': str(results_df['timestamp'].max())
    }
}

if 'actual_consumption' in results_df.columns:
    summary['metrics'] = {
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2)
    }

summary_path = 'output/predictions_summary.json'
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print(f"✅ Đã lưu summary vào: {summary_path}")

print("\n" + "=" * 80)
print("HOÀN THÀNH PREDICTION!")
print("=" * 80)
print(f"✅ Kết quả đã được lưu trong: {output_path}")

