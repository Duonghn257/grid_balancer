#!/usr/bin/env python3
"""
Script 7: Prediction với XGBoost Model
Sử dụng XGBoost model đã train để dự đoán lượng điện tiêu thụ
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
print("PREDICTION VỚI XGBOOST MODEL")
print("=" * 80)

# ============================================================================
# 1. LOAD MODEL VÀ THÔNG TIN
# ============================================================================

print("\n" + "=" * 80)
print("BƯỚC 1: LOAD MODEL")
print("=" * 80)

# Load model info
with open('output/models/model_info_dice.json', 'r') as f:
    model_info = json.load(f)

# Load features info
with open('output/features_info.json', 'r') as f:
    features_info = json.load(f)

# Load wrapped model (tương thích với DiCE)
with open('output/models/xgboost_wrapped_dice.pkl', 'rb') as f:
    model = pickle.load(f)

# Load label encoders
with open('output/models/label_encoders_dice.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

print(f"\n📊 Model: {model_info['model_type']}")
print(f"   - Test R²: {model_info['performance']['test_r2']:.4f}")
print(f"   - Test RMSE: {model_info['performance']['test_rmse']:.2f} kWh")
print(f"   - DiCE Compatible: {model_info['dice_compatible']}")

# ============================================================================
# 2. LOAD DỮ LIỆU ĐỂ PREDICT
# ============================================================================

print("\n" + "=" * 80)
print("BƯỚC 2: LOAD DỮ LIỆU")
print("=" * 80)

# Load dữ liệu đã xử lý
print("\n📂 Đang load dữ liệu...")
df = pd.read_parquet("./output/processed_data.parquet")

# Sample một số buildings để predict (hoặc có thể predict toàn bộ)
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
all_features = model_info['continuous_features']
categorical_features = model_info['categorical_features']

# Lấy features từ dữ liệu
X_predict = df_predict[all_features + categorical_features].copy()

# Loại bỏ duplicate columns
if X_predict.columns.duplicated().any():
    X_predict = X_predict.loc[:, ~X_predict.columns.duplicated()]

# Đảm bảo tất cả các cột đều là Series 1D
for col in X_predict.columns:
    col_data = X_predict[col]
    if isinstance(col_data, pd.DataFrame):
        X_predict[col] = col_data.iloc[:, 0]

print(f"✅ Features shape: {X_predict.shape}")

# ============================================================================
# 4. PREDICT
# ============================================================================

print("\n" + "=" * 80)
print("BƯỚC 4: PREDICT")
print("=" * 80)

print("\n📊 Đang thực hiện prediction...")
predictions = model.predict(X_predict)

print(f"✅ Đã predict {len(predictions)} samples")
print(f"   - Min prediction: {predictions.min():.2f} kWh")
print(f"   - Max prediction: {predictions.max():.2f} kWh")
print(f"   - Mean prediction: {predictions.mean():.2f} kWh")

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

# Thêm thông tin building
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
    
    # Tính metrics
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

output_path = 'output/predictions_xgboost.csv'
results_df.to_csv(output_path, index=False)
print(f"✅ Đã lưu kết quả vào: {output_path}")

# Lưu summary
summary = {
    'model_type': model_info['model_type'],
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

summary_path = 'output/predictions_xgboost_summary.json'
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print(f"✅ Đã lưu summary vào: {summary_path}")

print("\n" + "=" * 80)
print("HOÀN THÀNH PREDICTION!")
print("=" * 80)
print(f"✅ Kết quả đã được lưu trong: {output_path}")
