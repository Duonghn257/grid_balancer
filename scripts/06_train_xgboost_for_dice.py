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
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

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

# Xác định features (bao gồm interaction và dynamic features)
all_features = (
    features_info['continuous_features'] + 
    features_info['time_features'] + 
    features_info['lag_features']
)

# Thêm interaction và dynamic features nếu có
if 'interaction_features' in features_info:
    all_features.extend(features_info['interaction_features'])
if 'dynamic_features' in features_info:
    all_features.extend(features_info['dynamic_features'])

# Loại bỏ các features không có trong dataset
all_features = [f for f in all_features if f in df_train.columns]
categorical_features = [f for f in features_info['categorical_features'] if f in df_train.columns]

print(f"\n📊 Features được sử dụng:")
print(f"   - Continuous/Time/Lag/Interaction/Dynamic: {len(all_features)}")
print(f"   - Categorical: {len(categorical_features)}")
print(f"   - Total: {len(all_features) + len(categorical_features)}")

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

# Chuẩn bị monotone constraints
# Map feature names to constraints: 1 = tăng, -1 = giảm, 0 = không ràng buộc
monotone_constraints_dict = {
    'active_occupants': 1,      # Người tăng -> Điện tăng
    'occupants': 1,             # Người tăng -> Điện tăng
    'sqm': 1,                   # Diện tích tăng -> Điện tăng
    'airTemperature': 1,        # Nhiệt độ tăng -> Điện tăng (làm mát)
    'cooling_load': 1,          # Cooling load tăng -> Điện tăng
    'people_density': 1,        # Mật độ người tăng -> Điện tăng
    'occupancy_ratio': 1,       # Tỷ lệ sử dụng tăng -> Điện tăng
    'hour': 0,                  # Giờ giấc lên xuống tùy ý
    'day_of_week': 0,           # Ngày trong tuần lên xuống tùy ý
    'month': 0,                 # Tháng lên xuống tùy ý
}

# Tạo tuple monotone constraints theo thứ tự features trong X_train
# Sau khi encode categorical, chúng ta sẽ có feature_names_in_
# Tạm thời tạo constraints cho tất cả features = 0 (không ràng buộc)
# Sau khi fit, sẽ cập nhật lại nếu cần

print("\n🔧 Monotone Constraints:")
print("   - active_occupants: +1 (tăng)")
print("   - sqm: +1 (tăng)")
print("   - airTemperature: +1 (tăng)")
print("   - cooling_load: +1 (tăng)")
print("   - people_density: +1 (tăng)")
print("   - occupancy_ratio: +1 (tăng)")

xgb_model = xgb.XGBRegressor(
    n_estimators=500,  # Tăng từ 200 lên 500
    max_depth=10,      # Tăng từ 8 lên 10
    learning_rate=0.03,  # Giảm từ 0.05 xuống 0.03 (cần nhiều trees hơn)
    subsample=0.85,     # Tăng từ 0.8 lên 0.85
    colsample_bytree=0.85,  # Tăng từ 0.8 lên 0.85
    min_child_weight=2,  # Giảm từ 3 xuống 2 (cho phép splits nhỏ hơn)
    gamma=0.1,          # Thêm regularization
    reg_alpha=0.1,      # L1 regularization
    reg_lambda=1.0,     # L2 regularization
    random_state=42,
    n_jobs=-1,
    objective='reg:squarederror',
    eval_metric='rmse',
    # Monotone constraints sẽ được set sau khi có feature names
)

print("\n📊 Đang training...")

# Fit model (lần đầu không có monotone constraints để lấy feature names)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=50
)

# Sau khi fit, tạo monotone constraints tuple theo thứ tự feature names
if hasattr(xgb_model, 'feature_names_in_'):
    feature_names = xgb_model.feature_names_in_
    monotone_constraints_tuple = []
    
    for feat_name in feature_names:
        # Tìm constraint trong dict
        constraint = 0  # Mặc định không ràng buộc
        for key, value in monotone_constraints_dict.items():
            if key in feat_name:
                constraint = value
                break
        monotone_constraints_tuple.append(constraint)
    
    # Retrain với monotone constraints nếu có ít nhất 1 constraint != 0
    if any(c != 0 for c in monotone_constraints_tuple):
        print(f"\n🔄 Retraining với monotone constraints...")
        print(f"   - Số features có constraints: {sum(1 for c in monotone_constraints_tuple if c != 0)}")
        
        xgb_model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=10,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=2,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            objective='reg:squarederror',
            eval_metric='rmse',
            monotone_constraints=tuple(monotone_constraints_tuple),
        )
        
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=50
        )
        print("✅ Đã retrain với monotone constraints")

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

# ============================================================================
# 7. VISUALIZATION
# ============================================================================

print("\n" + "=" * 80)
print("BƯỚC 7: TẠO VISUALIZATIONS")
print("=" * 80)

os.makedirs('output/visualizations', exist_ok=True)

# 7.1. Feature Importance
print("\n📊 Tạo feature importance plot...")

feature_importance = pd.DataFrame({
    'feature': xgb_model.feature_names_in_,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(14, 10))
top_features = feature_importance.head(25)
colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
bars = plt.barh(range(len(top_features)), top_features['importance'].values, color=colors)
plt.yticks(range(len(top_features)), top_features['feature'].values, fontsize=10)
plt.xlabel('Feature Importance', fontsize=12, fontweight='bold')
plt.title('Top 25 Feature Importance (XGBoost for DiCE)', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)

# Thêm giá trị importance vào bars
for i, (idx, row) in enumerate(top_features.iterrows()):
    plt.text(row['importance'], i, f' {row["importance"]:.4f}', 
             va='center', fontsize=8)

plt.tight_layout()
plt.savefig('output/visualizations/feature_importance_dice.png', dpi=150, bbox_inches='tight')
print("✅ Đã lưu: output/visualizations/feature_importance_dice.png")
plt.close()

# In top features ra console
print(f"\n📊 Top 15 Features:")
print(feature_importance.head(15).to_string(index=False))

# Check lag features và occupants importance
lag_features = [f for f in feature_importance['feature'] if 'lag' in f or 'rolling' in f]
lag_importance = feature_importance[feature_importance['feature'].isin(lag_features)]['importance'].sum()
print(f"\n📈 Lag features tổng importance: {lag_importance:.4f} ({lag_importance*100:.1f}%)")

if 'occupants' in feature_importance['feature'].values:
    occ_imp = feature_importance[feature_importance['feature'] == 'occupants']['importance'].values[0]
    occ_rank = feature_importance[feature_importance['feature'] == 'occupants'].index[0] + 1
    print(f"👥 Occupants importance: {occ_imp:.6f} (rank: {occ_rank}/{len(feature_importance)})")

# 7.2. Actual vs Predicted Scatter Plot
print("\n📊 Tạo scatter plot: Actual vs Predicted...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Train set
ax = axes[0]
sample_size = min(5000, len(y_train))
sample_idx = np.random.choice(len(y_train), sample_size, replace=False)
y_train_array = np.array(y_train)
y_pred_train_array = np.array(y_pred_train)

ax.scatter(y_train_array[sample_idx], y_pred_train_array[sample_idx], 
          alpha=0.4, s=15, color='blue', label='Train Predictions')

# Perfect prediction line
min_val = min(y_train.min(), y_pred_train.min())
max_val = max(y_train.max(), y_pred_train.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

ax.set_xlabel('Actual Electricity Consumption (kWh)', fontsize=12, fontweight='bold')
ax.set_ylabel('Predicted Electricity Consumption (kWh)', fontsize=12, fontweight='bold')
ax.set_title(f'Train Set\nRMSE: {train_rmse:.2f} kWh, R²: {train_r2:.4f}', 
             fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Test set
ax = axes[1]
sample_size = min(5000, len(y_test))
sample_idx = np.random.choice(len(y_test), sample_size, replace=False)
y_test_array = np.array(y_test)
y_pred_test_array = np.array(y_pred_test)

ax.scatter(y_test_array[sample_idx], y_pred_test_array[sample_idx], 
          alpha=0.4, s=15, color='green', label='Test Predictions')

# Perfect prediction line
min_val = min(y_test.min(), y_pred_test.min())
max_val = max(y_test.max(), y_pred_test.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

ax.set_xlabel('Actual Electricity Consumption (kWh)', fontsize=12, fontweight='bold')
ax.set_ylabel('Predicted Electricity Consumption (kWh)', fontsize=12, fontweight='bold')
ax.set_title(f'Test Set\nRMSE: {test_rmse:.2f} kWh, R²: {test_r2:.4f}', 
             fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/visualizations/scatter_actual_vs_predicted_dice.png', dpi=150, bbox_inches='tight')
print("✅ Đã lưu: output/visualizations/scatter_actual_vs_predicted_dice.png")
plt.close()

# 7.3. Time Series Predictions
print("\n📊 Tạo time series prediction plots...")

# Chọn một vài buildings để visualize
sample_buildings_for_viz = df_train['building_id'].unique()[:4]
df_test_only = df_train.iloc[split_idx:].copy()

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
axes = axes.flatten()

for idx, building_id in enumerate(sample_buildings_for_viz):
    if idx >= 4:
        break
    
    ax = axes[idx]
    
    # Lấy data cho building này trong test set
    building_mask = df_test_only['building_id'] == building_id
    building_data = df_test_only[building_mask].head(200)  # Lấy 200 điểm đầu tiên
    
    if len(building_data) == 0:
        # Nếu không có trong test set, lấy từ train set
        building_mask = df_train['building_id'] == building_id
        building_data = df_train[building_mask].tail(200)
    
    if len(building_data) > 0:
        # Lấy indices trong original dataframe để map với predictions
        building_indices = building_data.index
        
        # Map với test set indices
        if building_mask.sum() > 0 and building_mask.sum() <= len(y_test):
            # Building này có trong test set
            test_indices = df_test_only[building_mask].index
            # Tìm vị trí trong y_test
            test_positions = [df_test_only.index.get_loc(idx) for idx in test_indices[:200]]
            building_pred = y_pred_test[test_positions] if len(test_positions) > 0 else []
            building_actual = building_data[features_info['target']].values[:len(building_pred)]
        else:
            # Building này không có trong test set, dùng train predictions
            train_indices = df_train[df_train['building_id'] == building_id].index
            train_positions = [df_train.index.get_loc(idx) for idx in train_indices[-200:]]
            building_pred = y_pred_train[train_positions] if len(train_positions) > 0 else []
            building_actual = building_data[features_info['target']].values[:len(building_pred)]
        
        if len(building_actual) > 0 and len(building_pred) > 0:
            timestamps = building_data['timestamp'].values[:len(building_actual)]
            
            ax.plot(timestamps, building_actual, 'b-', label='Actual', 
                   linewidth=2, marker='o', markersize=2, alpha=0.7)
            ax.plot(timestamps, building_pred, 'r-', label='Predicted', 
                   linewidth=2, marker='s', markersize=2, alpha=0.7)
            
            ax.set_xlabel('Timestamp', fontsize=11, fontweight='bold')
            ax.set_ylabel('Electricity Consumption (kWh)', fontsize=11, fontweight='bold')
            ax.set_title(f'Building: {building_id}\n(First 200 hours)', 
                        fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        else:
            ax.text(0.5, 0.5, f'No data for\n{building_id}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'Building: {building_id}', fontsize=12, fontweight='bold')
    else:
        ax.text(0.5, 0.5, f'No data for\n{building_id}', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title(f'Building: {building_id}', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('output/visualizations/time_series_predictions_dice.png', dpi=150, bbox_inches='tight')
print("✅ Đã lưu: output/visualizations/time_series_predictions_dice.png")
plt.close()

# 7.4. Residual Plot
print("\n📊 Tạo residual plot...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Train residuals
ax = axes[0]
residuals_train = y_train - y_pred_train
sample_size = min(5000, len(residuals_train))
sample_idx = np.random.choice(len(residuals_train), sample_size, replace=False)

ax.scatter(y_pred_train_array[sample_idx], residuals_train.iloc[sample_idx] if isinstance(residuals_train, pd.Series) else residuals_train[sample_idx], 
          alpha=0.4, s=15, color='blue')
ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax.set_xlabel('Predicted Values (kWh)', fontsize=12, fontweight='bold')
ax.set_ylabel('Residuals (kWh)', fontsize=12, fontweight='bold')
ax.set_title(f'Train Set Residuals\nMean: {residuals_train.mean():.2f}, Std: {residuals_train.std():.2f}', 
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# Test residuals
ax = axes[1]
residuals_test = y_test - y_pred_test
sample_size = min(5000, len(residuals_test))
sample_idx = np.random.choice(len(residuals_test), sample_size, replace=False)

ax.scatter(y_pred_test_array[sample_idx], residuals_test.iloc[sample_idx] if isinstance(residuals_test, pd.Series) else residuals_test[sample_idx], 
          alpha=0.4, s=15, color='green')
ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax.set_xlabel('Predicted Values (kWh)', fontsize=12, fontweight='bold')
ax.set_ylabel('Residuals (kWh)', fontsize=12, fontweight='bold')
ax.set_title(f'Test Set Residuals\nMean: {residuals_test.mean():.2f}, Std: {residuals_test.std():.2f}', 
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/visualizations/residual_plots_dice.png', dpi=150, bbox_inches='tight')
print("✅ Đã lưu: output/visualizations/residual_plots_dice.png")
plt.close()

print("\n" + "=" * 80)
print("HOÀN THÀNH TRAINING VÀ VISUALIZATION!")
print("=" * 80)
print(f"✅ Model đã được train và lưu")
print(f"📊 Test R²: {test_r2:.4f}, Test RMSE: {test_rmse:.2f} kWh")
print(f"✅ Model đã sẵn sàng cho DiCE integration")
print(f"📁 Models đã được lưu trong: output/models/")
print(f"📊 Visualizations đã được lưu trong: output/visualizations/")
print(f"   - feature_importance_dice.png")
print(f"   - scatter_actual_vs_predicted_dice.png")
print(f"   - time_series_predictions_dice.png")
print(f"   - residual_plots_dice.png")