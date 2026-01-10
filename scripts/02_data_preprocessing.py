#!/usr/bin/env python3
"""
Script 2: Data Preprocessing và Feature Engineering
Xử lý dữ liệu, merge các file, tạo features, xử lý missing values
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
import os

warnings.filterwarnings('ignore')

print("=" * 80)
print("XỬ LÝ DỮ LIỆU VÀ FEATURE ENGINEERING")
print("=" * 80)

# ============================================================================
# 1. LOAD DỮ LIỆU
# ============================================================================

print("\n" + "=" * 80)
print("BƯỚC 1: LOAD DỮ LIỆU")
print("=" * 80)

base_path = Path("./datasets")

print("\n📂 Đang load dữ liệu...")
df_metadata = pd.read_csv(base_path / "metadata.csv")
df_electricity = pd.read_csv(base_path / "electricity_cleaned.csv", parse_dates=['timestamp'])
df_weather = pd.read_csv(base_path / "weather.csv", parse_dates=['timestamp'])

print(f"✅ Metadata: {df_metadata.shape}")
print(f"✅ Electricity: {df_electricity.shape}")
print(f"✅ Weather: {df_weather.shape}")

# ============================================================================
# 2. CHUYỂN ĐỔI ELECTRICITY DATA SANG LONG FORMAT
# ============================================================================

print("\n" + "=" * 80)
print("BƯỚC 2: CHUYỂN ĐỔI ELECTRICITY DATA")
print("=" * 80)

print("\n📊 Chuyển đổi từ wide format sang long format...")

df_electricity_long = pd.melt(
    df_electricity,
    id_vars=['timestamp'],
    var_name='building_id',
    value_name='electricity_consumption'
)

# Loại bỏ NaN
df_electricity_long = df_electricity_long.dropna(subset=['electricity_consumption'])

# Chỉ giữ lại các buildings có electricity meter
buildings_with_electricity = df_metadata[df_metadata['electricity'] == 'Yes']['building_id'].tolist()
df_electricity_long = df_electricity_long[df_electricity_long['building_id'].isin(buildings_with_electricity)]

print(f"✅ Long format: {df_electricity_long.shape}")
print(f"   - Số buildings: {df_electricity_long['building_id'].nunique()}")
print(f"   - Số records: {len(df_electricity_long)}")

# ============================================================================
# 3. MERGE VỚI METADATA
# ============================================================================

print("\n" + "=" * 80)
print("BƯỚC 3: MERGE VỚI METADATA")
print("=" * 80)

print("\n📊 Đang merge electricity data với metadata...")

df_merged = pd.merge(
    df_electricity_long,
    df_metadata,
    on='building_id',
    how='inner'
)

print(f"✅ Sau khi merge metadata: {df_merged.shape}")
print(f"   - Số buildings: {df_merged['building_id'].nunique()}")
print(f"   - Số timestamps: {df_merged['timestamp'].nunique()}")

# ============================================================================
# 4. MERGE VỚI WEATHER DATA
# ============================================================================

print("\n" + "=" * 80)
print("BƯỚC 4: MERGE VỚI WEATHER DATA")
print("=" * 80)

print("\n📊 Đang merge với weather data (theo site_id và timestamp)...")

df_final = pd.merge(
    df_merged,
    df_weather,
    on=['timestamp', 'site_id'],
    how='left'
)

print(f"✅ Sau khi merge weather: {df_final.shape}")

# ============================================================================
# 5. FEATURE ENGINEERING - TẠO FEATURES THỜI GIAN
# ============================================================================

print("\n" + "=" * 80)
print("BƯỚC 5: FEATURE ENGINEERING - THỜI GIAN")
print("=" * 80)

print("\n📊 Đang tạo các features thời gian...")

# Sắp xếp theo building_id và timestamp
df_final = df_final.sort_values(['building_id', 'timestamp']).reset_index(drop=True)

# Features thời gian cơ bản
df_final['hour'] = df_final['timestamp'].dt.hour
df_final['day_of_week'] = df_final['timestamp'].dt.dayofweek
df_final['day_of_month'] = df_final['timestamp'].dt.day
df_final['month'] = df_final['timestamp'].dt.month
df_final['year'] = df_final['timestamp'].dt.year
df_final['is_weekend'] = (df_final['day_of_week'] >= 5).astype(int)

# Tạo season feature
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

df_final['season'] = df_final['month'].apply(get_season)

# Cyclical encoding cho hour và day_of_week (để model hiểu tính tuần hoàn)
df_final['hour_sin'] = np.sin(2 * np.pi * df_final['hour'] / 24)
df_final['hour_cos'] = np.cos(2 * np.pi * df_final['hour'] / 24)
df_final['day_of_week_sin'] = np.sin(2 * np.pi * df_final['day_of_week'] / 7)
df_final['day_of_week_cos'] = np.cos(2 * np.pi * df_final['day_of_week'] / 7)
df_final['month_sin'] = np.sin(2 * np.pi * df_final['month'] / 12)
df_final['month_cos'] = np.cos(2 * np.pi * df_final['month'] / 12)

print("✅ Đã tạo features thời gian cơ bản")

# ============================================================================
# 6. FEATURE ENGINEERING - LAG FEATURES
# ============================================================================

print("\n" + "=" * 80)
print("BƯỚC 6: FEATURE ENGINEERING - LAG FEATURES")
print("=" * 80)

print("\n📊 Đang tạo lag features...")

# Lag features (điện tiêu thụ giờ trước)
df_final['electricity_lag1'] = df_final.groupby('building_id')['electricity_consumption'].shift(1)
df_final['electricity_lag24'] = df_final.groupby('building_id')['electricity_consumption'].shift(24)  # Cùng giờ ngày hôm trước
df_final['electricity_lag168'] = df_final.groupby('building_id')['electricity_consumption'].shift(168)  # Cùng giờ tuần trước

# Rolling statistics
df_final['electricity_rolling_mean_24h'] = df_final.groupby('building_id')['electricity_consumption'].transform(
    lambda x: x.rolling(window=24, min_periods=1).mean()
)
df_final['electricity_rolling_std_24h'] = df_final.groupby('building_id')['electricity_consumption'].transform(
    lambda x: x.rolling(window=24, min_periods=1).std()
)
df_final['electricity_rolling_mean_7d'] = df_final.groupby('building_id')['electricity_consumption'].transform(
    lambda x: x.rolling(window=168, min_periods=1).mean()
)

print("✅ Đã tạo lag features và rolling statistics")

# ============================================================================
# 7. XỬ LÝ MISSING VALUES
# ============================================================================

print("\n" + "=" * 80)
print("BƯỚC 7: XỬ LÝ MISSING VALUES")
print("=" * 80)

print("\n📊 Đang xử lý missing values...")

# Xác định các features cần xử lý
continuous_features = [
    'sqm', 'yearbuilt', 'numberoffloors', 'occupants',
    'airTemperature', 'cloudCoverage', 'dewTemperature', 
    'windSpeed', 'seaLvlPressure', 'precipDepth1HR'
]

categorical_features = [
    'primaryspaceusage', 'sub_primaryspaceusage', 
    'site_id', 'timezone', 'season'
]

# Fill missing values cho continuous features
for col in continuous_features:
    if col in df_final.columns:
        # Fill bằng median theo building_id nếu có thể, nếu không thì fill bằng median tổng thể
        if col in ['sqm', 'yearbuilt', 'numberoffloors', 'occupants']:
            # Features của building (không đổi theo thời gian)
            df_final[col] = df_final.groupby('building_id')[col].transform(
                lambda x: x.fillna(x.median() if not x.isna().all() else 0)
            )
            # Nếu vẫn còn NaN, fill bằng median tổng thể
            df_final[col] = df_final[col].fillna(df_final[col].median() if not df_final[col].isna().all() else 0)
        else:
            # Features thời tiết - fill bằng median theo site_id
            df_final[col] = df_final.groupby('site_id')[col].transform(
                lambda x: x.fillna(x.median() if not x.isna().all() else 0)
            )
            # Nếu vẫn còn NaN, fill bằng median tổng thể
            df_final[col] = df_final[col].fillna(df_final[col].median() if not df_final[col].isna().all() else 0)

# Fill missing values cho categorical features
for col in categorical_features:
    if col in df_final.columns:
        df_final[col] = df_final[col].fillna(
            df_final[col].mode()[0] if len(df_final[col].mode()) > 0 else 'Unknown'
        )

# Fill missing values cho lag features (bằng 0 hoặc giá trị hiện tại)
for col in ['electricity_lag1', 'electricity_lag24', 'electricity_lag168']:
    if col in df_final.columns:
        df_final[col] = df_final[col].fillna(0)

for col in ['electricity_rolling_mean_24h', 'electricity_rolling_std_24h', 'electricity_rolling_mean_7d']:
    if col in df_final.columns:
        df_final[col] = df_final[col].fillna(df_final['electricity_consumption'])

print("✅ Đã xử lý missing values")

# Kiểm tra missing values còn lại
missing_after = df_final.isnull().sum().sum()
print(f"   - Missing values còn lại: {missing_after}")

# ============================================================================
# 8. LỌC DỮ LIỆU (LOẠI BỎ OUTLIERS VÀ DỮ LIỆU KHÔNG HỢP LỆ)
# ============================================================================

print("\n" + "=" * 80)
print("BƯỚC 8: LỌC DỮ LIỆU")
print("=" * 80)

print(f"\n📊 Dữ liệu trước khi lọc: {len(df_final)} records")

# Loại bỏ các dòng có electricity_consumption <= 0 hoặc quá lớn (outliers)
# Giữ lại các giá trị hợp lý (0 < consumption < percentile 99.9)
q99_9 = df_final['electricity_consumption'].quantile(0.999)
df_final = df_final[
    (df_final['electricity_consumption'] > 0) & 
    (df_final['electricity_consumption'] < q99_9)
].copy()

print(f"✅ Dữ liệu sau khi lọc: {len(df_final)} records")
print(f"   - Đã loại bỏ: {missing_after} records có vấn đề")

# ============================================================================
# 9. LƯU DỮ LIỆU ĐÃ XỬ LÝ
# ============================================================================

print("\n" + "=" * 80)
print("BƯỚC 9: LƯU DỮ LIỆU")
print("=" * 80)

output_path = Path("./output/processed_data.parquet")
os.makedirs('output', exist_ok=True)

print(f"\n📊 Đang lưu dữ liệu đã xử lý...")
df_final.to_parquet(output_path, index=False, compression='snappy')

print(f"✅ Đã lưu vào: {output_path}")
print(f"   - Shape: {df_final.shape}")
print(f"   - Columns: {len(df_final.columns)}")

# Lưu thông tin về features
features_info = {
    'continuous_features': [f for f in continuous_features if f in df_final.columns],
    'categorical_features': [f for f in categorical_features if f in df_final.columns],
    'time_features': ['hour', 'day_of_week', 'month', 'year', 'is_weekend', 'season',
                      'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos', 
                      'month_sin', 'month_cos'],
    'lag_features': ['electricity_lag1', 'electricity_lag24', 'electricity_lag168',
                     'electricity_rolling_mean_24h', 'electricity_rolling_std_24h', 
                     'electricity_rolling_mean_7d'],
    'target': 'electricity_consumption'
}

import json
with open('output/features_info.json', 'w') as f:
    json.dump(features_info, f, indent=2)

print(f"✅ Đã lưu thông tin features vào: output/features_info.json")

# Tóm tắt
print("\n" + "=" * 80)
print("TÓM TẮT")
print("=" * 80)
print(f"✅ Đã xử lý xong dữ liệu!")
print(f"   - Tổng số records: {len(df_final)}")
print(f"   - Số buildings: {df_final['building_id'].nunique()}")
print(f"   - Số features: {len(df_final.columns)}")
print(f"   - Khoảng thời gian: {df_final['timestamp'].min()} đến {df_final['timestamp'].max()}")
print(f"\n📁 Dữ liệu đã được lưu vào: {output_path}")

