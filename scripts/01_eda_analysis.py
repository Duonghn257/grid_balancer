#!/usr/bin/env python3
"""
Script 1: Exploratory Data Analysis (EDA)
Phân tích dataset để hiểu cấu trúc, missing values, distributions, correlations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
import os

warnings.filterwarnings('ignore')

# Setup
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Tạo thư mục output nếu chưa có
os.makedirs('analysis', exist_ok=True)
os.makedirs('output', exist_ok=True)

print("=" * 80)
print("PHÂN TÍCH EDA - DATASET ĐIỆN TIÊU THỤ")
print("=" * 80)

# ============================================================================
# 1. LOAD DỮ LIỆU
# ============================================================================

print("\n" + "=" * 80)
print("BƯỚC 1: LOAD DỮ LIỆU")
print("=" * 80)

base_path = Path("./datasets")

# Load metadata
print("\n📂 Đang load metadata.csv...")
df_metadata = pd.read_csv(base_path / "metadata.csv")
print(f"✅ Metadata shape: {df_metadata.shape}")
print(f"   - Số buildings: {len(df_metadata)}")
print(f"   - Số features: {len(df_metadata.columns)}")

# Load electricity data (sample để phân tích nhanh)
print("\n📂 Đang load electricity_cleaned.csv...")
df_electricity = pd.read_csv(base_path / "electricity_cleaned.csv", parse_dates=['timestamp'])
print(f"✅ Electricity shape: {df_electricity.shape}")
print(f"   - Số timestamps: {len(df_electricity)}")
print(f"   - Số buildings: {len(df_electricity.columns) - 1}")  # Trừ cột timestamp
print(f"   - Khoảng thời gian: {df_electricity['timestamp'].min()} đến {df_electricity['timestamp'].max()}")

# Load weather data
print("\n📂 Đang load weather.csv...")
df_weather = pd.read_csv(base_path / "weather.csv", parse_dates=['timestamp'])
print(f"✅ Weather shape: {df_weather.shape}")
print(f"   - Số records: {len(df_weather)}")
print(f"   - Số sites: {df_weather['site_id'].nunique()}")

# ============================================================================
# 2. PHÂN TÍCH METADATA
# ============================================================================

print("\n" + "=" * 80)
print("BƯỚC 2: PHÂN TÍCH METADATA")
print("=" * 80)

print("\n📊 Thông tin cơ bản về Metadata:")
print(f"   - Tổng số buildings: {len(df_metadata)}")
print(f"   - Số buildings có electricity meter: {len(df_metadata[df_metadata['electricity'] == 'Yes'])}")

# Missing values analysis
print("\n📊 Phân tích Missing Values:")
missing_metadata = df_metadata.isnull().sum().sort_values(ascending=False)
missing_pct = (missing_metadata / len(df_metadata) * 100).round(2)
missing_df = pd.DataFrame({
    'Missing Count': missing_metadata,
    'Missing %': missing_pct
})
missing_df = missing_df[missing_df['Missing Count'] > 0]
print(missing_df.head(15))

# Phân tích các features quan trọng
print("\n📊 Phân tích các Features Quan trọng:")

# Continuous features
continuous_cols = ['sqm', 'yearbuilt', 'numberoffloors', 'occupants']
for col in continuous_cols:
    if col in df_metadata.columns:
        non_null = df_metadata[col].notna().sum()
        pct = (non_null / len(df_metadata) * 100)
        if non_null > 0:
            mean_val = df_metadata[col].mean()
            median_val = df_metadata[col].median()
            print(f"\n   {col}:")
            print(f"      - Có dữ liệu: {non_null}/{len(df_metadata)} ({pct:.1f}%)")
            print(f"      - Mean: {mean_val:.2f}")
            print(f"      - Median: {median_val:.2f}")
            print(f"      - Min: {df_metadata[col].min():.2f}")
            print(f"      - Max: {df_metadata[col].max():.2f}")

# Categorical features
print("\n📊 Phân tích Categorical Features:")

# primaryspaceusage
if 'primaryspaceusage' in df_metadata.columns:
    usage_counts = df_metadata['primaryspaceusage'].value_counts()
    print(f"\n   primaryspaceusage (Top 10):")
    for usage, count in usage_counts.head(10).items():
        pct = (count / len(df_metadata) * 100)
        print(f"      - {usage}: {count} ({pct:.1f}%)")

# site_id
if 'site_id' in df_metadata.columns:
    site_counts = df_metadata['site_id'].value_counts()
    print(f"\n   site_id ({len(site_counts)} sites):")
    for site, count in site_counts.head(10).items():
        pct = (count / len(df_metadata) * 100)
        print(f"      - {site}: {count} ({pct:.1f}%)")

# Visualization: Distribution của sqm
if 'sqm' in df_metadata.columns:
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    df_metadata['sqm'].hist(bins=50, edgecolor='black')
    plt.xlabel('Diện tích (sqm)', fontsize=12)
    plt.ylabel('Số lượng buildings', fontsize=12)
    plt.title('Phân bố Diện tích Buildings', fontsize=14, fontweight='bold')
    plt.yscale('log')
    
    plt.subplot(1, 2, 2)
    plt.boxplot(df_metadata['sqm'].dropna())
    plt.ylabel('Diện tích (sqm)', fontsize=12)
    plt.title('Boxplot Diện tích', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('analysis/metadata_sqm_distribution.png', dpi=150, bbox_inches='tight')
    print("\n✅ Đã lưu biểu đồ phân bố sqm vào: analysis/metadata_sqm_distribution.png")
    plt.close()

# ============================================================================
# 3. PHÂN TÍCH ELECTRICITY DATA
# ============================================================================

print("\n" + "=" * 80)
print("BƯỚC 3: PHÂN TÍCH ELECTRICITY DATA")
print("=" * 80)

# Chuyển sang long format (sample một số buildings để phân tích nhanh)
print("\n📊 Chuyển đổi sang long format (sample 10 buildings)...")
sample_buildings = df_electricity.columns[1:11].tolist()  # Lấy 10 buildings đầu tiên
df_electricity_sample = df_electricity[['timestamp'] + sample_buildings].copy()

df_electricity_long = pd.melt(
    df_electricity_sample,
    id_vars=['timestamp'],
    var_name='building_id',
    value_name='electricity_consumption'
)

df_electricity_long = df_electricity_long.dropna(subset=['electricity_consumption'])

print(f"✅ Long format shape: {df_electricity_long.shape}")
print(f"   - Số records: {len(df_electricity_long)}")
print(f"   - Số buildings: {df_electricity_long['building_id'].nunique()}")

# Thống kê cơ bản
print("\n📊 Thống kê Electricity Consumption:")
print(df_electricity_long['electricity_consumption'].describe())

# Phân tích theo thời gian
print("\n📊 Phân tích theo thời gian:")
df_electricity_long['hour'] = df_electricity_long['timestamp'].dt.hour
df_electricity_long['day_of_week'] = df_electricity_long['timestamp'].dt.dayofweek
df_electricity_long['month'] = df_electricity_long['timestamp'].dt.month

# Trung bình theo giờ
hourly_avg = df_electricity_long.groupby('hour')['electricity_consumption'].mean()
print(f"\n   Trung bình theo giờ trong ngày:")
print(f"      - Giờ cao điểm: {hourly_avg.idxmax()}h ({hourly_avg.max():.2f} kWh)")
print(f"      - Giờ thấp điểm: {hourly_avg.idxmin()}h ({hourly_avg.min():.2f} kWh)")

# Visualization: Pattern theo giờ
plt.figure(figsize=(15, 10))

# Plot 1: Average consumption by hour
plt.subplot(2, 2, 1)
hourly_avg.plot(kind='line', marker='o')
plt.xlabel('Giờ trong ngày', fontsize=12)
plt.ylabel('Trung bình điện tiêu thụ (kWh)', fontsize=12)
plt.title('Pattern Tiêu thụ Điện theo Giờ', fontsize=14, fontweight='bold')
plt.grid(True)

# Plot 2: Average consumption by day of week
plt.subplot(2, 2, 2)
daily_avg = df_electricity_long.groupby('day_of_week')['electricity_consumption'].mean()
daily_avg.plot(kind='bar')
plt.xlabel('Ngày trong tuần (0=Monday)', fontsize=12)
plt.ylabel('Trung bình điện tiêu thụ (kWh)', fontsize=12)
plt.title('Pattern Tiêu thụ Điện theo Ngày', fontsize=14, fontweight='bold')
plt.xticks(rotation=0)
plt.grid(True, axis='y')

# Plot 3: Average consumption by month
plt.subplot(2, 2, 3)
monthly_avg = df_electricity_long.groupby('month')['electricity_consumption'].mean()
monthly_avg.plot(kind='bar', color='orange')
plt.xlabel('Tháng', fontsize=12)
plt.ylabel('Trung bình điện tiêu thụ (kWh)', fontsize=12)
plt.title('Pattern Tiêu thụ Điện theo Tháng', fontsize=14, fontweight='bold')
plt.xticks(rotation=0)
plt.grid(True, axis='y')

# Plot 4: Distribution of consumption
plt.subplot(2, 2, 4)
df_electricity_long['electricity_consumption'].hist(bins=100, edgecolor='black')
plt.xlabel('Điện tiêu thụ (kWh)', fontsize=12)
plt.ylabel('Tần suất', fontsize=12)
plt.title('Phân bố Điện tiêu thụ', fontsize=14, fontweight='bold')
plt.yscale('log')

plt.tight_layout()
plt.savefig('analysis/electricity_patterns.png', dpi=150, bbox_inches='tight')
print("\n✅ Đã lưu biểu đồ patterns vào: analysis/electricity_patterns.png")
plt.close()

# ============================================================================
# 4. PHÂN TÍCH WEATHER DATA
# ============================================================================

print("\n" + "=" * 80)
print("BƯỚC 4: PHÂN TÍCH WEATHER DATA")
print("=" * 80)

print("\n📊 Thông tin Weather Data:")
print(f"   - Số records: {len(df_weather)}")
print(f"   - Số sites: {df_weather['site_id'].nunique()}")
print(f"   - Sites: {df_weather['site_id'].unique().tolist()}")

# Missing values
print("\n📊 Missing Values trong Weather Data:")
missing_weather = df_weather.isnull().sum().sort_values(ascending=False)
missing_weather_pct = (missing_weather / len(df_weather) * 100).round(2)
missing_weather_df = pd.DataFrame({
    'Missing Count': missing_weather,
    'Missing %': missing_weather_pct
})
missing_weather_df = missing_weather_df[missing_weather_df['Missing Count'] > 0]
print(missing_weather_df)

# Thống kê các features thời tiết
weather_features = ['airTemperature', 'cloudCoverage', 'windSpeed', 'dewTemperature']
print("\n📊 Thống kê Weather Features:")
for feature in weather_features:
    if feature in df_weather.columns:
        print(f"\n   {feature}:")
        print(f"      - Mean: {df_weather[feature].mean():.2f}")
        print(f"      - Median: {df_weather[feature].median():.2f}")
        print(f"      - Min: {df_weather[feature].min():.2f}")
        print(f"      - Max: {df_weather[feature].max():.2f}")
        print(f"      - Missing: {df_weather[feature].isnull().sum()} ({df_weather[feature].isnull().sum()/len(df_weather)*100:.1f}%)")

# Visualization: Temperature over time
if 'airTemperature' in df_weather.columns:
    plt.figure(figsize=(15, 5))
    
    # Sample một site
    sample_site = df_weather['site_id'].iloc[0]
    df_weather_sample = df_weather[df_weather['site_id'] == sample_site].head(1000)
    
    plt.plot(df_weather_sample['timestamp'], df_weather_sample['airTemperature'], linewidth=1)
    plt.xlabel('Thời gian', fontsize=12)
    plt.ylabel('Nhiệt độ (°C)', fontsize=12)
    plt.title(f'Nhiệt độ theo thời gian - Site: {sample_site}', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('analysis/weather_temperature.png', dpi=150, bbox_inches='tight')
    print("\n✅ Đã lưu biểu đồ nhiệt độ vào: analysis/weather_temperature.png")
    plt.close()

# ============================================================================
# 5. TỔNG HỢP VÀ KẾT LUẬN
# ============================================================================

print("\n" + "=" * 80)
print("BƯỚC 5: TỔNG HỢP VÀ KẾT LUẬN")
print("=" * 80)

print("\n📋 TÓM TẮT EDA:")
print("=" * 60)
print(f"1. Metadata:")
print(f"   - Tổng số buildings: {len(df_metadata)}")
print(f"   - Buildings có electricity: {len(df_metadata[df_metadata['electricity'] == 'Yes'])}")
print(f"   - Missing values nhiều nhất: {missing_df.index[0] if len(missing_df) > 0 else 'N/A'}")

print(f"\n2. Electricity Data:")
print(f"   - Số timestamps: {len(df_electricity)}")
print(f"   - Số buildings: {len(df_electricity.columns) - 1}")
print(f"   - Khoảng thời gian: {df_electricity['timestamp'].min()} đến {df_electricity['timestamp'].max()}")

print(f"\n3. Weather Data:")
print(f"   - Số records: {len(df_weather)}")
print(f"   - Số sites: {df_weather['site_id'].nunique()}")

print("\n✅ Hoàn thành EDA Analysis!")
print("   Các biểu đồ đã được lưu trong thư mục 'analysis/'")

