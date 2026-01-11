#!/usr/bin/env python3
"""
Debugging script for DiCE IndexError issue
This script will help identify exactly which column and value is causing the problem
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path

print("=" * 80)
print("DiCE DEBUGGING SCRIPT")
print("=" * 80)

# Load data
processed_data_path = "output/processed_data.parquet"
features_info_path = "output/features_info.json"
encoders_path = "output/models/label_encoders_dice.pkl"

print("\n1. Loading data...")
df = pd.read_parquet(processed_data_path)
print(f"   ✅ Loaded data: {df.shape}")

with open(features_info_path, 'r') as f:
    features_info = json.load(f)

all_features = (
    features_info['continuous_features'] + 
    features_info['time_features'] + 
    features_info['lag_features']
)
categorical_features = features_info['categorical_features']

all_features = [f for f in all_features if f in df.columns]
categorical_features = [f for f in categorical_features if f in df.columns]

print(f"   ✅ Features: {len(all_features)} continuous, {len(categorical_features)} categorical")

# Sample data
sample_size = min(5000, len(df))
df_sample = df.sample(n=sample_size, random_state=42).copy()
df_sample = df_sample.dropna(subset=['electricity_consumption'])

print(f"\n2. Sample data: {df_sample.shape}")

# Prepare features
available_features = [f for f in all_features if f in df_sample.columns]
available_categorical = [f for f in categorical_features if f in df_sample.columns]

df_for_dice = df_sample[available_features + available_categorical + ['electricity_consumption']].copy()

# Decode categoricals
with open(encoders_path, 'rb') as f:
    label_encoders = pickle.load(f)

for col in available_categorical:
    if col in df_for_dice.columns and col in label_encoders:
        le = label_encoders[col]
        col_series = df_for_dice[col]
        sample_vals = col_series.dropna()
        if len(sample_vals) > 0:
            sample_val = sample_vals.iloc[0]
            if isinstance(sample_val, str) and sample_val in le.classes_:
                continue
        
        def decode_value(x):
            try:
                if pd.isna(x):
                    return 'Unknown'
                if isinstance(x, str):
                    if x in le.classes_:
                        return x
                    try:
                        int_val = int(float(x))
                        if int_val in le.classes_:
                            return le.inverse_transform([int_val])[0]
                    except:
                        pass
                    return 'Unknown'
                if isinstance(x, (int, float, np.integer, np.floating)):
                    int_val = int(x)
                    if int_val in le.classes_:
                        return le.inverse_transform([int_val])[0]
                return 'Unknown'
            except:
                return 'Unknown'
        
        df_for_dice[col] = col_series.apply(decode_value)

# Separate continuous and categorical
continuous_features = []
for f in available_features:
    if f not in available_categorical:
        if f not in ['hour', 'day_of_week', 'month', 'year', 'is_weekend']:
            continuous_features.append(f)

continuous_features = [f for f in continuous_features if f in df_for_dice.columns]
dice_categorical_features = [f for f in available_categorical if f in df_for_dice.columns]

if 'electricity_consumption' in continuous_features:
    continuous_features.remove('electricity_consumption')
if 'electricity_consumption' in dice_categorical_features:
    dice_categorical_features.remove('electricity_consumption')

# Clean dataframe
df_for_dice = df_for_dice.loc[:, ~df_for_dice.columns.duplicated()].copy()

for col in dice_categorical_features:
    if col in df_for_dice.columns:
        df_for_dice[col] = df_for_dice[col].astype(str)

if 'electricity_consumption' in df_for_dice.columns:
    df_for_dice['electricity_consumption'] = pd.to_numeric(
        df_for_dice['electricity_consumption'], errors='coerce'
    ).fillna(0)

# Filter to numeric
numeric_continuous = []
for f in continuous_features:
    if f in df_for_dice.columns:
        if pd.api.types.is_numeric_dtype(df_for_dice[f]):
            numeric_continuous.append(f)

continuous_features = numeric_continuous

print(f"\n3. Continuous features: {len(continuous_features)}")
print(f"   Categorical features: {len(dice_categorical_features)}")

# DEBUG: Check each continuous feature for the problematic mode issue
print("\n" + "=" * 80)
print("DEBUGGING: Checking each continuous feature for mode() issues")
print("=" * 80)

problematic_columns = []

for col in continuous_features:
    if col not in df_for_dice.columns:
        continue
    
    if not pd.api.types.is_float_dtype(df_for_dice[col]):
        continue
    
    print(f"\n📊 Column: {col}")
    print(f"   Type: {df_for_dice[col].dtype}")
    
    col_data = df_for_dice[col].dropna()
    if len(col_data) == 0:
        print(f"   ⚠️  No data")
        continue
    
    # Check for integer-like values
    integer_like = col_data == col_data.astype(int)
    integer_like_count = integer_like.sum()
    print(f"   Integer-like values: {integer_like_count} / {len(col_data)}")
    
    # Get mode
    try:
        modes = col_data.mode()
        print(f"   Mode count: {len(modes)}")
        
        if len(modes) > 0:
            mode_val = modes.iloc[0]
            mode_str = str(mode_val)
            print(f"   Mode value: {mode_val}")
            print(f"   Mode string: '{mode_str}'")
            print(f"   Mode type: {type(mode_val)}")
            
            # Check if mode string has decimal point
            has_decimal = '.' in mode_str
            print(f"   Has '.' in string: {has_decimal}")
            
            if has_decimal:
                try:
                    split_result = mode_str.split('.')
                    print(f"   Split result: {split_result}")
                    if len(split_result) > 1:
                        decimal_part = split_result[1]
                        print(f"   Decimal part: '{decimal_part}'")
                        print(f"   Decimal part length: {len(decimal_part)}")
                    else:
                        print(f"   ⚠️  WARNING: Split by '.' but no second element!")
                        problematic_columns.append({
                            'column': col,
                            'mode': mode_val,
                            'mode_str': mode_str,
                            'issue': 'No decimal part after split'
                        })
                except Exception as e:
                    print(f"   ❌ ERROR splitting: {e}")
                    problematic_columns.append({
                        'column': col,
                        'mode': mode_val,
                        'mode_str': mode_str,
                        'issue': f'Split error: {e}'
                    })
            else:
                print(f"   ❌ PROBLEM: Mode string has no decimal point!")
                print(f"   This will cause IndexError in DiCE's get_decimal_precisions")
                problematic_columns.append({
                    'column': col,
                    'mode': mode_val,
                    'mode_str': mode_str,
                    'issue': 'No decimal point in string representation'
                })
                
                # Show sample values
                print(f"   Sample values (first 5): {col_data.head().tolist()}")
                print(f"   Sample value strings: {[str(v) for v in col_data.head().tolist()]}")
                
                # Check if all values are integer-like
                all_integer_like = integer_like.all()
                print(f"   All values integer-like: {all_integer_like}")
                
    except Exception as e:
        print(f"   ❌ ERROR calculating mode: {e}")
        problematic_columns.append({
            'column': col,
            'mode': None,
            'mode_str': None,
            'issue': f'Mode calculation error: {e}'
        })

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

if problematic_columns:
    print(f"\n❌ Found {len(problematic_columns)} problematic column(s):")
    for item in problematic_columns:
        print(f"\n  Column: {item['column']}")
        print(f"  Issue: {item['issue']}")
        print(f"  Mode value: {item['mode']}")
        print(f"  Mode string: '{item['mode_str']}'")
else:
    print("\n✅ No problematic columns found in mode calculation")

print("\n" + "=" * 80)
print("TESTING: Simulating DiCE's get_decimal_precisions logic")
print("=" * 80)

# Simulate what DiCE does
for col in continuous_features:
    if col not in df_for_dice.columns:
        continue
    
    if not (df_for_dice[col].dtype == np.float32 or df_for_dice[col].dtype == np.float64):
        continue
    
    try:
        modes = df_for_dice[col].mode()
        if len(modes) > 0:
            mode_str = str(modes[0])
            print(f"\nColumn: {col}")
            print(f"  Mode: {modes[0]}")
            print(f"  Mode string: '{mode_str}'")
            
            # This is the line that fails in DiCE
            try:
                split_result = mode_str.split('.')
                if len(split_result) > 1:
                    decimal_part = split_result[1]
                    maxp = len(decimal_part)
                    print(f"  ✅ Success: decimal part length = {maxp}")
                else:
                    print(f"  ❌ FAIL: split('.') returned {split_result}, no second element!")
                    print(f"     This will cause IndexError: list index out of range")
            except IndexError as e:
                print(f"  ❌ IndexError: {e}")
    except Exception as e:
        print(f"\nColumn: {col}")
        print(f"  ❌ Error: {e}")

print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)

if problematic_columns:
    print("\nTo fix the issue, you need to ensure that:")
    print("1. All float columns have at least one value with a decimal part")
    print("2. The mode value's string representation contains a '.' character")
    print("3. After splitting by '.', there is a second element")
    print("\nSuggested fix:")
    print("  - Add a small epsilon (e.g., 0.0001) to integer-like values")
    print("  - Ensure the epsilon is large enough to avoid scientific notation")
    print("  - Verify the mode after modification has a decimal part")
else:
    print("\nNo issues found in mode calculation.")
    print("The problem might be elsewhere in DiCE initialization.")

print("\n" + "=" * 80)
print("DEBUGGING COMPLETE")
print("=" * 80)
