#!/usr/bin/env python3
"""
Detailed debugging script - tests DiCE initialization step by step
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
import dice_ml
from dice_ml import Dice

print("=" * 80)
print("DETAILED DiCE DEBUGGING - Step by Step")
print("=" * 80)

# Load and prepare data (same as dice_explainer)
processed_data_path = "output/processed_data.parquet"
features_info_path = "output/features_info.json"
encoders_path = "output/models/label_encoders_dice.pkl"

print("\nStep 1: Loading data...")
df = pd.read_parquet(processed_data_path)
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

sample_size = min(5000, len(df))
df_sample = df.sample(n=sample_size, random_state=42).copy()
df_sample = df_sample.dropna(subset=['electricity_consumption'])

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

# Separate features
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

df_for_dice = df_for_dice.loc[:, ~df_for_dice.columns.duplicated()].copy()

for col in dice_categorical_features:
    if col in df_for_dice.columns:
        df_for_dice[col] = df_for_dice[col].astype(str)

if 'electricity_consumption' in df_for_dice.columns:
    df_for_dice['electricity_consumption'] = pd.to_numeric(
        df_for_dice['electricity_consumption'], errors='coerce'
    ).fillna(0)

numeric_continuous = []
for f in continuous_features:
    if f in df_for_dice.columns:
        if pd.api.types.is_numeric_dtype(df_for_dice[f]):
            numeric_continuous.append(f)

continuous_features = numeric_continuous

print(f"   Continuous: {len(continuous_features)}")
print(f"   Categorical: {len(dice_categorical_features)}")

# Test DiCE Data creation with detailed error tracking
print("\nStep 2: Testing DiCE Data creation...")

try:
    dice_data = dice_ml.Data(
        dataframe=df_for_dice,
        continuous_features=continuous_features,
        categorical_features=dice_categorical_features if dice_categorical_features else None,
        outcome_name='electricity_consumption'
    )
    print("   ✅ DiCE Data created successfully")
except Exception as e:
    print(f"   ❌ Error creating DiCE Data: {e}")
    print(f"   Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
    
    # Try to identify which column is causing the issue
    print("\n   Attempting to identify problematic column...")
    for col in continuous_features:
        if col not in df_for_dice.columns:
            continue
        if not (df_for_dice[col].dtype == np.float32 or df_for_dice[col].dtype == np.float64):
            continue
        try:
            # Simulate what DiCE does in get_decimal_precisions
            modes = df_for_dice[col].mode()
            if len(modes) > 0:
                mode_str = str(modes[0])
                split_result = mode_str.split('.')
                if len(split_result) <= 1:
                    print(f"   ❌ PROBLEM COLUMN: {col}")
                    print(f"      Mode: {modes[0]}")
                    print(f"      Mode string: '{mode_str}'")
                    print(f"      Split result: {split_result}")
        except Exception as col_e:
            print(f"   ❌ ERROR with column {col}: {col_e}")
    
    exit(1)

# Test DiCE Model creation
print("\nStep 3: Testing DiCE Model creation...")
try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from src.inference import ElectricityConsumptionInference
    inference = ElectricityConsumptionInference()
    
    dice_model = dice_ml.Model(
        model=inference.model,
        backend='sklearn',
        model_type='regressor'
    )
    print("   ✅ DiCE Model created successfully")
except Exception as e:
    print(f"   ❌ Error creating DiCE Model: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test DiCE Explainer creation (this is where the error occurs)
print("\nStep 4: Testing DiCE Explainer creation...")
print("   (This is where the IndexError typically occurs)")

try:
    explainer = Dice(
        dice_data,
        dice_model,
        method='random'
    )
    print("   ✅ DiCE Explainer created successfully!")
    print("\n" + "=" * 80)
    print("SUCCESS! DiCE is working correctly.")
    print("=" * 80)
except IndexError as e:
    print(f"   ❌ IndexError occurred: {e}")
    print("\n   This is the error we're trying to fix.")
    print("   It happens in dice_ml's get_decimal_precisions method.")
    print("\n   Let's check the exact column causing the issue...")
    
    # Check each column's mode more carefully
    print("\n   Checking each column's mode string representation:")
    for col in continuous_features:
        if col not in df_for_dice.columns:
            continue
        if not (df_for_dice[col].dtype == np.float32 or df_for_dice[col].dtype == np.float64):
            continue
        
        try:
            modes = df_for_dice[col].mode()
            if len(modes) > 0:
                mode_val = modes.iloc[0]
                mode_str = str(mode_val)
                
                # Try the exact operation that DiCE does
                try:
                    split_result = mode_str.split('.')
                    if len(split_result) > 1:
                        decimal_part = split_result[1]
                        # This is what DiCE does - it should work
                        pass
                    else:
                        print(f"   ❌ COLUMN: {col}")
                        print(f"      Mode value: {mode_val}")
                        print(f"      Mode string: '{mode_str}'")
                        print(f"      Split result: {split_result}")
                        print(f"      This will cause IndexError!")
                except IndexError:
                    print(f"   ❌ COLUMN: {col} - IndexError when accessing split result")
        except Exception as col_e:
            print(f"   ❌ Column {col} error: {col_e}")
    
    import traceback
    traceback.print_exc()
    
except Exception as e:
    print(f"   ❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("DEBUGGING COMPLETE")
print("=" * 80)
