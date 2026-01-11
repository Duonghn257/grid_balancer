#!/usr/bin/env python3
"""
Advanced debugging script - intercepts DiCE's get_decimal_precisions to catch the exact column
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
import dice_ml
from dice_ml import Dice
import sys

# Monkey patch to catch the exact column causing the issue
original_get_decimal_precisions = None

def debug_get_decimal_precisions(self, output_type="list"):
    """Wrapped version that catches the problematic column"""
    precisions = [0] * len(self.continuous_feature_names)
    precisions_dict = {}
    
    for ix, col in enumerate(self.continuous_feature_names):
        try:
            if (self.continuous_features_precision is not None) and (col in self.continuous_features_precision):
                precisions[ix] = self.continuous_features_precision[col]
                precisions_dict[col] = self.continuous_features_precision[col]
            elif self.data_df[col].dtype == np.float32 or self.data_df[col].dtype == np.float64:
                print(f"\n🔍 Processing column: {col}")
                modes = self.data_df[col].mode()
                print(f"   Mode count: {len(modes)}")
                
                if len(modes) > 0:
                    mode_val = modes.iloc[0]
                    mode_str = str(mode_val)
                    print(f"   Mode value: {mode_val}")
                    print(f"   Mode string: '{mode_str}'")
                    print(f"   Mode type: {type(mode_val)}")
                    
                    # This is the line that fails
                    try:
                        split_result = mode_str.split('.')
                        print(f"   Split result: {split_result}")
                        print(f"   Split length: {len(split_result)}")
                        
                        if len(split_result) > 1:
                            decimal_part = split_result[1]
                            maxp = len(decimal_part)
                            print(f"   ✅ Success: decimal part length = {maxp}")
                            
                            for mx in range(len(modes)):
                                mode_mx_str = str(modes[mx])
                                split_mx = mode_mx_str.split('.')
                                if len(split_mx) > 1:
                                    prec = len(split_mx[1])
                                    if prec > maxp:
                                        maxp = prec
                                else:
                                    print(f"   ⚠️  WARNING: Mode[{mx}] = '{mode_mx_str}' has no decimal part!")
                            
                            precisions[ix] = maxp
                            precisions_dict[col] = maxp
                        else:
                            print(f"   ❌ ERROR: Split by '.' returned {split_result}, no second element!")
                            print(f"   This will cause IndexError!")
                            print(f"   Column data sample:")
                            print(f"     First 5 values: {self.data_df[col].head().tolist()}")
                            print(f"     First 5 as strings: {[str(v) for v in self.data_df[col].head().tolist()]}")
                            print(f"     All values integer-like: {(self.data_df[col] == self.data_df[col].astype(int)).all()}")
                            raise IndexError(f"Column '{col}' mode string '{mode_str}' has no decimal part after split")
                    except IndexError as e:
                        print(f"   ❌ IndexError: {e}")
                        raise
        except Exception as e:
            print(f"   ❌ Error processing column {col}: {e}")
            raise
    
    if output_type == "list":
        return precisions
    return precisions_dict

print("=" * 80)
print("ADVANCED DiCE DEBUGGING - With Column Interception")
print("=" * 80)

# Load and prepare data
processed_data_path = "output/processed_data.parquet"
features_info_path = "output/features_info.json"
encoders_path = "output/models/label_encoders_dice.pkl"

print("\n1. Loading and preparing data...")
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

print(f"   ✅ Prepared {len(continuous_features)} continuous and {len(dice_categorical_features)} categorical features")

# Create DiCE Data
print("\n2. Creating DiCE Data...")
dice_data = dice_ml.Data(
    dataframe=df_for_dice,
    continuous_features=continuous_features,
    categorical_features=dice_categorical_features if dice_categorical_features else None,
    outcome_name='electricity_consumption'
)
print("   ✅ DiCE Data created")

# Create DiCE Model
print("\n3. Creating DiCE Model...")
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.inference import ElectricityConsumptionInference
inference = ElectricityConsumptionInference()

dice_model = dice_ml.Model(
    model=inference.model,
    backend='sklearn',
    model_type='regressor'
)
print("   ✅ DiCE Model created")

# Monkey patch to intercept the problematic method
print("\n4. Setting up debugging interception...")
from dice_ml.data_interfaces import public_data_interface
original_get_decimal_precisions = public_data_interface.PublicData.get_decimal_precisions
public_data_interface.PublicData.get_decimal_precisions = debug_get_decimal_precisions
print("   ✅ Interception set up")

# Now try to create the explainer - this will show us exactly which column fails
print("\n5. Creating DiCE Explainer (this will show the problematic column)...")
print("=" * 80)

try:
    explainer = Dice(
        dice_data,
        dice_model,
        method='random'
    )
    print("\n✅ SUCCESS! DiCE Explainer created!")
except Exception as e:
    print(f"\n❌ Error occurred: {e}")
    print(f"   Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()

# Restore original method
if original_get_decimal_precisions:
    public_data_interface.PublicData.get_decimal_precisions = original_get_decimal_precisions

print("\n" + "=" * 80)
print("DEBUGGING COMPLETE")
print("=" * 80)
print("\nPlease report:")
print("1. Which column was being processed when the error occurred")
print("2. The mode value and its string representation")
print("3. Whether the split result had a second element")
print("4. Any other information shown in the output above")
