# Step-by-Step Debugging Guide for DiCE IndexError

## Quick Start

Run these commands in order to identify the exact issue:

### Step 1: Basic Column Check
```bash
cd /Users/duonghn/Documents/mnt/data/cmcglobal/grid_balancer
source .venv/bin/activate
python src/debug_dice.py > debug_output_step1.txt 2>&1
```

**What to report:**
- Look for any columns marked with ❌
- Note which columns have "Has '.' in string: False"
- Check the "SUMMARY" section at the end

### Step 2: Detailed DiCE Test
```bash
python src/debug_dice_detailed.py > debug_output_step2.txt 2>&1
```

**What to report:**
- Which step fails (Step 2, 3, or 4)
- The full error traceback
- Any columns identified in the error checking section

### Step 3: Advanced Interception (Most Important)
```bash
python src/debug_dice_advanced.py > debug_output_step3.txt 2>&1
```

**What to report:**
- The last column being processed before the error
- The mode value and string for that column
- Whether the split result had 2 elements
- Any warnings about modes with no decimal part

### Step 4: Check DiCE's Internal Data Processing

DiCE might be modifying the data. Check what DiCE sees:

```python
import pandas as pd
import numpy as np
import dice_ml

# After creating dice_data, check what DiCE sees
dice_data = dice_ml.Data(...)

# Check the internal dataframe
internal_df = dice_data.data_df

for col in dice_data.continuous_feature_names:
    if col in internal_df.columns:
        if internal_df[col].dtype == np.float32 or internal_df[col].dtype == np.float64:
            modes = internal_df[col].mode()
            if len(modes) > 0:
                mode_str = str(modes[0])
                print(f"{col}: mode='{mode_str}', has_dot={'.' in mode_str}")
                split_result = mode_str.split('.')
                if len(split_result) <= 1:
                    print(f"  ❌ PROBLEM: {col} mode string has no decimal part!")
```

## What to Look For

### Red Flags:
1. **Mode string without decimal point**: `"0"` instead of `"0.0"`
2. **Scientific notation**: `"1e-07"` instead of `"0.0000001"`
3. **Empty mode list**: `len(modes) == 0`
4. **Type conversion issues**: Column changes from float64 to float32

### Common Problem Patterns:

**Pattern 1: All values are 0.0**
```
Column: occupants
Mode: 0.0
Mode string: '0'  ← Problem!
```

**Pattern 2: Scientific notation**
```
Column: some_column
Mode: 1e-07
Mode string: '1e-07'  ← Problem! No '.' in expected format
```

**Pattern 3: Integer values**
```
Column: numberoffloors
Mode: 1.0
Mode string: '1'  ← Problem!
```

## Expected vs Actual Behavior

### Expected (Working):
```
Mode value: 0.0
Mode string: '0.0'
Split result: ['0', '0']
✅ Works
```

### Problem (Failing):
```
Mode value: 0.0
Mode string: '0'  ← Missing decimal point!
Split result: ['0']
❌ Fails: IndexError when accessing [1]
```

## Reporting Checklist

When you report back, please include:

- [ ] Output from `debug_dice.py` (especially the SUMMARY section)
- [ ] Output from `debug_dice_detailed.py` (the full traceback)
- [ ] Output from `debug_dice_advanced.py` (the last column processed)
- [ ] List of any columns identified as problematic
- [ ] The exact mode values and strings for problematic columns
- [ ] Whether the issue occurs with a smaller dataset (try sample_size=1000)
- [ ] DiCE version: `pip show dice-ml | grep Version`
- [ ] Python version: `python --version`

## Quick Test Commands

Test with smaller dataset:
```python
# In debug_dice_advanced.py, change:
sample_size = min(1000, len(df))  # Instead of 5000
```

Test with minimal features:
```python
# Only use core features
continuous_features = ['sqm', 'occupants', 'airTemperature']
```

## Next Steps After Identification

Once you identify the problematic column:

1. **Note the column name**
2. **Check its values**: Are they all integer-like?
3. **Check the mode**: What is the exact mode value and string?
4. **Apply fix**: Add epsilon to integer-like values
5. **Verify**: Re-run debug script to confirm fix

## Example Fix

If column `occupants` is problematic:

```python
# Before DiCE Data creation, add:
if 'occupants' in df_for_dice.columns:
    integer_like = df_for_dice['occupants'] == df_for_dice['occupants'].astype(int)
    if integer_like.any():
        df_for_dice.loc[integer_like, 'occupants'] = \
            df_for_dice.loc[integer_like, 'occupants'] + 0.0001
```

## Contact Information

After running all debug scripts, share:
- The three output files (debug_output_step*.txt)
- Any observations you made
- Which columns seem suspicious

This will help me create a targeted fix for your specific data.
