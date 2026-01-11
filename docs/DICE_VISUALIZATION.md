# DiCE Recommendations Visualization Guide

## Overview

The `DiceExplainer` class now includes methods to convert recommendations to DataFrames and create visualizations. This makes it easy to analyze and present counterfactual recommendations.

## Quick Start

```python
from src.dice_explainer import DiceExplainer

# Initialize
explainer = DiceExplainer()

# Generate recommendations
result = explainer.generate_recommendations(
    json_data=your_data,
    threshold=100.0,
    total_cfs=5
)

# Convert to DataFrame
df_summary = explainer.recommendations_to_dataframe(result)
print(df_summary)

# Create visualization
explainer.visualize_recommendations(result, save_path='recommendations.png')
```

## Available Methods

### 1. `recommendations_to_dataframe(result)`

Convert recommendations to a summary DataFrame.

**Returns:**
- DataFrame with columns:
  - `Counterfactual ID`
  - `Predicted Consumption (kWh)`
  - `Reduction (kWh)`
  - `Reduction (%)`
  - `Below Threshold`
  - `Number of Changes`

**Example:**
```python
df = explainer.recommendations_to_dataframe(result)
print(df)
# Output:
#    Counterfactual ID  Predicted Consumption (kWh)  Reduction (kWh)  ...
# 0                  0                     95.2        55.3            ...
# 1                  1                     88.5        62.0            ...
```

### 2. `changes_to_dataframe(result, counterfactual_id=None)`

Convert detailed changes to a DataFrame.

**Parameters:**
- `result`: Result from `generate_recommendations()`
- `counterfactual_id`: Optional specific counterfactual ID (if None, returns all)

**Returns:**
- DataFrame with columns:
  - `Counterfactual ID`
  - `Feature`
  - `Description`
  - `Original Value`
  - `Suggested Value`
  - `Change`
  - `Change (%)`
  - `Action`

**Example:**
```python
# All changes
df_all = explainer.changes_to_dataframe(result)

# Changes for specific counterfactual
df_cf0 = explainer.changes_to_dataframe(result, counterfactual_id=0)
```

### 3. `visualize_recommendations(result, save_path=None, figsize=(12, 8))`

Create comprehensive visualization of all recommendations.

**Features:**
- Consumption comparison chart (with threshold line)
- Reduction amount chart
- Reduction percentage chart
- Most frequently changed features

**Example:**
```python
# Display interactively
explainer.visualize_recommendations(result)

# Save to file
explainer.visualize_recommendations(
    result, 
    save_path='output/recommendations.png'
)
```

### 4. `visualize_changes(result, counterfactual_id=0, save_path=None, figsize=(10, 6))`

Visualize changes for a specific counterfactual.

**Features:**
- Percentage change by feature (bar chart)
- Original vs suggested values comparison

**Example:**
```python
explainer.visualize_changes(
    result, 
    counterfactual_id=0,
    save_path='output/changes_cf0.png'
)
```

### 5. `export_to_csv(result, base_path)`

Export recommendations to CSV files.

**Returns:**
- Dictionary with paths to created files:
  - `'summary'`: Summary CSV
  - `'changes'`: Detailed changes CSV

**Example:**
```python
files = explainer.export_to_csv(result, 'output/recommendations')
print(files)
# {'summary': 'output/recommendations_summary.csv',
#  'changes': 'output/recommendations_changes.csv'}
```

### 6. `export_to_excel(result, excel_path)`

Export recommendations to Excel with multiple sheets.

**Returns:**
- Path to created Excel file (or empty string if openpyxl not available)

**Sheets:**
- `Summary`: Recommendations summary
- `Changes`: All changes
- `CF_0`, `CF_1`, ...: Individual counterfactual sheets

**Example:**
```python
path = explainer.export_to_excel(result, 'output/recommendations.xlsx')
# Requires: pip install openpyxl
```

## DataFrame Formatting Options

Pandas DataFrames support multiple output formats:

```python
df = explainer.recommendations_to_dataframe(result)

# String representation
print(df.to_string(index=False))

# HTML (for web display)
html = df.to_html(index=False)

# JSON
json_str = df.to_json(orient='records', indent=2)

# Markdown (requires tabulate)
try:
    markdown = df.to_markdown(index=False)
except ImportError:
    print("Install tabulate: pip install tabulate")

# CSV
df.to_csv('output.csv', index=False)
```

## Complete Example

See `src/dice_visualization_example.py` for a complete working example.

**Run it:**
```bash
cd /Users/duonghn/Documents/mnt/data/cmcglobal/grid_balancer
source .venv/bin/activate
python src/dice_visualization_example.py
```

This will:
1. Generate recommendations
2. Convert to DataFrames
3. Create visualizations
4. Export to CSV
5. Show various formatting options

## Visualization Output

The `visualize_recommendations()` method creates a 2x2 subplot with:

1. **Top Left**: Predicted consumption by counterfactual (with threshold and current consumption lines)
2. **Top Right**: Reduction amount in kWh (with percentage labels)
3. **Bottom Left**: Reduction percentage
4. **Bottom Right**: Most frequently changed features

The `visualize_changes()` method creates a 1x2 subplot with:

1. **Left**: Percentage change by feature (color-coded: red for decrease, green for increase)
2. **Right**: Original vs suggested values comparison

## Requirements

**Required:**
- `pandas`
- `matplotlib` (for visualizations)
- `numpy`

**Optional:**
- `openpyxl` (for Excel export): `pip install openpyxl`
- `tabulate` (for markdown export): `pip install tabulate`

## Tips

1. **Threshold Selection**: Use a reasonable threshold (e.g., 80% of current consumption) to ensure counterfactuals can be generated.

2. **Method Selection**: 
   - `'random'`: Faster, good for quick analysis
   - `'genetic'`: Slower but better quality counterfactuals

3. **Visualization Size**: Adjust `figsize` parameter if figures are too small/large for your display.

4. **Export Formats**: CSV is always available. Excel requires `openpyxl`, markdown requires `tabulate`.

5. **Empty Results**: If no counterfactuals are generated, DataFrames will be empty. Check the error message in the result dictionary.
