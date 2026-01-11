#!/usr/bin/env python3
"""
Example: Visualizing DiCE Recommendations
Shows how to convert recommendations to DataFrames and create visualizations
"""

import json
import sys
from pathlib import Path

# Add src to path if needed
sys.path.insert(0, str(Path(__file__).parent))

from dice_explainer import DiceExplainer

# Initialize DiCE Explainer
print("=" * 80)
print("DiCE RECOMMENDATIONS VISUALIZATION EXAMPLE")
print("=" * 80)

explainer = DiceExplainer()

# Example building data
json_data = {
    'time': '2016-01-01T21:00:00',
    'building_id': 'Bear_education_Sharon',
    'site_id': 'Bear',
    'primaryspaceusage': 'Education',
    'sub_primaryspaceusage': 'Education',
    'sqm': 5261.7,
    'yearbuilt': 1953,
    'numberoffloors': 5,
    'occupants': 200,
    'timezone': 'US/Pacific',
    'airTemperature': 25.0,
    'cloudCoverage': 30.0,
    'dewTemperature': 18.0,
    'windSpeed': 2.6,
    'seaLvlPressure': 1020.7,
    'precipDepth1HR': 0.0
}

# First, check current prediction
current_pred = explainer.inference.predict(json_data)
print(f"\n📊 Current predicted consumption: {current_pred:.2f} kWh")

# Set threshold (e.g., 80% of current consumption)
threshold = current_pred * 0.8
print(f"🎯 Target threshold: {threshold:.2f} kWh")

# Generate recommendations
print("\n📊 Generating recommendations...")
result = explainer.generate_recommendations(
    json_data=json_data,
    threshold=threshold,
    total_cfs=5,
    method='random'  # Use 'random' for faster results
)

if not result['success']:
    print(f"❌ Error: {result.get('error', 'Unknown error')}")
    if 'No counterfactuals' in result.get('error', ''):
        print("\n💡 Tip: Try a higher threshold or adjust the method to 'genetic'")
    sys.exit(1)

print(f"✅ Generated {result['total_counterfactuals']} recommendations\n")

# ============================================================================
# 1. Convert to DataFrame - Summary
# ============================================================================
print("=" * 80)
print("1. RECOMMENDATIONS SUMMARY DATAFRAME")
print("=" * 80)

df_summary = explainer.recommendations_to_dataframe(result)
print("\n📋 Summary DataFrame:")
print(df_summary.to_string(index=False))

# ============================================================================
# 2. Convert to DataFrame - Detailed Changes
# ============================================================================
print("\n" + "=" * 80)
print("2. DETAILED CHANGES DATAFRAME")
print("=" * 80)

df_changes = explainer.changes_to_dataframe(result)
print(f"\n📋 Changes DataFrame ({len(df_changes)} rows):")
print(df_changes.head(10).to_string(index=False))
if len(df_changes) > 10:
    print(f"\n... and {len(df_changes) - 10} more rows")

# ============================================================================
# 3. Changes for Specific Counterfactual
# ============================================================================
print("\n" + "=" * 80)
print("3. CHANGES FOR COUNTERFACTUAL 0")
print("=" * 80)

df_cf0 = explainer.changes_to_dataframe(result, counterfactual_id=0)
print(f"\n📋 Counterfactual 0 Changes ({len(df_cf0)} rows):")
print(df_cf0.to_string(index=False))

# ============================================================================
# 4. Create Visualizations
# ============================================================================
print("\n" + "=" * 80)
print("4. CREATING VISUALIZATIONS")
print("=" * 80)

# Save visualizations
output_dir = Path(__file__).parent.parent / "output" / "visualizations"
output_dir.mkdir(parents=True, exist_ok=True)

# Overall recommendations visualization
viz_path = output_dir / "dice_recommendations.png"
print(f"\n📊 Creating recommendations visualization...")
explainer.visualize_recommendations(result, save_path=str(viz_path))
print(f"✅ Saved to: {viz_path}")

# Changes visualization for first counterfactual
viz_changes_path = output_dir / "dice_changes_cf0.png"
print(f"\n📊 Creating changes visualization for Counterfactual 0...")
explainer.visualize_changes(result, counterfactual_id=0, save_path=str(viz_changes_path))
print(f"✅ Saved to: {viz_changes_path}")

# ============================================================================
# 5. Export to CSV
# ============================================================================
print("\n" + "=" * 80)
print("5. EXPORTING TO CSV")
print("=" * 80)

csv_base = output_dir / "dice_recommendations"
files_created = explainer.export_to_csv(result, str(csv_base))
print(f"\n✅ Exported files:")
for file_type, file_path in files_created.items():
    print(f"   • {file_type}: {file_path}")

# ============================================================================
# 6. Export to Excel (if openpyxl is available)
# ============================================================================
print("\n" + "=" * 80)
print("6. EXPORTING TO EXCEL")
print("=" * 80)

excel_path = output_dir / "dice_recommendations.xlsx"
result_path = explainer.export_to_excel(result, str(excel_path))
if result_path:
    print(f"✅ Exported to: {result_path}")
else:
    print("⚠️  Excel export requires openpyxl: pip install openpyxl")

# ============================================================================
# 7. Display DataFrame in different formats
# ============================================================================
print("\n" + "=" * 80)
print("7. DATAFRAME FORMATTING EXAMPLES")
print("=" * 80)

# Markdown format (if tabulate is available)
print("\n📋 Summary as Markdown:")
try:
    print(df_summary.to_markdown(index=False))
except ImportError:
    print("⚠️  tabulate not available. Install with: pip install tabulate")
    print("   Using string representation instead:")
    print(df_summary.to_string(index=False))

# HTML format
print("\n📋 Summary as HTML (first 3 rows):")
print(df_summary.head(3).to_html(index=False))

# JSON format
print("\n📋 Summary as JSON:")
print(df_summary.head(3).to_json(orient='records', indent=2))

print("\n" + "=" * 80)
print("✅ ALL VISUALIZATION EXAMPLES COMPLETED!")
print("=" * 80)
