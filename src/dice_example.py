#!/usr/bin/env python3
"""
Example usage of DiCE Explainer for electricity consumption reduction
"""

import json
import sys
from pathlib import Path

# Add src to path if needed
sys.path.insert(0, str(Path(__file__).parent))

from dice_explainer import DiceExplainer

# Initialize DiCE Explainer
print("=" * 80)
print("DiCE EXPLAINER - ELECTRICITY CONSUMPTION REDUCTION")
print("=" * 80)

explainer = DiceExplainer()

# Example: Building with high electricity consumption
print("\n" + "=" * 80)
print("EXAMPLE: Building with High Consumption")
print("=" * 80)

json_data = {
    'time': '2017-01-01T21:00:00',
    'building_id': 'Bear_education_Sharon',
    'site_id': 'Bear',
    'primaryspaceusage': 'Education',
    'sub_primaryspaceusage': 'Education',
    'sqm': 5261.7,
    'yearbuilt': 1953,
    'numberoffloors': 5,
    'occupants': 200,  # High number of occupants
    'timezone': 'US/Pacific',
    'airTemperature': 25.0,  # High temperature (needs cooling)
    'cloudCoverage': 30.0,
    'dewTemperature': 18.0,
    'windSpeed': 2.6,
    'seaLvlPressure': 1020.7,
    'precipDepth1HR': 0.0
}

# First, check current prediction
current_pred = explainer.inference.predict(json_data)
print(f"\n📊 Current predicted consumption: {current_pred:.2f} kWh")

# Set threshold (e.g., 80% of current consumption for realistic reduction)
threshold = current_pred * 0.8  # 80% of current = 20% reduction target
print(f"🎯 Target threshold: {threshold:.2f} kWh (20% reduction target)")

if current_pred > threshold:
    print(f"⚠️  Consumption exceeds threshold by {current_pred - threshold:.2f} kWh")
    print(f"\n🔍 Generating counterfactual recommendations...")
    
    # Generate recommendations
    # Use 'random' method for faster results (change to 'genetic' for better quality but slower)
    result = explainer.generate_recommendations(
        json_data=json_data,
        threshold=threshold,
        total_cfs=5,
        method='genetic'  # Use 'random' for faster results, 'genetic' for better quality
    )
    
    if result['success']:
        print(f"\n✅ Generated {result['total_counterfactuals']} recommendations")
        print(f"\n📋 Top Recommendations:")
        
        for i, rec in enumerate(result['recommendations'][:3], 1):
            print(f"\n--- Recommendation {i} ---")
            print(f"Predicted consumption: {rec['predicted_consumption']:.2f} kWh")
            print(f"Reduction: {rec['reduction']:.2f} kWh ({rec['reduction_pct']:.1f}%)")
            print(f"Below threshold: {'✅ Yes' if rec['below_threshold'] else '❌ No'}")
            
            if rec['changes']:
                print(f"\nKey changes needed:")
                for change in rec['changes'][:5]:  # Top 5 changes
                    print(f"  • {change['action']}")
                    print(f"    ({change['description']})")
    else:
        print(f"\n❌ Error: {result.get('error', 'Unknown error')}")
        if 'error_details' in result:
            print(f"\nError details:\n{result['error_details']}")
else:
    print(f"✅ Consumption is already below threshold!")

# Example 2: Get simplified recommendations
print("\n" + "=" * 80)
print("EXAMPLE 2: Simplified Recommendations")
print("=" * 80)

simple_recs = explainer.get_simple_recommendations(
    json_data=json_data,
    threshold=threshold,
    top_n=3
)

if simple_recs['success'] and simple_recs.get('top_recommendations'):
    print(f"\n📊 Current: {simple_recs['current_prediction']:.2f} kWh")
    print(f"🎯 Target: {simple_recs['threshold']} kWh")
    print(f"📉 Need to reduce: {simple_recs['needs_reduction']:.2f} kWh")
    
    print(f"\n💡 Top Recommendations:")
    for i, rec in enumerate(simple_recs['top_recommendations'], 1):
        print(f"\n{i}. Reduce to {rec['predicted_consumption']:.2f} kWh ({rec['reduction_pct']:.1f}% reduction)")
        print(f"   Key actions:")
        for change in rec['key_changes']:
            print(f"   • {change['action']}")

# Example 3: Show actionable features
print("\n" + "=" * 80)
print("EXAMPLE 3: Actionable Features")
print("=" * 80)

actionable = explainer.get_actionable_features()
print(f"\n✅ Features that can be adjusted ({len(actionable)}):")
for feat in actionable:
    info = explainer.actionable_features.get(feat, {})
    print(f"  • {feat}: {info.get('description', 'N/A')}")
    if 'direction' in info:
        print(f"    Direction: {info['direction']}")

print("\n" + "=" * 80)
print("✅ ALL EXAMPLES COMPLETED!")
print("=" * 80)
