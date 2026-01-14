#!/usr/bin/env python3
"""
Example usage of DiceExplainer v2
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src_v2.dice_explainer import DiceExplainer

# Configuration and default input are global for all functions
MODEL_DIR = "models_1578_csv"
ENCODE_PATH = "models_1578_csv/categorical_encoder.pkl"
DATA_PATH = "data_1578_csv/train_encode.csv"

default_input = {
    'sub_primaryspaceusage': 'Education',
    'industry': None,
    'subindustry': None,
    'lat': 37.871903400000036,
    'lng': -122.26072860000008,
    'timezone': 'US/Pacific',
    'heatingtype': None,
    'yearbuilt': 1953,
    'date_opened': None,
    'numberoffloors': 5,
    'occupants': None,
    'energystarscore': None,
    'eui': None,
    'site_eui': None,
    'source_eui': None,
    'leed_level': None,
    'rating': None,
    'air_temperature': 16.11578947368421,
    'cloud_coverage': None,
    'dew_temperature': 13.3,
    'precip_depth_1hr': 0.0,
    'precip_depth_6hr': None,
    'sea_lvl_pressure': 1020.7,
    'wind_direction': 80.0,
    'wind_speed': 2.131578947368421,
    'id': 66,
    'electricity': 98.25,
    'hotwater': 0.0,
    'chilledwater': 0.0,
    'steam': 0.0,
    'water': 0.0,
    'irrigation': 0.0,
    'solar': 0.0,
    'gas': 0.0,
    'time': '2017-08-31T06:00:00',
    'building_code': 'Fox_education_Wendell',
    'site_id': 'Fox',
    'sqm': 20402.2,
    'sqft': 219608,
    'primaryspaceusage': 'Education',
    'Chilledwater': 4151.7687,
    'Hotwater': 10306.8675,
}

def example1_full_recommendations(explainer: DiceExplainer, default_input: dict, threshold: float):
    print("=" * 80)
    print("EXAMPLE 1: Full Recommendations (t+1)")
    print("=" * 80)

    current_pred = threshold / 0.8  # Since threshold = current_pred * 0.8, so this reverses it

    if current_pred > threshold:
        print(f"   ⚠️  Consumption exceeds threshold by {current_pred - threshold:.2f} kWh")
        print(f"\n🔍 Generating counterfactual recommendations...")

        try:
            recommendations = explainer.generate_recommendations(
                json_data=default_input,
                threshold=threshold,
                hour_offset=3,
                total_cfs=5,
                method='random'
            )
            if recommendations['success']:
                print(f"\n✅ Generated {recommendations['total_counterfactuals']} recommendations")
                print(f"\n📋 Top Recommendations:")
                for i, rec in enumerate(recommendations['recommendations'][:3], 1):
                    print(f"\n--- Recommendation {i} ---")
                    print(f"Predicted consumption: {rec['predicted_consumption']:.2f} kWh")
                    print(f"Reduction: {rec['reduction']:.2f} kWh ({rec['reduction_pct']:.1f}%)")
                    print(f"Below threshold: {'✅ Yes' if rec['below_threshold'] else '❌ No'}")
                    if rec['changes']:
                        print(f"\nKey changes needed:")
                        for change in rec['changes'][:5]:
                            print(f"  • {change['action']}")
                            if change.get('description'):
                                print(f"    ({change['description']})")
            else:
                print(f"\n❌ Error: {recommendations.get('error', 'Unknown error')}")
                if 'error_details' in recommendations:
                    print(f"\nError details:\n{recommendations['error_details']}")
        except Exception as e:
            print(f"❌ Error generating recommendations: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"✅ Consumption is already below threshold!")

def example2_simplified_recommendations(explainer: DiceExplainer, default_input: dict, threshold: float):
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Simplified Recommendations")
    print("=" * 80)

    try:
        simple_recs = explainer.get_simple_recommendations(
            json_data=default_input,
            threshold=threshold,
            top_n=3
        )

        if simple_recs['success'] and simple_recs.get('top_recommendations'):
            print(f"\n✅ Top {len(simple_recs['top_recommendations'])} simplified recommendations:")
            for i, rec in enumerate(simple_recs['top_recommendations'], 1):
                print(f"\n--- Option {i} ---")
                print(f"Predicted: {rec['predicted_consumption']:.2f} kWh")
                print(f"Reduction: {rec['reduction']:.2f} kWh ({rec['reduction_pct']:.1f}%)")
                print("Key changes:")
                for change in rec['key_changes']:
                    print(f"  • {change['action']} (impact: {change['impact']})")
        else:
            print(f"\n⚠️  No simplified recommendations available")
    except Exception as e:
        print(f"❌ Error getting simplified recommendations: {e}")
        import traceback
        traceback.print_exc()

def example3_specific_hour_recommendations(explainer: DiceExplainer, default_input: dict, threshold: float, hour_offset: int = 5):
    print("\n" + "=" * 80)
    print(f"EXAMPLE 3: Recommendations for Specific Hour (t+{hour_offset})")
    print("=" * 80)

    try:
        hour_recs = explainer.generate_recommendations_for_hour(
            json_data=default_input,
            threshold=threshold,
            hour_offset=hour_offset,
            total_cfs=3,
            method='random'
        )

        if hour_recs['success']:
            print(f"\n✅ Recommendations for hour t+{hour_offset} ({hour_recs['time']}):")
            print(f"   Current prediction: {hour_recs['current_prediction']:.2f} kWh")
            print(f"   Threshold: {hour_recs['threshold']:.2f} kWh")

            if hour_recs.get('recommendations'):
                top_rec = hour_recs['recommendations'][0]
                print(f"\n   Top recommendation:")
                print(f"   Predicted: {top_rec['predicted_consumption']:.2f} kWh")
                print(f"   Reduction: {top_rec['reduction']:.2f} kWh ({top_rec['reduction_pct']:.1f}%)")
                print(f"   Key changes:")
                for change in top_rec['changes'][:3]:
                    print(f"     • {change['action']}")
        else:
            print(f"\n⚠️  {hour_recs.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def example4_monitor_24_hours(explainer: DiceExplainer, default_input: dict, threshold: float):
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Monitor All 24 Hours")
    print("=" * 80)

    try:
        monitoring_result = explainer.monitor_24_hours(
            json_data=default_input,
            threshold=threshold,
            total_cfs=2,
            method='random',
            only_problematic_hours=True
        )

        if monitoring_result['success']:
            print(f"\n✅ Monitoring complete!")
            print(f"   Hours monitored: {monitoring_result['hours_monitored']}")
            print(f"   Hours exceeding threshold: {monitoring_result['hours_exceeding_threshold']}")
            print(f"   Hours with recommendations: {monitoring_result['hours_with_recommendations']}")

            problematic = [h for h in monitoring_result['hourly_results'] if h['exceeds_threshold']]
            if problematic:
                print(f"\n   📊 Problematic Hours Summary:")
                for hour_info in problematic[:5]:
                    print(f"      Hour t+{hour_info['hour_offset']} ({hour_info['time']}): "
                          f"{hour_info['current_prediction']:.2f} kWh "
                          f"(exceeds by {hour_info['current_prediction'] - threshold:.2f} kWh)")
        else:
            print(f"\n⚠️  Monitoring failed")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("=" * 80)
    print("DiCE Explainer v2 - Example Usage")
    print("=" * 80)

    # Initialize explainer
    print("\n🔧 Initializing DiceExplainer...")
    try:
        explainer = DiceExplainer(
            model_dir=MODEL_DIR,
            encode_path=ENCODE_PATH,
            processed_data_path=DATA_PATH
        )
        print("✅ DiceExplainer initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing explainer: {e}")
        import traceback
        traceback.print_exc()
        return

    # Get current prediction
    print("\n📊 Getting current prediction...")
    try:
        result = explainer.forecaster(default_input)
        current_pred = result[0]['electric']  # First hour prediction
        print(f"   Current prediction (t+1): {current_pred:.2f} kWh")
    except Exception as e:
        print(f"❌ Error getting prediction: {e}")
        import traceback
        traceback.print_exc()
        return

    # Set threshold (80% of current = 20% reduction target)
    threshold = current_pred * 0.8
    print(f"   Target threshold: {threshold:.2f} kWh (20% reduction target)")

    # Run examples
    example1_full_recommendations(explainer, default_input, threshold)
    # example2_simplified_recommendations(explainer, default_input, threshold)
    # example3_specific_hour_recommendations(explainer, default_input, threshold, hour_offset=5)
    # example4_monitor_24_hours(explainer, default_input, threshold)

if __name__ == "__main__":
    main()
