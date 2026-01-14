#!/usr/bin/env python3
"""
Test DiCE với nhiều scenarios khác nhau để kiểm tra khả năng tìm recommendations thực tế
"""

import json
import sys
from pathlib import Path
import pandas as pd

# Add src to path if needed
sys.path.insert(0, str(Path(__file__).parent))

from dice_explainer import DiceExplainer

def create_test_scenarios():
    """Tạo nhiều test scenarios với các mức quá tải khác nhau"""
    
    base_data = {
        'time': '2016-01-01T21:00:00',
        'building_id': 'Bear_education_Sharon',
        'site_id': 'Bear',
        'primaryspaceusage': 'Education',
        'sub_primaryspaceusage': 'Education',
        'sqm': 5261.7,
        'yearbuilt': 1953,
        'numberoffloors': 5,
        'timezone': 'US/Pacific',
        'airTemperature': 25.0,
        'cloudCoverage': 30.0,
        'dewTemperature': 18.0,
        'windSpeed': 2.6,
        'seaLvlPressure': 1020.7,
        'precipDepth1HR': 0.0
    }
    
    scenarios = [
        {
            'name': 'Scenario 1: Quá tải nhẹ (20%)',
            'data': {**base_data, 'occupants': 200},
            'threshold': 70.0  # 20% reduction from ~87.87
        },
        {
            'name': 'Scenario 2: Quá tải vừa (30%)',
            'data': {**base_data, 'occupants': 200},
            'threshold': 60.0  # 30% reduction
        },
        {
            'name': 'Scenario 3: Quá tải nặng (43%)',
            'data': {**base_data, 'occupants': 200},
            'threshold': 50.0  # 43% reduction
        },
        {
            'name': 'Scenario 4: Quá tải rất nặng (50%)',
            'data': {**base_data, 'occupants': 200},
            'threshold': 44.0  # 50% reduction
        },
        {
            'name': 'Scenario 5: Ít người hơn, quá tải nhẹ',
            'data': {**base_data, 'occupants': 150},
            'threshold': 60.0
        },
        {
            'name': 'Scenario 6: Nhiều người, quá tải nặng',
            'data': {**base_data, 'occupants': 250},
            'threshold': 80.0
        },
        {
            'name': 'Scenario 7: Nhiệt độ cao, quá tải',
            'data': {**base_data, 'occupants': 200, 'airTemperature': 30.0},
            'threshold': 70.0
        },
        {
            'name': 'Scenario 8: Giờ cao điểm, quá tải',
            'data': {**base_data, 'occupants': 200, 'time': '2016-07-15T14:00:00'},  # Giữa trưa mùa hè
            'threshold': 70.0
        }
    ]
    
    return scenarios

def test_scenario(explainer, scenario, verbose=True):
    """Test một scenario và trả về kết quả"""
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"{scenario['name']}")
        print(f"{'='*80}")
    
    # Predict
    prediction = explainer.inference.predict(scenario['data'], include_lag=True)
    threshold = scenario['threshold']
    
    if verbose:
        print(f"\n📊 Dự đoán: {prediction:.2f} kWh")
        print(f"🎯 Threshold: {threshold:.2f} kWh")
    
    if prediction <= threshold:
        if verbose:
            print(f"✅ Không vượt ngưỡng - Không cần điều chỉnh")
        return {
            'scenario': scenario['name'],
            'prediction': prediction,
            'threshold': threshold,
            'exceeds': False,
            'realistic_found': False,
            'recommendations_count': 0
        }
    
    excess = prediction - threshold
    reduction_pct = (excess / prediction) * 100
    
    if verbose:
        print(f"⚠️ Vượt ngưỡng: {excess:.2f} kWh ({reduction_pct:.1f}%)")
        print(f"🔍 Đang tạo gợi ý...")
    
    # Generate recommendations
    result = explainer.generate_recommendations(
        json_data=scenario['data'],
        threshold=threshold,
        total_cfs=10,  # Generate more to have better chance
        method='random'
    )
    
    if not result['success']:
        if verbose:
            print(f"❌ Lỗi: {result.get('error', 'Unknown')}")
        return {
            'scenario': scenario['name'],
            'prediction': prediction,
            'threshold': threshold,
            'exceeds': True,
            'realistic_found': False,
            'recommendations_count': 0,
            'error': result.get('error')
        }
    
    recommendations = result.get('recommendations', [])
    
    # Filter realistic recommendations (80-100% of threshold)
    realistic_min = threshold * 0.8
    realistic_recs = [
        rec for rec in recommendations
        if rec['predicted_consumption'] >= realistic_min
        and rec['predicted_consumption'] <= threshold
    ]
    
    # Sort by proximity to threshold
    realistic_recs.sort(key=lambda r: abs(r['predicted_consumption'] - threshold))
    
    if verbose:
        print(f"\n📋 Tổng số recommendations: {len(recommendations)}")
        print(f"✅ Recommendations thực tế (80-100% threshold): {len(realistic_recs)}")
        
        if realistic_recs:
            print(f"\n💡 Top 3 recommendations thực tế:")
            for i, rec in enumerate(realistic_recs[:3], 1):
                print(f"   {i}. {rec['predicted_consumption']:.2f} kWh "
                      f"(giảm {rec['reduction']:.2f} kWh, {rec['reduction_pct']:.1f}%)")
                if rec.get('changes'):
                    for change in rec['changes'][:2]:  # Top 2 changes
                        if change['feature'] == 'occupants':
                            print(f"      • {change['action']}")
        else:
            print(f"\n⚠️ Không tìm thấy recommendations thực tế")
            if recommendations:
                print(f"   Tất cả đều quá cực đoan:")
                for i, rec in enumerate(recommendations[:3], 1):
                    print(f"   {i}. {rec['predicted_consumption']:.2f} kWh "
                          f"(giảm {rec['reduction_pct']:.1f}%)")
    
    return {
        'scenario': scenario['name'],
        'prediction': prediction,
        'threshold': threshold,
        'exceeds': True,
        'excess': excess,
        'reduction_needed_pct': reduction_pct,
        'realistic_found': len(realistic_recs) > 0,
        'realistic_count': len(realistic_recs),
        'recommendations_count': len(recommendations),
        'best_realistic': realistic_recs[0] if realistic_recs else None
    }

def main():
    print("="*80)
    print("TEST DiCE VỚI NHIỀU SCENARIOS")
    print("="*80)
    
    # Initialize explainer
    print("\n🔧 Đang khởi tạo DiCE Explainer...")
    explainer = DiceExplainer()
    
    # Get scenarios
    scenarios = create_test_scenarios()
    
    print(f"\n📋 Tổng số scenarios: {len(scenarios)}")
    
    # Test all scenarios
    results = []
    for scenario in scenarios:
        result = test_scenario(explainer, scenario, verbose=True)
        results.append(result)
    
    # Summary
    print("\n" + "="*80)
    print("TÓM TẮT KẾT QUẢ")
    print("="*80)
    
    df_results = pd.DataFrame(results)
    
    # Filter only scenarios that exceed threshold
    exceeded = df_results[df_results['exceeds'] == True]
    
    print(f"\n📊 Tổng số scenarios: {len(results)}")
    print(f"⚠️ Scenarios vượt ngưỡng: {len(exceeded)}")
    
    if len(exceeded) > 0:
        print(f"\n📈 Tỷ lệ tìm được recommendations thực tế:")
        realistic_found = exceeded['realistic_found'].sum()
        print(f"   ✅ Có recommendations thực tế: {realistic_found}/{len(exceeded)} ({realistic_found/len(exceeded)*100:.1f}%)")
        print(f"   ❌ Không có recommendations thực tế: {len(exceeded) - realistic_found}/{len(exceeded)} ({(len(exceeded)-realistic_found)/len(exceeded)*100:.1f}%)")
        
        print(f"\n📋 Chi tiết:")
        for _, row in exceeded.iterrows():
            status = "✅" if row['realistic_found'] else "❌"
            print(f"   {status} {row['scenario']}")
            print(f"      Prediction: {row['prediction']:.2f} kWh, Threshold: {row['threshold']:.2f} kWh")
            print(f"      Cần giảm: {row['reduction_needed_pct']:.1f}%")
            if row['realistic_found']:
                best = row['best_realistic']
                print(f"      ✅ Tìm được: {best['predicted_consumption']:.2f} kWh (giảm {best['reduction_pct']:.1f}%)")
            else:
                print(f"      ❌ Không tìm được recommendations thực tế")
    
    # Analysis
    print(f"\n" + "="*80)
    print("PHÂN TÍCH")
    print("="*80)
    
    if len(exceeded) > 0:
        avg_reduction_needed = exceeded['reduction_needed_pct'].mean()
        print(f"\n📊 Mức giảm trung bình cần thiết: {avg_reduction_needed:.1f}%")
        
        # Check if there's a pattern
        realistic_scenarios = exceeded[exceeded['realistic_found'] == True]
        unrealistic_scenarios = exceeded[exceeded['realistic_found'] == False]
        
        if len(realistic_scenarios) > 0:
            avg_realistic = realistic_scenarios['reduction_needed_pct'].mean()
            print(f"   ✅ Scenarios có recommendations: giảm trung bình {avg_realistic:.1f}%")
        
        if len(unrealistic_scenarios) > 0:
            avg_unrealistic = unrealistic_scenarios['reduction_needed_pct'].mean()
            print(f"   ❌ Scenarios không có recommendations: giảm trung bình {avg_unrealistic:.1f}%")
            
            if len(realistic_scenarios) > 0:
                if avg_unrealistic > avg_realistic:
                    print(f"\n💡 Nhận xét: DiCE khó tìm recommendations khi cần giảm > {avg_unrealistic:.1f}%")
                    print(f"   Có thể cần hướng giải quyết khác cho các trường hợp này")
            else:
                print(f"\n💡 Nhận xét: DiCE KHÔNG TÌM ĐƯỢC recommendations thực tế cho BẤT KỲ scenario nào!")
                print(f"   Có thể do:")
                print(f"   - Model không nhạy cảm với thay đổi của occupants")
                print(f"   - Có vấn đề với cách predict counterfactual (lag features)")
                print(f"   - Cần kiểm tra model behavior với test_model_behavior.py")
    
    # Save results
    output_file = "output/dice_test_results.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\n💾 Đã lưu kết quả vào: {output_file}")
    
    return df_results

if __name__ == "__main__":
    results = main()
