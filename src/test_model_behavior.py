#!/usr/bin/env python3
"""
Test model behavior để kiểm tra:
1. Model có phản ứng đúng với thay đổi của features không?
2. Feature importance - features nào quan trọng nhất?
3. Có vấn đề với lag features khi predict counterfactual không?
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from inference import ElectricityConsumptionInference
from dice_explainer import DiceExplainer

def test_feature_importance(explainer):
    """Kiểm tra feature importance của model"""
    print("="*80)
    print("1. FEATURE IMPORTANCE")
    print("="*80)
    
    model = explainer.inference.model.model  # Get underlying XGBoost model
    
    if hasattr(model, 'feature_importances_'):
        # Get feature names - try different ways
        feature_names = None
        if hasattr(model, 'feature_names_in_'):
            feature_names = model.feature_names_in_
        elif hasattr(explainer.inference.model, 'model') and hasattr(explainer.inference.model.model, 'feature_names_in_'):
            feature_names = explainer.inference.model.model.feature_names_in_
        
        # Fallback: use feature names from features_info
        if feature_names is None:
            with open(explainer.inference.features_info_path, 'r') as f:
                import json
                features_info = json.load(f)
            all_features = (features_info['continuous_features'] + 
                          features_info['time_features'] + 
                          features_info['lag_features'] + 
                          features_info['categorical_features'])
            # Match length with importances
            num_features = len(model.feature_importances_)
            feature_names = all_features[:num_features] if len(all_features) >= num_features else all_features
        
        importances = model.feature_importances_
        
        # Create DataFrame
        df_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print(f"\n📊 Top 20 features quan trọng nhất:")
        print(df_importance.head(20).to_string(index=False))
        
        # Check if occupants is important
        if 'occupants' in df_importance['feature'].values:
            occ_importance = df_importance[df_importance['feature'] == 'occupants']['importance'].values[0]
            rank = df_importance[df_importance['feature'] == 'occupants'].index[0] + 1
            print(f"\n🔍 'occupants' importance: {occ_importance:.6f} (rank: {rank}/{len(df_importance)})")
            
            if occ_importance < 0.01:
                print(f"   ⚠️ WARNING: 'occupants' có importance rất thấp!")
                print(f"   Model có thể không học được mối quan hệ giữa occupants và consumption")
        else:
            print(f"\n⚠️ WARNING: 'occupants' không có trong feature list!")
        
        # Check lag features importance
        lag_features = [f for f in df_importance['feature'] if 'lag' in f.lower() or 'rolling' in f.lower()]
        if lag_features:
            lag_importance = df_importance[df_importance['feature'].isin(lag_features)]['importance'].sum()
            print(f"\n📈 Lag features tổng importance: {lag_importance:.6f}")
            print(f"   Top lag features:")
            for feat in lag_features[:5]:
                imp = df_importance[df_importance['feature'] == feat]['importance'].values[0]
                print(f"     • {feat}: {imp:.6f}")
    
    return df_importance

def test_occupants_sensitivity(inference, json_data):
    """Test xem model có phản ứng đúng với thay đổi của occupants không"""
    print("\n" + "="*80)
    print("2. TEST MODEL SENSITIVITY TO OCCUPANTS")
    print("="*80)
    
    base_prediction = inference.predict(json_data, include_lag=True)
    print(f"\n📊 Prediction với occupants={json_data['occupants']}: {base_prediction:.2f} kWh")
    
    # Test different occupants values
    occupants_values = [200, 180, 160, 140, 120, 100, 80, 60, 40, 20]
    predictions = []
    
    print(f"\n🔍 Testing với các giá trị occupants khác nhau:")
    print(f"{'Occupants':<12} {'Prediction':<15} {'Change':<15} {'Change %':<15}")
    print("-" * 60)
    
    for occ in occupants_values:
        test_data = json_data.copy()
        test_data['occupants'] = occ
        
        # Predict WITHOUT lag (like in counterfactual)
        pred_no_lag = inference.predict(test_data, include_lag=False)
        
        # Predict WITH lag (for comparison)
        pred_with_lag = inference.predict(test_data, include_lag=True)
        
        change = pred_no_lag - base_prediction
        change_pct = (change / base_prediction * 100) if base_prediction != 0 else 0
        
        predictions.append({
            'occupants': occ,
            'prediction_no_lag': pred_no_lag,
            'prediction_with_lag': pred_with_lag,
            'change': change,
            'change_pct': change_pct
        })
        
        print(f"{occ:<12} {pred_no_lag:<15.2f} {change:<15.2f} {change_pct:<15.1f}%")
    
    df_sensitivity = pd.DataFrame(predictions)
    
    # Check if model is sensitive
    max_change = abs(df_sensitivity['change_pct'].max())
    min_change = abs(df_sensitivity['change_pct'].min())
    
    print(f"\n📈 Phân tích:")
    print(f"   • Thay đổi lớn nhất: {max_change:.1f}%")
    print(f"   • Thay đổi nhỏ nhất: {min_change:.1f}%")
    
    if max_change < 5:
        print(f"   ⚠️ WARNING: Model không nhạy cảm với thay đổi của occupants!")
        print(f"   Giảm 50% occupants chỉ thay đổi prediction <5%")
    elif max_change < 20:
        print(f"   ⚠️ CAUTION: Model ít nhạy cảm với thay đổi của occupants")
        print(f"   Cần giảm nhiều occupants để có tác động đáng kể")
    else:
        print(f"   ✅ Model nhạy cảm với thay đổi của occupants")
    
    # Check if we can reach threshold
    threshold = 50.0
    print(f"\n🎯 Kiểm tra khả năng đạt threshold={threshold} kWh:")
    below_threshold = df_sensitivity[df_sensitivity['prediction_no_lag'] <= threshold]
    
    if len(below_threshold) > 0:
        min_occ = below_threshold['occupants'].min()
        pred_at_min = below_threshold[below_threshold['occupants'] == min_occ]['prediction_no_lag'].values[0]
        print(f"   ✅ Có thể đạt threshold bằng cách giảm occupants xuống {min_occ}")
        print(f"      Prediction tại {min_occ} occupants: {pred_at_min:.2f} kWh")
    else:
        print(f"   ❌ KHÔNG THỂ đạt threshold ngay cả khi giảm occupants xuống 20")
        print(f"      Prediction thấp nhất: {df_sensitivity['prediction_no_lag'].min():.2f} kWh")
        print(f"   ⚠️ Có thể do:")
        print(f"      - Lag features vẫn có giá trị cao")
        print(f"      - Model không học được mối quan hệ đúng")
        print(f"      - Các features khác (như sqm, weather) đang chi phối prediction")
    
    return df_sensitivity

def test_lag_features_impact(inference, json_data):
    """Test impact của lag features"""
    print("\n" + "="*80)
    print("3. TEST LAG FEATURES IMPACT")
    print("="*80)
    
    # Predict with lag
    pred_with_lag = inference.predict(json_data, include_lag=True)
    
    # Predict without lag
    pred_no_lag = inference.predict(json_data, include_lag=False)
    
    print(f"\n📊 So sánh prediction:")
    print(f"   • Với lag features: {pred_with_lag:.2f} kWh")
    print(f"   • Không có lag features: {pred_no_lag:.2f} kWh")
    print(f"   • Chênh lệch: {abs(pred_with_lag - pred_no_lag):.2f} kWh ({abs(pred_with_lag - pred_no_lag)/pred_with_lag*100:.1f}%)")
    
    if abs(pred_with_lag - pred_no_lag) > pred_with_lag * 0.3:
        print(f"\n   ⚠️ WARNING: Lag features có tác động rất lớn!")
        print(f"   Khi predict counterfactual với include_lag=False,")
        print(f"   lag features vẫn có giá trị từ query instance gốc")
        print(f"   Điều này có thể làm prediction không chính xác")
    
    # Check lag feature values
    X = inference._preprocess_input(json_data, include_lag=True)
    lag_features = [col for col in X.columns if 'lag' in col.lower() or 'rolling' in col.lower()]
    
    if lag_features:
        print(f"\n📈 Giá trị lag features hiện tại:")
        for feat in lag_features[:5]:
            val = X[feat].iloc[0] if feat in X.columns else 0
            print(f"   • {feat}: {val:.2f}")
    
    return pred_with_lag, pred_no_lag

def test_counterfactual_prediction(inference, json_data):
    """Test cách predict counterfactual"""
    print("\n" + "="*80)
    print("4. TEST COUNTERFACTUAL PREDICTION")
    print("="*80)
    
    # Original prediction
    original_pred = inference.predict(json_data, include_lag=True)
    print(f"\n📊 Original prediction: {original_pred:.2f} kWh")
    
    # Test counterfactual: reduce occupants
    cf_data = json_data.copy()
    cf_data['occupants'] = 100  # Giảm 50%
    
    # Predict with include_lag=False (như trong DiCE)
    cf_pred_no_lag = inference.predict(cf_data, include_lag=False)
    
    # Predict with include_lag=True (để so sánh)
    cf_pred_with_lag = inference.predict(cf_data, include_lag=True)
    
    print(f"\n🔍 Counterfactual: occupants = 100 (giảm 50%)")
    print(f"   • Prediction (no lag): {cf_pred_no_lag:.2f} kWh")
    print(f"   • Prediction (with lag): {cf_pred_with_lag:.2f} kWh")
    print(f"   • Reduction (no lag): {original_pred - cf_pred_no_lag:.2f} kWh")
    print(f"   • Reduction (with lag): {original_pred - cf_pred_with_lag:.2f} kWh")
    
    # Check if we can reach threshold
    threshold = 50.0
    print(f"\n🎯 Threshold: {threshold} kWh")
    
    if cf_pred_no_lag <= threshold:
        print(f"   ✅ Có thể đạt threshold với prediction no lag")
    else:
        print(f"   ❌ Không thể đạt threshold với prediction no lag")
        print(f"      Cần giảm thêm: {cf_pred_no_lag - threshold:.2f} kWh")
    
    if cf_pred_with_lag <= threshold:
        print(f"   ✅ Có thể đạt threshold với prediction with lag")
    else:
        print(f"   ❌ Không thể đạt threshold với prediction with lag")
        print(f"      Cần giảm thêm: {cf_pred_with_lag - threshold:.2f} kWh")
    
    return cf_pred_no_lag, cf_pred_with_lag

def main():
    print("="*80)
    print("TEST MODEL BEHAVIOR - KIỂM TRA VẤN ĐỀ")
    print("="*80)
    
    # Initialize
    print("\n🔧 Đang khởi tạo...")
    explainer = DiceExplainer()
    inference = explainer.inference
    
    # Test data
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
    
    # Run tests
    df_importance = test_feature_importance(explainer)
    df_sensitivity = test_occupants_sensitivity(inference, json_data)
    pred_with_lag, pred_no_lag = test_lag_features_impact(inference, json_data)
    cf_pred_no_lag, cf_pred_with_lag = test_counterfactual_prediction(inference, json_data)
    
    # Summary
    print("\n" + "="*80)
    print("TÓM TẮT VÀ KẾT LUẬN")
    print("="*80)
    
    print(f"\n📋 Các vấn đề có thể:")
    
    # Check 1: Feature importance
    if 'occupants' in df_importance['feature'].values:
        occ_imp = df_importance[df_importance['feature'] == 'occupants']['importance'].values[0]
        if occ_imp < 0.01:
            print(f"\n1. ⚠️ 'occupants' có importance thấp ({occ_imp:.6f})")
            print(f"   → Model không học được mối quan hệ giữa occupants và consumption")
            print(f"   → Giải pháp: Kiểm tra lại data preprocessing hoặc feature engineering")
    
    # Check 2: Model sensitivity
    max_change = abs(df_sensitivity['change_pct'].max())
    if max_change < 20:
        print(f"\n2. ⚠️ Model ít nhạy cảm với thay đổi của occupants (max change: {max_change:.1f}%)")
        print(f"   → Giảm 50% occupants chỉ thay đổi prediction <20%")
        print(f"   → Giải pháp: Cần giảm rất nhiều occupants để có tác động")
    
    # Check 3: Lag features impact
    lag_impact = abs(pred_with_lag - pred_no_lag) / pred_with_lag * 100
    if lag_impact > 30:
        print(f"\n3. ⚠️ Lag features có tác động rất lớn ({lag_impact:.1f}%)")
        print(f"   → Khi predict counterfactual với include_lag=False,")
        print(f"     lag features vẫn có giá trị từ instance gốc")
        print(f"   → Giải pháp: Cần xử lý lag features đúng cách khi predict counterfactual")
    
    # Check 4: Can reach threshold?
    threshold = 50.0
    min_pred = df_sensitivity['prediction_no_lag'].min()
    if min_pred > threshold:
        print(f"\n4. ⚠️ KHÔNG THỂ đạt threshold {threshold} kWh")
        print(f"   → Prediction thấp nhất: {min_pred:.2f} kWh")
        print(f"   → Ngay cả khi giảm occupants xuống 20")
        print(f"   → Giải pháp: Cần điều chỉnh threshold hoặc tìm features khác")
    
    print(f"\n💡 Khuyến nghị:")
    print(f"   1. Kiểm tra lại feature importance - đảm bảo occupants có tác động")
    print(f"   2. Xử lý lag features đúng cách khi predict counterfactual")
    print(f"   3. Cân nhắc sử dụng SimpleRecommender thay vì DiCE")
    print(f"   4. Kiểm tra lại data preprocessing và feature engineering")

if __name__ == "__main__":
    main()
