#!/usr/bin/env python3
"""
Simple Recommender - Hướng giải quyết thay thế cho DiCE
Sử dụng optimization đơn giản để tìm recommendations thực tế
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

try:
    from .inference import ElectricityConsumptionInference
except ImportError:
    from inference import ElectricityConsumptionInference


@dataclass
class Recommendation:
    """Một recommendation để giảm tiêu thụ điện"""
    predicted_consumption: float
    reduction: float
    reduction_pct: float
    changes: List[Dict]
    below_threshold: bool


class SimpleRecommender:
    """
    Simple recommender sử dụng grid search và optimization
    để tìm recommendations thực tế gần threshold
    """
    
    def __init__(self, inference: ElectricityConsumptionInference):
        self.inference = inference
        
        # Define actionable features and their ranges
        self.actionable_features = {
            'occupants': {
                'min_pct': 0.1,  # Minimum 10% of original
                'max_pct': 1.0,  # Maximum 100% (no change)
                'step_pct': 0.05,  # Step 5%
                'description': 'Number of occupants'
            }
        }
    
    def generate_recommendations(self,
                                 json_data: Dict,
                                 threshold: float,
                                 max_recommendations: int = 5) -> Dict:
        """
        Generate recommendations using simple optimization.
        
        Args:
            json_data: Original building data
            threshold: Target consumption threshold
            max_recommendations: Maximum number of recommendations
            
        Returns:
            Dictionary with recommendations
        """
        # Predict current consumption
        current_prediction = self.inference.predict(json_data, include_lag=True)
        
        if current_prediction <= threshold:
            return {
                'success': True,
                'current_prediction': float(current_prediction),
                'threshold': float(threshold),
                'below_threshold': True,
                'message': f'Current consumption ({current_prediction:.2f} kWh) is already below threshold',
                'recommendations': []
            }
        
        recommendations = []
        
        # Try different combinations of actionable features
        for feat, config in self.actionable_features.items():
            if feat not in json_data:
                continue
            
            current_value = float(json_data[feat])
            min_value = current_value * config['min_pct']
            max_value = current_value * config['max_pct']
            step = current_value * config['step_pct']
            
            # Generate candidate values
            candidates = np.arange(max_value, min_value - step, -step)
            
            for candidate_value in candidates:
                # Create modified data
                modified_data = json_data.copy()
                modified_data[feat] = float(candidate_value)
                
                # Predict
                try:
                    # Use include_lag=False for counterfactual (like DiCE does)
                    new_prediction = self.inference.predict(modified_data, include_lag=False)
                    
                    # Debug: print first few to see what's happening
                    if len(recommendations) == 0 and candidate_value <= current_value * 0.7:
                        print(f"      Debug: occupants={candidate_value:.0f} -> prediction={new_prediction:.2f} kWh")
                    
                    # Check if below threshold and realistic
                    if new_prediction <= threshold and new_prediction >= threshold * 0.8:
                        reduction = current_prediction - new_prediction
                        reduction_pct = (reduction / current_prediction) * 100
                        
                        # Calculate change
                        change = candidate_value - current_value
                        change_pct = (change / current_value) * 100 if current_value != 0 else 0
                        
                        changes = [{
                            'feature': feat,
                            'description': config['description'],
                            'original_value': current_value,
                            'suggested_value': candidate_value,
                            'change': change,
                            'change_pct': change_pct,
                            'action': self._get_action_description(feat, current_value, candidate_value)
                        }]
                        
                        recommendations.append(Recommendation(
                            predicted_consumption=float(new_prediction),
                            reduction=float(reduction),
                            reduction_pct=float(reduction_pct),
                            changes=changes,
                            below_threshold=True
                        ))
                        
                        # Stop if we have enough recommendations
                        if len(recommendations) >= max_recommendations:
                            break
                            
                except Exception as e:
                    continue
            
            # Stop if we have enough recommendations
            if len(recommendations) >= max_recommendations:
                break
        
        # Sort by proximity to threshold (closest first)
        recommendations.sort(key=lambda r: abs(r.predicted_consumption - threshold))
        
        # Convert to dict format
        recommendations_dict = []
        for rec in recommendations:
            recommendations_dict.append({
                'predicted_consumption': rec.predicted_consumption,
                'reduction': rec.reduction,
                'reduction_pct': rec.reduction_pct,
                'below_threshold': rec.below_threshold,
                'changes': rec.changes
            })
        
        return {
            'success': True,
            'current_prediction': float(current_prediction),
            'threshold': float(threshold),
            'below_threshold': False,
            'needs_reduction': float(current_prediction - threshold),
            'total_recommendations': len(recommendations_dict),
            'recommendations': recommendations_dict
        }
    
    def _get_action_description(self, feat: str, orig_val: float, new_val: float) -> str:
        """Get human-readable action description"""
        change = new_val - orig_val
        change_pct = (change / orig_val * 100) if orig_val != 0 else 0
        
        if feat == 'occupants':
            if change < 0:
                return f"Reduce occupants by {abs(change):.0f} ({abs(change_pct):.1f}%)"
            else:
                return f"Increase occupants by {change:.0f} ({change_pct:.1f}%)"
        else:
            if change < 0:
                return f"Reduce {feat} by {abs(change):.2f} ({abs(change_pct):.1f}%)"
            else:
                return f"Increase {feat} by {change:.2f} ({change_pct:.1f}%)"


def main():
    """Example usage"""
    from dice_explainer import DiceExplainer
    
    print("="*80)
    print("SIMPLE RECOMMENDER - HƯỚNG GIẢI QUYẾT THAY THẾ")
    print("="*80)
    
    # Initialize (reuse inference from DiceExplainer)
    print("\n🔧 Đang khởi tạo...")
    dice_explainer = DiceExplainer()
    recommender = SimpleRecommender(dice_explainer.inference)
    
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
    
    # Predict
    prediction = recommender.inference.predict(json_data)
    threshold = 50.0
    
    print(f"\n📊 Dự đoán: {prediction:.2f} kWh")
    print(f"🎯 Threshold: {threshold:.2f} kWh")
    
    if prediction > threshold:
        print(f"⚠️ Vượt ngưỡng: {prediction - threshold:.2f} kWh")
        print(f"\n🔍 Đang tạo recommendations...")
        
        result = recommender.generate_recommendations(
            json_data=json_data,
            threshold=threshold,
            max_recommendations=5
        )
        
        if result['success']:
            print(f"\n✅ Tìm được {result['total_recommendations']} recommendations:")
            
            for i, rec in enumerate(result['recommendations'], 1):
                print(f"\n   Recommendation {i}:")
                print(f"   • Tiêu thụ sau điều chỉnh: {rec['predicted_consumption']:.2f} kWh")
                print(f"   • Giảm: {rec['reduction']:.2f} kWh ({rec['reduction_pct']:.1f}%)")
                print(f"   • Cần điều chỉnh:")
                for change in rec['changes']:
                    print(f"     - {change['action']}")
        else:
            print(f"\n❌ Không tìm được recommendations")
    else:
        print(f"✅ Không vượt ngưỡng")

if __name__ == "__main__":
    main()
