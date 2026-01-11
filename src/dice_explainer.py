#!/usr/bin/env python3
"""
DiCE (Diverse Counterfactual Explanations) Integration
Provides counterfactual explanations to reduce electricity consumption below threshold
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import warnings

try:
    import dice_ml
    from dice_ml import Dice
    DICE_AVAILABLE = True
except ImportError:
    DICE_AVAILABLE = False
    warnings.warn("dice-ml not installed. Install with: pip install dice-ml")

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("matplotlib not installed. Install with: pip install matplotlib")

try:
    from .inference import ElectricityConsumptionInference, XGBoostWrapper
except ImportError:
    # Fallback for direct execution
    from inference import ElectricityConsumptionInference, XGBoostWrapper

warnings.filterwarnings('ignore')


class DiceExplainer:
    """
    DiCE Explainer for electricity consumption reduction.
    
    Provides counterfactual explanations to suggest feature adjustments
    that would reduce electricity consumption below a threshold.
    """
    
    def __init__(self,
                 inference: Optional[ElectricityConsumptionInference] = None,
                 processed_data_path: str = "output/processed_data.parquet",
                 model_path: str = "output/models/xgboost_wrapped_dice.pkl",
                 encoders_path: str = "output/models/label_encoders_dice.pkl",
                 features_info_path: str = "output/features_info.json"):
        """
        Initialize DiCE Explainer.
        
        Args:
            inference: Optional ElectricityConsumptionInference instance
            processed_data_path: Path to processed data for DiCE
            model_path: Path to wrapped model
            encoders_path: Path to label encoders
            features_info_path: Path to features info
        """
        if not DICE_AVAILABLE:
            raise ImportError("dice-ml is required. Install with: pip install dice-ml")
        
        # Initialize inference if not provided
        if inference is None:
            self.inference = ElectricityConsumptionInference(
                model_path=model_path,
                encoders_path=encoders_path,
                features_info_path=features_info_path,
                processed_data_path=processed_data_path
            )
        else:
            self.inference = inference
        
        self.processed_data_path = Path(processed_data_path)
        self.model_path = Path(model_path)
        self.encoders_path = Path(encoders_path)
        self.features_info_path = Path(features_info_path)
        
        # Load features info
        with open(self.features_info_path, 'r') as f:
            self.features_info = json.load(f)
        
        # Define actionable features (features that can be adjusted)
        self.actionable_features = self._define_actionable_features()
        
        # Initialize DiCE components
        self.dice_data = None
        self.dice_model = None
        self.explainer = None
        
        # Load and setup DiCE
        self._setup_dice()
    
    def _define_actionable_features(self) -> Dict[str, Dict]:
        """
        Define which features can be adjusted and their constraints.
        
        Returns:
            Dictionary mapping feature names to their constraints
        """
        return {
            # Building features that can be adjusted
            'sqm': {
                'adjustable': True,
                'direction': 'decrease',  # Can only decrease (reduce space)
                'min_change_pct': 0.05,  # Minimum 5% change
                'max_change_pct': 0.30,  # Maximum 30% change
                'description': 'Building area (square meters)'
            },
            'occupants': {
                'adjustable': True,
                'direction': 'decrease',
                'min_change_pct': 0.10,
                'max_change_pct': 0.50,
                'description': 'Number of occupants'
            },
            'numberoffloors': {
                'adjustable': False,  # Cannot change building structure
                'description': 'Number of floors (fixed)'
            },
            'yearbuilt': {
                'adjustable': False,
                'description': 'Year built (fixed)'
            },
            # Weather/operational features
            'airTemperature': {
                'adjustable': True,
                'direction': 'both',  # Can adjust HVAC
                'min_change': 1.0,  # Minimum 1°C change
                'max_change': 5.0,  # Maximum 5°C change
                'description': 'Air temperature (can adjust HVAC)'
            },
            # Time features (can adjust schedule)
            'hour': {
                'adjustable': True,
                'direction': 'both',
                'description': 'Hour of day (can adjust schedule)'
            },
            'is_weekend': {
                'adjustable': True,
                'direction': 'both',
                'description': 'Weekend flag (can adjust schedule)'
            },
            # Fixed features
            'primaryspaceusage': {
                'adjustable': False,
                'description': 'Primary space usage (fixed)'
            },
            'site_id': {
                'adjustable': False,
                'description': 'Site ID (fixed)'
            },
            'timezone': {
                'adjustable': False,
                'description': 'Timezone (fixed)'
            },
            # Lag features (cannot change past)
            'electricity_lag1': {
                'adjustable': False,
                'description': 'Lag 1 hour (depends on past)'
            },
            'electricity_lag24': {
                'adjustable': False,
                'description': 'Lag 24 hours (depends on past)'
            },
            'electricity_lag168': {
                'adjustable': False,
                'description': 'Lag 168 hours (depends on past)'
            },
            'electricity_rolling_mean_24h': {
                'adjustable': False,
                'description': 'Rolling mean 24h (depends on past)'
            },
            'electricity_rolling_std_24h': {
                'adjustable': False,
                'description': 'Rolling std 24h (depends on past)'
            },
            'electricity_rolling_mean_7d': {
                'adjustable': False,
                'description': 'Rolling mean 7d (depends on past)'
            }
        }
    
    def _setup_dice(self):
        """Setup DiCE data and model objects."""
        print("🔧 Setting up DiCE...")
        
        # Load processed data
        df = pd.read_parquet(self.processed_data_path)
        
        # Get features used by model
        all_features = (
            self.features_info['continuous_features'] + 
            self.features_info['time_features'] + 
            self.features_info['lag_features']
        )
        categorical_features = self.features_info['categorical_features']
        
        # Filter to features that exist in data
        all_features = [f for f in all_features if f in df.columns]
        categorical_features = [f for f in categorical_features if f in df.columns]
        
        # Create a sample dataset for DiCE (use a smaller subset for stability)
        sample_size = min(5000, len(df))
        df_sample = df.sample(n=sample_size, random_state=42).copy()
        
        # Drop any rows with NaN in critical columns
        df_sample = df_sample.dropna(subset=['electricity_consumption'])
        
        # Prepare features for DiCE (need to decode categoricals back to strings)
        # Only include features that exist in the dataframe
        available_features = [f for f in all_features if f in df_sample.columns]
        available_categorical = [f for f in categorical_features if f in df_sample.columns]
        
        df_for_dice = df_sample[available_features + available_categorical + ['electricity_consumption']].copy()
        
        # Decode categorical features back to original strings for DiCE
        # Check if they're already strings or need decoding
        with open(self.encoders_path, 'rb') as f:
            label_encoders = pickle.load(f)
        
        for col in available_categorical:
            if col not in df_for_dice.columns or col not in label_encoders:
                continue
                
            le = label_encoders[col]
            col_series = df_for_dice[col]
            
            # Check if column is already decoded (strings in encoder classes)
            sample_vals = col_series.dropna()
            if len(sample_vals) > 0:
                sample_val = sample_vals.iloc[0]
                # If it's already a string and in the encoder classes, it's already decoded
                if isinstance(sample_val, str) and sample_val in le.classes_:
                    # Already decoded, skip
                    continue
            
            # Decode back to original strings
            def decode_value(x):
                try:
                    # Handle NaN
                    if pd.isna(x):
                        return 'Unknown'
                    
                    # If already a string and valid, return as is
                    if isinstance(x, str):
                        if x in le.classes_:
                            return x
                        # Try to see if it's a string representation of an int
                        try:
                            int_val = int(float(x))
                            if int_val in le.classes_:
                                return le.inverse_transform([int_val])[0]
                        except (ValueError, TypeError):
                            pass
                        return 'Unknown'
                    
                    # Try to convert to int if it's numeric
                    if isinstance(x, (int, float, np.integer, np.floating)):
                        int_val = int(x)
                        if int_val in le.classes_:
                            return le.inverse_transform([int_val])[0]
                    
                    return 'Unknown'
                except (ValueError, TypeError, KeyError, AttributeError):
                    return 'Unknown'
            
            df_for_dice[col] = col_series.apply(decode_value)
        
        # Create DiCE Data object
        # Separate continuous and categorical features
        continuous_features = []
        for f in available_features:
            if f not in available_categorical:
                # Check if it's actually continuous (not a time feature that should be treated as categorical)
                if f not in ['hour', 'day_of_week', 'month', 'year', 'is_weekend']:
                    continuous_features.append(f)
        
        # Ensure all features in continuous_features exist in df_for_dice
        continuous_features = [f for f in continuous_features if f in df_for_dice.columns]
        dice_categorical_features = [f for f in available_categorical if f in df_for_dice.columns]
        
        # Remove outcome from feature lists
        if 'electricity_consumption' in continuous_features:
            continuous_features.remove('electricity_consumption')
        if 'electricity_consumption' in dice_categorical_features:
            dice_categorical_features.remove('electricity_consumption')
        
        # Clean dataframe: remove duplicates, ensure proper types
        df_for_dice = df_for_dice.loc[:, ~df_for_dice.columns.duplicated()].copy()
        
        # Ensure categorical features are object type (string)
        for col in dice_categorical_features:
            if col in df_for_dice.columns:
                df_for_dice[col] = df_for_dice[col].astype(str)
        
        # Remove any columns that might cause issues
        # Ensure outcome column exists and is numeric
        if 'electricity_consumption' in df_for_dice.columns:
            df_for_dice['electricity_consumption'] = pd.to_numeric(
                df_for_dice['electricity_consumption'], errors='coerce'
            ).fillna(0)
        
        # Filter continuous features to only numeric columns
        numeric_continuous = []
        for f in continuous_features:
            if f in df_for_dice.columns:
                if pd.api.types.is_numeric_dtype(df_for_dice[f]):
                    numeric_continuous.append(f)
        
        continuous_features = numeric_continuous
        
        # Fix issue with DiCE's get_decimal_precisions: ensure float columns have decimal parts
        # DiCE's get_decimal_precisions tries to access str(modes[0]).split('.')[1] which fails when:
        # 1. Mode value uses scientific notation (e.g., "1e-04" instead of "0.0001") - FOUND: electricity_lag1
        # 2. Mode value is integer-like and string is "0" instead of "0.0"
        # Solution: Fix values that would result in scientific notation in mode calculation
        for col in continuous_features:
            if col not in df_for_dice.columns or not pd.api.types.is_float_dtype(df_for_dice[col]):
                continue
            
            col_data = df_for_dice[col].dropna()
            if len(col_data) == 0:
                continue
            
            # FIX: Replace very small values that would use scientific notation
            # This is the main issue - values < 0.0001 become "1e-04" in string representation
            # which causes IndexError when DiCE tries to split by '.'
            very_small_mask = (abs(col_data) < 0.0001) & (col_data != 0) & col_data.notna()
            if very_small_mask.any():
                print(f"   ⚠️  Fixing {col}: {very_small_mask.sum()} very small values (scientific notation issue)")
                very_small_indices = col_data[very_small_mask].index
                # Replace with 0.0001 which is large enough to avoid scientific notation
                df_for_dice.loc[very_small_indices, col] = 0.0001
            
            # Also fix integer-like values
            integer_like_mask = (col_data == col_data.astype(int)) & col_data.notna()
            if integer_like_mask.any():
                epsilon = 0.0001
                df_for_dice.loc[integer_like_mask, col] = df_for_dice.loc[integer_like_mask, col] + epsilon
        
        # For DiCE, try without explicit categorical_features first
        # DiCE should auto-detect them, but we can specify if needed
        # Note: We've already fixed integer-like floats above to prevent IndexError in get_decimal_precisions
        try:
            self.dice_data = dice_ml.Data(
                dataframe=df_for_dice,
                continuous_features=continuous_features,
                categorical_features=dice_categorical_features if dice_categorical_features else None,
                outcome_name='electricity_consumption'
            )
        except Exception as e:
            # Fallback: let DiCE auto-detect categoricals
            print(f"⚠️  Warning: Error with explicit categorical features: {e}")
            print("   Trying with auto-detection...")
            try:
                self.dice_data = dice_ml.Data(
                    dataframe=df_for_dice,
                    continuous_features=continuous_features,
                    outcome_name='electricity_consumption'
                )
            except Exception as e2:
                # Last resort: minimal setup with core features only
                print(f"⚠️  Warning: Error with auto-detection: {e2}")
                print("   Trying minimal setup with core features only...")
                # Use only clearly continuous, actionable features
                core_features = ['sqm', 'occupants', 'airTemperature', 'yearbuilt', 'numberoffloors']
                minimal_continuous = [f for f in core_features if f in df_for_dice.columns]
                
                if len(minimal_continuous) > 0:
                    self.dice_data = dice_ml.Data(
                        dataframe=df_for_dice[minimal_continuous + ['electricity_consumption']].copy(),
                        continuous_features=minimal_continuous,
                        outcome_name='electricity_consumption'
                    )
                else:
                    raise RuntimeError("Could not setup DiCE with any feature configuration. Please check your data.")
        
        # Create DiCE Model object
        self.dice_model = dice_ml.Model(
            model=self.inference.model,
            backend='sklearn',
            model_type='regressor'
        )
        
        # Monkey-patch DiCE's get_decimal_precisions to handle scientific notation edge case
        # This ensures it works even when DiCE recalculates precisions during query processing
        self._patch_dice_precisions()
        
        # Create DiCE Explainer (use 'random' for speed, 'genetic' for better results)
        self.explainer = Dice(
            self.dice_data, 
            self.dice_model, 
            method='random'  # Default to 'random' to avoid precision recalculation issues
        )
        
        print("✅ DiCE setup complete!")
    
    def _patch_dice_precisions(self):
        """
        Monkey-patch DiCE's get_decimal_precisions method to handle scientific notation.
        This fixes the IndexError that occurs when mode values use scientific notation.
        """
        from dice_ml.data_interfaces import public_data_interface
        import numpy as np
        
        # Store original method
        original_get_decimal_precisions = public_data_interface.PublicData.get_decimal_precisions
        
        def patched_get_decimal_precisions(self, output_type="list"):
            """Patched version that handles scientific notation in mode values."""
            precisions = [0] * len(self.continuous_feature_names)
            precisions_dict = {}
            
            for ix, col in enumerate(self.continuous_feature_names):
                try:
                    if (self.continuous_features_precision is not None) and (col in self.continuous_features_precision):
                        precisions[ix] = self.continuous_features_precision[col]
                        precisions_dict[col] = self.continuous_features_precision[col]
                    elif self.data_df[col].dtype == np.float32 or self.data_df[col].dtype == np.float64:
                        modes = self.data_df[col].mode()
                        if len(modes) > 0:
                            maxp = 0
                            for mx in range(len(modes)):
                                mode_str = str(modes[mx])
                                # Handle scientific notation (e.g., "1e-04") and integer-like values
                                if '.' in mode_str and 'e' not in mode_str.lower():
                                    # Standard decimal notation
                                    split_result = mode_str.split('.')
                                    if len(split_result) > 1:
                                        prec = len(split_result[1])
                                        if prec > maxp:
                                            maxp = prec
                                elif 'e-' in mode_str.lower() or 'e+' in mode_str.lower():
                                    # Scientific notation - extract precision from exponent
                                    # e.g., "1e-04" -> precision is 4
                                    try:
                                        if 'e-' in mode_str.lower():
                                            exp_part = mode_str.lower().split('e-')[1]
                                            # Precision is the exponent value
                                            maxp = max(maxp, int(exp_part.split('.')[0]) if '.' in exp_part else int(exp_part))
                                        elif 'e+' in mode_str.lower():
                                            # Positive exponent, use default precision
                                            maxp = max(maxp, 6)
                                    except:
                                        maxp = max(maxp, 6)  # Default precision
                                else:
                                    # Integer-like value without decimal point (e.g., "0", "1")
                                    # Use default precision
                                    maxp = max(maxp, 1)
                            
                            precisions[ix] = maxp if maxp > 0 else 1  # Ensure at least precision 1
                            precisions_dict[col] = maxp if maxp > 0 else 1
                except Exception as e:
                    # If anything fails, use default precision
                    precisions[ix] = 6
                    precisions_dict[col] = 6
            
            if output_type == "list":
                return precisions
            return precisions_dict
        
        # Apply the patch
        public_data_interface.PublicData.get_decimal_precisions = patched_get_decimal_precisions
    
    def get_actionable_features(self) -> List[str]:
        """
        Get list of features that can be adjusted.
        
        Returns:
            List of actionable feature names
        """
        return [
            feat for feat, info in self.actionable_features.items()
            if info.get('adjustable', False)
        ]
    
    def generate_recommendations(self,
                                json_data: Dict,
                                threshold: float,
                                total_cfs: int = 5,
                                method: str = 'random') -> Dict:
        """
        Generate counterfactual recommendations to reduce consumption below threshold.
        
        Args:
            json_data: Dictionary with building and weather data
            threshold: Target consumption threshold (kWh)
            total_cfs: Number of counterfactual examples to generate
            method: DiCE method ('random' or 'genetic')
            
        Returns:
            Dictionary with recommendations and counterfactuals
        """
        # First, predict current consumption
        current_prediction = self.inference.predict(json_data, include_lag=True)
        
        # Check if already below threshold
        if current_prediction <= threshold:
            return {
                'success': True,
                'current_prediction': float(current_prediction),
                'threshold': float(threshold),
                'below_threshold': True,
                'message': f'Current consumption ({current_prediction:.2f} kWh) is already below threshold ({threshold} kWh)',
                'recommendations': []
            }
        
        # Preprocess input for DiCE
        X = self.inference._preprocess_input(json_data, include_lag=True)
        
        # Convert to original format (decode categoricals) for DiCE
        query_instance = self._prepare_query_instance(json_data, X)
        
        # Ensure time features that DiCE treats as categorical are proper integer strings
        # DiCE expects '0', '1', etc., not '0.0' or 0.0
        # CRITICAL: Must set dtype to object (string) to prevent DiCE from converting back to float
        time_categorical_features = ['hour', 'day_of_week', 'month', 'year', 'is_weekend']
        for col in time_categorical_features:
            if col in query_instance.columns and col not in self.dice_data.continuous_feature_names:
                # DiCE treats this as categorical - ensure it's an integer string with object dtype
                try:
                    val = query_instance[col].iloc[0]
                    # Convert to int then string to avoid '0.0'
                    if isinstance(val, str):
                        # If it's already a string, check if it has decimal
                        if '.' in val:
                            int_val = int(float(val))
                            query_instance[col] = str(int_val)
                        # Already an integer string, keep it
                    else:
                        # Convert numeric to integer string
                        int_val = int(float(val))
                        query_instance[col] = str(int_val)
                    
                    # CRITICAL: Set dtype to object (string) to prevent float conversion
                    query_instance[col] = query_instance[col].astype('object')
                except (ValueError, TypeError):
                    # If conversion fails, use original value from json_data
                    if col in json_data:
                        try:
                            int_val = int(float(json_data[col]))
                            query_instance[col] = str(int_val)
                            query_instance[col] = query_instance[col].astype('object')
                        except:
                            pass
        
            # Get permitted ranges for actionable features
            permitted_range = self._get_permitted_ranges(query_instance, json_data)
            
            # Remove target column from query instance (DiCE doesn't want it)
            query_instance_for_dice = query_instance.drop(columns=['electricity_consumption'], errors='ignore')
            
            # Generate counterfactuals
            try:
                # Update explainer method if needed
                if method != 'random':
                    self.explainer = Dice(
                        self.dice_data,
                        self.dice_model,
                        method=method
                    )
                
                # Prepare parameters based on method
                cf_params = {
                    'query_instances': query_instance_for_dice,
                    'total_CFs': total_cfs,
                    'desired_range': [0, threshold],
                    'permitted_range': permitted_range
                }
                
                # Add weight parameters only for genetic method
                if method == 'genetic':
                    cf_params.update({
                        'proximity_weight': 0.5,
                        'diversity_weight': 1.0,
                        'sparsity_weight': 0.1
                    })
                
                counterfactuals = self.explainer.generate_counterfactuals(**cf_params)
                
                # Extract counterfactual data
                if not hasattr(counterfactuals, 'cf_examples_list') or not counterfactuals.cf_examples_list:
                    return {
                        'success': False,
                        'error': 'No counterfactuals generated by DiCE (cf_examples_list is empty)',
                        'current_prediction': float(current_prediction),
                        'threshold': float(threshold),
                        'recommendations': []
                    }
                
                if len(counterfactuals.cf_examples_list) == 0:
                    return {
                        'success': False,
                        'error': 'No counterfactuals generated by DiCE (list is empty)',
                        'current_prediction': float(current_prediction),
                        'threshold': float(threshold),
                        'recommendations': []
                    }
                
                try:
                    cf_example = counterfactuals.cf_examples_list[0]
                    if not hasattr(cf_example, 'final_cfs_df'):
                        return {
                            'success': False,
                            'error': 'Counterfactual example has no final_cfs_df attribute',
                            'current_prediction': float(current_prediction),
                            'threshold': float(threshold),
                            'recommendations': []
                        }
                    cf_df = cf_example.final_cfs_df
                except IndexError as e:
                    return {
                        'success': False,
                        'error': f'IndexError accessing counterfactuals: {str(e)}',
                        'current_prediction': float(current_prediction),
                        'threshold': float(threshold),
                        'recommendations': []
                    }
                
                if cf_df is None or len(cf_df) == 0:
                    return {
                        'success': False,
                        'error': 'DiCE generated empty counterfactuals DataFrame',
                        'current_prediction': float(current_prediction),
                        'threshold': float(threshold),
                        'recommendations': []
                    }
                
                # Get predictions for counterfactuals
                cf_predictions = []
                recommendations = []
                
                for idx, cf_row in cf_df.iterrows():
                    # Convert counterfactual back to dict format
                    cf_dict = cf_row.to_dict()
                    
                    # Predict consumption for this counterfactual
                    # Need to prepare it properly for prediction
                    cf_data = json_data.copy()
                    
                    # Update with counterfactual values
                    for feat, value in cf_dict.items():
                        if feat != 'electricity_consumption' and feat in cf_data:
                            # Handle different naming conventions
                            if feat in cf_data:
                                cf_data[feat] = value
                            elif feat.replace('_', '') in [k.replace('_', '') for k in cf_data.keys()]:
                                # Find matching key
                                for key in cf_data.keys():
                                    if key.replace('_', '') == feat.replace('_', ''):
                                        cf_data[key] = value
                                        break
                    
                    # Predict
                    try:
                        cf_pred = self.inference.predict(cf_data, include_lag=False)
                        cf_predictions.append(cf_pred)
                        
                        # Calculate changes
                        changes = self._calculate_changes(json_data, cf_dict, current_prediction, cf_pred)
                        
                        recommendations.append({
                            'counterfactual_id': idx,
                            'predicted_consumption': float(cf_pred),
                            'reduction': float(current_prediction - cf_pred),
                            'reduction_pct': float((current_prediction - cf_pred) / current_prediction * 100),
                            'below_threshold': cf_pred <= threshold,
                            'changes': changes
                        })
                    except Exception as e:
                        print(f"⚠️  Error predicting counterfactual {idx}: {e}")
                        continue
                
                # Sort by reduction amount
                recommendations.sort(key=lambda x: x['reduction'], reverse=True)
                
                return {
                    'success': True,
                    'current_prediction': float(current_prediction),
                    'threshold': float(threshold),
                    'below_threshold': False,
                    'needs_reduction': float(current_prediction - threshold),
                    'total_counterfactuals': len(recommendations),
                    'recommendations': recommendations
                }
                
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                return {
                    'success': False,
                    'error': f'{str(e)}',
                    'error_details': error_details,
                    'current_prediction': float(current_prediction),
                    'threshold': float(threshold),
                    'recommendations': []
                }
    
    def _prepare_query_instance(self, json_data: Dict, X: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare query instance for DiCE (with decoded categoricals).
        
        Args:
            json_data: Original JSON data
            X: Preprocessed DataFrame
            
        Returns:
            DataFrame ready for DiCE
        """
        # Start with preprocessed data
        query_df = X.copy()
        
        # Decode categorical features back to strings
        for col in self.inference.categorical_features:
            if col in query_df.columns and col in self.inference.label_encoders:
                le = self.inference.label_encoders[col]
                try:
                    encoded_val = int(query_df[col].iloc[0])
                    if encoded_val in le.classes_:
                        query_df[col] = le.inverse_transform([encoded_val])[0]
                    else:
                        # Use original value from json_data
                        query_df[col] = json_data.get(col, 'Unknown')
                except:
                    query_df[col] = json_data.get(col, 'Unknown')
        
        # Handle time features that DiCE might treat as categorical
        # DiCE auto-detects hour, day_of_week, month, etc. as categorical if they're not in continuous_features
        # We need to convert them to strings to match DiCE's expectations
        time_features_that_might_be_categorical = ['hour', 'day_of_week', 'month', 'year', 'is_weekend']
        for col in time_features_that_might_be_categorical:
            if col in query_df.columns:
                # Check if DiCE treats this as categorical (it's not in continuous_features)
                if col not in self.dice_data.continuous_feature_names:
                    # DiCE treats it as categorical, convert to integer string (e.g., '0' not '0.0')
                    try:
                        # Convert to int first, then to string to avoid '0.0' -> '0'
                        int_val = int(float(query_df[col].iloc[0]))
                        query_df[col] = str(int_val)
                    except (ValueError, TypeError):
                        # Fallback: convert to string directly
                        query_df[col] = query_df[col].astype(str)
                else:
                    # DiCE treats it as continuous, keep as numeric
                    query_df[col] = pd.to_numeric(query_df[col], errors='coerce')
        
        # Add target column (will be ignored by DiCE but needed for structure)
        query_df['electricity_consumption'] = self.inference.predict(json_data, include_lag=True)
        
        return query_df
    
    def _get_permitted_ranges(self, query_instance: pd.DataFrame, json_data: Dict) -> Dict:
        """
        Get permitted ranges for features based on actionable features.
        
        Args:
            query_instance: Query instance DataFrame
            json_data: Original JSON data
            
        Returns:
            Dictionary of permitted ranges
        """
        permitted_range = {}
        
        for feat, info in self.actionable_features.items():
            if not info.get('adjustable', False):
                continue
            
            if feat not in query_instance.columns:
                continue
            
            # Check if DiCE treats this feature as categorical
            is_categorical = feat not in self.dice_data.continuous_feature_names
            
            # For categorical features, don't set permitted_range (let DiCE use allowed categories)
            if is_categorical:
                # Skip setting permitted_range for categorical features
                # DiCE will use the allowed categories from training data
                continue
            
            current_value = query_instance[feat].iloc[0]
            
            # Convert to numeric if it's a string
            try:
                current_value = float(current_value)
            except (ValueError, TypeError):
                # If conversion fails, try to get from json_data
                current_value = json_data.get(feat, 0)
                try:
                    current_value = float(current_value)
                except (ValueError, TypeError):
                    # Skip this feature if we can't convert to numeric
                    continue
            
            if info.get('direction') == 'decrease':
                # Can only decrease
                min_val = current_value * (1 - info.get('max_change_pct', 0.3))
                max_val = current_value
            elif info.get('direction') == 'increase':
                # Can only increase
                min_val = current_value
                max_val = current_value * (1 + info.get('max_change_pct', 0.3))
            else:  # 'both'
                # Can increase or decrease
                if 'min_change' in info:
                    # Absolute change
                    min_val = current_value - info.get('max_change', 5.0)
                    max_val = current_value + info.get('max_change', 5.0)
                else:
                    # Percentage change
                    min_val = current_value * (1 - info.get('max_change_pct', 0.3))
                    max_val = current_value * (1 + info.get('max_change_pct', 0.3))
            
            # Ensure reasonable bounds
            if feat == 'sqm':
                min_val = max(min_val, 100)  # Minimum 100 sqm
            elif feat == 'occupants':
                min_val = max(min_val, 1)  # Minimum 1 occupant
            elif feat == 'airTemperature':
                min_val = max(min_val, -10)  # Reasonable temperature range
                max_val = min(max_val, 50)
            elif feat == 'hour':
                min_val = 0
                max_val = 23
            
            permitted_range[feat] = [float(min_val), float(max_val)]
        
        return permitted_range
    
    def _calculate_changes(self, 
                          original_data: Dict,
                          counterfactual: Dict,
                          original_pred: float,
                          cf_pred: float) -> List[Dict]:
        """
        Calculate and format changes between original and counterfactual.
        
        Args:
            original_data: Original data
            counterfactual: Counterfactual data
            original_pred: Original prediction
            cf_pred: Counterfactual prediction
            
        Returns:
            List of change descriptions
        """
        changes = []
        
        # Get actionable features that changed
        actionable = self.get_actionable_features()
        
        for feat in actionable:
            if feat not in counterfactual or feat == 'electricity_consumption':
                continue
            
            # Try to get original value
            orig_val = original_data.get(feat)
            if orig_val is None:
                # Try alternative names
                alt_names = {
                    'airTemperature': ['air_temperature', 'temperature'],
                    'cloudCoverage': ['cloud_coverage'],
                    'dewTemperature': ['dew_temperature'],
                    'windSpeed': ['wind_speed'],
                    'seaLvlPressure': ['sea_lvl_pressure'],
                    'precipDepth1HR': ['precip_depth_1hr']
                }
                if feat in alt_names:
                    for alt in alt_names[feat]:
                        if alt in original_data:
                            orig_val = original_data[alt]
                            break
            
            if orig_val is None:
                continue
            
            cf_val = counterfactual[feat]
            
            # Check if value changed significantly
            if pd.isna(orig_val) or pd.isna(cf_val):
                continue
            
            try:
                orig_val = float(orig_val)
                cf_val = float(cf_val)
                
                change = cf_val - orig_val
                change_pct = (change / orig_val * 100) if orig_val != 0 else 0
                
                # Only include significant changes (>1% or >0.1 absolute)
                if abs(change_pct) > 1 or abs(change) > 0.1:
                    feat_info = self.actionable_features.get(feat, {})
                    changes.append({
                        'feature': feat,
                        'description': feat_info.get('description', feat),
                        'original_value': orig_val,
                        'suggested_value': cf_val,
                        'change': change,
                        'change_pct': change_pct,
                        'action': self._get_action_description(feat, orig_val, cf_val)
                    })
            except (ValueError, TypeError):
                # Handle categorical or non-numeric features
                if str(orig_val) != str(cf_val):
                    feat_info = self.actionable_features.get(feat, {})
                    changes.append({
                        'feature': feat,
                        'description': feat_info.get('description', feat),
                        'original_value': str(orig_val),
                        'suggested_value': str(cf_val),
                        'change': None,
                        'change_pct': None,
                        'action': f"Change {feat} from '{orig_val}' to '{cf_val}'"
                    })
        
        return changes
    
    def _get_action_description(self, feat: str, orig_val: float, cf_val: float) -> str:
        """
        Get human-readable action description.
        
        Args:
            feat: Feature name
            orig_val: Original value
            cf_val: Counterfactual value
            
        Returns:
            Action description string
        """
        change = cf_val - orig_val
        change_pct = (change / orig_val * 100) if orig_val != 0 else 0
        
        if feat == 'sqm':
            if change < 0:
                return f"Reduce building area by {abs(change):.0f} sqm ({abs(change_pct):.1f}%)"
            else:
                return f"Increase building area by {change:.0f} sqm ({change_pct:.1f}%)"
        elif feat == 'occupants':
            if change < 0:
                return f"Reduce occupants by {abs(change):.0f} ({abs(change_pct):.1f}%)"
            else:
                return f"Increase occupants by {change:.0f} ({change_pct:.1f}%)"
        elif feat == 'airTemperature':
            if change < 0:
                return f"Lower temperature by {abs(change):.1f}°C (adjust HVAC)"
            else:
                return f"Raise temperature by {change:.1f}°C (adjust HVAC)"
        elif feat == 'hour':
            return f"Change operating hour from {int(orig_val)} to {int(cf_val)}"
        else:
            if change < 0:
                return f"Reduce {feat} by {abs(change):.2f} ({abs(change_pct):.1f}%)"
            else:
                return f"Increase {feat} by {change:.2f} ({change_pct:.1f}%)"
    
    def get_simple_recommendations(self,
                                   json_data: Dict,
                                   threshold: float,
                                   top_n: int = 3) -> Dict:
        """
        Get simplified recommendations (faster, less detailed).
        
        Args:
            json_data: Dictionary with building and weather data
            threshold: Target consumption threshold
            top_n: Number of top recommendations to return
            
        Returns:
            Simplified recommendations dictionary
        """
        result = self.generate_recommendations(
            json_data=json_data,
            threshold=threshold,
            total_cfs=top_n,
            method='genetic'
        )
        
        if not result['success'] or result.get('below_threshold', False):
            return result
        
        # Simplify recommendations
        simplified = {
            'success': True,
            'current_prediction': result['current_prediction'],
            'threshold': result['threshold'],
            'needs_reduction': result['needs_reduction'],
            'top_recommendations': []
        }
        
        for rec in result['recommendations'][:top_n]:
            if rec['below_threshold']:
                simplified['top_recommendations'].append({
                    'predicted_consumption': rec['predicted_consumption'],
                    'reduction': rec['reduction'],
                    'reduction_pct': rec['reduction_pct'],
                    'key_changes': [
                        {
                            'feature': ch['feature'],
                            'action': ch['action'],
                            'impact': f"{ch['change_pct']:.1f}%" if ch['change_pct'] else "significant"
                        }
                        for ch in rec['changes'][:3]  # Top 3 changes
                    ]
                })
        
        return simplified
    
    def recommendations_to_dataframe(self, result: Dict) -> pd.DataFrame:
        """
        Convert recommendations result to a pandas DataFrame.
        
        Args:
            result: Result dictionary from generate_recommendations()
            
        Returns:
            DataFrame with recommendations summary
        """
        if not result.get('success') or not result.get('recommendations'):
            return pd.DataFrame()
        
        rows = []
        for rec in result['recommendations']:
            row = {
                'Counterfactual ID': rec.get('counterfactual_id', 'N/A'),
                'Predicted Consumption (kWh)': rec['predicted_consumption'],
                'Reduction (kWh)': rec['reduction'],
                'Reduction (%)': rec['reduction_pct'],
                'Below Threshold': '✅ Yes' if rec['below_threshold'] else '❌ No',
                'Number of Changes': len(rec.get('changes', []))
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        return df
    
    def changes_to_dataframe(self, result: Dict, counterfactual_id: Optional[int] = None) -> pd.DataFrame:
        """
        Convert changes from recommendations to a DataFrame.
        
        Args:
            result: Result dictionary from generate_recommendations()
            counterfactual_id: Specific counterfactual ID (if None, returns all)
            
        Returns:
            DataFrame with detailed changes
        """
        if not result.get('success') or not result.get('recommendations'):
            return pd.DataFrame()
        
        rows = []
        for rec in result['recommendations']:
            if counterfactual_id is not None and rec.get('counterfactual_id') != counterfactual_id:
                continue
            
            for change in rec.get('changes', []):
                row = {
                    'Counterfactual ID': rec.get('counterfactual_id', 'N/A'),
                    'Feature': change.get('feature', 'N/A'),
                    'Description': change.get('description', 'N/A'),
                    'Original Value': change.get('original_value', 'N/A'),
                    'Suggested Value': change.get('suggested_value', 'N/A'),
                    'Change': change.get('change', 'N/A'),
                    'Change (%)': change.get('change_pct', 'N/A'),
                    'Action': change.get('action', 'N/A')
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        return df
    
    def visualize_recommendations(self, 
                                  result: Dict,
                                  save_path: Optional[str] = None,
                                  figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Create visualizations for recommendations.
        
        Args:
            result: Result dictionary from generate_recommendations()
            save_path: Optional path to save the figure
            figsize: Figure size (width, height)
        """
        if not MATPLOTLIB_AVAILABLE:
            print("⚠️  matplotlib not available. Install with: pip install matplotlib")
            return
        
        if not result.get('success') or not result.get('recommendations'):
            print("⚠️  No recommendations to visualize")
            return
        
        recommendations = result['recommendations']
        current_pred = result['current_prediction']
        threshold = result['threshold']
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('DiCE Recommendations Analysis', fontsize=16, fontweight='bold')
        
        # 1. Consumption Comparison (Bar Chart)
        ax1 = axes[0, 0]
        cf_ids = [f"CF {i+1}" for i in range(len(recommendations))]
        predicted = [rec['predicted_consumption'] for rec in recommendations]
        
        x_pos = np.arange(len(cf_ids))
        bars = ax1.bar(x_pos, predicted, color=['green' if p <= threshold else 'orange' for p in predicted])
        ax1.axhline(y=threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold ({threshold} kWh)')
        ax1.axhline(y=current_pred, color='blue', linestyle='--', linewidth=2, label=f'Current ({current_pred:.1f} kWh)')
        ax1.set_xlabel('Counterfactual')
        ax1.set_ylabel('Predicted Consumption (kWh)')
        ax1.set_title('Predicted Consumption by Counterfactual')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(cf_ids, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. Reduction Amount (Bar Chart)
        ax2 = axes[0, 1]
        reductions = [rec['reduction'] for rec in recommendations]
        reduction_pcts = [rec['reduction_pct'] for rec in recommendations]
        
        bars2 = ax2.bar(x_pos, reductions, color='steelblue')
        ax2.set_xlabel('Counterfactual')
        ax2.set_ylabel('Reduction (kWh)')
        ax2.set_title('Consumption Reduction')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(cf_ids, rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)
        
        # Add percentage labels on bars
        for i, (bar, pct) in enumerate(zip(bars2, reduction_pcts)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{pct:.1f}%',
                    ha='center', va='bottom', fontsize=9)
        
        # 3. Reduction Percentage (Bar Chart)
        ax3 = axes[1, 0]
        bars3 = ax3.bar(x_pos, reduction_pcts, color='coral')
        ax3.set_xlabel('Counterfactual')
        ax3.set_ylabel('Reduction (%)')
        ax3.set_title('Percentage Reduction')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(cf_ids, rotation=45, ha='right')
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Feature Changes Summary (Top features across all CFs)
        ax4 = axes[1, 1]
        # Count how many times each feature appears in recommendations
        feature_counts = {}
        for rec in recommendations:
            for change in rec.get('changes', []):
                feat = change.get('feature', 'Unknown')
                feature_counts[feat] = feature_counts.get(feat, 0) + 1
        
        if feature_counts:
            features = list(feature_counts.keys())[:10]  # Top 10
            counts = [feature_counts[f] for f in features]
            
            bars4 = ax4.barh(range(len(features)), counts, color='mediumseagreen')
            ax4.set_yticks(range(len(features)))
            ax4.set_yticklabels(features)
            ax4.set_xlabel('Frequency in Recommendations')
            ax4.set_title('Most Frequently Changed Features')
            ax4.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Visualization saved to: {save_path}")
        else:
            plt.show()
    
    def visualize_changes(self,
                         result: Dict,
                         counterfactual_id: int = 0,
                         save_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Visualize changes for a specific counterfactual.
        
        Args:
            result: Result dictionary from generate_recommendations()
            counterfactual_id: ID of counterfactual to visualize
            save_path: Optional path to save the figure
            figsize: Figure size (width, height)
        """
        if not MATPLOTLIB_AVAILABLE:
            print("⚠️  matplotlib not available. Install with: pip install matplotlib")
            return
        
        if not result.get('success') or not result.get('recommendations'):
            print("⚠️  No recommendations to visualize")
            return
        
        # Find the counterfactual
        rec = None
        for r in result['recommendations']:
            if r.get('counterfactual_id') == counterfactual_id:
                rec = r
                break
        
        if not rec or not rec.get('changes'):
            print(f"⚠️  No changes found for counterfactual {counterfactual_id}")
            return
        
        changes = rec['changes']
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(f'Counterfactual {counterfactual_id} - Feature Changes', 
                    fontsize=14, fontweight='bold')
        
        # 1. Change Magnitude (Bar Chart)
        features = [ch['feature'] for ch in changes]
        change_pcts = [ch.get('change_pct', 0) for ch in changes]
        
        colors = ['red' if pct < 0 else 'green' for pct in change_pcts]
        bars = ax1.barh(range(len(features)), change_pcts, color=colors)
        ax1.set_yticks(range(len(features)))
        ax1.set_yticklabels(features)
        ax1.set_xlabel('Change (%)')
        ax1.set_title('Percentage Change by Feature')
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax1.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, pct) in enumerate(zip(bars, change_pcts)):
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{pct:.1f}%',
                    ha='left' if width > 0 else 'right', va='center', fontsize=9)
        
        # 2. Original vs Suggested Values
        original_vals = [ch.get('original_value', 0) for ch in changes]
        suggested_vals = [ch.get('suggested_value', 0) for ch in changes]
        
        x_pos = np.arange(len(features))
        width = 0.35
        
        bars1 = ax2.bar(x_pos - width/2, original_vals, width, label='Original', color='lightblue')
        bars2 = ax2.bar(x_pos + width/2, suggested_vals, width, label='Suggested', color='lightcoral')
        
        ax2.set_xlabel('Feature')
        ax2.set_ylabel('Value')
        ax2.set_title('Original vs Suggested Values')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(features, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Visualization saved to: {save_path}")
        else:
            plt.show()
    
    def export_to_csv(self, result: Dict, base_path: str) -> Dict[str, str]:
        """
        Export recommendations to CSV files.
        
        Args:
            result: Result dictionary from generate_recommendations()
            base_path: Base path for output files (without extension)
            
        Returns:
            Dictionary with paths to created files
        """
        files_created = {}
        
        # Export recommendations summary
        df_summary = self.recommendations_to_dataframe(result)
        if not df_summary.empty:
            summary_path = f"{base_path}_summary.csv"
            df_summary.to_csv(summary_path, index=False)
            files_created['summary'] = summary_path
        
        # Export detailed changes
        df_changes = self.changes_to_dataframe(result)
        if not df_changes.empty:
            changes_path = f"{base_path}_changes.csv"
            df_changes.to_csv(changes_path, index=False)
            files_created['changes'] = changes_path
        
        return files_created
    
    def export_to_excel(self, result: Dict, excel_path: str) -> str:
        """
        Export recommendations to Excel file with multiple sheets.
        
        Args:
            result: Result dictionary from generate_recommendations()
            excel_path: Path to Excel file
            
        Returns:
            Path to created Excel file
        """
        try:
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # Summary sheet
                df_summary = self.recommendations_to_dataframe(result)
                if not df_summary.empty:
                    df_summary.to_excel(writer, sheet_name='Summary', index=False)
                
                # Changes sheet
                df_changes = self.changes_to_dataframe(result)
                if not df_changes.empty:
                    df_changes.to_excel(writer, sheet_name='Changes', index=False)
                
                # Individual counterfactual sheets
                if result.get('success') and result.get('recommendations'):
                    for rec in result['recommendations']:
                        cf_id = rec.get('counterfactual_id', 'unknown')
                        df_cf = self.changes_to_dataframe(result, counterfactual_id=cf_id)
                        if not df_cf.empty:
                            sheet_name = f'CF_{cf_id}'[:31]  # Excel sheet name limit
                            df_cf.to_excel(writer, sheet_name=sheet_name, index=False)
            
            return excel_path
        except ImportError:
            print("⚠️  openpyxl not available. Install with: pip install openpyxl")
            return ""