#!/usr/bin/env python3
"""
DiCE (Diverse Counterfactual Explanations) Integration - Version 2
Provides counterfactual explanations to reduce electricity consumption below threshold
Adapted for new XGBoost model architecture with multi-horizon predictions
"""

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import warnings
import sys
import os

# Add parent directory to path to import inference modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

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

# Import new inference modules
from inference.model import ElectricityForecaster
from inference.preprocess import preprocess, X_FEATURE_INPUT
from inference.encoder import LabelEncoder
from inference.helper import time_features

warnings.filterwarnings('ignore')


class MultiHorizonModelWrapper:
    """
    Wrapper for multi-horizon XGBoost models to work with DiCE.
    DiCE expects a single model, but we have 24 separate models (one per hour).
    This wrapper can use any specific hour model.
    """
    
    def __init__(self, forecaster: ElectricityForecaster, hour_offset: int = 1):
        """
        Initialize wrapper.
        
        Args:
            forecaster: ElectricityForecaster instance with loaded models
            hour_offset: Which hour model to use (1-24, default 1 for t+1)
        """
        self.forecaster = forecaster
        self.hour_offset = hour_offset
        
        if hour_offset < 1 or hour_offset > forecaster.horizon:
            raise ValueError(f"hour_offset must be between 1 and {forecaster.horizon}, got {hour_offset}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using the specified hour model.
        
        Args:
            X: DataFrame with features (may be decoded strings from DiCE or encoded numbers)
            
        Returns:
            Array of predictions for the specified hour
        """
        # Check if we need to encode categorical features
        # DiCE may pass decoded strings, but the model needs encoded numbers
        X_encoded = X.copy()
        categorical_cols = self.forecaster.encoder.categorical_cols
        
        # Check if any categorical columns contain strings (need encoding)
        needs_encoding = False
        for col in categorical_cols:
            if col in X_encoded.columns:
                # Check if column contains strings
                sample_val = X_encoded[col].iloc[0] if len(X_encoded) > 0 else None
                if isinstance(sample_val, str):
                    needs_encoding = True
                    break
        
        # Encode categorical features if needed
        if needs_encoding:
            for col in categorical_cols:
                if col in X_encoded.columns:
                    le = self.forecaster.encoder.encoders[col]
                    values = X_encoded[col].fillna(self.forecaster.encoder.unknown_token).astype(str)
                    # Map values to encoder classes, use unknown_token for unseen values
                    values = values.where(
                        values.isin(le.classes_),
                        self.forecaster.encoder.unknown_token
                    )
                    X_encoded[col] = le.transform(values)
        
        # Ensure all columns are numeric (handle any remaining non-numeric)
        for col in X_encoded.columns:
            if X_encoded[col].dtype == 'object':
                # Try to convert to numeric, if fails use 0
                try:
                    X_encoded[col] = pd.to_numeric(X_encoded[col], errors='coerce').fillna(0)
                except:
                    X_encoded[col] = 0
        
        # Get the model for the specified hour
        model = self.forecaster.models[self.hour_offset]
        
        # Ensure we have the right feature order (use X_FEATURE_INPUT order)
        # This is critical - the model expects features in a specific order
        from inference.preprocess import X_FEATURE_INPUT
        
        # Make sure all required features are present
        missing_features = set(X_FEATURE_INPUT) - set(X_encoded.columns)
        if missing_features:
            # Add missing features with default values
            for feat in missing_features:
                X_encoded[feat] = 0
        
        # Reorder columns to match X_FEATURE_INPUT
        X_encoded = X_encoded[X_FEATURE_INPUT]
        
        # Convert to numpy array and ensure numeric types
        X_array = X_encoded.values.astype(float)
        
        # Predict using the specified hour model
        predictions = model.predict(X_array)
        
        return predictions


class DiceExplainer:
    """
    DiCE Explainer for electricity consumption reduction - Version 2.
    
    Provides counterfactual explanations to suggest feature adjustments
    that would reduce electricity consumption below a threshold.
    Adapted for new XGBoost model architecture.
    """
    
    def __init__(self,
                 model_dir: str = "models_1578_csv",
                 encode_path: str = "models_1578_csv/categorical_encoder.pkl",
                 processed_data_path: str = "data_1578_csv/train_encode.csv",
                 forecast_horizon: int = 24):
        """
        Initialize DiCE Explainer.
        
        Args:
            model_dir: Directory containing model files (model_hour_1.pkl, etc.)
            encode_path: Path to categorical encoder pickle file
            processed_data_path: Path to processed training data for DiCE
            forecast_horizon: Number of forecast horizons (default 24)
        """
        if not DICE_AVAILABLE:
            raise ImportError("dice-ml is required. Install with: pip install dice-ml")
        
        # Initialize forecaster
        self.forecaster = ElectricityForecaster(
            model_dir=model_dir,
            encode_path=encode_path,
            forecast_horizon=forecast_horizon
        )
        
        self.model_dir = Path(model_dir)
        self.encode_path = Path(encode_path)
        self.processed_data_path = Path(processed_data_path)
        self.horizon = forecast_horizon
        
        # Feature columns used by the model
        self.feature_cols = X_FEATURE_INPUT.copy()
        
        # Categorical columns (from encoder)
        self.categorical_cols = self.forecaster.encoder.categorical_cols
        
        # Define actionable features
        self.actionable_features = self._define_actionable_features()
        
        # Initialize DiCE components
        self.dice_data = None
        self.dice_models = {}  # Dictionary of 24 dice models, one for each hour
        self.explainer = None  # Default explainer for hour 1 (for backward compatibility)
        
        # Load and setup DiCE
        self._setup_dice()
    
    def _define_actionable_features(self) -> Dict[str, Dict]:
        """
        Define which features can be adjusted and their constraints.
        
        Returns:
            Dictionary mapping feature names to their constraints
        """
        return {
            # Time features - FIXED (cannot change timestamp)
            'hour': {
                'adjustable': False,
                'description': 'Hour of day (fixed - determined by timestamp)'
            },
            'dayofweek': {
                'adjustable': False,
                'description': 'Day of week (fixed - determined by timestamp)'
            },
            'is_weekend': {
                'adjustable': False,
                'description': 'Weekend flag (fixed - determined by timestamp)'
            },
            'month': {
                'adjustable': False,
                'description': 'Month (fixed - determined by timestamp)'
            },
            # Weather features - FIXED (cannot control weather)
            'airTemperature': {
                'adjustable': False,
                'description': 'Air temperature (weather data - cannot control)'
            },
            'dewTemperature': {
                'adjustable': True,
                'direction': 'decrease',
                'min_change_pct': 0.10,
                'max_change_pct': 0.50,
                'description': 'Dew temperature (weather data - cannot control)'
            },
            'windSpeed': {
                'adjustable': True,
                'direction': 'decrease',
                'min_change_pct': 0.10,
                'max_change_pct': 0.50,
                'description': 'Wind speed (weather data - can reduce usage)'
            },
            # Building features - FIXED (cannot change building properties)
            'sqm': {
                'adjustable': True,
                'direction': 'decrease',  # Can only decrease (reduce space)
                'min_change_pct': 0.05,  # Minimum 5% change
                'max_change_pct': 0.30,  # Maximum 30% change
                'description': 'Building area (square meters)'
            },
            'primaryspaceusage': {
                'adjustable': False,
                'description': 'Primary space usage (fixed)'
            },
            'site_id': {
                'adjustable': False,
                'description': 'Site ID (fixed)'
            },
            'building_id': {
                'adjustable': False,
                'description': 'Building ID (fixed)'
            },
            # Energy consumption features - ADJUSTABLE (can control usage)
            'Chilledwater': {
                'adjustable': True,
                'direction': 'decrease',
                'min_change_pct': 0.10,
                'max_change_pct': 0.50,
                'description': 'Chilled water consumption (can reduce HVAC usage)'
            },
            'Hotwater': {
                'adjustable': True,
                'direction': 'decrease',
                'min_change_pct': 0.10,
                'max_change_pct': 0.50,
                'description': 'Hot water consumption (can reduce usage)'
            },
            # Note: numberoffloors is not in current feature set, but marked as actionable for future use
            # 'numberoffloors': {
            #     'adjustable': True,
            #     'direction': 'decrease',
            #     'min_change_pct': 0.0,
            #     'max_change_pct': 0.30,
            #     'description': 'Number of floors (can reduce usage by reducing active floors)'
            # }
        }
    
    def _setup_dice(self):
        """Setup DiCE data and model objects."""
        print("🔧 Setting up DiCE...")
        
        # Load processed data
        if not self.processed_data_path.exists():
            raise FileNotFoundError(
                f"Processed data not found at {self.processed_data_path}. "
                "Please ensure train_encode.csv exists."
            )
        
        df = pd.read_csv(self.processed_data_path)
        
        # Get target column (we'll use target_t+1 for DiCE)
        # Check if target columns exist in the data
        target_col = 'target_t+1'
        
        # Check if target column exists
        if target_col not in df.columns:
            print(f"⚠️  Warning: {target_col} not found in data.")
            print("   Creating target from model predictions...")
            
            # Get features (ensure all feature columns exist)
            available_features = [f for f in self.feature_cols if f in df.columns]
            if len(available_features) < len(self.feature_cols):
                missing = set(self.feature_cols) - set(available_features)
                print(f"   ⚠️  Missing features: {missing}")
            
            X = df[available_features].copy()
            
            # Fill any missing values with 0 or mode
            for col in X.columns:
                if X[col].isna().any():
                    if col in self.categorical_cols:
                        X[col] = X[col].fillna(0)  # Use 0 for encoded categoricals
                    else:
                        X[col] = X[col].fillna(X[col].median() if X[col].dtype in ['float64', 'int64'] else 0)
            
            # Predict using first hour model
            model_h1 = self.forecaster.models[1]
            df[target_col] = model_h1.predict(X.values)
            print(f"   ✅ Created target column with {len(df)} predictions")
        else:
            print(f"   ✅ Using existing target column: {target_col}")
        
        # Create a sample dataset for DiCE (use a smaller subset for stability)
        sample_size = min(5000, len(df))
        df_sample = df.sample(n=sample_size, random_state=42).copy()
        
        # Drop any rows with NaN in critical columns
        df_sample = df_sample.dropna(subset=[target_col] + self.feature_cols)
        
        # Prepare features for DiCE
        # Need to decode categoricals back to strings for DiCE
        df_for_dice = df_sample[self.feature_cols + [target_col]].copy()
        
        # Decode categorical features back to original strings for DiCE
        for col in self.categorical_cols:
            if col not in df_for_dice.columns:
                continue
            
            le = self.forecaster.encoder.encoders[col]
            col_series = df_for_dice[col]
            
            # Check if column is already decoded (strings in encoder classes)
            sample_vals = col_series.dropna()
            if len(sample_vals) > 0:
                sample_val = sample_vals.iloc[0]
                # If it's already a string and in the encoder classes, it's already decoded
                if isinstance(sample_val, str) and sample_val in le.classes_:
                    continue
                
                # Decode back to original strings
                def decode_value(x):
                    try:
                        if pd.isna(x):
                            return self.forecaster.encoder.unknown_token
                        
                        if isinstance(x, str):
                            if x in le.classes_:
                                return x
                            try:
                                int_val = int(float(x))
                                if int_val < len(le.classes_):
                                    return le.inverse_transform([int_val])[0]
                            except (ValueError, TypeError):
                                pass
                            return self.forecaster.encoder.unknown_token
                        
                        if isinstance(x, (int, float, np.integer, np.floating)):
                            int_val = int(x)
                            if int_val < len(le.classes_):
                                return le.inverse_transform([int_val])[0]
                            return self.forecaster.encoder.unknown_token
                        
                        return self.forecaster.encoder.unknown_token
                    except Exception:
                        return self.forecaster.encoder.unknown_token
                
                df_for_dice[col] = col_series.apply(decode_value)
        
        # Identify continuous and categorical features
        continuous_features = [
            f for f in self.feature_cols 
            if f not in self.categorical_cols
        ]
        categorical_features = [
            f for f in self.categorical_cols 
            if f in df_for_dice.columns
        ]
        
        # Create DiCE Data object
        try:
            self.dice_data = dice_ml.Data(
                dataframe=df_for_dice,
                continuous_features=continuous_features,
                categorical_features=categorical_features,
                outcome_name=target_col
            )
        except Exception as e:
            print(f"⚠️  Error creating DiCE Data object: {e}")
            print("   Trying with minimal configuration...")
            # Fallback: use only continuous features
            self.dice_data = dice_ml.Data(
                dataframe=df_for_dice[continuous_features + [target_col]].copy(),
                continuous_features=continuous_features,
                outcome_name=target_col
            )
        
        # Create 24 DiCE models (one for each hour)
        print("   Creating DiCE models for all 24 hours...")
        for hour in range(1, self.horizon + 1):
            wrapped_model = MultiHorizonModelWrapper(self.forecaster, hour_offset=hour)
            dice_model = dice_ml.Model(
                model=wrapped_model,
                backend='sklearn',
                model_type='regressor'
            )
            self.dice_models[hour] = dice_model
        
        # Create default explainer for hour 1 (for backward compatibility)
        self.explainer = Dice(
            self.dice_data,
            self.dice_models[1],  # Use hour 1 model
            method='random'  # Default to 'random' for speed
        )
        
        print(f"✅ DiCE setup complete! Created {len(self.dice_models)} models (one for each hour)")
    
    def _get_dice_explainer_for_hour(self, hour_offset: int, method: str = 'random'):
        """
        Get or create a DiCE explainer for a specific hour using pre-created dice model.
        
        Args:
            hour_offset: Hour offset (1-24)
            method: DiCE method ('random' or 'genetic')
        
        Returns:
            DiCE explainer configured for the specific hour
        """
        if hour_offset < 1 or hour_offset > self.horizon:
            raise ValueError(f"hour_offset must be between 1 and {self.horizon}, got {hour_offset}")
        
        # Use pre-created dice model for this hour
        dice_model = self.dice_models[hour_offset]
        
        # Create DiCE Explainer with the pre-created model
        explainer = Dice(
            self.dice_data,
            dice_model,
            method=method
        )
        
        return explainer
    
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
    
    def _prepare_query_instance(self, json_data: Dict, X: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare query instance for DiCE from JSON data and preprocessed DataFrame.
        
        Args:
            json_data: Original JSON input
            X: Preprocessed DataFrame (encoded)
            
        Returns:
            DataFrame ready for DiCE (with decoded categoricals)
        """
        # Start with encoded DataFrame
        query_df = X.copy()
        
        # Decode categorical features back to strings for DiCE
        for col in self.categorical_cols:
            if col not in query_df.columns:
                continue
            
            le = self.forecaster.encoder.encoders[col]
            col_series = query_df[col]
            
            # Decode
            def decode_value(x):
                try:
                    if pd.isna(x):
                        return self.forecaster.encoder.unknown_token
                    
                    if isinstance(x, (int, float, np.integer, np.floating)):
                        int_val = int(x)
                        if int_val < len(le.classes_):
                            return le.inverse_transform([int_val])[0]
                        return self.forecaster.encoder.unknown_token
                    
                    return self.forecaster.encoder.unknown_token
                except Exception:
                    return self.forecaster.encoder.unknown_token
            
            query_df[col] = col_series.apply(decode_value)
        
        # Add target column (will be predicted)
        # For DiCE, we'll predict and add it (using hour 1 by default)
        wrapped_model = MultiHorizonModelWrapper(self.forecaster, hour_offset=1)
        prediction = wrapped_model.predict(X)[0]
        query_df['target_t+1'] = prediction
        
        return query_df
    
    def _prepare_query_instance_for_hour(self, json_data: Dict, X_encoded: pd.DataFrame, hour_offset: int) -> pd.DataFrame:
        """
        Prepare query instance for DiCE from JSON data and preprocessed DataFrame for a specific hour.
        
        Args:
            json_data: Original JSON input
            X_encoded: Preprocessed and encoded DataFrame
            hour_offset: Hour offset (1-24)
        
        Returns:
            DataFrame ready for DiCE (with decoded categoricals)
        """
        # Start with encoded DataFrame
        query_df = X_encoded.copy()
        
        # Decode categorical features back to strings for DiCE
        for col in self.categorical_cols:
            if col not in query_df.columns:
                continue
            
            le = self.forecaster.encoder.encoders[col]
            col_series = query_df[col]
            
            # Decode
            def decode_value(x):
                try:
                    if pd.isna(x):
                        return self.forecaster.encoder.unknown_token
                    
                    if isinstance(x, (int, float, np.integer, np.floating)):
                        int_val = int(x)
                        if int_val < len(le.classes_):
                            return le.inverse_transform([int_val])[0]
                        return self.forecaster.encoder.unknown_token
                    
                    return self.forecaster.encoder.unknown_token
                except Exception:
                    return self.forecaster.encoder.unknown_token
            
            query_df[col] = col_series.apply(decode_value)
        
        # Add target column (will be predicted for this specific hour)
        wrapped_model = MultiHorizonModelWrapper(self.forecaster, hour_offset=hour_offset)
        prediction = wrapped_model.predict(X_encoded)[0]
        query_df[f'target_t+{hour_offset}'] = prediction
        
        return query_df
    
    def _get_permitted_ranges(self, query_instance: pd.DataFrame, json_data: Dict) -> Dict[str, Tuple[float, float]]:
        """
        Get permitted ranges for actionable features.
        
        Args:
            query_instance: Query instance DataFrame
            json_data: Original JSON input
            
        Returns:
            Dictionary mapping feature names to (min, max) tuples
        """
        permitted_range = {}
        
        for feat, info in self.actionable_features.items():
            if not info.get('adjustable', False):
                continue
            
            if feat not in query_instance.columns:
                continue
            
            current_val = query_instance[feat].iloc[0]
            
            # Handle numeric features
            if isinstance(current_val, (int, float, np.integer, np.floating)):
                direction = info.get('direction', 'both')
                min_change_pct = info.get('min_change_pct', 0.0)
                max_change_pct = info.get('max_change_pct', 1.0)
                
                if direction == 'decrease':
                    min_val = current_val * (1 - max_change_pct)
                    max_val = current_val * (1 - min_change_pct)
                elif direction == 'increase':
                    min_val = current_val * (1 + min_change_pct)
                    max_val = current_val * (1 + max_change_pct)
                else:  # both
                    min_val = current_val * (1 - max_change_pct)
                    max_val = current_val * (1 + max_change_pct)
                
                # Ensure non-negative for consumption features
                if feat in ['Chilledwater', 'Hotwater']:
                    min_val = max(0.0, min_val)
                
                # For weather features that can be negative, ensure reasonable bounds
                # (dewTemperature can be negative, windSpeed should be non-negative)
                if feat == 'windSpeed':
                    min_val = max(0.0, min_val)  # Wind speed can't be negative
                elif feat == 'sqm':
                    min_val = max(100, min_val)  # Minimum 100 sqm
                elif feat == 'dewTemperature':
                    # Dew temperature can be negative, but set a reasonable lower bound
                    # Allow it to go down but not too extreme (e.g., not below -50°C)
                    min_val = max(-50.0, min_val)
                
                # For numberoffloors, ensure it's at least 1
                # if feat == 'numberoffloors':
                #     min_val = max(1.0, min_val)
                #     max_val = max(1.0, max_val)  # Ensure max is also at least 1
                
                permitted_range[feat] = (float(min_val), float(max_val))
        
        return permitted_range
    
    def generate_recommendations(self,
                                json_data: Dict,
                                threshold: float,
                                hour_offset: int = 1,
                                total_cfs: int = 5,
                                method: str = 'random') -> Dict:
        """
        Generate counterfactual recommendations to reduce consumption below threshold.
        
        Args:
            json_data: Dictionary with building and weather data
            threshold: Target consumption threshold (kWh) for the specified hour
            hour_offset: Hours ahead to predict (1-24, where 1 = next hour, default 1 for backward compatibility)
            total_cfs: Number of counterfactual examples to generate
            method: DiCE method ('random' or 'genetic')
            
        Returns:
            Dictionary with recommendations and counterfactuals for the specified hour
        """
        if hour_offset < 1 or hour_offset > self.horizon:
            raise ValueError(f"hour_offset must be between 1 and {self.horizon}, got {hour_offset}")
        
        # Predict current consumption for the specified hour
        current_prediction_result = self.forecaster.predict_hour(json_data, hour_offset)
        current_prediction = current_prediction_result['electric']
        
        # Check if already below threshold
        if current_prediction <= threshold:
            return {
                'success': True,
                'hour_offset': hour_offset,
                'time': current_prediction_result['time'],
                'current_prediction': float(current_prediction),
                'threshold': float(threshold),
                'below_threshold': True,
                'message': f'Current consumption for hour t+{hour_offset} ({current_prediction:.2f} kWh) is already below threshold ({threshold} kWh)',
                'recommendations': []
            }
        
        # Preprocess input
        X, start_time = preprocess(json_data)
        X_encoded = self.forecaster.encoder.transform(X)
        
        # Prepare query instance for DiCE (for the specified hour)
        query_instance = self._prepare_query_instance_for_hour(json_data, X_encoded, hour_offset)
        
        # Get permitted ranges for actionable features
        permitted_range = self._get_permitted_ranges(query_instance, json_data)
        
        # Check if we have any actionable features
        actionable_features_list = [
            feat for feat in self.get_actionable_features() 
            if feat in query_instance.columns
        ]
        
        if not actionable_features_list:
            return {
                'success': False,
                'hour_offset': hour_offset,
                'time': current_prediction_result['time'],
                'error': 'No actionable features available. Cannot generate counterfactuals.',
                'current_prediction': float(current_prediction),
                'threshold': float(threshold),
                'recommendations': []
            }
        
        # Remove target column from query instance (DiCE doesn't want it)
        target_col = f'target_t+{hour_offset}'
        query_instance_for_dice = query_instance.drop(columns=[target_col], errors='ignore')
        
        # Get DiCE explainer for the specified hour
        explainer = self._get_dice_explainer_for_hour(hour_offset, method)
        
        # Generate counterfactuals
        try:
            
            # Desired range for counterfactuals
            desired_range_min = max(0.0, threshold * 0.85)
            desired_range_max = threshold
            
            print(f"   📋 Generating recommendations for hour t+{hour_offset} ({current_prediction_result['time']})")
            print(f"   📋 Actionable features: {actionable_features_list}")
            if permitted_range:
                print(f"   📋 Permitted ranges: {list(permitted_range.keys())}")
                for feat, (min_val, max_val) in permitted_range.items():
                    current_val = query_instance_for_dice[feat].iloc[0] if feat in query_instance_for_dice.columns else "N/A"
                    print(f"      • {feat}: [{min_val:.2f}, {max_val:.2f}] (current: {current_val})")
            
            print(f"   🎯 Desired range: [{desired_range_min:.2f}, {desired_range_max:.2f}] kWh")
            
            # Prepare parameters
            cf_params = {
                'query_instances': query_instance_for_dice,
                'total_CFs': total_cfs * 2,  # Generate more to have better chance
                'desired_range': [desired_range_min, desired_range_max],
                'features_to_vary': actionable_features_list  # Only vary actionable features
            }
            
            if permitted_range:
                cf_params['permitted_range'] = permitted_range
            
            if method == 'genetic':
                cf_params.update({
                    'proximity_weight': 10.0,
                    'diversity_weight': 0.1,
                    'sparsity_weight': 1.0
                })
            
            print(f"   🔍 Generating counterfactuals with method='{method}'...")
            
            # Generate counterfactuals
            counterfactuals = None
            try:
                counterfactuals = explainer.generate_counterfactuals(**cf_params)
            except Exception as e:
                error_msg = str(e)
                # If features_to_vary is too restrictive, try without it but filter results later
                if 'No counterfactuals found' in error_msg or 'features_to_vary' in error_msg.lower():
                    print(f"   ⚠️  Could not find counterfactuals with restricted features. Trying without restriction...")
                    # Remove features_to_vary and try again
                    cf_params_no_restrict = cf_params.copy()
                    cf_params_no_restrict.pop('features_to_vary', None)
                    try:
                        counterfactuals = explainer.generate_counterfactuals(**cf_params_no_restrict)
                        print(f"   ✅ Found counterfactuals without feature restriction (will filter to actionable only)")
                    except Exception as e2:
                        if method == 'genetic':
                            print(f"   ⚠️  Genetic method failed: {e2}")
                            print(f"   🔄 Falling back to 'random' method...")
                            explainer = self._get_dice_explainer_for_hour(hour_offset, 'random')
                            cf_params_no_restrict = cf_params.copy()
                            cf_params_no_restrict.pop('features_to_vary', None)
                            counterfactuals = explainer.generate_counterfactuals(**cf_params_no_restrict)
                        else:
                            raise
                elif method == 'genetic':
                    print(f"   ⚠️  Genetic method failed: {e}")
                    print(f"   🔄 Falling back to 'random' method...")
                    explainer = self._get_dice_explainer_for_hour(hour_offset, 'random')
                    counterfactuals = explainer.generate_counterfactuals(**cf_params)
                else:
                    raise
            
            # Extract counterfactual data
            if not hasattr(counterfactuals, 'cf_examples_list') or not counterfactuals.cf_examples_list:
                return {
                    'success': False,
                    'hour_offset': hour_offset,
                    'time': current_prediction_result['time'],
                    'error': 'No counterfactuals generated by DiCE',
                    'current_prediction': float(current_prediction),
                    'threshold': float(threshold),
                    'recommendations': []
                }
            
            cf_example = counterfactuals.cf_examples_list[0]
            if not hasattr(cf_example, 'final_cfs_df'):
                return {
                    'success': False,
                    'hour_offset': hour_offset,
                    'time': current_prediction_result['time'],
                    'error': 'Counterfactual example has no final_cfs_df attribute',
                    'current_prediction': float(current_prediction),
                    'threshold': float(threshold),
                    'recommendations': []
                }
            
            cf_df = cf_example.final_cfs_df
            
            # Process counterfactuals
            recommendations = []
            for idx, row in cf_df.iterrows():
                # Convert counterfactual back to encoded format for prediction
                cf_dict = row.to_dict()
                
                # IMPORTANT: Restore non-actionable features to original values
                # This ensures DiCE doesn't change features we can't control
                actionable_set = set(actionable_features_list)
                for col in self.feature_cols:
                    if col not in actionable_set and col in query_instance_for_dice.columns:
                        # Restore original value for non-actionable features
                        cf_dict[col] = query_instance_for_dice[col].iloc[0]
                
                # Encode categorical features
                cf_encoded = pd.DataFrame([cf_dict])
                for col in self.categorical_cols:
                    if col in cf_encoded.columns:
                        le = self.forecaster.encoder.encoders[col]
                        values = cf_encoded[col].fillna(self.forecaster.encoder.unknown_token).astype(str)
                        values = values.where(
                            values.isin(le.classes_),
                            self.forecaster.encoder.unknown_token
                        )
                        cf_encoded[col] = le.transform(values)
                
                # Predict with counterfactual using the specified hour model
                cf_prediction = self.forecaster.models[hour_offset].predict(cf_encoded[self.feature_cols].values)[0]
                
                # Calculate changes - ONLY include actionable features
                changes = []
                # actionable_set already defined above
                
                for col in self.feature_cols:
                    # Only process actionable features
                    if col not in actionable_set:
                        continue
                    
                    if col in query_instance_for_dice.columns and col in cf_dict:
                        original_val = query_instance_for_dice[col].iloc[0]
                        cf_val = cf_dict[col]
                        
                        # Handle different types
                        try:
                            if isinstance(original_val, (int, float)) and isinstance(cf_val, (int, float)):
                                change = cf_val - original_val
                                change_pct = (change / original_val * 100) if original_val != 0 else 0
                                
                                if abs(change) > 1e-6:  # Significant change
                                    changes.append({
                                        'feature': col,
                                        'original': original_val,
                                        'new': cf_val,
                                        'change': change,
                                        'change_pct': change_pct,
                                        'action': f"{col}: {original_val:.2f} → {cf_val:.2f} ({change_pct:+.1f}%)",
                                        'description': self.actionable_features.get(col, {}).get('description', '')
                                    })
                            elif isinstance(original_val, str) and isinstance(cf_val, str):
                                if original_val != cf_val:
                                    changes.append({
                                        'feature': col,
                                        'original': original_val,
                                        'new': cf_val,
                                        'change': None,
                                        'change_pct': None,
                                        'action': f"{col}: '{original_val}' → '{cf_val}'",
                                        'description': self.actionable_features.get(col, {}).get('description', '')
                                    })
                        except Exception:
                            pass
                
                # Sort changes by absolute change percentage
                changes.sort(key=lambda x: abs(x.get('change_pct', 0)) if x.get('change_pct') is not None else 0, reverse=True)
                
                reduction = current_prediction - cf_prediction
                reduction_pct = (reduction / current_prediction * 100) if current_prediction > 0 else 0
                
                recommendations.append({
                    'counterfactual_id': idx + 1,
                    'predicted_consumption': float(cf_prediction),
                    'reduction': float(reduction),
                    'reduction_pct': float(reduction_pct),
                    'below_threshold': cf_prediction <= threshold,
                    'changes': changes
                })
            
            # Sort by reduction (descending)
            recommendations.sort(key=lambda x: x['reduction'], reverse=True)
            
            return {
                'success': True,
                'hour_offset': hour_offset,
                'time': current_prediction_result['time'],
                'current_prediction': float(current_prediction),
                'threshold': float(threshold),
                'below_threshold': False,
                'needs_reduction': float(current_prediction - threshold),
                'total_counterfactuals': len(recommendations),
                'recommendations': recommendations
            }
            
        except Exception as e:
            import traceback
            return {
                'success': False,
                'hour_offset': hour_offset,
                'time': current_prediction_result.get('time', ''),
                'error': str(e),
                'error_details': traceback.format_exc(),
                'current_prediction': float(current_prediction),
                'threshold': float(threshold),
                'recommendations': []
            }
    
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
            method='random'  # Use random for speed
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
    
    def generate_recommendations_for_hour(self,
                                          json_data: Dict,
                                          threshold: float,
                                          hour_offset: int,
                                          total_cfs: int = 5,
                                          method: str = 'random') -> Dict:
        """
        Generate counterfactual recommendations for a specific hour ahead.
        
        Args:
            json_data: Dictionary with building and weather data
            threshold: Target consumption threshold (kWh) for the specified hour
            hour_offset: Hours ahead to predict (1-24, where 1 = next hour, 2 = hour after next, etc.)
            total_cfs: Number of counterfactual examples to generate
            method: DiCE method ('random' or 'genetic')
            
        Returns:
            Dictionary with recommendations and counterfactuals for the specific hour
        """
        if hour_offset < 1 or hour_offset > self.horizon:
            raise ValueError(f"hour_offset must be between 1 and {self.horizon}, got {hour_offset}")
        
        # Predict current consumption for this specific hour
        current_prediction_result = self.forecaster.predict_hour(json_data, hour_offset)
        current_prediction = current_prediction_result['electric']
        
        # Check if already below threshold
        if current_prediction <= threshold:
            return {
                'success': True,
                'hour_offset': hour_offset,
                'time': current_prediction_result['time'],
                'current_prediction': float(current_prediction),
                'threshold': float(threshold),
                'below_threshold': True,
                'message': f'Current consumption for hour t+{hour_offset} ({current_prediction:.2f} kWh) is already below threshold ({threshold} kWh)',
                'recommendations': []
            }
        
        # Preprocess input
        X, start_time = preprocess(json_data)
        X_encoded = self.forecaster.encoder.transform(X)
        
        # Prepare query instance for DiCE (for this specific hour)
        query_instance = self._prepare_query_instance_for_hour(json_data, X_encoded, hour_offset)
        
        # Get permitted ranges for actionable features
        permitted_range = self._get_permitted_ranges(query_instance, json_data)
        
        # Check if we have any actionable features
        actionable_features_list = [
            feat for feat in self.get_actionable_features() 
            if feat in query_instance.columns
        ]
        
        if not actionable_features_list:
            return {
                'success': False,
                'hour_offset': hour_offset,
                'time': current_prediction_result['time'],
                'error': 'No actionable features available. Cannot generate counterfactuals.',
                'current_prediction': float(current_prediction),
                'threshold': float(threshold),
                'recommendations': []
            }
        
        # Remove target column from query instance
        target_col = f'target_t+{hour_offset}'
        query_instance_for_dice = query_instance.drop(columns=[target_col], errors='ignore')
        
        # Get DiCE explainer for this specific hour
        explainer = self._get_dice_explainer_for_hour(hour_offset, method)
        
        # Generate counterfactuals
        try:
            # Desired range for counterfactuals
            desired_range_min = max(0.0, threshold * 0.85)
            desired_range_max = threshold
            
            print(f"   📋 Generating recommendations for hour t+{hour_offset} ({current_prediction_result['time']})")
            print(f"   📋 Actionable features: {actionable_features_list}")
            
            # Prepare parameters
            # Try with features_to_vary first, but if it fails, try without it
            cf_params = {
                'query_instances': query_instance_for_dice,
                'total_CFs': total_cfs * 2,
                'desired_range': [desired_range_min, desired_range_max],
                'features_to_vary': actionable_features_list  # Only vary actionable features
            }
            
            if permitted_range:
                cf_params['permitted_range'] = permitted_range
            
            if method == 'genetic':
                cf_params.update({
                    'proximity_weight': 10.0,
                    'diversity_weight': 0.1,
                    'sparsity_weight': 1.0
                })
            
            print(f"   🔍 Generating counterfactuals with method='{method}'...")
            
            # Generate counterfactuals
            counterfactuals = None
            try:
                counterfactuals = explainer.generate_counterfactuals(**cf_params)
            except Exception as e:
                error_msg = str(e)
                # If features_to_vary is too restrictive, try without it but filter results later
                if 'No counterfactuals found' in error_msg or 'features_to_vary' in error_msg.lower():
                    print(f"   ⚠️  Could not find counterfactuals with restricted features. Trying without restriction...")
                    # Remove features_to_vary and try again
                    cf_params_no_restrict = cf_params.copy()
                    cf_params_no_restrict.pop('features_to_vary', None)
                    try:
                        counterfactuals = explainer.generate_counterfactuals(**cf_params_no_restrict)
                        print(f"   ✅ Found counterfactuals without feature restriction (will filter to actionable only)")
                    except Exception as e2:
                        if method == 'genetic':
                            print(f"   ⚠️  Genetic method failed: {e2}")
                            print(f"   🔄 Falling back to 'random' method...")
                            explainer = self._get_dice_explainer_for_hour(hour_offset, 'random')
                            cf_params_no_restrict = cf_params.copy()
                            cf_params_no_restrict.pop('features_to_vary', None)
                            counterfactuals = explainer.generate_counterfactuals(**cf_params_no_restrict)
                        else:
                            raise
                elif method == 'genetic':
                    print(f"   ⚠️  Genetic method failed: {e}")
                    print(f"   🔄 Falling back to 'random' method...")
                    explainer = self._get_dice_explainer_for_hour(hour_offset, 'random')
                    counterfactuals = explainer.generate_counterfactuals(**cf_params)
                else:
                    raise
            
            # Extract counterfactual data
            if not hasattr(counterfactuals, 'cf_examples_list') or not counterfactuals.cf_examples_list:
                return {
                    'success': False,
                    'hour_offset': hour_offset,
                    'time': current_prediction_result['time'],
                    'error': 'No counterfactuals generated by DiCE',
                    'current_prediction': float(current_prediction),
                    'threshold': float(threshold),
                    'recommendations': []
                }
            
            cf_example = counterfactuals.cf_examples_list[0]
            if not hasattr(cf_example, 'final_cfs_df'):
                return {
                    'success': False,
                    'hour_offset': hour_offset,
                    'time': current_prediction_result['time'],
                    'error': 'Counterfactual example has no final_cfs_df attribute',
                    'current_prediction': float(current_prediction),
                    'threshold': float(threshold),
                    'recommendations': []
                }
            
            cf_df = cf_example.final_cfs_df
            
            # Process counterfactuals
            recommendations = []
            for idx, row in cf_df.iterrows():
                # Convert counterfactual back to encoded format for prediction
                cf_dict = row.to_dict()
                
                # IMPORTANT: Restore non-actionable features to original values
                # This ensures DiCE doesn't change features we can't control
                actionable_set = set(actionable_features_list)
                for col in self.feature_cols:
                    if col not in actionable_set and col in query_instance_for_dice.columns:
                        # Restore original value for non-actionable features
                        cf_dict[col] = query_instance_for_dice[col].iloc[0]
                
                # Encode categorical features
                cf_encoded = pd.DataFrame([cf_dict])
                for col in self.categorical_cols:
                    if col in cf_encoded.columns:
                        le = self.forecaster.encoder.encoders[col]
                        values = cf_encoded[col].fillna(self.forecaster.encoder.unknown_token).astype(str)
                        values = values.where(
                            values.isin(le.classes_),
                            self.forecaster.encoder.unknown_token
                        )
                        cf_encoded[col] = le.transform(values)
                
                # Predict with counterfactual using the specific hour model
                cf_prediction = self.forecaster.models[hour_offset].predict(cf_encoded[self.feature_cols].values)[0]
                
                # Calculate changes - ONLY include actionable features
                changes = []
                # actionable_set already defined above
                
                for col in self.feature_cols:
                    # Only process actionable features
                    if col not in actionable_set:
                        continue
                    
                    if col in query_instance_for_dice.columns and col in cf_dict:
                        original_val = query_instance_for_dice[col].iloc[0]
                        cf_val = cf_dict[col]
                        
                        try:
                            if isinstance(original_val, (int, float)) and isinstance(cf_val, (int, float)):
                                change = cf_val - original_val
                                change_pct = (change / original_val * 100) if original_val != 0 else 0
                                
                                if abs(change) > 1e-6:
                                    changes.append({
                                        'feature': col,
                                        'original': original_val,
                                        'new': cf_val,
                                        'change': change,
                                        'change_pct': change_pct,
                                        'action': f"{col}: {original_val:.2f} → {cf_val:.2f} ({change_pct:+.1f}%)",
                                        'description': self.actionable_features.get(col, {}).get('description', '')
                                    })
                            elif isinstance(original_val, str) and isinstance(cf_val, str):
                                if original_val != cf_val:
                                    changes.append({
                                        'feature': col,
                                        'original': original_val,
                                        'new': cf_val,
                                        'change': None,
                                        'change_pct': None,
                                        'action': f"{col}: '{original_val}' → '{cf_val}'",
                                        'description': self.actionable_features.get(col, {}).get('description', '')
                                    })
                        except Exception:
                            pass
                
                # Sort changes by absolute change percentage
                changes.sort(key=lambda x: abs(x.get('change_pct', 0)) if x.get('change_pct') is not None else 0, reverse=True)
                
                reduction = current_prediction - cf_prediction
                reduction_pct = (reduction / current_prediction * 100) if current_prediction > 0 else 0
                
                recommendations.append({
                    'counterfactual_id': idx + 1,
                    'predicted_consumption': float(cf_prediction),
                    'reduction': float(reduction),
                    'reduction_pct': float(reduction_pct),
                    'below_threshold': cf_prediction <= threshold,
                    'changes': changes
                })
            
            # Sort by reduction (descending)
            recommendations.sort(key=lambda x: x['reduction'], reverse=True)
            
            return {
                'success': True,
                'hour_offset': hour_offset,
                'time': current_prediction_result['time'],
                'current_prediction': float(current_prediction),
                'threshold': float(threshold),
                'below_threshold': False,
                'needs_reduction': float(current_prediction - threshold),
                'total_counterfactuals': len(recommendations),
                'recommendations': recommendations
            }
            
        except Exception as e:
            import traceback
            return {
                'success': False,
                'hour_offset': hour_offset,
                'time': current_prediction_result.get('time', ''),
                'error': str(e),
                'error_details': traceback.format_exc(),
                'current_prediction': float(current_prediction),
                'threshold': float(threshold),
                'recommendations': []
            }
    
    def monitor_24_hours(self,
                        json_data: Dict,
                        threshold: float,
                        total_cfs: int = 3,
                        method: str = 'random',
                        only_problematic_hours: bool = True) -> Dict:
        """
        Monitor all 24 hours ahead and generate recommendations for hours that exceed threshold.
        
        Args:
            json_data: Dictionary with building and weather data
            threshold: Target consumption threshold (kWh) for each hour
            total_cfs: Number of counterfactual examples to generate per hour
            method: DiCE method ('random' or 'genetic')
            only_problematic_hours: If True, only generate recommendations for hours exceeding threshold
        
        Returns:
            Dictionary with monitoring results for all 24 hours:
            {
                'success': True,
                'hours_monitored': 24,
                'hours_exceeding_threshold': 5,
                'hours_with_recommendations': 5,
                'hourly_results': [
                    {
                        'hour_offset': 1,
                        'time': '...',
                        'current_prediction': 150.5,
                        'threshold': 120.0,
                        'exceeds_threshold': True,
                        'recommendations': {...} or None
                    },
                    ...
                ]
            }
        """
        print(f"🔍 Monitoring 24 hours ahead from current time...")
        print(f"   Threshold: {threshold:.2f} kWh")
        
        # Get predictions for all 24 hours
        all_predictions = self.forecaster.predict_all_hours(json_data)
        
        hourly_results = []
        hours_exceeding = 0
        hours_with_recommendations = 0
        
        for hour_offset in range(1, self.horizon + 1):
            pred_info = all_predictions[hour_offset]
            current_pred = pred_info['electric']
            exceeds = current_pred > threshold
            
            if exceeds:
                hours_exceeding += 1
            
            result_entry = {
                'hour_offset': hour_offset,
                'time': pred_info['time'],
                'current_prediction': float(current_pred),
                'threshold': float(threshold),
                'exceeds_threshold': exceeds,
                'recommendations': None
            }
            
            # Generate recommendations if exceeds threshold (or if only_problematic_hours is False)
            if exceeds or not only_problematic_hours:
                print(f"\n   📊 Hour t+{hour_offset} ({pred_info['time']}): {current_pred:.2f} kWh {'⚠️ EXCEEDS' if exceeds else '✅ OK'}")
                
                if exceeds:
                    print(f"      Generating recommendations...")
                    recommendations = self.generate_recommendations_for_hour(
                        json_data=json_data,
                        threshold=threshold,
                        hour_offset=hour_offset,
                        total_cfs=total_cfs,
                        method=method
                    )
                    
                    result_entry['recommendations'] = recommendations
                    
                    if recommendations.get('success') and recommendations.get('recommendations'):
                        hours_with_recommendations += 1
                        top_rec = recommendations['recommendations'][0]
                        print(f"      ✅ Top recommendation: {top_rec['predicted_consumption']:.2f} kWh "
                              f"(reduction: {top_rec['reduction']:.2f} kWh, {top_rec['reduction_pct']:.1f}%)")
                    else:
                        print(f"      ⚠️  Could not generate recommendations: {recommendations.get('error', 'Unknown error')}")
            
            hourly_results.append(result_entry)
        
        return {
            'success': True,
            'hours_monitored': self.horizon,
            'hours_exceeding_threshold': hours_exceeding,
            'hours_with_recommendations': hours_with_recommendations,
            'threshold': float(threshold),
            'hourly_results': hourly_results
        }


if __name__ == "__main__":
    # Example usage
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
    
    # Initialize explainer
    explainer = DiceExplainer(
        model_dir=MODEL_DIR,
        encode_path=ENCODE_PATH,
        processed_data_path=DATA_PATH
    )
    
    # Get current prediction
    result = explainer.forecaster(default_input)
    current_pred = result[0]['electric']
    threshold = current_pred * 0.8  # 80% of current
    
    print(f"Current prediction: {current_pred:.2f} kWh")
    print(f"Target threshold: {threshold:.2f} kWh")
    
    # Generate recommendations
    recommendations = explainer.generate_recommendations(
        json_data=default_input,
        threshold=threshold,
        total_cfs=5,
        method='random'
    )
    
    if recommendations['success']:
        print(f"\n✅ Generated {recommendations['total_counterfactuals']} recommendations")
        for rec in recommendations['recommendations'][:3]:
            print(f"\nPredicted: {rec['predicted_consumption']:.2f} kWh")
            print(f"Reduction: {rec['reduction']:.2f} kWh ({rec['reduction_pct']:.1f}%)")
            for change in rec['changes'][:3]:
                print(f"  • {change['action']}")
