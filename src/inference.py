#!/usr/bin/env python3
"""
Inference Class for Electricity Consumption Prediction
Provides methods to predict electricity consumption for new data
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
import warnings

warnings.filterwarnings('ignore')


class XGBoostWrapper:
    """
    Wrapper class để tự động encode categorical features trước khi predict
    Tương thích với DiCE (Diverse Counterfactual Explanations)
    
    This class must be defined here so pickle can load saved models.
    """
    def __init__(self, model, label_encoders, categorical_features):
        self.model = model
        self.label_encoders = label_encoders
        self.categorical_features = categorical_features
    
    def predict(self, X):
        """Predict với tự động encode categorical features"""
        # Convert to DataFrame nếu là array hoặc Series
        if isinstance(X, np.ndarray):
            # Nếu là array, cần column names
            X = pd.DataFrame(X, columns=self.model.feature_names_in_)
        elif isinstance(X, pd.Series):
            X = X.to_frame().T
        
        X_encoded = X.copy()
        
        # Encode categorical features
        for col in self.categorical_features:
            if col in X_encoded.columns:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    # Chuyển đổi về string và encode
                    X_encoded[col] = X_encoded[col].astype(str)
                    # Xử lý các giá trị chưa thấy (unknown values)
                    mask = ~X_encoded[col].isin(le.classes_)
                    unknown_count = int(np.sum(mask.values)) if isinstance(mask, pd.Series) else int(np.sum(mask))
                    if unknown_count > 0:
                        X_encoded.loc[mask, col] = le.classes_[0]
                    X_encoded[col] = le.transform(X_encoded[col])
                else:
                    # Nếu không có encoder, giữ nguyên (có thể là integer rồi)
                    if X_encoded[col].dtype == 'object':
                        X_encoded[col] = 0
        
        # Đảm bảo tất cả columns là numeric
        for col in X_encoded.columns:
            if X_encoded[col].dtype == 'object':
                X_encoded[col] = pd.to_numeric(X_encoded[col], errors='coerce').fillna(0)
        
        # Đảm bảo thứ tự columns đúng với model
        if hasattr(self.model, 'feature_names_in_'):
            X_encoded = X_encoded.reindex(columns=self.model.feature_names_in_, fill_value=0)
        
        return self.model.predict(X_encoded)


class ElectricityConsumptionInference:
    """
    Inference class for predicting electricity consumption.
    
    This class loads the trained XGBoost model and provides methods to:
    - Predict consumption from new data
    - Handle data preprocessing
    - Provide utility functions for dashboard applications
    """
    
    def __init__(self, 
                 model_path: str = "output/models/xgboost_wrapped_dice.pkl",
                 encoders_path: str = "output/models/label_encoders_dice.pkl",
                 model_info_path: str = "output/models/model_info_dice.json",
                 features_info_path: str = "output/features_info.json",
                 processed_data_path: str = "output/processed_data.parquet"):
        """
        Initialize the inference class.
        
        Args:
            model_path: Path to the trained wrapped model
            encoders_path: Path to label encoders
            model_info_path: Path to model info JSON
            features_info_path: Path to features info JSON
            processed_data_path: Path to processed data (for historical context)
        """
        self.model_path = Path(model_path)
        self.encoders_path = Path(encoders_path)
        self.model_info_path = Path(model_info_path)
        self.features_info_path = Path(features_info_path)
        self.processed_data_path = Path(processed_data_path)
        
        # Load model and encoders
        self._load_model()
        self._load_encoders()
        self._load_info()
        
        # Load historical data for lag features (optional, can be None)
        self._historical_data = None
        try:
            self._historical_data = pd.read_parquet(self.processed_data_path)
            print(f"✅ Loaded historical data: {self._historical_data.shape}")
        except Exception as e:
            print(f"⚠️  Could not load historical data: {e}")
            print("   Lag features will be set to 0 or mean values")
    
    def _load_model(self):
        """Load the trained model."""
        # Register XGBoostWrapper class for pickle to find it
        # This is needed because the class was defined in the training script
        import sys
        import types
        
        # Make XGBoostWrapper available in __main__ module
        # This is where pickle will look for it when loading
        import __main__
        if not hasattr(__main__, 'XGBoostWrapper'):
            __main__.XGBoostWrapper = XGBoostWrapper
        
        # Also register in the scripts module path (where it was originally defined)
        try:
            scripts_module = sys.modules.get('scripts.06_train_xgboost_for_dice')
            if scripts_module is None:
                # Create a mock module
                scripts_module = types.ModuleType('scripts.06_train_xgboost_for_dice')
                sys.modules['scripts.06_train_xgboost_for_dice'] = scripts_module
            scripts_module.XGBoostWrapper = XGBoostWrapper
        except Exception:
            pass
        
        # Try loading the model
        with open(self.model_path, 'rb') as f:
            try:
                self.model = pickle.load(f)
            except AttributeError as e:
                if 'XGBoostWrapper' in str(e):
                    # Try with a custom unpickler that uses our class
                    f.seek(0)
                    # Create a custom unpickler
                    class CustomUnpickler(pickle.Unpickler):
                        def find_class(self, module, name):
                            if name == 'XGBoostWrapper':
                                return XGBoostWrapper
                            return super().find_class(module, name)
                    
                    unpickler = CustomUnpickler(f)
                    self.model = unpickler.load()
                else:
                    raise
        
        print(f"✅ Loaded model from: {self.model_path}")
    
    def _load_encoders(self):
        """Load label encoders."""
        with open(self.encoders_path, 'rb') as f:
            self.label_encoders = pickle.load(f)
        print(f"✅ Loaded {len(self.label_encoders)} label encoders")
    
    def _load_info(self):
        """Load model and features info."""
        with open(self.model_info_path, 'r') as f:
            self.model_info = json.load(f)
        
        with open(self.features_info_path, 'r') as f:
            self.features_info = json.load(f)
        
        # Get feature lists
        self.all_features = self.model_info['continuous_features']
        self.categorical_features = self.model_info['categorical_features']
        self.required_features = self.all_features + self.categorical_features
        
        print(f"✅ Loaded model info: {self.model_info['model_type']}")
        print(f"   - Test R²: {self.model_info['performance']['test_r2']:.4f}")
        print(f"   - Test RMSE: {self.model_info['performance']['test_rmse']:.2f} kWh")
    
    def _create_time_features(self, timestamp: Union[str, datetime, pd.Timestamp]) -> Dict:
        """
        Create time-based features from timestamp.
        
        Args:
            timestamp: Timestamp as string, datetime, or pd.Timestamp
            
        Returns:
            Dictionary of time features
        """
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)
        elif isinstance(timestamp, datetime):
            timestamp = pd.Timestamp(timestamp)
        
        features = {
            'hour': timestamp.hour,
            'day_of_week': timestamp.dayofweek,
            'day_of_month': timestamp.day,
            'month': timestamp.month,
            'year': timestamp.year,
            'is_weekend': 1 if timestamp.dayofweek >= 5 else 0,
        }
        
        # Season
        month = timestamp.month
        if month in [12, 1, 2]:
            features['season'] = 'Winter'
        elif month in [3, 4, 5]:
            features['season'] = 'Spring'
        elif month in [6, 7, 8]:
            features['season'] = 'Summer'
        else:
            features['season'] = 'Fall'
        
        # Cyclical encoding
        features['hour_sin'] = np.sin(2 * np.pi * timestamp.hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * timestamp.hour / 24)
        features['day_of_week_sin'] = np.sin(2 * np.pi * timestamp.dayofweek / 7)
        features['day_of_week_cos'] = np.cos(2 * np.pi * timestamp.dayofweek / 7)
        features['month_sin'] = np.sin(2 * np.pi * timestamp.month / 12)
        features['month_cos'] = np.cos(2 * np.pi * timestamp.month / 12)
        
        return features
    
    def _get_lag_features(self, building_id: str, timestamp: Union[str, datetime, pd.Timestamp]) -> Dict:
        """
        Get lag features from historical data.
        
        Args:
            building_id: Building ID
            timestamp: Current timestamp
            
        Returns:
            Dictionary of lag features
        """
        lag_features = {
            'electricity_lag1': 0.0,
            'electricity_lag24': 0.0,
            'electricity_lag168': 0.0,
            'electricity_rolling_mean_24h': 0.0,
            'electricity_rolling_std_24h': 0.0,
            'electricity_rolling_mean_7d': 0.0
        }
        
        if self._historical_data is None:
            return lag_features
        
        try:
            if isinstance(timestamp, str):
                timestamp = pd.to_datetime(timestamp)
            elif isinstance(timestamp, datetime):
                timestamp = pd.Timestamp(timestamp)
            
            # Filter data for this building
            building_data = self._historical_data[
                self._historical_data['building_id'] == building_id
            ].sort_values('timestamp')
            
            if len(building_data) == 0:
                return lag_features
            
            # Get lag values
            current_idx = building_data[building_data['timestamp'] <= timestamp]
            if len(current_idx) > 0:
                # Lag 1 hour
                if len(current_idx) > 1:
                    lag_features['electricity_lag1'] = current_idx.iloc[-2]['electricity_consumption'] if 'electricity_consumption' in current_idx.columns else 0.0
                
                # Lag 24 hours
                target_time = timestamp - timedelta(hours=24)
                lag24_data = building_data[building_data['timestamp'] <= target_time]
                if len(lag24_data) > 0:
                    lag_features['electricity_lag24'] = lag24_data.iloc[-1]['electricity_consumption'] if 'electricity_consumption' in lag24_data.columns else 0.0
                
                # Lag 168 hours (1 week)
                target_time = timestamp - timedelta(hours=168)
                lag168_data = building_data[building_data['timestamp'] <= target_time]
                if len(lag168_data) > 0:
                    lag_features['electricity_lag168'] = lag168_data.iloc[-1]['electricity_consumption'] if 'electricity_consumption' in lag168_data.columns else 0.0
                
                # Rolling statistics (last 24 hours)
                last_24h = current_idx.tail(24)
                if len(last_24h) > 0 and 'electricity_consumption' in last_24h.columns:
                    lag_features['electricity_rolling_mean_24h'] = last_24h['electricity_consumption'].mean()
                    lag_features['electricity_rolling_std_24h'] = last_24h['electricity_consumption'].std() if len(last_24h) > 1 else 0.0
                
                # Rolling mean 7 days (168 hours)
                last_7d = current_idx.tail(168)
                if len(last_7d) > 0 and 'electricity_consumption' in last_7d.columns:
                    lag_features['electricity_rolling_mean_7d'] = last_7d['electricity_consumption'].mean()
            
            # If no historical data, use mean from all buildings
            if all(v == 0.0 for v in lag_features.values()):
                if 'electricity_consumption' in self._historical_data.columns:
                    mean_consumption = self._historical_data['electricity_consumption'].mean()
                    lag_features['electricity_rolling_mean_24h'] = mean_consumption
                    lag_features['electricity_rolling_mean_7d'] = mean_consumption
        
        except Exception as e:
            print(f"⚠️  Error getting lag features: {e}")
        
        return lag_features
    
    def _preprocess_input(self, data: Dict, include_lag: bool = True) -> pd.DataFrame:
        """
        Preprocess input data for prediction.
        
        Args:
            data: Dictionary containing building and weather data
            include_lag: Whether to include lag features (requires historical data)
            
        Returns:
            Preprocessed DataFrame ready for prediction
        """
        # Start with time features
        timestamp = data.get('time') or data.get('timestamp')
        if timestamp is None:
            raise ValueError("'time' or 'timestamp' is required in input data")
        
        time_features = self._create_time_features(timestamp)
        
        # Get lag features if requested
        lag_features = {}
        if include_lag:
            # Check if lag features are already provided in data (e.g., from previous predictions)
            lag_feature_names = ['electricity_lag1', 'electricity_lag24', 'electricity_lag168',
                                'electricity_rolling_mean_24h', 'electricity_rolling_std_24h',
                                'electricity_rolling_mean_7d']
            if any(feat in data for feat in lag_feature_names):
                # Use provided lag features
                for feat in lag_feature_names:
                    if feat in data:
                        lag_features[feat] = data[feat]
            else:
                # Get lag features from historical data
                building_id = data.get('building_id') or data.get('building_code')
                if building_id:
                    lag_features = self._get_lag_features(building_id, timestamp)
        
        # Map input data to feature names (handle different naming conventions)
        feature_mapping = {
            'air_temperature': 'airTemperature',
            'cloud_coverage': 'cloudCoverage',
            'dew_temperature': 'dewTemperature',
            'wind_speed': 'windSpeed',
            'sea_lvl_pressure': 'seaLvlPressure',
            'precip_depth_1hr': 'precipDepth1HR',
            'building_code': 'building_id',
        }
        
        # Create feature dictionary
        features = {}
        
        # Continuous features
        for feat in self.all_features:
            if feat in time_features:
                features[feat] = time_features[feat]
            elif feat in lag_features:
                features[feat] = lag_features[feat]
            else:
                # Try direct mapping
                value = data.get(feat)
                if value is None:
                    # Try alternative names
                    for alt_name, mapped_name in feature_mapping.items():
                        if mapped_name == feat:
                            value = data.get(alt_name)
                            break
                
                # Fill missing with 0 or median (you might want to use actual medians)
                if value is None:
                    # Use default values based on feature type
                    if feat in ['sqm', 'yearbuilt', 'numberoffloors', 'occupants']:
                        features[feat] = 0.0  # Building features - should be provided
                    elif feat in ['airTemperature', 'cloudCoverage', 'dewTemperature', 
                                  'windSpeed', 'seaLvlPressure', 'precipDepth1HR']:
                        features[feat] = 0.0  # Weather features - should be provided
                    else:
                        features[feat] = 0.0
                else:
                    features[feat] = float(value) if pd.notna(value) else 0.0
        
        # Categorical features
        for feat in self.categorical_features:
            value = data.get(feat)
            if value is None:
                # Try to infer from other fields
                if feat == 'season' and 'season' in time_features:
                    value = time_features['season']
                else:
                    value = 'Unknown'
            
            features[feat] = str(value) if value is not None else 'Unknown'
        
        # Create DataFrame
        df = pd.DataFrame([features])
        
        # Encode categorical features
        for col in self.categorical_features:
            if col in df.columns and col in self.label_encoders:
                le = self.label_encoders[col]
                # Handle unknown values
                if df[col].iloc[0] not in le.classes_:
                    df[col] = le.classes_[0]  # Use first class as default
                else:
                    df[col] = le.transform([df[col].iloc[0]])[0]
        
        # Ensure correct column order
        if hasattr(self.model.model, 'feature_names_in_'):
            df = df.reindex(columns=self.model.model.feature_names_in_, fill_value=0)
        else:
            df = df.reindex(columns=self.required_features, fill_value=0)
        
        return df
    
    def predict(self, data: Union[Dict, pd.DataFrame], include_lag: bool = True) -> float:
        """
        Predict electricity consumption for a single data point.
        
        Args:
            data: Dictionary or DataFrame with building and weather data
            include_lag: Whether to include lag features
            
        Returns:
            Predicted electricity consumption in kWh
        """
        if isinstance(data, pd.DataFrame):
            # If DataFrame, convert first row to dict
            data = data.iloc[0].to_dict()
        
        X = self._preprocess_input(data, include_lag=include_lag)
        prediction = self.model.predict(X)[0]
        return max(0.0, prediction)  # Ensure non-negative
    
    def predict_batch(self, data_list: List[Dict], include_lag: bool = True) -> np.ndarray:
        """
        Predict electricity consumption for multiple data points.
        
        Args:
            data_list: List of dictionaries with building and weather data
            include_lag: Whether to include lag features
            
        Returns:
            Array of predictions
        """
        X_list = []
        for data in data_list:
            X = self._preprocess_input(data, include_lag=include_lag)
            X_list.append(X)
        
        X_batch = pd.concat(X_list, ignore_index=True)
        predictions = self.model.predict(X_batch)
        return np.maximum(0.0, predictions)  # Ensure non-negative
    
    def predict_from_json(self, json_data: Dict) -> Dict:
        """
        Predict from JSON data and return detailed result.
        
        Args:
            json_data: Dictionary with building and weather data
            
        Returns:
            Dictionary with prediction and metadata
        """
        try:
            prediction = self.predict(json_data)
            
            return {
                'success': True,
                'prediction': float(prediction),
                'unit': 'kWh',
                'timestamp': json_data.get('time') or json_data.get('timestamp'),
                'building_id': json_data.get('building_id') or json_data.get('building_code'),
                'model_info': {
                    'model_type': self.model_info['model_type'],
                    'test_r2': self.model_info['performance']['test_r2'],
                    'test_rmse': self.model_info['performance']['test_rmse']
                }
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'prediction': None
            }
    
    def predict_with_threshold(self, json_data: Dict, threshold: float) -> Dict:
        """
        Predict and classify based on threshold.
        
        Args:
            json_data: Dictionary with building and weather data
            threshold: Threshold for classification
            
        Returns:
            Dictionary with prediction and classification
        """
        result = self.predict_from_json(json_data)
        
        if result['success']:
            prediction = result['prediction']
            result['threshold'] = threshold
            result['classification'] = 'high' if prediction > threshold else 'low'
            result['is_above_threshold'] = prediction > threshold
        
        return result
    
    def validate_input(self, json_data: Dict) -> Dict:
        """
        Validate input data for prediction.
        
        Args:
            json_data: Dictionary with building and weather data
            
        Returns:
            Dictionary with validation results
        """
        errors = []
        warnings_list = []
        
        # Required fields
        required_fields = ['time', 'building_id']
        for field in required_fields:
            if field not in json_data and field.replace('_', '') not in json_data:
                errors.append(f"Missing required field: {field}")
        
        # Important fields (warnings if missing)
        important_fields = {
            'sqm': 'Building area',
            'airTemperature': 'Air temperature',
            'primaryspaceusage': 'Primary space usage',
            'site_id': 'Site ID'
        }
        
        for field, description in important_fields.items():
            if field not in json_data:
                # Try alternative names
                alt_names = {
                    'airTemperature': ['air_temperature', 'temperature'],
                    'primaryspaceusage': ['primary_space_usage', 'usage_type']
                }
                found = False
                if field in alt_names:
                    for alt in alt_names[field]:
                        if alt in json_data:
                            found = True
                            break
                
                if not found:
                    warnings_list.append(f"Missing important field: {description} ({field})")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings_list
        }
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance from the model.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if hasattr(self.model.model, 'feature_importances_'):
            importances = self.model.model.feature_importances_
            feature_names = self.model.model.feature_names_in_
        else:
            # Fallback if feature names not available
            importances = np.zeros(len(self.required_features))
            feature_names = self.required_features
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        return df
    
    def predict_with_confidence_interval(self, json_data: Dict, confidence: float = 0.95) -> Dict:
        """
        Predict with confidence interval (using model's uncertainty estimation).
        
        Note: XGBoost doesn't provide uncertainty estimates directly.
        This is a simplified version using model performance metrics.
        
        Args:
            json_data: Dictionary with building and weather data
            confidence: Confidence level (0.95 = 95%)
            
        Returns:
            Dictionary with prediction and confidence interval
        """
        result = self.predict_from_json(json_data)
        
        if result['success']:
            # Use test RMSE as uncertainty estimate
            test_rmse = self.model_info['performance']['test_rmse']
            
            # For 95% confidence, use ~2 standard deviations
            z_score = 1.96 if confidence == 0.95 else 2.576  # 99% confidence
            
            prediction = result['prediction']
            margin = z_score * test_rmse
            
            result['confidence_interval'] = {
                'lower': max(0.0, prediction - margin),
                'upper': prediction + margin,
                'confidence': confidence,
                'margin': margin
            }
        
        return result
    
    def get_prediction_explanation(self, json_data: Dict) -> Dict:
        """
        Get explanation for prediction (feature contributions).
        
        Args:
            json_data: Dictionary with building and weather data
            
        Returns:
            Dictionary with prediction explanation
        """
        result = self.predict_from_json(json_data)
        
        if result['success']:
            # Get feature importance
            importance_df = self.get_feature_importance(top_n=10)
            
            # Get preprocessed features
            X = self._preprocess_input(json_data, include_lag=False)
            
            explanation = {
                'prediction': result['prediction'],
                'top_features': importance_df.to_dict('records'),
                'input_features': X.iloc[0].to_dict(),
                'model_performance': {
                    'test_r2': self.model_info['performance']['test_r2'],
                    'test_rmse': self.model_info['performance']['test_rmse']
                }
            }
            
            result['explanation'] = explanation
        
        return result
    
    def predict_future(self, 
                      building_id: str,
                      start_time: Union[str, datetime],
                      hours: int = 24,
                      weather_data: Optional[List[Dict]] = None,
                      building_data: Optional[Dict] = None) -> pd.DataFrame:
        """
        Predict electricity consumption for future hours.
        
        Args:
            building_id: Building ID
            start_time: Start timestamp
            hours: Number of hours to predict
            weather_data: Optional list of weather data for each hour
            building_data: Optional building metadata
            
        Returns:
            DataFrame with predictions for each hour
        """
        if isinstance(start_time, str):
            start_time = pd.to_datetime(start_time)
        elif isinstance(start_time, datetime):
            start_time = pd.Timestamp(start_time)
        
        # Load building data if not provided
        if building_data is None:
            if self._historical_data is not None:
                building_info = self._historical_data[
                    self._historical_data['building_id'] == building_id
                ].iloc[0].to_dict() if len(self._historical_data[
                    self._historical_data['building_id'] == building_id
                ]) > 0 else {}
            else:
                building_info = {}
        else:
            building_info = building_data.copy()
        
        building_info['building_id'] = building_id
        
        predictions = []
        previous_predictions = []  # Store previous predictions for lag features
        
        for i in range(hours):
            current_time = start_time + timedelta(hours=i)
            
            # Prepare data for this hour
            data = building_info.copy()
            data['time'] = current_time
            
            # Add weather data if provided
            if weather_data and i < len(weather_data):
                weather = weather_data[i]
                data.update({
                    'airTemperature': weather.get('airTemperature') or weather.get('air_temperature'),
                    'cloudCoverage': weather.get('cloudCoverage') or weather.get('cloud_coverage'),
                    'dewTemperature': weather.get('dewTemperature') or weather.get('dew_temperature'),
                    'windSpeed': weather.get('windSpeed') or weather.get('wind_speed'),
                    'seaLvlPressure': weather.get('seaLvlPressure') or weather.get('sea_lvl_pressure'),
                    'precipDepth1HR': weather.get('precipDepth1HR') or weather.get('precip_depth_1hr'),
                })
            
            # Use previous predictions as lag features if available
            if i > 0 and len(previous_predictions) > 0:
                # Override lag features with previous predictions
                data['electricity_lag1'] = previous_predictions[-1] if len(previous_predictions) >= 1 else 0.0
                data['electricity_lag24'] = previous_predictions[-24] if len(previous_predictions) >= 24 else 0.0
                data['electricity_lag168'] = previous_predictions[-168] if len(previous_predictions) >= 168 else 0.0
                
                # Rolling statistics
                if len(previous_predictions) >= 24:
                    last_24 = previous_predictions[-24:]
                    data['electricity_rolling_mean_24h'] = np.mean(last_24)
                    data['electricity_rolling_std_24h'] = np.std(last_24) if len(last_24) > 1 else 0.0
                else:
                    data['electricity_rolling_mean_24h'] = np.mean(previous_predictions) if previous_predictions else 0.0
                    data['electricity_rolling_std_24h'] = 0.0
                
                if len(previous_predictions) >= 168:
                    last_168 = previous_predictions[-168:]
                    data['electricity_rolling_mean_7d'] = np.mean(last_168)
                else:
                    data['electricity_rolling_mean_7d'] = np.mean(previous_predictions) if previous_predictions else 0.0
            
            # Predict (include historical lag only for first prediction)
            include_lag = (i == 0)
            prediction = self.predict(data, include_lag=include_lag)
            
            # Store prediction for future lag features
            previous_predictions.append(prediction)
            
            predictions.append({
                'building_id': building_id,
                'timestamp': current_time,
                'predicted_consumption': prediction,
                'hour': i + 1
            })
        
        return pd.DataFrame(predictions)
    
    def predict_by_city(self, 
                       site_id: str,
                       timestamp: Union[str, datetime],
                       weather_data: Optional[Dict] = None) -> pd.DataFrame:
        """
        Predict electricity consumption for all buildings in a city/site.
        
        Args:
            site_id: Site/City ID
            timestamp: Timestamp for prediction
            weather_data: Optional weather data for the site
            
        Returns:
            DataFrame with predictions for all buildings in the site
        """
        if self._historical_data is None:
            raise ValueError("Historical data is required for city-level predictions")
        
        # Get all buildings in this site
        site_buildings = self._historical_data[
            self._historical_data['site_id'] == site_id
        ]['building_id'].unique()
        
        if len(site_buildings) == 0:
            return pd.DataFrame()
        
        predictions = []
        
        for building_id in site_buildings:
            # Get building info
            building_info = self._historical_data[
                self._historical_data['building_id'] == building_id
            ].iloc[0].to_dict()
            
            # Add weather data if provided
            if weather_data:
                building_info.update(weather_data)
            
            building_info['time'] = timestamp
            
            try:
                prediction = self.predict(building_info, include_lag=True)
                
                predictions.append({
                    'building_id': building_id,
                    'site_id': site_id,
                    'timestamp': timestamp,
                    'predicted_consumption': prediction,
                    'primaryspaceusage': building_info.get('primaryspaceusage'),
                    'sqm': building_info.get('sqm')
                })
            except Exception as e:
                print(f"⚠️  Error predicting for {building_id}: {e}")
        
        return pd.DataFrame(predictions)
    
    def get_city_summary(self, 
                        site_id: str,
                        timestamp: Union[str, datetime],
                        weather_data: Optional[Dict] = None) -> Dict:
        """
        Get summary statistics for a city/site.
        
        Args:
            site_id: Site/City ID
            timestamp: Timestamp for prediction
            weather_data: Optional weather data for the site
            
        Returns:
            Dictionary with city-level summary statistics
        """
        predictions_df = self.predict_by_city(site_id, timestamp, weather_data)
        
        if len(predictions_df) == 0:
            return {
                'site_id': site_id,
                'timestamp': str(timestamp),
                'total_buildings': 0,
                'total_consumption': 0.0,
                'average_consumption': 0.0
            }
        
        summary = {
            'site_id': site_id,
            'timestamp': str(timestamp),
            'total_buildings': len(predictions_df),
            'total_consumption': float(predictions_df['predicted_consumption'].sum()),
            'average_consumption': float(predictions_df['predicted_consumption'].mean()),
            'median_consumption': float(predictions_df['predicted_consumption'].median()),
            'min_consumption': float(predictions_df['predicted_consumption'].min()),
            'max_consumption': float(predictions_df['predicted_consumption'].max()),
            'std_consumption': float(predictions_df['predicted_consumption'].std()),
            'by_usage_type': {
                usage_type: {
                    'count': int(group['predicted_consumption'].count()),
                    'total': float(group['predicted_consumption'].sum()),
                    'average': float(group['predicted_consumption'].mean())
                }
                for usage_type, group in predictions_df.groupby('primaryspaceusage')
            }
        }
        
        return summary
