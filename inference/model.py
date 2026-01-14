from inference.helper import load_models
from inference.preprocess import preprocess
from inference.postprocess import postprocess
from inference.encoder import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class ElectricityForecaster:
    def __init__(
        self,
        model_dir: str,
        encode_path: str = None,
        forecast_horizon: int = 24
    ):
        self.encoder = LabelEncoder.load(encode_path)
        self.model_dir = model_dir
        self.horizon = forecast_horizon
        self.models = self.load_models()

    def load_models(self):
        return load_models(self.model_dir, self.horizon)

    def __call__(self, raw_input: dict):
        X, start_time = preprocess(raw_input)
        print(X)
        # print("____"*5)
        X = self.encoder.transform(X)
        # print(X)
        preds = {}
        for h, model in self.models.items():
            preds[h] = model.predict(X)[0]

        return postprocess(start_time, preds)
    
    def predict_hour(self, raw_input: dict, hour_offset: int):
        """
        Predict electricity consumption for a specific hour ahead.
        
        Args:
            raw_input: Dictionary with building and weather data
            hour_offset: Hours ahead to predict (1-24, where 1 = next hour, 2 = hour after next, etc.)
        
        Returns:
            Dictionary with prediction details:
            {
                'hour_offset': hour_offset,
                'time': ISO format timestamp,
                'electric': predicted consumption (kWh)
            }
        """
        if hour_offset < 1 or hour_offset > self.horizon:
            raise ValueError(f"hour_offset must be between 1 and {self.horizon}, got {hour_offset}")
        
        X, start_time = preprocess(raw_input)
        X = self.encoder.transform(X)
        
        # Get the model for this specific hour
        model = self.models[hour_offset]
        prediction = model.predict(X)[0]
        
        from datetime import timedelta
        target_time = start_time + timedelta(hours=hour_offset)
        
        return {
            'hour_offset': hour_offset,
            'time': target_time.isoformat(),
            'electric': float(prediction)
        }
    
    def predict_all_hours(self, raw_input: dict):
        """
        Predict electricity consumption for all 24 hours ahead.
        Returns predictions in a structured format.
        
        Args:
            raw_input: Dictionary with building and weather data
        
        Returns:
            Dictionary with hour_offset as keys:
            {
                1: {'hour_offset': 1, 'time': '...', 'electric': 100.5},
                2: {'hour_offset': 2, 'time': '...', 'electric': 105.2},
                ...
            }
        """
        X, start_time = preprocess(raw_input)
        X = self.encoder.transform(X)
        
        predictions = {}
        from datetime import timedelta
        
        for h, model in self.models.items():
            prediction = model.predict(X)[0]
            target_time = start_time + timedelta(hours=h)
            predictions[h] = {
                'hour_offset': h,
                'time': target_time.isoformat(),
                'electric': float(prediction)
            }
        
        return predictions
    
    def get_preprocessed_features(self, raw_input: dict):
        """
        Get preprocessed and encoded features for a given input.
        Useful for DiCE and other analysis tools.
        
        Args:
            raw_input: Dictionary with building and weather data
        
        Returns:
            Tuple of (encoded_dataframe, start_time)
        """
        X, start_time = preprocess(raw_input)
        X_encoded = self.encoder.transform(X)
        return X_encoded, start_time

    
if __name__ == "__main__":
    MODEL_DIR = "./models_1578_csv"
    encode_path = "./models_1578_csv/categorical_encoder.pkl"
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

    forecaster = ElectricityForecaster(MODEL_DIR, encode_path, forecast_horizon=24)

    result = forecaster(default_input)
    
    import pandas as pd
    df = pd.DataFrame(result)
    df["time"] = pd.to_datetime(df["time"])

    # print(df)