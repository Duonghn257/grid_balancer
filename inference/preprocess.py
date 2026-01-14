import pandas as pd
from inference.helper import time_features


# feature columns
X_FEATURE_INPUT = [
    "hour","dayofweek","is_weekend","month",
    # "lag_24","rolling_24",
    # "electricity_current",
    "airTemperature", "dewTemperature", "windSpeed",
    # "temp_lag_1h","dewTemperature_lag_1h", "windSpeed_lag_1h",
    # "sqft", 
    "sqm", "primaryspaceusage", "site_id", "building_id",
    "Chilledwater", "Hotwater"
]

def preprocess(raw_input: dict) -> pd.DataFrame:
    tf = time_features(raw_input["time"])

    feat = {
        "hour": tf["hour"],
        "dayofweek": tf["dayofweek"],
        "is_weekend": tf["is_weekend"],
        "month": tf["month"],

        # "lag_24": raw_input["lag_24"],
        # "rolling_24": raw_input["rolling_24"],
        
        "airTemperature": raw_input.get("air_temperature", 0.0),
        "dewTemperature": raw_input.get("dew_temperature", 0.0),
        "windSpeed": raw_input.get("wind_speed", 0.0),

        
        # "temp_lag_1h": raw_input["temp_lag_1h"],
        # "dewTemperature_lag_1h": raw_input["dewTemperature_lag_1h"],
        # "windSpeed_lag_1h": raw_input["windSpeed_lag_1h"],
        
        # "sqft": raw_input.get("sqft", 0.0),
        "sqm": raw_input.get("sqm", 0.0),
        "primaryspaceusage": raw_input.get("primaryspaceusage", "Unknown"),
        "site_id": raw_input.get("site_id", "Unknown"),
        "building_id": raw_input.get("building_code", "Unknown"),
        "Chilledwater": raw_input.get("Chilledwater", 0.0),
        "Hotwater": raw_input.get("Hotwater", 0.0)
    }

    return pd.DataFrame([[feat[col] for col in X_FEATURE_INPUT]],
                        columns=X_FEATURE_INPUT), tf["datetime"]
