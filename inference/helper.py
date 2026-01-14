import os
import joblib
import pandas as pd


def load_models(model_dir: str, horizon: int):
    models = {}
    for h in range(1, horizon + 1):
        path = os.path.join(model_dir, f"model_hour_{h}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing model {path}")
        model = joblib.load(path)
        model.set_params(verbose=-1)  # tắt log GPU
        models[h] = model
    return models


def time_features(ts: str):
    dt = pd.to_datetime(ts)
    return {
        "hour": dt.hour,
        "dayofweek": dt.dayofweek,
        "is_weekend": int(dt.dayofweek >= 5),
        "month": dt.month,
        "datetime": dt
    }
