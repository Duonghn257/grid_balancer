import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder


class LabelEncoder:
    def __init__(self, categorical_cols, unknown_token="unknown"):
        self.categorical_cols = categorical_cols
        self.unknown_token = unknown_token
        self.encoders = {}

    def fit(self, df: pd.DataFrame):
        for col in self.categorical_cols:
            le = LabelEncoder()

            values = (
                df[col]
                .fillna(self.unknown_token)
                .astype(str)
            )

            classes = values.unique().tolist()
            if self.unknown_token not in classes:
                classes.append(self.unknown_token)

            le.fit(classes)
            self.encoders[col] = le
        return self

    def transform(self, df: pd.DataFrame):
        df = df.copy()
        for col, le in self.encoders.items():
            values = (
                df[col]
                .fillna(self.unknown_token)
                .astype(str)
            )

            values = values.where(
                values.isin(le.classes_),
                self.unknown_token
            )

            df[col] = le.transform(values)
        return df

    def fit_transform(self, df: pd.DataFrame):
        self.fit(df)
        return self.transform(df)

    def save(self, path: str):
        joblib.dump({
            "categorical_cols": self.categorical_cols,
            "unknown_token": self.unknown_token,
            "encoders": self.encoders
        }, path)

    @classmethod
    def load(cls, path: str):
        data = joblib.load(path)
        obj = cls(
            categorical_cols=data["categorical_cols"],
            unknown_token=data["unknown_token"]
        )
        obj.encoders = data["encoders"]
        return obj
