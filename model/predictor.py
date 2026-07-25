import pandas as pd


class Predictor:
    def __init__(self, models: list, dates: list):
        self.models = models
        self.dates = dates

    def predict(self, features: pd.DataFrame) -> pd.Series:
        preds = []
        for model, date in zip(self.models, self.dates):
            X = features.loc[features.index.get_level_values(0) == date]
            if len(X) == 0:
                continue
            pred = model.predict(X.values)
            preds.append(pd.Series(pred, index=X.index))
        return pd.concat(preds) if preds else pd.Series(dtype=float)
