import numpy as np
import pandas as pd


class FeatureProcessor:
    @staticmethod
    def winsorize_series(s: pd.Series, limits=(0.01, 0.99)) -> pd.Series:
        lower, upper = s.quantile(limits[0]), s.quantile(limits[1])
        return s.clip(lower, upper)

    @staticmethod
    def winsorize_3sigma(s: pd.Series) -> pd.Series:
        mean, std = s.mean(), s.std()
        return s.clip(mean - 3 * std, mean + 3 * std)

    @staticmethod
    def cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
        return df.subtract(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)

    @staticmethod
    def quantile_transform(s: pd.Series, n_quantiles=100) -> pd.Series:
        ranks = s.rank(method="min")
        return (ranks / ranks.max() * n_quantiles).astype(int)

    def process(self, df: pd.DataFrame, method: str = "3sigma") -> pd.DataFrame:
        result = df.copy()
        for col in result.columns:
            if method == "3sigma":
                result[col] = self.winsorize_3sigma(result[col])
            elif method == "quantile":
                result[col] = self.winsorize_series(result[col])
            result[col] = self.quantile_transform(result[col])
        return self.cs_zscore(result)
