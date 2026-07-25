import pandas as pd
from .base import FactorPipeline
from .preprocessing import winsorize, cross_sectional_standardize, neutralize, orthogonalize


class FactorPreprocessingPipeline:
    def __init__(self, config: dict):
        self.config = config

    def process(self, factor_df: pd.DataFrame, market_data: pd.DataFrame = None) -> pd.DataFrame:
        result = factor_df.copy()

        # Step 1: Winsorize
        for col in result.columns:
            result[col] = winsorize(result[col], self.config.get("winsorize", "3sigma"))

        # Step 2: Cross-sectional standardization
        result = cross_sectional_standardize(result)

        # Step 3: Neutralization
        if self.config.get("neutralize") and market_data is not None:
            exog_cols = self.config["neutralize"]
            exog_data = market_data[[c for c in exog_cols if c in market_data.columns]]
            for col in result.columns:
                result[col] = neutralize(result[col], exog_data)

        # Step 4: Orthogonalization
        if self.config.get("orthogonalize"):
            result = orthogonalize(result, self.config["orthogonalize"])

        return result
