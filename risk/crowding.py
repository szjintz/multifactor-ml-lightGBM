import pandas as pd


class CrowdingMonitor:
    def __init__(self, lookback=60):
        self.lookback = lookback

    def compute_crowding(self, factor_returns: pd.DataFrame, flow_data: pd.DataFrame = None) -> pd.Series:
        if flow_data is not None:
            crowding = {}
            for col in factor_returns.columns:
                if col in flow_data.columns:
                    corr = factor_returns[col].rolling(self.lookback).corr(flow_data[col])
                    crowding[col] = float(corr.abs().mean())
                else:
                    crowding[col] = 0.0
            return pd.Series(crowding)
        return pd.Series(0.0, index=factor_returns.columns)

    def filter_crowded(self, factor_list: list, crowding_scores: pd.Series, threshold=0.6) -> list:
        return [f for f in factor_list if crowding_scores.get(f, 0) < threshold]
