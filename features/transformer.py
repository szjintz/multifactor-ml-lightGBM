import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def build_cross_features(factor_df: pd.DataFrame, market_cap: pd.Series) -> pd.DataFrame:
    result = factor_df.copy()
    for col in factor_df.columns:
        result[f"{col}_x_CAP"] = factor_df[col] * market_cap
    return result


def build_momentum_features(factor_df: pd.DataFrame, windows=[7, 15, 30]) -> pd.DataFrame:
    result = factor_df.copy()
    for col in factor_df.columns:
        for w in windows:
            result[f"{col}_MOM_{w}"] = factor_df[col].rolling(w).mean()
            result[f"{col}_VOL_{w}"] = factor_df[col].rolling(w).std()
    return result


def build_ic_time_series(factor_df: pd.DataFrame, returns: pd.DataFrame,
                         ic_windows=[20, 60]) -> pd.DataFrame:
    result = factor_df.copy()
    factor_arr = factor_df.values if hasattr(factor_df, 'values') else factor_df
    ret_arr = returns.values if hasattr(returns, 'values') else returns
    for col_idx, col in enumerate(factor_df.columns):
        ic_series = []
        for t in range(len(factor_df)):
            f = factor_arr[:t+1, col_idx]
            r = ret_arr[:t+1] if ret_arr.ndim == 1 else ret_arr[:t+1, col_idx]
            if len(f) >= 20:
                ic, _ = spearmanr(f[-20:], r[-20:])
                ic_series.append(ic)
            else:
                ic_series.append(0)
        result[f"{col}_IC"] = ic_series if len(ic_series) == len(factor_df) else 0
    return result
