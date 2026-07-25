import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def compute_ic(factor: pd.Series, forward_return: pd.Series) -> float:
    valid = factor.notna() & forward_return.notna()
    if valid.sum() < 30:
        return 0.0
    return spearmanr(factor[valid], forward_return[valid])[0]


def compute_icir(ic_series: pd.Series) -> float:
    if len(ic_series) < 5:
        return 0.0
    return ic_series.mean() / ic_series.std()


def ic_prefilter(factor_df: pd.DataFrame, returns: pd.DataFrame,
                 min_ic=0.02, min_icir=0.5, p_threshold=0.05) -> list[str]:
    selected = []
    for col in factor_df.columns:
        ic_values = []
        for date in factor_df.index.levels[0][:min(252, len(factor_df.index.levels[0]))]:
            try:
                f = factor_df.loc[date, col]
                r = returns.loc[date]
                ic = compute_ic(f, r)
                ic_values.append(ic)
            except (KeyError, AttributeError):
                continue
        ic_series = pd.Series(ic_values)
        mean_ic = ic_series.mean()
        icir = compute_icir(ic_series)
        if abs(mean_ic) > min_ic and abs(icir) > min_icir:
            selected.append(col)
    return selected
