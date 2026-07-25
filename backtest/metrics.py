import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def compute_sharpe(returns: pd.Series, rf: float = 0.025) -> float:
    excess = returns - rf / 252
    return np.sqrt(252) * excess.mean() / excess.std() if excess.std() > 0 else 0.0


def compute_max_drawdown(equity: pd.Series) -> float:
    peak = equity.expanding().max()
    dd = (equity - peak) / peak
    return dd.min()


def compute_calmar(returns: pd.Series) -> float:
    ann_ret = (1 + returns).prod() ** (252 / len(returns)) - 1
    equity = (1 + returns).cumprod()
    mdd = compute_max_drawdown(equity)
    return ann_ret / abs(mdd) if mdd != 0 else 0.0


def compute_ic(pred: pd.Series, actual: pd.Series) -> float:
    valid = pred.notna() & actual.notna()
    if valid.sum() < 10:
        return 0.0
    return spearmanr(pred[valid], actual[valid])[0]


def compute_ir(ic_series: pd.Series) -> float:
    return ic_series.mean() / ic_series.std() if ic_series.std() > 0 else 0.0


def compute_turnover(weights_history: pd.DataFrame) -> float:
    if len(weights_history) < 2:
        return 0.0
    to = weights_history.diff().abs().sum(axis=1)
    return to.mean()


class MetricsReport:
    def __init__(self, returns: pd.Series, benchmark: pd.Series,
                 predictions: pd.Series, actuals: pd.Series,
                 weights: pd.DataFrame):
        self.returns = returns
        self.benchmark = benchmark
        self.predictions = predictions
        self.actuals = actuals
        self.weights = weights
        self.excess = returns - benchmark

    def generate(self) -> dict:
        equity = (1 + self.returns).cumprod()
        ann_ret = (1 + self.returns).prod() ** (252 / len(self.returns)) - 1
        bench_ann_ret = (1 + self.benchmark).prod() ** (252 / len(self.benchmark)) - 1

        ic_values = []
        for date in self.predictions.index.get_level_values(0).unique():
            pred_date = self.predictions.loc[self.predictions.index.get_level_values(0) == date]
            actual_date = self.actuals.loc[self.actuals.index.get_level_values(0) == date]
            if len(pred_date) > 10 and len(actual_date) > 10:
                ic_values.append(compute_ic(pred_date, actual_date))

        ic_series = pd.Series(ic_values)

        return {
            "annualized_return": ann_ret,
            "benchmark_return": bench_ann_ret,
            "excess_return": ann_ret - bench_ann_ret,
            "annualized_vol": self.returns.std() * np.sqrt(252),
            "sharpe_ratio": compute_sharpe(self.returns),
            "calmar_ratio": compute_calmar(self.returns),
            "max_drawdown": compute_max_drawdown(equity),
            "annual_turnover": compute_turnover(self.weights),
            "ic_mean": ic_series.mean(),
            "ic_std": ic_series.std(),
            "ir": compute_ir(ic_series),
            "hit_rate": (ic_series > 0).mean(),
        }
