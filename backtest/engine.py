import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from portfolio.optimizer import CVXPYOptimizer


class BacktestEngine:
    def __init__(self, config: dict):
        self.config = config
        backtest_cfg = config.get("backtest", {})
        self.slippage = backtest_cfg.get("slippage", 0.003)
        self.market_impact = backtest_cfg.get("market_impact", 0.001)
        self.optimizer = CVXPYOptimizer(config.get("portfolio", {}))
        self.lookback = 60
        self.daily_returns_cache = {}

    def _get_covariance(self, prices: pd.DataFrame, date, n_stocks):
        if date not in self.daily_returns_cache:
            daily_ret = prices.pct_change().iloc[-self.lookback:]
            self.daily_returns_cache[date] = daily_ret
        else:
            daily_ret = self.daily_returns_cache[date]

        common = daily_ret.columns.intersection(
            prices.columns if hasattr(prices, 'columns') else prices.index
        )
        if len(common) < 2:
            return np.eye(n_stocks) * 0.01

        lw = LedoitWolf().fit(daily_ret[common].fillna(0))
        cov = pd.DataFrame(lw.covariance_, index=common, columns=common)
        return cov.reindex(index=range(n_stocks), columns=range(n_stocks)).fillna(0).values

    def run(self, predictions: pd.Series, prices: pd.DataFrame,
            benchmark_returns: pd.Series = None, use_optimizer=True) -> tuple:
        dates = sorted(predictions.index.get_level_values(0).unique())
        portfolio_returns = []
        weights_history = []
        current_weights_arr = None

        for i, date in enumerate(dates):
            date_preds = predictions.loc[predictions.index.get_level_values(0) == date]
            if len(date_preds) == 0:
                continue

            n = self.config.get("portfolio", {}).get("top_n", 50)
            top_stocks = date_preds.nlargest(min(n, len(date_preds)))
            stock_names = top_stocks.index.get_level_values(-1)

            if use_optimizer and len(stock_names) >= 5:
                pred_arr = date_preds.loc[top_stocks.index].values
                cov = self._get_covariance(prices, date, len(pred_arr))
                w_arr = self.optimizer.optimize(
                    predicted_returns=pred_arr,
                    cov_matrix=cov,
                    current_weights=current_weights_arr,
                )
                if w_arr is not None:
                    w_arr = w_arr / w_arr.sum()
                    w = pd.Series(w_arr, index=stock_names)
                else:
                    w = pd.Series(1.0 / len(stock_names), index=stock_names)
            else:
                w = pd.Series(1.0 / len(stock_names), index=stock_names)

            cost = 0.0
            if current_weights_arr is not None:
                current_w = current_weights_arr
                new_w = w.reindex(pd.Index(range(len(w))), fill_value=0).values[:len(current_w)]
                turnover_val = np.abs(new_w - current_w[:len(new_w)]).sum() / 2
                cost = turnover_val * (self.slippage + self.market_impact)

            weights_history.append(w)
            current_weights_arr = w.values.copy()

            if i + 1 < len(dates):
                next_date = dates[i + 1]
                ret = prices.loc[next_date] / prices.loc[date] - 1
                ret_aligned = ret.reindex(stock_names).fillna(0).values
                portfolio_ret = float(w.values @ ret_aligned) - cost
                portfolio_returns.append(portfolio_ret)

        returns_idx = dates[1:] if len(dates) > 1 else dates
        returns = pd.Series(portfolio_returns, index=returns_idx)

        if benchmark_returns is not None:
            bench = benchmark_returns.reindex(returns.index).fillna(0)
        else:
            bench = pd.Series(0, index=returns.index)

        weights_df = pd.DataFrame(weights_history, index=dates[:len(weights_history)])
        return returns, bench, weights_df
