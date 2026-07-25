import numpy as np
import pandas as pd


class BacktestEngine:
    def __init__(self, config: dict):
        self.config = config
        backtest_cfg = config.get("backtest", {})
        self.slippage = backtest_cfg.get("slippage", 0.003)
        self.market_impact = backtest_cfg.get("market_impact", 0.001)

    def run(self, predictions: pd.Series, prices: pd.DataFrame,
            benchmark_returns: pd.Series = None) -> tuple:
        dates = sorted(predictions.index.get_level_values(0).unique())
        portfolio_returns = []
        weights_history = []
        current_weights = None

        for i, date in enumerate(dates):
            date_preds = predictions.loc[predictions.index.get_level_values(0) == date]
            if len(date_preds) == 0:
                continue

            n = self.config.get("portfolio", {}).get("top_n", 50)
            top_stocks = date_preds.nlargest(n)

            stock_names = top_stocks.index.get_level_values(-1)
            w = pd.Series(1.0 / len(stock_names), index=stock_names)

            cost = 0.0
            if current_weights is not None:
                current_w = current_weights.reindex(w.index, fill_value=0)
                new_w = w.reindex(current_weights.index, fill_value=0)
                turnover = new_w.sub(current_w).abs().sum() / 2
                cost = turnover * (self.slippage + self.market_impact)

            weights_history.append(w)

            if i + 1 < len(dates):
                next_date = dates[i + 1]
                ret = prices.loc[next_date] / prices.loc[date] - 1
                portfolio_ret = (w * ret.reindex(w.index)).sum() - cost
                portfolio_returns.append(portfolio_ret)

            current_weights = w

        returns_idx = dates[1:] if len(dates) > 1 else dates
        returns = pd.Series(portfolio_returns, index=returns_idx)

        if benchmark_returns is not None:
            bench = benchmark_returns.reindex(returns.index).fillna(0)
        else:
            bench = pd.Series(0, index=returns.index)

        weights_df = pd.DataFrame(weights_history, index=dates)
        return returns, bench, weights_df
