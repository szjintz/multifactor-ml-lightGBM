import numpy as np
import pandas as pd
from ..backtest.engine import BacktestEngine


class MonteCarloSimulator:
    def __init__(self, config: dict = None):
        self.config = config or {}
        mc_cfg = self.config.get("monte_carlo", {})
        self.noise_levels = mc_cfg.get("noise_levels", [0.001, 0.005, 0.01, 0.02, 0.05])
        self.n_simulations = mc_cfg.get("n_simulations", 200)
        self.seed = mc_cfg.get("seed", 42)

    def run(self, factor_df: pd.DataFrame, prices: pd.DataFrame,
            benchmark_returns: pd.Series = None,
            models: list = None, dates: list = None) -> dict:
        results = {}
        for noise_level in self.noise_levels:
            sharpe_list = []
            max_dd_list = []
            rng = np.random.default_rng(self.seed)

            for _ in range(self.n_simulations):
                noise = rng.normal(0, noise_level, factor_df.shape)
                perturbed = factor_df * (1 + noise)
                perturbed.index = factor_df.index
                perturbed.columns = factor_df.columns

                if models and dates:
                    predictions_list = []
                    for model, date in zip(models, dates):
                        mask = perturbed.index.get_level_values(0) == date
                        X = perturbed[mask]
                        if len(X) == 0:
                            continue
                        pred = model.predict(X.values)
                        predictions_list.append(pd.Series(pred, index=X.index))
                    predictions = pd.concat(predictions_list) if predictions_list else pd.Series(dtype=float)
                else:
                    predictions = pd.Series(
                        perturbed.mean(axis=1).values,
                        index=factor_df.index
                    )

                engine = BacktestEngine(self.config)
                returns, bench, weights = engine.run(predictions, prices, benchmark_returns)

                sharpe_list.append(self._compute_sharpe(returns))
                max_dd_list.append(self._compute_max_dd(returns))

            results[noise_level] = {
                "sharpe_10pct": float(np.percentile(sharpe_list, 10)),
                "sharpe_50pct": float(np.percentile(sharpe_list, 50)),
                "sharpe_90pct": float(np.percentile(sharpe_list, 90)),
                "max_dd_10pct": float(np.percentile(max_dd_list, 10)),
                "max_dd_50pct": float(np.percentile(max_dd_list, 50)),
                "max_dd_90pct": float(np.percentile(max_dd_list, 90)),
                "collapse_prob": float(
                    sum(1 for dd in max_dd_list if dd < -0.20) / len(max_dd_list)
                ),
            }

        return results

    def _compute_sharpe(self, returns: pd.Series) -> float:
        if len(returns) < 10:
            return 0.0
        excess = returns - 0.025 / 252
        return float(np.sqrt(252) * excess.mean() / excess.std()) if excess.std() > 0 else 0.0

    def _compute_max_dd(self, returns: pd.Series) -> float:
        equity = (1 + returns).cumprod()
        peak = equity.expanding().max()
        dd = (equity - peak) / peak
        return float(dd.min())

    def report(self, results: dict) -> str:
        lines = ["=== 蒙特卡洛模拟结果 ==="]
        for level, stats in results.items():
            lines.append(f"\n噪声水平: {level*100:.1f}%")
            lines.append(f"  Sharpe: [{stats['sharpe_10pct']:.2f}, {stats['sharpe_50pct']:.2f}, {stats['sharpe_90pct']:.2f}]")
            lines.append(f"  最大回撤: [{stats['max_dd_10pct']:.1%}, {stats['max_dd_50pct']:.1%}, {stats['max_dd_90pct']:.1%}]")
            lines.append(f"  崩塌概率: {stats['collapse_prob']:.1%}")
        return "\n".join(lines)
