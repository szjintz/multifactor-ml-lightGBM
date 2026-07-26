"""
蒙特卡洛模拟模块

通过向因子数据添加随机噪声，评估策略的稳健性。

方法论：
1. 对每个噪声水平 ε ∈ {0.1%, 0.5%, 1%, 2%, 5%}
2. 对原始因子添加 N(0, ε·σ_f) 的随机扰动
3. 重复 200 次模拟，记录每次的夏普比率和最大回撤
4. 分析绩效分布的统计特征

输出：
- Sharpe 分位数：10%/50%/90%
- 最大回撤分位数：10%/50%/90%
- 崩塌概率：最大回撤 > 20% 的概率
"""

import logging
import numpy as np
import pandas as pd
from backtest.engine import BacktestEngine

logger = logging.getLogger(__name__)


class MonteCarloSimulator:
    """
    蒙特卡洛稳健性模拟器

    通过扰动因子数据来测试策略的稳健性。
    如果策略在噪声下仍然表现良好，说明策略具有较强的泛化能力。
    """

    def __init__(self, config: dict = None):
        """
        初始化模拟器

        Args:
            config: 配置字典，包含：
                - monte_carlo.noise_levels: 噪声水平列表
                - monte_carlo.n_simulations: 每种噪声水平的模拟次数
                - monte_carlo.seed: 随机种子
        """
        self.config = config or {}
        mc_cfg = self.config.get("monte_carlo", {})
        self.noise_levels = mc_cfg.get("noise_levels", [0.001, 0.005, 0.01, 0.02, 0.05])
        self.n_simulations = mc_cfg.get("n_simulations", 200)
        self.seed = mc_cfg.get("seed", 42)
        logger.info(f"[MC] MonteCarloSimulator initialized: noise_levels={self.noise_levels}, n_simulations={self.n_simulations}, seed={self.seed}")

    def run(self, factor_df: pd.DataFrame, prices: pd.DataFrame,
            benchmark_returns: pd.Series = None,
            models: list = None, dates: list = None) -> dict:
        """
        执行蒙特卡洛模拟

        Args:
            factor_df: 因子 DataFrame
            prices: 价格 DataFrame
            benchmark_returns: 基准收益序列
            models: 训练好的模型列表（如果使用模型预测）
            dates: 模型对应的日期列表

        Returns:
            结果字典，每个噪声水平对应一组统计量
        """
        logger.info(f"[MC] Starting Monte Carlo simulation: {self.n_simulations} runs x {len(self.noise_levels)} noise levels, factors={factor_df.shape}")

        if factor_df.size == 0:
            logger.warning("[MC] Empty factor_df, skipping simulation")
            return {}

        if not models or not dates:
            logger.warning("[MC] No models available, skipping Monte Carlo simulation")
            return {}

        n_rows, n_cols = factor_df.shape
        mem_per_sim = n_rows * n_cols * 8 * 2 / 1024**3
        streaming = mem_per_sim > 0.3
        if mem_per_sim > 0.5:
            adjusted_sims = max(1, int(0.5 / mem_per_sim))
            if adjusted_sims < 20:
                logger.warning(f"[MC] Estimated memory {mem_per_sim:.1f}GB/sim, insufficient for meaningful MC (need ≥20 sims). Skipping.")
                return {}
            logger.warning(f"[MC] Estimated memory per simulation {mem_per_sim:.1f}GB, reducing simulations from {self.n_simulations} to {adjusted_sims}")
            self.n_simulations = adjusted_sims

        results = {}
        date_level = 0 if factor_df.index.nlevels >= 2 and np.issubdtype(
            factor_df.index.get_level_values(0).dtype, np.datetime64
        ) else 1
        model_features_list = [m.feature_name() for m in models]

        for noise_level in self.noise_levels:
            logger.info(f"[MC] Running {self.n_simulations} simulations with noise_level={noise_level}, streaming={streaming}")
            sharpe_list = []
            max_dd_list = []
            rng = np.random.default_rng(self.seed)

            for sim_idx in range(self.n_simulations):
                if streaming:
                    predictions_list = []
                    for model, date, mf in zip(models, dates, model_features_list):
                        mask = factor_df.index.get_level_values(date_level) == date
                        X_orig = factor_df.loc[mask, mf].values
                        if len(X_orig) == 0:
                            continue
                        noise = rng.normal(0, noise_level, size=X_orig.shape).astype(np.float32)
                        X_perturbed = X_orig * (1 + noise)
                        pred = model.predict(X_perturbed)
                        predictions_list.append(pd.Series(pred, index=factor_df.index[mask]))
                        del X_orig, noise, X_perturbed
                else:
                    noise = rng.normal(0, noise_level, factor_df.shape).astype(np.float32)
                    perturbed = factor_df.values * (1 + noise)
                    perturbed_df = pd.DataFrame(perturbed, index=factor_df.index, columns=factor_df.columns)
                    predictions_list = []
                    for model, date, mf in zip(models, dates, model_features_list):
                        mask = perturbed_df.index.get_level_values(date_level) == date
                        X = perturbed_df.loc[mask, mf]
                        if len(X) == 0:
                            continue
                        pred = model.predict(X.values)
                        predictions_list.append(pd.Series(pred, index=X.index))
                    del perturbed_df, noise

                predictions = pd.concat(predictions_list) if predictions_list else pd.Series(dtype=float)
                del predictions_list

                if len(predictions) == 0:
                    continue

                engine = BacktestEngine(self.config)
                returns, bench, weights = engine.run(predictions, prices, benchmark_returns)
                sharpe_list.append(self._compute_sharpe(returns))
                max_dd_list.append(self._compute_max_dd(returns))
                del predictions, engine, returns

            if not sharpe_list:
                logger.warning(f"[MC] Noise {noise_level*100:.1f}%: all simulations produced empty results")
                results[noise_level] = {
                    "sharpe_10pct": 0.0, "sharpe_50pct": 0.0, "sharpe_90pct": 0.0,
                    "max_dd_10pct": 0.0, "max_dd_50pct": 0.0, "max_dd_90pct": 0.0,
                    "collapse_prob": 0.0,
                }
                continue

            results[noise_level] = {
                "sharpe_10pct": float(np.percentile(sharpe_list, 10)),
                "sharpe_50pct": float(np.percentile(sharpe_list, 50)),
                "sharpe_90pct": float(np.percentile(sharpe_list, 90)),
                "max_dd_10pct": float(np.percentile(max_dd_list, 10)),
                "max_dd_50pct": float(np.percentile(max_dd_list, 50)),
                "max_dd_90pct": float(np.percentile(max_dd_list, 90)),
                "collapse_prob": float(
                    sum(1 for dd in max_dd_list if dd < -0.20) / max(len(max_dd_list), 1)
                ),
            }
            logger.info(f"[MC] Noise {noise_level*100:.1f}% -> Sharpe50={results[noise_level]['sharpe_50pct']:.4f}, Sharpe90={results[noise_level]['sharpe_90pct']:.4f}, DD50={results[noise_level]['max_dd_50pct']:.1%}, collapse={results[noise_level]['collapse_prob']:.1%}")

        logger.info("[MC] Monte Carlo simulation complete")
        return results

    def _compute_sharpe(self, returns: pd.Series) -> float:
        """
        计算夏普比率

        Args:
            returns: 收益率序列

        Returns:
            夏普比率（年化）
        """
        if len(returns) < 10:
            return 0.0
        excess = returns - 0.025 / 252  # 无风险利率 2.5%
        return float(np.sqrt(252) * excess.mean() / excess.std()) if excess.std() > 0 else 0.0

    def _compute_max_dd(self, returns: pd.Series) -> float:
        """
        计算最大回撤

        Args:
            returns: 收益率序列

        Returns:
            最大回撤（负值）
        """
        equity = (1 + returns).cumprod()
        peak = equity.expanding().max()
        dd = (equity - peak) / peak
        return float(dd.min())

    def report(self, results: dict) -> str:
        """
        生成文本报告

        Args:
            results: run() 返回的结果字典

        Returns:
            格式化的报告字符串
        """
        lines = ["=== 蒙特卡洛模拟结果 ==="]
        logger.info("[MC] Monte Carlo Report:")
        for level, stats in results.items():
            lines.append(f"\n噪声水平: {level*100:.1f}%")
            lines.append(f"  Sharpe: [{stats['sharpe_10pct']:.2f}, {stats['sharpe_50pct']:.2f}, {stats['sharpe_90pct']:.2f}]")
            lines.append(f"  最大回撤: [{stats['max_dd_10pct']:.1%}, {stats['max_dd_50pct']:.1%}, {stats['max_dd_90pct']:.1%}]")
            lines.append(f"  崩塌概率: {stats['collapse_prob']:.1%}")
            logger.info(f"  Noise {level*100:.1f}% -> Sharpe: [{stats['sharpe_10pct']:.2f}, {stats['sharpe_50pct']:.2f}, {stats['sharpe_90pct']:.2f}], DD: [{stats['max_dd_10pct']:.1%}, {stats['max_dd_50pct']:.1%}, {stats['max_dd_90pct']:.1%}], collapse: {stats['collapse_prob']:.1%}")
        return "\n".join(lines)