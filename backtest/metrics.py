"""
回测指标模块

计算投资组合的绩效评估指标，包括：
- 收益指标：年化收益率、超额收益率
- 风险指标：年化波动率、最大回撤
- 风险调整收益：夏普比率、卡尔玛比率
- 预测能力：IC、IR、胜率
- 交易指标：换手率

基准假设：
- 无风险利率：2.5%（年化）
- 交易天数：252
"""

import logging
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


def compute_sharpe(returns: pd.Series, rf: float = 0.025, hp: int = 1) -> float:
    """
    计算夏普比率

    Sharpe = (年化收益 - 无风险利率) / 年化波动率
    衡量单位风险获得的超额收益。

    Args:
        returns: 收益率序列（支持任意频率，需配合 hp 做年化）
        rf: 年化无风险利率，默认 2.5%
        hp: 持有期天数（用于 rf 调整和年化），默认 1

    Returns:
        夏普比率
    """
    rf_per_period = rf * hp / 252
    excess = returns - rf_per_period
    periods_per_year = 252.0 / hp
    std = excess.std(ddof=1)
    sharpe = np.sqrt(periods_per_year) * excess.mean() / std if std > 0 else 0.0
    logger.debug(f"[METRICS] Sharpe: {sharpe:.4f} (rf={rf}, hp={hp}, rf_per_period={rf_per_period:.6f}, mean_excess={excess.mean():.6f}, std={std:.6f})")
    return sharpe


def compute_max_drawdown(equity: pd.Series) -> float:
    """
    计算最大回撤

    MaxDD = (谷值 - 峰值) / 峰值
    衡量投资组合从历史高点的最大损失。

    Args:
        equity: 净值序列（从 1 开始累计）

    Returns:
        最大回撤（负值）
    """
    peak = equity.expanding().max()
    dd = (equity - peak) / peak
    mdd = dd.min()
    logger.debug(f"[METRICS] Max Drawdown: {mdd:.4f}")
    return mdd


def compute_calmar(returns: pd.Series, hp: int = 1) -> float:
    """
    计算卡尔玛比率

    Calmar = 年化收益率 / 最大回撤
    衡量单位最大回撤获得的收益。

    Args:
        returns: 收益率序列（支持任意频率）
        hp: 持有期天数，默认 1

    Returns:
        卡尔玛比率
    """
    periods_per_year = 252.0 / hp
    ann_ret = (1 + returns).prod() ** (periods_per_year / len(returns)) - 1
    equity = (1 + returns).cumprod()
    mdd = compute_max_drawdown(equity)
    calmar = ann_ret / abs(mdd) if mdd != 0 else 0.0
    logger.debug(f"[METRICS] Calmar: {calmar:.4f} (ann_ret={ann_ret:.4f}, mdd={mdd:.4f})")
    return calmar


def compute_ic(pred: pd.Series, actual: pd.Series) -> float:
    """
    计算信息系数（IC）

    使用 Spearman 秩相关系数衡量预测与实际收益的相关性。
    IC > 0 表示正相关，IC < 0 表示负相关。

    Args:
        pred: 预测值序列
        actual: 实际收益序列

    Returns:
        IC 值（-1 到 1）
    """
    valid = pred.notna() & actual.notna()
    if valid.sum() < 10:
        logger.debug(f"[METRICS] IC: insufficient valid points ({valid.sum()}<10)")
        return 0.0
    ic = spearmanr(pred[valid], actual[valid])[0]
    logger.debug(f"[METRICS] IC: {ic:.4f} from {valid.sum()} valid points")
    return ic


def compute_ir(ic_series: pd.Series) -> float:
    """
    计算信息比率（IR）

    IR = IC 均值 / IC 标准差
    类似于夏普比率，衡量预测能力的稳定性。

    Args:
        ic_series: IC 时间序列

    Returns:
        信息比率
    """
    ir = ic_series.mean() / ic_series.std() if ic_series.std() > 0 else 0.0
    logger.debug(f"[METRICS] IR: {ir:.4f} (mean={ic_series.mean():.4f}, std={ic_series.std():.4f})")
    return ir


def compute_turnover(weights_history: pd.DataFrame, hp: int = 1) -> float:
    """
    计算年化换手率

    Turnover = sum(|w_t - w_{t-1}|) / 2
    衡量组合的活跃度。

    Args:
        weights_history: 权重历史 DataFrame，每行是一个日期的权重
        hp: 持有期天数（用于年化），默认 1

    Returns:
        年化换手率
    """
    if len(weights_history) < 2:
        logger.debug(f"[METRICS] Turnover: insufficient periods ({len(weights_history)}<2)")
        return 0.0
    to = weights_history.diff().abs().sum(axis=1)
    avg_to = to.mean()
    periods_per_year = 252.0 / hp
    ann_to = avg_to * periods_per_year
    logger.debug(f"[METRICS] Turnover: {ann_to:.4f} (annualized, per_period={avg_to:.4f}, periods_per_year={periods_per_year:.1f})")
    return ann_to


class MetricsReport:
    """
    绩效指标报告生成器

    整合所有指标的计算，生成完整的绩效评估报告。
    """

    def __init__(self, returns: pd.Series, benchmark: pd.Series,
                 predictions: pd.Series, actuals: pd.Series,
                 weights: pd.DataFrame, holding_period: int = 1):
        """
        初始化指标报告生成器

        Args:
            returns: 组合收益率序列（持有期收益率，非日频）
            benchmark: 基准收益率序列
            predictions: 预测值序列
            actuals: 实际收益序列
            weights: 权重历史 DataFrame
            holding_period: 持有期天数（用于年化计算），默认 1
        """
        self.returns = returns
        self.benchmark = benchmark
        self.predictions = predictions
        self.actuals = actuals
        self.weights = weights
        self.hp = holding_period
        self.excess = returns - benchmark
        logger.info(f"[METRICS] MetricsReport initialized: {len(returns)} periods (hp={holding_period}), {len(predictions)} predictions")

    def generate(self) -> dict:
        """
        生成完整绩效指标报告

        Returns:
            包含所有指标的字典
        """
        logger.info("[METRICS] Computing performance metrics...")
        if len(self.returns) == 0:
            logger.warning("[METRICS] Empty returns, returning zero metrics")
            return {k: 0.0 for k in ["annualized_return", "annualized_vol", "sharpe_ratio",
                                      "max_drawdown", "calmar_ratio", "win_rate",
                                      "profit_loss_ratio", "total_return"]}
        equity = (1 + self.returns).cumprod()
        ann_factor = 252.0 / self.hp
        periods_per_year = ann_factor
        ann_ret = (1 + self.returns).prod() ** (periods_per_year / len(self.returns)) - 1
        bench_ann_ret = (1 + self.benchmark).prod() ** (periods_per_year / len(self.benchmark)) - 1 if len(self.benchmark) > 0 else 0.0

        # 计算每日 IC
        ic_values = []
        for date in self.predictions.index.get_level_values(0).unique():
            pred_date = self.predictions.loc[self.predictions.index.get_level_values(0) == date]
            actual_date = self.actuals.loc[self.actuals.index.get_level_values(0) == date]
            if len(pred_date) > 10 and len(actual_date) > 10:
                ic_values.append(compute_ic(pred_date, actual_date))

        ic_series = pd.Series(ic_values)

        metrics = {
            "annualized_return": ann_ret,  # 年化收益率
            "benchmark_return": bench_ann_ret,  # 基准年化收益率
            "excess_return": ann_ret - bench_ann_ret,  # 超额收益率
            "annualized_vol": self.returns.std(ddof=1) * np.sqrt(ann_factor),  # 年化波动率（样本标准差）
            "sharpe_ratio": compute_sharpe(self.returns, hp=self.hp),  # 夏普比率
            "calmar_ratio": compute_calmar(self.returns, hp=self.hp),  # 卡尔玛比率
            "max_drawdown": compute_max_drawdown(equity),  # 最大回撤
            "annual_turnover": compute_turnover(self.weights, hp=self.hp),  # 年化换手率
            "ic_mean": ic_series.mean(),  # IC 均值
            "ic_std": ic_series.std(),  # IC 标准差
            "ir": compute_ir(ic_series),  # 信息比率
            "hit_rate": (ic_series > 0).mean(),  # 胜率（IC > 0 的比例）
        }

        logger.info("[METRICS] Performance metrics computed:")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        return metrics