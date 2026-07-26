"""
另类因子模块

定义非传统数据源的因子，包括：
- 换手率异常（TurnoverAnomaly）
- 特质波动率（IdiosyncraticVolatility）
- Amihud 非流动性指标（AmihudILLIQ）
- 已实现偏度和峰度（RealizedSkew, RealizedKurt）

这些因子基于金融学术研究成果，通常捕捉市场微观结构效应或投资者行为偏差。
"""

import logging
import numpy as np
import pandas as pd
from .base import BaseFactor

logger = logging.getLogger(__name__)


class TurnoverAnomaly(BaseFactor):
    """
    换手率异常因子

    TURN_ANOM = (turnover - MA(turnover, N)) / std(turnover, N)
    标准化后的换手率偏离度，捕捉交易活跃度的异常变化。
    高换手率异常可能预示着投机性交易。
    """

    def __init__(self, window: int = 20):
        self._window = window
        self._name = f"TURN_ANOM_{window}"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        amount = data.get("amount", data["volume"] * data["close"])
        cap = data.get("market_cap", pd.Series(1, index=data.index))
        amount_filled = amount.groupby(level=0).ffill()
        cap_filled = cap.groupby(level=0).ffill().replace(0, np.nan)
        t = (amount_filled / cap_filled).groupby(level=0).ffill()
        mean = t.groupby(level=0).rolling(self._window).mean().droplevel(0)
        std = t.groupby(level=0).rolling(self._window).std().droplevel(0)
        result = (t - mean) / std.replace(0, np.nan)
        result = result.fillna(0)
        nan_pct = result.isna().sum() / len(result) * 100
        logger.debug(f"[FACTOR] {self._name}: computed, {nan_pct:.1f}% NaN")
        return result


class IdiosyncraticVolatility(BaseFactor):
    """
    特质波动率

    股票收益中不能被市场因子解释的部分的波动率。
    使用 Fama-French 三因子模型回归后的残差计算。

    计算方法：
    1. 对每只股票，用过去 N 天日收益对市场收益回归
    2. 取残差的绝对值作为特质波动率

    学术研究发现：低特质波动率的股票长期表现更好（ volatility puzzle）。
    """

    def __init__(self, window: int = 60):
        self._window = window
        self._name = f"IVOL_{window}"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        logger.debug(f"[FACTOR] {self._name}: Starting computation with window={self._window}")
        close = data["close"].unstack(level=0)
        ret = close.ffill().pct_change(fill_method=None).fillna(0)
        market_ret = ret.mean(axis=1)
        w = self._window

        # 向量化滚动 OLS 残差计算
        # 对每只股票, 用过去 w 天日收益对市场收益回归, 取残差绝对值
        # beta = Cov(X,Y) / Var(X) = (w*sum_xy - sum_x*sum_y) / (w*sum_x2 - sum_x^2)
        # residual = |y_t - (mean_y + beta * (x_t - mean_x))|

        # 滚动统计量（shift(1) 使窗口对齐到 [t-w, t-1)）
        sum_x = market_ret.rolling(w).sum().shift(1)    # (T,)
        sum_y = ret.rolling(w).sum().shift(1)            # (T, N)
        sum_x2 = (market_ret ** 2).rolling(w).sum().shift(1)  # (T,)
        sum_xy = ret.multiply(market_ret, axis=0).rolling(w).sum().shift(1)  # (T, N)

        # beta
        numer = w * sum_xy - sum_x.values.reshape(-1, 1) * sum_y
        denom = (w * sum_x2 - sum_x ** 2).values.reshape(-1, 1)
        denom = np.where(np.abs(denom) < 1e-10, np.nan, denom)
        beta = numer / denom

        # 残差 = |y_t - (mean_y + beta * (x_t - mean_x))|
        mean_x = sum_x / w
        mean_y_est = sum_y / w
        x_centered = (market_ret - mean_x).values.reshape(-1, 1)
        y_centered = ret - mean_y_est
        residual = (y_centered - beta * x_centered).abs()

        res = residual.stack().swaplevel(0, 1).sort_index()
        nan_pct = res.isna().sum() / len(res) * 100
        logger.debug(f"[FACTOR] {self._name}: computed, {nan_pct:.1f}% NaN, range: {res.min():.6f} to {res.max():.6f}")
        return res


class AmihudILLIQ(BaseFactor):
    """
    Amihud 非流动性指标

    ILLIQ = |RET| / Volume
    衡量单位成交量对应的价格变动幅度。

    学术意义：
    - 高 ILLIQ：流动性差，微小交易就能导致价格大幅波动
    - 低 ILLIQ：流动性好，价格稳定

    Amihud (2002) 发现非流动性与股票收益正相关。
    """

    def __init__(self, window: int = 20):
        self._window = window
        self._name = f"ILLIQ_{window}"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        ret = data["close"].groupby(level=0).ffill().groupby(level=0).pct_change(fill_method=None).abs()
        volume = data["volume"].groupby(level=0).ffill().fillna(0)
        illiq = ret / (volume + 1e-8)
        result = illiq.groupby(level=0).rolling(self._window).mean().droplevel(0)
        nan_pct = result.isna().sum() / len(result) * 100
        logger.debug(f"[FACTOR] {self._name}: computed, {nan_pct:.1f}% NaN")
        return result


class RealizedSkew(BaseFactor):
    """
    已实现偏度

    收益分布偏度的滚动估计。
    - 正偏度：右尾长，更多正向极端收益
    - 负偏度：左尾长，更多负向极端收益（黑天鹅风险）

    学术研究发现：投资者偏好正偏度，导致对负偏度资产的定价偏低。
    """

    def __init__(self, window: int = 20):
        self._window = window
        self._name = f"RSKEW_{window}"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"].groupby(level=0).ffill()
        ret = close.groupby(level=0).pct_change(fill_method=None)
        result = ret.groupby(level=0).rolling(self._window).skew().droplevel(0)
        nan_pct = result.isna().sum() / len(result) * 100
        logger.debug(f"[FACTOR] {self._name}: computed, {nan_pct:.1f}% NaN")
        return result


class RealizedKurt(BaseFactor):
    """
    已实现峰度

    收益分布峰度的滚动估计。
    - 高峰度：尖峰肥尾，更多极端收益
    - 低峰度：平峰薄尾，收益分布更均匀

    高峰度通常与市场压力时期相关。
    """

    def __init__(self, window: int = 20):
        self._window = window
        self._name = f"RKURT_{window}"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"].groupby(level=0).ffill()
        ret = close.groupby(level=0).pct_change(fill_method=None)
        result = ret.groupby(level=0).rolling(self._window).kurt().droplevel(0)
        nan_pct = result.isna().sum() / len(result) * 100
        logger.debug(f"[FACTOR] {self._name}: computed, {nan_pct:.1f}% NaN")
        return result


def register_all_alternative_factors(pipeline):
    """
    注册所有另类因子到管线

    Args:
        pipeline: FactorPipeline 实例

    Returns:
        传入的 pipeline（支持链式调用）
    """
    factors = []
    for w in [20, 60]:
        factors.append(TurnoverAnomaly(w))
    factors.append(IdiosyncraticVolatility(60))
    for w in [20, 60]:
        factors.append(AmihudILLIQ(w))
    for w in [20, 60]:
        factors.append(RealizedSkew(w))
        factors.append(RealizedKurt(w))
    for f in factors:
        pipeline.register(f)
    logger.info(f"[FACTOR] Registered {len(factors)} alternative factors: {[f.name for f in factors]}")
    return pipeline