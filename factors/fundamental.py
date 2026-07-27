"""
基本面因子模块

定义各类基本面因子，包括：
- 估值因子：EP（市盈率倒数）、BP（市净率倒数）
- 质量因子：ROE、ROA、毛利率、资产负债率
- 成长因子：营收增长率（QoQ/YoY）

基本面因子通常更新频率较低（季度），需要使用前向填充（ffill）处理缺失值。
"""

import logging
import numpy as np
import pandas as pd

from .base import BaseFactor

logger = logging.getLogger(__name__)


class EPFactor(BaseFactor):
    """
    估值因子：市盈率倒数（E/P）

    EP = 每股收益 / 收盘价 = 1 / PE
    数值越高表示估值越低，价值属性越强。
    """

    def __init__(self):
        self._name = "EP"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        if "earnings_per_share" in data and "close" in data:
            result = data["earnings_per_share"] / data["close"]
            nan_pct = result.isna().sum() / len(result) * 100
            logger.debug(f"[FACTOR] {self._name}: computed from earnings_per_share/close, {nan_pct:.1f}% NaN")
            return result
        fallback = data.get("EP", 0)
        logger.warning(f"[FACTOR] {self._name}: data not available, using fallback=0 (columns available: {list(data.columns)})")
        return fallback


class BPFactor(BaseFactor):
    """
    估值因子：市净率倒数（B/P）

    BP = 每股净资产 / 收盘价 = 1 / PB
    数值越高表示估值越低，价值属性越强。
    """

    def __init__(self):
        self._name = "BP"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        if "book_value_per_share" in data and "close" in data:
            result = data["book_value_per_share"] / data["close"]
            nan_pct = result.isna().sum() / len(result) * 100
            logger.debug(f"[FACTOR] {self._name}: computed from book_value_per_share/close, {nan_pct:.1f}% NaN")
            return result
        fallback = data.get("BP", 0)
        logger.warning(f"[FACTOR] {self._name}: data not available, using fallback=0")
        return fallback


class ROEFactor(BaseFactor):
    """
    质量因子：净资产收益率（ROE TTM）

    ROE = 净利润 / 净资产
    衡量公司运用自有资本的效率，数值越高越好。
    TTM（Trailing Twelve Months）表示过去12个月的滚动数据。
    """

    def __init__(self):
        self._name = "ROE_TTM"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        result = data.get("ROE_TTM", pd.Series(0, index=data.index))
        nan_pct = result.isna().sum() / len(result) * 100
        logger.debug(f"[FACTOR] {self._name}: {nan_pct:.1f}% NaN")
        return result


class ROAFactor(BaseFactor):
    """
    质量因子：资产回报率（ROA）

    ROA = 净利润 / 总资产
    衡量公司整体资产的盈利能力。
    """

    def __init__(self):
        self._name = "ROA"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        result = data.get("ROA", pd.Series(0, index=data.index))
        nan_pct = result.isna().sum() / len(result) * 100
        logger.debug(f"[FACTOR] {self._name}: {nan_pct:.1f}% NaN")
        return result


class GrossMarginFactor(BaseFactor):
    """
    质量因子：毛利率

    毛利率 = (营业收入 - 营业成本) / 营业收入
    衡量公司核心业务的盈利能力，数值越高护城河越强。
    """

    def __init__(self):
        self._name = "GROSS_MARGIN"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        result = data.get("gross_margin", pd.Series(0, index=data.index))
        nan_pct = result.isna().sum() / len(result) * 100
        logger.debug(f"[FACTOR] {self._name}: {nan_pct:.1f}% NaN")
        return result


class DebtRatioFactor(BaseFactor):
    """
    质量因子：资产负债率

    资产负债率 = 总负债 / 总资产
    衡量公司的财务杠杆水平，过高可能带来风险。
    """

    def __init__(self):
        self._name = "DEBT_RATIO"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        result = data.get("debt_ratio", pd.Series(0, index=data.index))
        nan_pct = result.isna().sum() / len(result) * 100
        logger.debug(f"[FACTOR] {self._name}: {nan_pct:.1f}% NaN")
        return result


class RevenueGrowthFactor(BaseFactor):
    """
    成长因子：营收增长率

    支持两种周期：
    - QoQ：季度环比增长
    - YoY：年度同比增长
    """

    def __init__(self, period="QoQ"):
        self._period = period
        self._name = f"REV_GROW_{period}"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        key = f"revenue_growth_{self._period.lower()}"
        result = data.get(key)
        if result is None or result.isna().all():
            # 列不存在或全部 NaN 时，回退到 YoY 增长率，避免下游死值
            fallback = data.get("revenue_growth_yoy")
            if fallback is not None:
                logger.debug(f"[FACTOR] {self._name}: 列 {key} 不可用，回退至 revenue_growth_yoy")
                result = fallback.copy()
            else:
                logger.warning(f"[FACTOR] {self._name}: 列 {key} 与回退列均不可用，使用全 NaN")
                return pd.Series(np.nan, index=data.index)
        # 按股票向前填充缺失值（季度数据通常稀疏）
        if result.isna().any():
            result = result.groupby(level=0).ffill()
        nan_pct = result.isna().sum() / len(result) * 100
        logger.debug(f"[FACTOR] {self._name}: {nan_pct:.1f}% NaN")
        return result


def register_all_fundamental_factors(pipeline):
    """
    注册所有基本面因子到管线

    Args:
        pipeline: FactorPipeline 实例

    Returns:
        传入的 pipeline（支持链式调用）
    """
    factors = [EPFactor(), BPFactor(), ROEFactor(), ROAFactor(),
              GrossMarginFactor(), DebtRatioFactor()]
    for p in ["QoQ", "YoY"]:
        factors.append(RevenueGrowthFactor(p))
    for f in factors:
        pipeline.register(f)
    logger.info(f"[FACTOR] Registered {len(factors)} fundamental factors: {[f.name for f in factors]}")
    return pipeline