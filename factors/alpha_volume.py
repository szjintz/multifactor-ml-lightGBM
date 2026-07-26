"""
量价因子模块

定义基于价格和成交量的技术因子，包括：
- 动量因子：收益率、加权动量、RSI、BIAS、MACD
- 反转因子：反转、最大/最小收益
- 波动率因子：标准差、ATR、BETA、已实现波动率
- 技术因子：成交量比率、换手率、VWAP偏离、量价相关性

这些因子基于历史价格和成交量计算，反映市场行为特征。
"""

import logging
import numpy as np
import pandas as pd
from .base import BaseFactor

logger = logging.getLogger(__name__)


class MomentumFactor(BaseFactor):
    """
    动量因子：N日收益率

    RET_N = (close_t - close_{t-N}) / close_{t-N}
    最基础的动量指标，正值表示近期上涨，负值表示近期下跌。
    """

    def __init__(self, window: int, name: str = None):
        self._window = window
        self._name = name or f"RET_{window}"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"].groupby(level=0).ffill()
        return close.groupby(level=0).pct_change(self._window, fill_method=None)


class WeightedMomentum(BaseFactor):
    """
    加权动量因子

    对过去 N 天的日收益率加权平均，近期权重更大。
    权重线性递增：第 N 天权重为 N，前一天权重为 1。
    """

    def __init__(self, window: int = 20):
        self._window = window
        self._name = f"WMA_{window}"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close_wide = data["close"].unstack(level=0)
        ret_wide = close_wide.ffill().pct_change(fill_method=None)

        weights = np.arange(self._window, 0, -1)
        result = sum(
            ret_wide.shift(i + 1) * weights[i]
            for i in range(self._window)
        ) / weights.sum()

        return result.stack().swaplevel(0, 1).sort_index()


class RSIFactor(BaseFactor):
    """
    相对强弱指数（RSI）

    RSI = 100 - 100/(1 + RS)
    其中 RS = N日内上涨均值 / N日内下跌均值

    取值范围 0-100：
    - >70：超买区域
    - <30：超卖区域
    """

    def __init__(self, window: int = 14):
        self._window = window
        self._name = f"RS_{window}"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"].groupby(level=0).ffill()
        delta = close.groupby(level=0).diff()
        gain = delta.clip(lower=0).groupby(level=0).rolling(self._window).mean().droplevel(0)
        loss = (-delta.clip(upper=0)).groupby(level=0).rolling(self._window).mean().droplevel(0)
        rs = gain / loss.replace(0, np.nan)
        return 100 - 100 / (1 + rs)


class BIASFactor(BaseFactor):
    """
    乖离率（BIAS）

    BIAS = (close - MA) / MA
    衡量价格偏离均线的程度，可用于判断超买超卖。
    """

    def __init__(self, window: int = 5):
        self._window = window
        self._name = f"BIAS_{window}"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]
        ma = close.groupby(level=0).rolling(self._window).mean().droplevel(0)
        return (close - ma) / ma


class MACDFactor(BaseFactor):
    """
     Moving Average Convergence Divergence（MACD）

    计算方式：
    - DIF = EMA12 - EMA26（快线）
    - DEA = DIF 的 EMA9（慢线）
    - MACD = (DIF - DEA) * 2（柱状图）

    用于判断趋势方向和动量强弱。
    """

    def __init__(self):
        self._name = "MACD"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]
        ema12 = close.groupby(level=0).ewm(span=12).mean().droplevel(0)
        ema26 = close.groupby(level=0).ewm(span=26).mean().droplevel(0)
        macd = ema12 - ema26
        signal = macd.groupby(level=0).ewm(span=9).mean().droplevel(0)
        return macd - signal


class ReversalFactor(BaseFactor):
    """
    反转因子：N日负收益

    REV_N = -RET_N
    用于捕捉短期反转效应：过去跌的股票未来可能涨。
    """

    def __init__(self, window: int = 5):
        self._window = window
        self._name = f"REV_{window}"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"].groupby(level=0).ffill()
        return -close.groupby(level=0).pct_change(self._window, fill_method=None)


class MaxReturnFactor(BaseFactor):
    """
    最大收益因子

    取负值：max(RET_1, RET_2, ..., RET_N)
    捕捉极端上涨后的反转效应。
    """

    def __init__(self, window: int = 5):
        self._window = window
        self._name = f"MAXRET_{window}"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"].groupby(level=0).ffill()
        ret = close.groupby(level=0).pct_change(1, fill_method=None)
        return -ret.groupby(level=0).rolling(self._window).max().droplevel(0)


class MinReturnFactor(BaseFactor):
    """
    最小收益因子

    取负值：min(RET_1, RET_2, ..., RET_N)
    捕捉极端下跌后的反转效应。
    """

    def __init__(self, window: int = 5):
        self._window = window
        self._name = f"MINRET_{window}"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"].groupby(level=0).ffill()
        ret = close.groupby(level=0).pct_change(1, fill_method=None)
        return -ret.groupby(level=0).rolling(self._window).min().droplevel(0)


class VolatilityFactor(BaseFactor):
    """
    波动率因子：N日收益标准差

    STD_N = std(RET_1, RET_2, ..., RET_N)
    衡量收益率的离散程度，高波动率通常伴随高风险。
    """

    def __init__(self, window: int = 20):
        self._window = window
        self._name = f"STD_{window}"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"].groupby(level=0).ffill()
        ret = close.groupby(level=0).pct_change(1, fill_method=None)
        return ret.groupby(level=0).rolling(self._window).std().droplevel(0)


class ATRFactor(BaseFactor):
    """
    平均真实波幅（ATR）

    True Range = max(H-L, |H-PC|, |L-PC|)
    其中 PC 是前一日收盘价
    ATR = N日 TR 的均值

    衡量价格波动幅度，不受方向影响。
    """

    def __init__(self, window: int = 14):
        self._window = window
        self._name = f"ATR_{window}"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        high, low, close = data["high"], data["low"], data["close"]
        close_ff = close.groupby(level=0).ffill()
        prev_close = close_ff.groupby(level=0).shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.groupby(level=0).rolling(self._window).mean().droplevel(0)


class BetaFactor(BaseFactor):
    """
    市场贝塔

    BETA = Cov(stock_ret, market_ret) / Var(market_ret)

    衡量个股相对市场的敏感度：
    - BETA > 1：波动大于市场
    - BETA < 1：波动小于市场
    - BETA < 0：反向运动（较少见）
    """

    def __init__(self, window: int = 60):
        self._window = window
        self._name = f"BETA_{window}"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"].unstack(level=0)
        ret = close.ffill().pct_change(fill_method=None)
        market_ret = ret.mean(axis=1)
        market_ret.name = "__market__"
        cov = ret.rolling(self._window).cov(market_ret)
        var = market_ret.rolling(self._window).var()
        return cov.div(var, axis=0).stack().swaplevel(0, 1).sort_index()


class RealizedVolatility(BaseFactor):
    """
    已实现波动率

    RV = sqrt(sum(RET_i^2)) for i in [1, N]

    基于高频收益平方和计算的波动率估计，是 Ito 过程二次变差的估计。
    """

    def __init__(self, window: int = 20):
        self._window = window
        self._name = f"RVOL_{window}"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close_wide = data["close"].unstack(level=0)
        ret_wide = close_wide.ffill().pct_change(fill_method=None)
        rolled = (ret_wide ** 2).rolling(self._window, min_periods=self._window).sum()
        return np.sqrt(rolled).stack().swaplevel(0, 1).sort_index()


class VolumeRatioFactor(BaseFactor):
    """
    成交量比率因子

    VOLR = volume / MA(volume, N)
    衡量当前成交量相对均值的倍数。
    """

    def __init__(self, window: int = 5):
        self._window = window
        self._name = f"VOLR_{window}"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        volume = data["volume"]
        return volume / volume.groupby(level=0).rolling(self._window).mean().droplevel(0)


class TurnoverFactor(BaseFactor):
    """
    换手率因子

    换手率 = 成交金额 / 流通市值
    衡量股票的交易活跃程度。
    """

    def __init__(self, window: int = 5):
        self._window = window
        self._name = f"TURN_{window}"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        amount = data.get("amount", data["volume"] * data["close"])
        cap = data.get("market_cap", pd.Series(1, index=data.index))
        amount_filled = amount.groupby(level=0).ffill().fillna(0)
        cap_filled = cap.groupby(level=0).ffill().replace(0, np.nan).fillna(1)
        turnover = amount_filled / cap_filled
        return turnover.groupby(level=0).rolling(self._window).mean().droplevel(0)


class VWAPDeviation(BaseFactor):
    """
    VWAP 偏离因子

    VWAP_DEV = (close - VWAP) / VWAP
    衡量当前价格相对成交量加权平均价格的偏离。
    """

    def __init__(self):
        self._name = "VWAP_DEV"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        vwap = data.get("vwap", data[["high", "low", "close"]].mean(axis=1))
        close = data["close"]
        return (close - vwap) / vwap


class PriceVolumeCorr(BaseFactor):
    """
    量价相关性因子

    PVC = Corr(RET, volume_change)
    衡量价格上涨与成交量增加的关系。
    正相关表示上涨有量支撑，负相关表示缩量上涨。
    """

    def __init__(self, window: int = 20):
        self._window = window
        self._name = f"PVC_{window}"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"].unstack(level=0)
        volume = data["volume"].unstack(level=0)
        ret = close.ffill().pct_change(fill_method=None)
        vol_chg = volume.ffill().pct_change(fill_method=None)
        corr = ret.rolling(self._window).corr(vol_chg)
        return corr.stack().swaplevel(0, 1).sort_index()


def register_all_volume_price_factors(pipeline):
    """
    注册所有量价因子到管线

    因子分类：
    - 动量：收益率、加权动量、RSI、BIAS、MACD
    - 反转：反转、最大/最小收益
    - 波动率：标准差、ATR、BETA、已实现波动率
    - 技术：成交量比率、换手率、VWAP偏离、量价相关性

    Args:
        pipeline: FactorPipeline 实例

    Returns:
        传入的 pipeline（支持链式调用）
    """
    registered = []
    # 动量因子
    for w in [5, 10, 20, 60]:
        f = MomentumFactor(w); pipeline.register(f); registered.append(f.name)
    for w in [5, 10, 20]:
        f = WeightedMomentum(w); pipeline.register(f); registered.append(f.name)
    for w in [14, 28]:
        f = RSIFactor(w); pipeline.register(f); registered.append(f.name)
    for w in [5, 10]:
        f = BIASFactor(w); pipeline.register(f); registered.append(f.name)
    f = MACDFactor(); pipeline.register(f); registered.append(f.name)

    # 反转因子
    for w in [1, 2, 5]:
        f = ReversalFactor(w); pipeline.register(f); registered.append(f.name)
    for w in [5, 10]:
        f = MaxReturnFactor(w); pipeline.register(f); registered.append(f.name)
        f = MinReturnFactor(w); pipeline.register(f); registered.append(f.name)

    # 波动率因子
    for w in [5, 10, 20, 60]:
        f = VolatilityFactor(w); pipeline.register(f); registered.append(f.name)
    for w in [5, 14]:
        f = ATRFactor(w); pipeline.register(f); registered.append(f.name)
    f = BetaFactor(60); pipeline.register(f); registered.append(f.name)
    for w in [5, 20]:
        f = RealizedVolatility(w); pipeline.register(f); registered.append(f.name)

    # 技术因子
    for w in [5, 20]:
        f = VolumeRatioFactor(w); pipeline.register(f); registered.append(f.name)
        f = TurnoverFactor(w); pipeline.register(f); registered.append(f.name)
    f = VWAPDeviation(); pipeline.register(f); registered.append(f.name)
    f = PriceVolumeCorr(20); pipeline.register(f); registered.append(f.name)

    logger.info(f"[FACTOR] Registered {len(registered)} volume/price factors: {registered}")
    return pipeline