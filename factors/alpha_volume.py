import numpy as np
import pandas as pd
from .base import BaseFactor


class MomentumFactor(BaseFactor):
    def __init__(self, window: int, name: str = None):
        self._window = window
        self._name = name or f"RET_{window}"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]
        ret = close.pct_change(self._window)
        return ret


class WeightedMomentum(BaseFactor):
    def __init__(self, window: int = 20):
        self._window = window
        self._name = f"WMA_{window}"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]
        weights = np.arange(1, self._window + 1) / self._window
        ret = close.pct_change(1).rolling(self._window).apply(
            lambda x: np.dot(x, weights[:len(x)]) if len(x) == self._window else np.nan
        )
        return ret


class RSIFactor(BaseFactor):
    def __init__(self, window: int = 14):
        self._window = window
        self._name = f"RS_{window}"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        delta = data["close"].diff()
        gain = delta.clip(lower=0).rolling(self._window).mean()
        loss = (-delta.clip(upper=0)).rolling(self._window).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - 100 / (1 + rs)


class BIASFactor(BaseFactor):
    def __init__(self, window: int = 5):
        self._window = window
        self._name = f"BIAS_{window}"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]
        ma = close.rolling(self._window).mean()
        return (close - ma) / ma


class MACDFactor(BaseFactor):
    def __init__(self):
        self._name = "MACD"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        return macd - signal


class ReversalFactor(BaseFactor):
    def __init__(self, window: int = 5):
        self._window = window
        self._name = f"REV_{window}"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        return -data["close"].pct_change(self._window)


class MaxReturnFactor(BaseFactor):
    def __init__(self, window: int = 5):
        self._window = window
        self._name = f"MAXRET_{window}"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        return -data["close"].pct_change(1).rolling(self._window).max()


class MinReturnFactor(BaseFactor):
    def __init__(self, window: int = 5):
        self._window = window
        self._name = f"MINRET_{window}"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        return -data["close"].pct_change(1).rolling(self._window).min()


class VolatilityFactor(BaseFactor):
    def __init__(self, window: int = 20):
        self._window = window
        self._name = f"STD_{window}"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        return data["close"].pct_change(1).rolling(self._window).std()


class ATRFactor(BaseFactor):
    def __init__(self, window: int = 14):
        self._window = window
        self._name = f"ATR_{window}"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        high, low, close = data["high"], data["low"], data["close"]
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(self._window).mean()


class BetaFactor(BaseFactor):
    def __init__(self, window: int = 60):
        self._window = window
        self._name = f"BETA_{window}"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        returns = data["close"].pct_change(1)
        market_ret = returns.mean(axis=1) if returns.ndim > 1 else returns
        result = returns.rolling(self._window).cov(market_ret) / market_ret.rolling(self._window).var()
        return result


class RealizedVolatility(BaseFactor):
    def __init__(self, window: int = 20):
        self._window = window
        self._name = f"RVOL_{window}"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        ret = data["close"].pct_change(1)
        return ret.rolling(self._window).apply(lambda x: np.sqrt(np.sum(x**2)))


class VolumeRatioFactor(BaseFactor):
    def __init__(self, window: int = 5):
        self._window = window
        self._name = f"VOLR_{window}"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        volume = data["volume"]
        return volume / volume.rolling(self._window).mean()


class TurnoverFactor(BaseFactor):
    def __init__(self, window: int = 5):
        self._window = window
        self._name = f"TURN_{window}"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        amount = data.get("amount", data["volume"] * data["close"])
        cap = data.get("market_cap", 1)
        turnover = amount / cap
        return turnover.rolling(self._window).mean()


class VWAPDeviation(BaseFactor):
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
    def __init__(self, window: int = 20):
        self._window = window
        self._name = f"PVC_{window}"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        ret = data["close"].pct_change(1)
        vol_change = data["volume"].pct_change(1)
        return ret.rolling(self._window).corr(vol_change)


def register_all_volume_price_factors(pipeline):
    # Momentum
    for w in [5, 10, 20, 60]:
        pipeline.register(MomentumFactor(w))
    for w in [5, 10, 20]:
        pipeline.register(WeightedMomentum(w))
    for w in [14, 28]:
        pipeline.register(RSIFactor(w))
    for w in [5, 10]:
        pipeline.register(BIASFactor(w))
    pipeline.register(MACDFactor())

    # Reversal
    for w in [1, 2, 5]:
        pipeline.register(ReversalFactor(w))
    for w in [5, 10]:
        pipeline.register(MaxReturnFactor(w))
        pipeline.register(MinReturnFactor(w))

    # Volatility
    for w in [5, 10, 20, 60]:
        pipeline.register(VolatilityFactor(w))
    for w in [5, 14]:
        pipeline.register(ATRFactor(w))
    pipeline.register(BetaFactor(60))
    for w in [5, 20]:
        pipeline.register(RealizedVolatility(w))

    # Technical
    for w in [5, 20]:
        pipeline.register(VolumeRatioFactor(w))
        pipeline.register(TurnoverFactor(w))
    pipeline.register(VWAPDeviation())
    pipeline.register(PriceVolumeCorr(20))

    return pipeline
