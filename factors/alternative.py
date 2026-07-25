import numpy as np
import pandas as pd
from .base import BaseFactor


class TurnoverAnomaly(BaseFactor):
    def __init__(self, window: int = 20):
        self._window = window
        self._name = f"TURN_ANOM_{window}"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        if "turnover" not in data:
            return 0
        t = data["turnover"]
        return (t - t.rolling(self._window).mean()) / t.rolling(self._window).std()


class IdiosyncraticVolatility(BaseFactor):
    def __init__(self, window: int = 60):
        self._window = window
        self._name = f"IVOL_{window}"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        ret = data["close"].pct_change(1).fillna(0)
        market_ret = ret.mean(axis=1) if ret.ndim > 1 else ret
        from sklearn.linear_model import LinearRegression
        result = ret.copy() * np.nan
        for i in range(self._window, len(ret)):
            X = market_ret.iloc[i-self._window:i].values.reshape(-1, 1)
            y = ret.iloc[i-self._window:i].values
            if np.isnan(y).any():
                continue
            model = LinearRegression().fit(X, y)
            resid = y[-1] - model.predict(X[-1:])[0]
            result.iloc[i] = abs(resid)
        return result


class AmihudILLIQ(BaseFactor):
    def __init__(self, window: int = 20):
        self._window = window
        self._name = f"ILLIQ_{window}"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        ret = data["close"].pct_change(1).abs()
        volume = data["volume"]
        illiq = ret / volume
        return illiq.rolling(self._window).mean()


class RealizedSkew(BaseFactor):
    def __init__(self, window: int = 20):
        self._window = window
        self._name = f"RSKEW_{window}"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        return data["close"].pct_change(1).rolling(self._window).skew()


class RealizedKurt(BaseFactor):
    def __init__(self, window: int = 20):
        self._window = window
        self._name = f"RKURT_{window}"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        return data["close"].pct_change(1).rolling(self._window).kurt()


def register_all_alternative_factors(pipeline):
    for w in [20, 60]:
        pipeline.register(TurnoverAnomaly(w))
    pipeline.register(IdiosyncraticVolatility(60))
    for w in [20, 60]:
        pipeline.register(AmihudILLIQ(w))
    for w in [20, 60]:
        pipeline.register(RealizedSkew(w))
        pipeline.register(RealizedKurt(w))
    return pipeline
