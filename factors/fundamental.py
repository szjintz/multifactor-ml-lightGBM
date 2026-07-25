import pandas as pd

from .base import BaseFactor


class EPFactor(BaseFactor):
    def __init__(self):
        self._name = "EP"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        if "earnings_per_share" in data and "close" in data:
            return data["earnings_per_share"] / data["close"]
        return data.get("EP", 0)


class BPFactor(BaseFactor):
    def __init__(self):
        self._name = "BP"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        if "book_value_per_share" in data and "close" in data:
            return data["book_value_per_share"] / data["close"]
        return data.get("BP", 0)


class ROEFactor(BaseFactor):
    def __init__(self):
        self._name = "ROE_TTM"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        return data.get("ROE_TTM", 0)


class ROAFactor(BaseFactor):
    def __init__(self):
        self._name = "ROA"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        return data.get("ROA", 0)


class GrossMarginFactor(BaseFactor):
    def __init__(self):
        self._name = "GROSS_MARGIN"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        return data.get("gross_margin", 0)


class DebtRatioFactor(BaseFactor):
    def __init__(self):
        self._name = "DEBT_RATIO"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        return data.get("debt_ratio", 0)


class RevenueGrowthFactor(BaseFactor):
    def __init__(self, period="QoQ"):
        self._period = period
        self._name = f"REV_GROW_{period}"

    @property
    def name(self):
        return self._name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        key = f"revenue_growth_{self._period.lower()}"
        return data.get(key, 0)


def register_all_fundamental_factors(pipeline):
    for f in [EPFactor(), BPFactor(), ROEFactor(), ROAFactor(),
              GrossMarginFactor(), DebtRatioFactor()]:
        pipeline.register(f)
    for p in ["QoQ", "YoY"]:
        pipeline.register(RevenueGrowthFactor(p))
    return pipeline
