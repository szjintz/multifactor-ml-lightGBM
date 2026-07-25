from abc import ABC, abstractmethod
import pandas as pd


class BaseFactor(ABC):
    @abstractmethod
    def compute(self, data: pd.DataFrame) -> pd.Series:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class FactorPipeline:
    def __init__(self):
        self.factors: list[BaseFactor] = []

    def register(self, factor: BaseFactor):
        self.factors.append(factor)
        return self

    def compute_all(self, data: pd.DataFrame) -> pd.DataFrame:
        results = {}
        for f in self.factors:
            results[f.name] = f.compute(data)
        return pd.DataFrame(results)
