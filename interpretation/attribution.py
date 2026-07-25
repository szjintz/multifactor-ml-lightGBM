import pandas as pd
import numpy as np


class BrinsonAttribution:
    def __init__(self, sector_map: dict):
        self.sector_map = sector_map

    def attribute(self, portfolio_weights: pd.Series, benchmark_weights: pd.Series,
                  stock_returns: pd.Series) -> dict:
        sectors = list(set(self.sector_map.values()))

        pw = portfolio_weights.groupby(self.sector_map).sum()
        bw = benchmark_weights.groupby(self.sector_map).sum()

        sr = stock_returns.groupby(self.sector_map).mean()
        pr = (portfolio_weights * stock_returns).groupby(self.sector_map).sum() / pw

        allocation = (pw - bw) * sr
        selection = bw * (pr - sr)
        interaction = (pw - bw) * (pr - sr)

        return {
            "allocation": float(allocation.sum()),
            "selection": float(selection.sum()),
            "interaction": float(interaction.sum()),
            "total": float(allocation.sum() + selection.sum() + interaction.sum()),
            "allocation_detail": allocation.to_dict(),
            "selection_detail": selection.to_dict(),
        }
