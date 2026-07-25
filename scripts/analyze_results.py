import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path


class ResultAnalyzer:
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def plot_equity_curve(self, returns: pd.Series, benchmark: pd.Series = None):
        equity = (1 + returns).cumprod()
        plt.figure(figsize=(12, 6))
        plt.plot(equity.index, equity.values, label="Strategy", linewidth=2)
        if benchmark is not None:
            bench_equity = (1 + benchmark).cumprod()
            plt.plot(benchmark.index, bench_equity.values, label="Benchmark", linewidth=2, alpha=0.7)
        plt.title("Equity Curve")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / "equity_curve.png", dpi=150)
        plt.close()

    def plot_drawdown(self, returns: pd.Series):
        equity = (1 + returns).cumprod()
        peak = equity.expanding().max()
        dd = (equity - peak) / peak
        plt.figure(figsize=(12, 4))
        plt.fill_between(dd.index, dd.values, 0, color="red", alpha=0.3)
        plt.plot(dd.index, dd.values, color="red", linewidth=1)
        plt.title("Drawdown")
        plt.xlabel("Date")
        plt.ylabel("Drawdown")
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / "drawdown.png", dpi=150)
        plt.close()

    def plot_ic_series(self, predictions: pd.Series, actuals: pd.Series):
        from backtest.metrics import compute_ic
        ic_values = []
        dates = []
        for date in predictions.index.get_level_values(0).unique():
            pred_date = predictions.loc[predictions.index.get_level_values(0) == date]
            actual_date = actuals.loc[actuals.index.get_level_values(0) == date]
            if len(pred_date) > 10 and len(actual_date) > 10:
                ic_values.append(compute_ic(pred_date, actual_date))
                dates.append(date)
        plt.figure(figsize=(12, 4))
        plt.plot(dates, ic_values, label="Daily IC")
        plt.axhline(y=0, color="gray", linestyle="--")
        plt.title("Information Coefficient Time Series")
        plt.xlabel("Date")
        plt.ylabel("IC")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / "ic_series.png", dpi=150)
        plt.close()

    def generate_report(self, returns, benchmark, predictions, actuals, weights, metrics: dict):
        self.plot_equity_curve(returns, benchmark)
        self.plot_drawdown(returns)
        self.plot_ic_series(predictions, actuals)

        with open(self.output_dir / "summary.txt", "w") as f:
            f.write("=== Strategy Performance Summary ===\n\n")
            for k, v in metrics.items():
                f.write(f"{k}: {v:.4f}\n" if isinstance(v, float) else f"{k}: {v}\n")

        print(f"Reports saved to {self.output_dir}/")
