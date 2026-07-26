"""
结果分析脚本

生成回测结果的可视化报告，包括：
1. 净值曲线（Equity Curve）
2. 回撤曲线（Drawdown）
3. IC 时序图

用于快速评估策略表现。
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path


class ResultAnalyzer:
    """
    回测结果分析器

    生成策略表现的可视化报告和摘要。
    """

    def __init__(self, output_dir: str = "results"):
        """
        初始化分析器

        Args:
            output_dir: 输出目录，保存图表和报告
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def plot_equity_curve(self, returns: pd.Series, benchmark: pd.Series = None):
        """
        绘制净值曲线

        显示策略和基准的累计收益随时间的变化。

        Args:
            returns: 策略收益率序列
            benchmark: 基准收益率序列（可选）
        """
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
        """
        绘制回撤曲线

        显示策略的回撤（从峰值下跌）随时间的变化。

        Args:
            returns: 收益率序列
        """
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
        """
        绘制 IC 时序图

        显示每日 IC 的时间序列，用于评估预测能力的稳定性。

        Args:
            predictions: 预测序列
            actuals: 实际收益序列
        """
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
        """
        生成完整报告

        生成所有图表和摘要文本文件。

        Args:
            returns: 收益率序列
            benchmark: 基准收益率
            predictions: 预测序列
            actuals: 实际收益序列
            weights: 权重历史
            metrics: 绩效指标字典
        """
        # 绘制所有图表
        self.plot_equity_curve(returns, benchmark)
        self.plot_drawdown(returns)
        self.plot_ic_series(predictions, actuals)

        # 生成摘要文件
        with open(self.output_dir / "summary.txt", "w") as f:
            f.write("=== Strategy Performance Summary ===\n\n")
            for k, v in metrics.items():
                f.write(f"{k}: {v:.4f}\n" if isinstance(v, float) else f"{k}: {v}\n")

        print(f"Reports saved to {self.output_dir}/")