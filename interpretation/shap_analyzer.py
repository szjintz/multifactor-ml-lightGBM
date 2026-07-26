"""
SHAP 分析模块

使用 TreeSHAP 解释 LightGBM 模型的预测结果。

SHAP (SHapley Additive exPlanations) 是一种博弈论方法，
用于解释模型预测结果中各个特征的贡献。

分析类型：
1. 全局重要性：所有样本的特征贡献均值
2. 蜂群图：展示特征值与贡献的关系
3. 瀑布图：单个样本的预测分解
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


class SHAPAnalyzer:
    """
    SHAP 分析器

    使用 TreeSHAP 算法解释树模型（LightGBM）的预测结果。
    支持全局和局部的模型可解释性分析。
    """

    def __init__(self):
        """
        初始化 SHAP 分析器
        """
        self.shap_values = None
        self.feature_names = None
        Path("results").mkdir(exist_ok=True)

    def analyze(self, model, X: pd.DataFrame):
        """
        执行 SHAP 分析

        Args:
            model: 训练好的 LightGBM 模型
            X: 特征 DataFrame

        Returns:
            self（支持链式调用）
        """
        import shap
        explainer = shap.TreeExplainer(model)
        self.shap_values = explainer.shap_values(X.values)
        self.feature_names = list(X.columns)
        return self

    def plot_global_importance(self, top_n=20):
        """
        绘制全局特征重要性图（SHAP 条形图）

        显示前 top_n 个最重要的特征。

        Args:
            top_n: 显示前 N 个特征
        """
        shap.summary_plot(
            self.shap_values,
            feature_names=self.feature_names,
            plot_type="bar",
            max_display=top_n,
            show=False,
        )
        plt.title("Global SHAP Feature Importance")
        plt.tight_layout()
        plt.savefig("results/shap_global_importance.png", dpi=150)
        plt.close()

    def plot_beeswarm(self):
        """
        绘制蜂群图（Beeswarm Plot）

        展示特征值（颜色）与 SHAP 贡献（位置）的关系。
        - 颜色：特征值高低（红色=高，蓝色=低）
        - 水平位置：SHAP 贡献（正=正向贡献，负=负向贡献）
        """
        shap.summary_plot(
            self.shap_values,
            feature_names=self.feature_names,
            show=False,
        )
        plt.title("SHAP Beeswarm Plot")
        plt.tight_layout()
        plt.savefig("results/shap_beeswarm.png", dpi=150)
        plt.close()

    def plot_waterfall(self, idx=0):
        """
        绘制瀑布图（Waterfall Plot）

        解释单个样本的预测分解。
        显示各个特征如何推动预测值偏离基准值。

        Args:
            idx: 样本索引
        """
        shap.plots.waterfall(
            shap.Explanation(
                self.shap_values[idx],
                feature_names=self.feature_names
            ),
            show=False,
        )
        plt.savefig(f"results/shap_waterfall_{idx}.png", dpi=150, bbox_inches="tight")
        plt.close()

    def get_top_features(self, top_n=10) -> list:
        """
        获取最重要的特征

        Returns:
            特征名和平均 |SHAP| 的列表，如 [("feature1", 0.5), ("feature2", 0.3), ...]
        """
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        indices = np.argsort(mean_abs_shap)[::-1][:top_n]
        return [(self.feature_names[i], mean_abs_shap[i]) for i in indices]