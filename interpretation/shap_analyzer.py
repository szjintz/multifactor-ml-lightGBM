import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class SHAPAnalyzer:
    def __init__(self):
        self.shap_values = None
        self.feature_names = None

    def analyze(self, model, X: pd.DataFrame):
        explainer = shap.TreeExplainer(model)
        self.shap_values = explainer.shap_values(X.values)
        self.feature_names = list(X.columns)
        return self

    def plot_global_importance(self, top_n=20):
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
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        indices = np.argsort(mean_abs_shap)[::-1][:top_n]
        return [(self.feature_names[i], mean_abs_shap[i]) for i in indices]
