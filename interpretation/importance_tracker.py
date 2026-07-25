import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class ImportanceTracker:
    def __init__(self):
        self.importance_history = []

    def record(self, model, date: str, feature_names: list):
        importance = pd.Series(
            model.feature_importance(importance_type="gain"),
            index=feature_names
        )
        importance = importance / importance.sum()
        importance["date"] = date
        self.importance_history.append(importance)

    def get_importance_matrix(self) -> pd.DataFrame:
        df = pd.DataFrame(self.importance_history)
        df = df.set_index("date")
        return df

    def plot_heatmap(self, top_n=20):
        imp_matrix = self.get_importance_matrix()
        top_features = imp_matrix.mean().nlargest(top_n).index
        plt.figure(figsize=(12, 8))
        sns.heatmap(imp_matrix[top_features].T, cmap="YlOrRd", cbar_kws={"label": "Importance"})
        plt.title(f"Feature Importance Time-Series (Top {top_n})")
        plt.xlabel("Date")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig("results/importance_heatmap.png", dpi=150)
        plt.close()

    def get_decaying_features(self, window=10) -> list:
        imp_matrix = self.get_importance_matrix()
        if len(imp_matrix) < window * 2:
            return []
        recent = imp_matrix.iloc[-window:].mean()
        early = imp_matrix.iloc[:window].mean()
        decay = early - recent
        return list(decay.nlargest(5).index)
