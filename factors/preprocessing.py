import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def winsorize(factor: pd.Series, method="3sigma"):
    if method == "3sigma":
        mean, std = factor.mean(), factor.std()
        lower, upper = mean - 3 * std, mean + 3 * std
    elif method == "quantile":
        lower, upper = factor.quantile(0.01), factor.quantile(0.99)
    return factor.clip(lower, upper)


def cross_sectional_standardize(factor: pd.DataFrame) -> pd.DataFrame:
    return factor.subtract(factor.mean(axis=1), axis=0).div(factor.std(axis=1), axis=0)


def neutralize(factor: pd.Series, exog: pd.DataFrame) -> pd.Series:
    valid = exog.notna().all(axis=1) & factor.notna()
    if valid.sum() < 10:
        return factor
    X = exog[valid].values
    y = factor[valid].values
    model = LinearRegression().fit(X, y)
    residuals = y - model.predict(X)
    result = factor.copy()
    result[valid] = residuals
    return result


def orthogonalize(factors: pd.DataFrame, method="gram_schmidt") -> pd.DataFrame:
    if method == "gram_schmidt":
        result = factors.copy()
        for i in range(1, factors.shape[1]):
            for j in range(i):
                col_i = factors.iloc[:, i]
                col_j = result.iloc[:, j]
                result.iloc[:, i] = col_i - col_j * (col_i @ col_j) / (col_j @ col_j)
        return result
    elif method == "pca":
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(factors.shape[1], factors.shape[0]))
        components = pca.fit_transform(factors.fillna(0))
        return pd.DataFrame(components, index=factors.index, columns=factors.columns[:components.shape[1]])
    return factors
