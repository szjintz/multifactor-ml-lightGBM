import pandas as pd
import numpy as np


def compute_forward_return(close: pd.DataFrame, periods: int = 20, skip: int = 1) -> pd.DataFrame:
    shifted_close = close.shift(-skip)
    future_close = close.shift(-periods - skip + 1)
    returns = (future_close - shifted_close) / shifted_close
    return returns


def denoise_label(labels: pd.Series, method: str = None) -> pd.Series:
    if method == "ewm":
        return labels.ewm(span=5).mean()
    return labels


def compute_labels(close: pd.DataFrame, periods=20, skip=1, denoise=None) -> pd.DataFrame:
    labels = compute_forward_return(close, periods, skip)
    if denoise:
        labels = denoise_label(labels, denoise)
    return labels
