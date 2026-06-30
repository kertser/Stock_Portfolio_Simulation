from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_weights(weights: list[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(weights, dtype=float)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError("Weights must be a non-empty one-dimensional array.")
    total = array.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Weights must sum to a positive number.")
    return array / total


def align_weights(tickers: list[str], weights: list[float]) -> pd.Series:
    if len(tickers) != len(weights):
        raise ValueError("The number of tickers must match the number of weights.")
    return pd.Series(normalize_weights(weights), index=tickers, name="weight")


def portfolio_returns(asset_returns: pd.DataFrame, weights: list[float] | np.ndarray) -> pd.Series:
    clean = asset_returns.dropna(how="any")
    normalized = normalize_weights(weights)
    if clean.shape[1] != len(normalized):
        raise ValueError("Weights length must match the number of return columns.")
    return pd.Series(clean.to_numpy() @ normalized, index=clean.index, name="Portfolio")


def covariance_matrix(asset_returns: pd.DataFrame) -> pd.DataFrame:
    return asset_returns.dropna(how="any").cov()


def correlation_matrix(asset_returns: pd.DataFrame) -> pd.DataFrame:
    return asset_returns.dropna(how="any").corr()
