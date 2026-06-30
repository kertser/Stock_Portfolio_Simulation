from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_monte_carlo.core.scenario import Frequency, months_to_periods


def _clean_return_matrix(asset_returns: pd.DataFrame) -> np.ndarray:
    matrix = asset_returns.dropna(how="any").to_numpy(dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 12:
        raise ValueError("Not enough clean return observations for simulation.")
    return matrix


def historical_bootstrap(
    asset_returns: pd.DataFrame,
    simulations: int,
    periods: int,
    rng: np.random.Generator,
) -> np.ndarray:
    matrix = _clean_return_matrix(asset_returns)
    indices = rng.integers(0, matrix.shape[0], size=(simulations, periods))
    return matrix[indices]


def block_bootstrap(
    asset_returns: pd.DataFrame,
    simulations: int,
    periods: int,
    block_size_months: int,
    frequency: Frequency,
    rng: np.random.Generator,
) -> np.ndarray:
    matrix = _clean_return_matrix(asset_returns)
    block_size = min(months_to_periods(block_size_months, frequency), matrix.shape[0])
    max_start = matrix.shape[0] - block_size
    if max_start < 0:
        raise ValueError("Block size is larger than the available return history.")

    paths = np.empty((simulations, periods, matrix.shape[1]), dtype=float)
    for sim in range(simulations):
        cursor = 0
        while cursor < periods:
            start = rng.integers(0, max_start + 1)
            block = matrix[start : start + block_size]
            take = min(block_size, periods - cursor)
            paths[sim, cursor : cursor + take] = block[:take]
            cursor += take
    return paths


def parametric_normal(
    asset_returns: pd.DataFrame,
    simulations: int,
    periods: int,
    rng: np.random.Generator,
) -> np.ndarray:
    clean = asset_returns.dropna(how="any")
    mean = clean.mean().to_numpy()
    cov = clean.cov().to_numpy()
    return rng.multivariate_normal(mean, cov, size=(simulations, periods), check_valid="warn")


def fat_tail_student_t(
    asset_returns: pd.DataFrame,
    simulations: int,
    periods: int,
    df: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if df <= 2:
        raise ValueError("Student-t degrees of freedom must be greater than 2.")
    clean = asset_returns.dropna(how="any")
    mean = clean.mean().to_numpy()
    cov = clean.cov().to_numpy()
    adjusted_cov = cov * ((df - 2) / df)
    z = rng.multivariate_normal(np.zeros(len(mean)), adjusted_cov, size=(simulations, periods))
    scale = np.sqrt(df / rng.chisquare(df, size=(simulations, periods, 1)))
    return mean + z * scale


def regime_switching(
    asset_returns: pd.DataFrame,
    simulations: int,
    periods: int,
    rng: np.random.Generator,
) -> np.ndarray:
    clean = asset_returns.dropna(how="any")
    mean = clean.mean().to_numpy()
    cov = clean.cov().to_numpy()
    regimes = rng.choice(3, size=(simulations, periods), p=[0.78, 0.12, 0.10])
    multipliers = np.select(
        [regimes == 0, regimes == 1, regimes == 2],
        [1.0, 2.5, 1.4],
    )
    drifts = np.select(
        [regimes == 0, regimes == 1, regimes == 2],
        [0.0, -2.0, 1.2],
    )
    base = rng.multivariate_normal(mean, cov, size=(simulations, periods), check_valid="warn")
    return mean + (base - mean) * multipliers[..., None] + mean * drifts[..., None]
