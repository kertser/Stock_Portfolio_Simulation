from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_monte_carlo.core.scenario import Frequency, periods_per_year


def resample_prices(prices: pd.DataFrame, frequency: Frequency) -> pd.DataFrame:
    if frequency == "daily":
        return prices.sort_index()
    rule = "W-FRI" if frequency == "weekly" else "ME"
    return prices.sort_index().resample(rule).last().dropna(how="all")


def calculate_returns(prices: pd.DataFrame, frequency: Frequency = "daily") -> pd.DataFrame:
    sampled = resample_prices(prices, frequency)
    returns = sampled.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    return returns.dropna(how="all")


def annualized_return(returns: pd.Series | pd.DataFrame, frequency: Frequency) -> pd.Series | float:
    factor = periods_per_year(frequency)
    mean_period_return = returns.mean()
    return (1 + mean_period_return) ** factor - 1


def annualized_volatility(returns: pd.Series | pd.DataFrame, frequency: Frequency) -> pd.Series | float:
    return returns.std(ddof=1) * np.sqrt(periods_per_year(frequency))


def cagr(values: pd.Series | np.ndarray, frequency: Frequency) -> float:
    series = pd.Series(values).dropna()
    if len(series) < 2 or series.iloc[0] <= 0 or series.iloc[-1] <= 0:
        return np.nan
    years = (len(series) - 1) / periods_per_year(frequency)
    if years <= 0:
        return np.nan
    return (series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1


def rolling_returns(returns: pd.Series | pd.DataFrame, window: int) -> pd.Series | pd.DataFrame:
    return (1 + returns).rolling(window).apply(np.prod, raw=True) - 1


def rolling_volatility(
    returns: pd.Series | pd.DataFrame,
    window: int,
    frequency: Frequency,
) -> pd.Series | pd.DataFrame:
    return returns.rolling(window).std() * np.sqrt(periods_per_year(frequency))
