from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_monte_carlo.core.returns import (
    annualized_return,
    annualized_volatility,
    cagr,
)
from portfolio_monte_carlo.core.scenario import Frequency, periods_per_year


def drawdown(values: pd.Series | np.ndarray) -> pd.Series:
    series = pd.Series(values, dtype=float)
    peak = series.cummax()
    return series / peak - 1


def max_drawdown(values: pd.Series | np.ndarray) -> float:
    return float(drawdown(values).min())


def historical_var(returns: pd.Series, level: float = 0.05) -> float:
    return float(returns.dropna().quantile(level))


def historical_cvar(returns: pd.Series, level: float = 0.05) -> float:
    clean = returns.dropna()
    if clean.empty:
        return np.nan
    threshold = clean.quantile(level)
    tail = clean[clean <= threshold]
    return float(tail.mean()) if len(tail) else np.nan


def sharpe_ratio(returns: pd.Series, frequency: Frequency, risk_free: float = 0.0) -> float:
    ann_ret = annualized_return(returns, frequency)
    ann_vol = annualized_volatility(returns, frequency)
    if ann_vol == 0 or np.isnan(ann_vol):
        return np.nan
    return float((ann_ret - risk_free) / ann_vol)


def sortino_ratio(returns: pd.Series, frequency: Frequency, risk_free: float = 0.0) -> float:
    ann_ret = annualized_return(returns, frequency)
    factor = periods_per_year(frequency)
    downside = returns[returns < 0].std(ddof=1) * np.sqrt(factor)
    if downside == 0 or np.isnan(downside):
        return np.nan
    return float((ann_ret - risk_free) / downside)


def calmar_ratio(returns: pd.Series, frequency: Frequency) -> float:
    clean = returns.dropna()
    wealth = (1 + clean).cumprod()
    md = max_drawdown(wealth)
    if md == 0 or np.isnan(md):
        return np.nan
    ann_cagr = cagr(wealth, frequency)
    return float(ann_cagr / abs(md))


def return_statistics(returns: pd.DataFrame, frequency: Frequency) -> pd.DataFrame:
    rows: list[dict] = []
    for name, series in returns.items():
        clean = series.dropna()
        wealth = (1 + clean).cumprod()
        rows.append(
            {
                "asset": name,
                "mean_return": clean.mean(),
                "annualized_return": annualized_return(clean, frequency),
                "annualized_volatility": annualized_volatility(clean, frequency),
                "cagr": cagr(wealth, frequency),
                "max_drawdown": max_drawdown(wealth) if len(wealth) else np.nan,
                "skewness": clean.skew(),
                "kurtosis": clean.kurtosis(),
                "historical_var_5": historical_var(clean, 0.05),
                "historical_cvar_5": historical_cvar(clean, 0.05),
                "sharpe_ratio": sharpe_ratio(clean, frequency),
                "sortino_ratio": sortino_ratio(clean, frequency),
                "calmar_ratio": calmar_ratio(clean, frequency),
                "observations": len(clean),
            }
        )
    return pd.DataFrame(rows).set_index("asset")


def summarize_simulation(
    paths: np.ndarray,
    total_contributions: np.ndarray,
    target_value: float,
    annual_inflation: float,
    frequency: Frequency,
    cashflows: dict[str, np.ndarray] | None = None,
) -> dict[str, float]:
    final_values = paths[:, -1]
    initial_values = paths[:, 0]
    contribution_final = total_contributions[:, -1]
    years = (paths.shape[1] - 1) / {"daily": 252, "weekly": 52, "monthly": 12}[frequency]
    real_final = final_values / ((1 + annual_inflation) ** years)
    max_drawdowns = np.min(paths / np.maximum.accumulate(paths, axis=1) - 1, axis=1)
    sorted_final = np.sort(final_values)
    tail_count = max(1, int(np.ceil(len(sorted_final) * 0.05)))

    summary = {
        "median_final_value": float(np.median(final_values)),
        "mean_final_value": float(np.mean(final_values)),
        "p5": float(np.percentile(final_values, 5)),
        "p10": float(np.percentile(final_values, 10)),
        "p25": float(np.percentile(final_values, 25)),
        "p75": float(np.percentile(final_values, 75)),
        "p90": float(np.percentile(final_values, 90)),
        "p95": float(np.percentile(final_values, 95)),
        "probability_reaching_target": float(np.mean(final_values >= target_value)),
        "probability_below_contributions": float(np.mean(final_values < contribution_final)),
        "probability_negative_nominal_return": float(np.mean(final_values < initial_values)),
        "probability_negative_real_return": float(np.mean(real_final < contribution_final)),
        "expected_max_drawdown": float(np.mean(max_drawdowns)),
        "median_max_drawdown": float(np.median(max_drawdowns)),
        "worst_5pct_average_outcome": float(np.mean(sorted_final[:tail_count])),
        "best_5pct_average_outcome": float(np.mean(sorted_final[-tail_count:])),
        "total_contributions": float(np.median(contribution_final)),
        "median_gain_over_contributions": float(np.median(final_values - contribution_final)),
    }
    if cashflows:
        summary.update(
            {
                "median_net_profit_if_sold": float(np.median(cashflows["net_profit_if_sold"][:, -1])),
                "median_cumulative_gross_dividends": float(np.median(cashflows["cumulative_gross_dividends"][:, -1])),
                "median_cumulative_net_dividends": float(np.median(cashflows["cumulative_net_dividends"][:, -1])),
                "median_cumulative_dividend_taxes": float(np.median(cashflows["cumulative_dividend_taxes"][:, -1])),
                "median_liquidation_tax": float(np.median(cashflows["liquidation_taxes"][:, -1])),
            }
        )
    return summary
