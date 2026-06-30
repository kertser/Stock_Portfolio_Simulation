import numpy as np
import pandas as pd

from portfolio_monte_carlo.core.returns import annualized_return, annualized_volatility, cagr, calculate_returns


def test_calculate_returns_monthly_resamples_last_price():
    dates = pd.date_range("2024-01-01", periods=45, freq="D")
    prices = pd.DataFrame({"SPY": np.linspace(100, 145, len(dates))}, index=dates)

    returns = calculate_returns(prices, "monthly")

    assert len(returns) == 1
    assert returns["SPY"].iloc[0] > 0


def test_annualized_return_uses_frequency_factor():
    returns = pd.Series([0.01] * 12)

    result = annualized_return(returns, "monthly")

    assert result == np.float64((1.01**12) - 1)


def test_annualized_volatility_scales_by_sqrt_periods():
    returns = pd.Series([0.01, -0.01] * 6)

    result = annualized_volatility(returns, "monthly")

    assert np.isclose(result, returns.std(ddof=1) * np.sqrt(12))


def test_cagr_from_wealth_path():
    values = pd.Series([100, 110, 121])

    result = cagr(values, "yearly" if False else "monthly")

    expected = (121 / 100) ** (1 / (2 / 12)) - 1
    assert np.isclose(result, expected)
