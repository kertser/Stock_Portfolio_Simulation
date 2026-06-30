import numpy as np
import pandas as pd

from portfolio_monte_carlo.core.portfolio import normalize_weights, portfolio_returns
from portfolio_monte_carlo.core.risk import historical_cvar, historical_var, max_drawdown, summarize_simulation


def test_normalize_weights():
    result = normalize_weights([70, 30])

    assert np.allclose(result, [0.7, 0.3])


def test_portfolio_weighted_returns():
    returns = pd.DataFrame({"A": [0.10, 0.00], "B": [0.00, 0.10]})

    result = portfolio_returns(returns, [0.6, 0.4])

    assert np.allclose(result.values, [0.06, 0.04])


def test_max_drawdown():
    values = pd.Series([100, 120, 90, 95])

    assert np.isclose(max_drawdown(values), -0.25)


def test_historical_var_and_cvar():
    returns = pd.Series([-0.10, -0.05, 0.01, 0.03, 0.04])

    assert historical_var(returns, 0.2) <= -0.05
    assert historical_cvar(returns, 0.2) <= historical_var(returns, 0.2)


def test_summarize_simulation_percentiles():
    paths = np.array([[100, 110], [100, 90], [100, 130], [100, 120]], dtype=float)
    contributions = np.array([[100, 100], [100, 100], [100, 100], [100, 100]], dtype=float)

    summary = summarize_simulation(paths, contributions, 115, 0.0, "monthly")

    assert np.isclose(summary["median_final_value"], 115)
    assert np.isclose(summary["probability_reaching_target"], 0.5)
