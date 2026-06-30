import numpy as np
import pandas as pd

from portfolio_monte_carlo.core.contributions import contribution_schedule, inflation_indexed_basis_path
from portfolio_monte_carlo.core.scenario import Scenario
from portfolio_monte_carlo.core.simulation import generate_return_paths, run_simulation, simulate_paths


def test_contribution_schedule_applies_annual_increase():
    schedule = contribution_schedule(120, 0.10, 2, "monthly")

    assert len(schedule) == 24
    assert np.allclose(schedule[:12], 120)
    assert np.allclose(schedule[12:], 132)


def test_inflation_indexed_basis_path_has_one_value_per_path_point():
    schedule = np.array([100, 100, 100], dtype=float)

    basis = inflation_indexed_basis_path(1000, schedule, 0.0, "monthly")

    assert np.allclose(basis, [1000, 1100, 1200, 1300])


def test_generate_return_paths_shape():
    returns = pd.DataFrame({"A": [0.01, 0.02, -0.01] * 20, "B": [0.0, 0.01, 0.02] * 20})
    scenario = Scenario(tickers=["A", "B"], weights=[0.5, 0.5], horizon_years=2, simulations=50, frequency="monthly")

    paths = generate_return_paths(returns, scenario, "historical_bootstrap")

    assert paths.shape == (50, 24, 2)


def test_simulate_paths_shape_and_contributions():
    return_paths = np.zeros((10, 12, 2))
    scenario = Scenario(tickers=["A", "B"], weights=[0.5, 0.5], horizon_years=1, simulations=10, frequency="monthly", initial_capital=1000, monthly_contribution=100)

    values, contributions, cashflows = simulate_paths(return_paths, scenario.weights, scenario)

    assert values.shape == (10, 13)
    assert contributions.shape == (10, 13)
    assert cashflows["net_profit_if_sold"].shape == (10, 13)
    assert np.isclose(values[0, -1], 2200 - values[0, -1] * 0, atol=5)
    assert np.isclose(contributions[0, -1], 2200)


def test_taxed_liquidation_path_does_not_create_terminal_cliff():
    return_paths = np.full((2, 12, 1), 0.01)
    scenario = Scenario(
        tickers=["A"],
        weights=[1.0],
        horizon_years=1,
        simulations=2,
        frequency="monthly",
        initial_capital=1000,
        monthly_contribution=0,
        annual_inflation=0.0,
        tax_mode="israel_individual",
        capital_gains_tax_rate=0.25,
    )

    values, _, _ = simulate_paths(return_paths, scenario.weights, scenario)

    assert values[0, -1] > values[0, -2]


def test_dividend_cashflows_track_gross_tax_and_net_amounts():
    return_paths = np.zeros((1, 12, 1))
    scenario = Scenario(
        tickers=["A"],
        weights=[1.0],
        horizon_years=1,
        simulations=1,
        frequency="monthly",
        initial_capital=1200,
        monthly_contribution=0,
        annual_fee=0,
        annual_tax_drag=0,
        annual_dividend_yield=0.12,
        dividend_tax_rate=0.25,
        dividend_mode="track_only",
        tax_mode="none",
    )

    values, contributions, cashflows = simulate_paths(return_paths, scenario.weights, scenario)

    assert np.isclose(values[0, -1], 1200)
    assert np.isclose(contributions[0, -1], 1200)
    assert np.isclose(cashflows["cumulative_gross_dividends"][0, -1], 144)
    assert np.isclose(cashflows["cumulative_dividend_taxes"][0, -1], 36)
    assert np.isclose(cashflows["cumulative_net_dividends"][0, -1], 108)
    assert np.isclose(cashflows["net_profit_if_sold"][0, -1], 108)


def test_run_simulation_returns_summary_and_paths():
    returns = pd.DataFrame({"A": [0.01, 0.02, -0.01] * 20})
    scenario = Scenario(tickers=["A"], weights=[1.0], horizon_years=1, simulations=25, frequency="monthly")

    result = run_simulation(returns, scenario, "normal")

    assert result["paths"].shape == (25, 13)
    assert "cashflows" in result
    assert "median_final_value" in result["summary"]
