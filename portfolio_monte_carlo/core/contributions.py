from __future__ import annotations

import numpy as np

from portfolio_monte_carlo.core.scenario import Frequency, periods_per_year


def contribution_schedule(
    monthly_contribution: float,
    annual_increase: float,
    horizon_years: float,
    frequency: Frequency,
) -> np.ndarray:
    factor = periods_per_year(frequency)
    periods = max(1, int(round(horizon_years * factor)))
    period_contribution = monthly_contribution * 12 / factor
    schedule = np.zeros(periods, dtype=float)
    for period in range(periods):
        year = period // factor
        schedule[period] = period_contribution * ((1 + annual_increase) ** year)
    return schedule


def inflation_indexed_basis(
    initial_capital: float,
    schedule: np.ndarray,
    annual_inflation: float,
    frequency: Frequency,
) -> float:
    factor = periods_per_year(frequency)
    period_inflation = (1 + annual_inflation) ** (1 / factor) - 1
    periods = len(schedule)
    basis = initial_capital * ((1 + period_inflation) ** periods)
    for period, contribution in enumerate(schedule):
        remaining = periods - period - 1
        basis += contribution * ((1 + period_inflation) ** remaining)
    return float(basis)


def inflation_indexed_basis_path(
    initial_capital: float,
    schedule: np.ndarray,
    annual_inflation: float,
    frequency: Frequency,
) -> np.ndarray:
    factor = periods_per_year(frequency)
    period_inflation = (1 + annual_inflation) ** (1 / factor) - 1
    periods = len(schedule)
    basis = np.empty(periods + 1, dtype=float)
    basis[0] = initial_capital
    for time_index in range(1, periods + 1):
        value = initial_capital * ((1 + period_inflation) ** time_index)
        for period in range(time_index):
            value += schedule[period] * ((1 + period_inflation) ** (time_index - period - 1))
        basis[time_index] = value
    return basis
