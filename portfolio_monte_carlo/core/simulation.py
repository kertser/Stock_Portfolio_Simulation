from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_monte_carlo.core.contributions import contribution_schedule, inflation_indexed_basis_path
from portfolio_monte_carlo.core.models import (
    block_bootstrap,
    fat_tail_student_t,
    historical_bootstrap,
    parametric_normal,
    regime_switching,
)
from portfolio_monte_carlo.core.portfolio import normalize_weights
from portfolio_monte_carlo.core.risk import summarize_simulation
from portfolio_monte_carlo.core.scenario import Scenario, horizon_periods, periods_per_year


def generate_return_paths(asset_returns: pd.DataFrame, scenario: Scenario, model: str | None = None) -> np.ndarray:
    selected_model = model or scenario.model
    rng = np.random.default_rng(scenario.random_seed)
    periods = horizon_periods(scenario.horizon_years, scenario.frequency)

    if selected_model == "historical_bootstrap":
        return historical_bootstrap(asset_returns, scenario.simulations, periods, rng)
    if selected_model == "block_bootstrap":
        return block_bootstrap(
            asset_returns,
            scenario.simulations,
            periods,
            scenario.block_size_months,
            scenario.frequency,
            rng,
        )
    if selected_model == "normal":
        return parametric_normal(asset_returns, scenario.simulations, periods, rng)
    if selected_model == "fat_tail":
        return fat_tail_student_t(asset_returns, scenario.simulations, periods, scenario.fat_tail_df, rng)
    if selected_model == "regime":
        return regime_switching(asset_returns, scenario.simulations, periods, rng)
    raise ValueError(f"Unsupported simulation model: {selected_model}")


def _is_rebalance_period(period_index: int, scenario: Scenario) -> bool:
    if scenario.rebalancing == "none":
        return False
    factor = periods_per_year(scenario.frequency)
    if scenario.rebalancing == "monthly":
        interval = max(1, round(factor / 12))
    elif scenario.rebalancing == "quarterly":
        interval = max(1, round(factor / 4))
    else:
        interval = factor
    return (period_index + 1) % interval == 0


def simulate_paths(
    asset_return_paths: np.ndarray,
    weights: list[float],
    scenario: Scenario,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    simulations, periods, assets = asset_return_paths.shape
    normalized_weights = normalize_weights(weights)
    if len(normalized_weights) != assets:
        raise ValueError("Weights length must match simulated asset count.")

    values_by_asset = np.zeros((simulations, assets), dtype=float)
    values_by_asset[:] = scenario.initial_capital * normalized_weights

    portfolio_values = np.empty((simulations, periods + 1), dtype=float)
    cumulative_contributions = np.empty_like(portfolio_values)
    cumulative_gross_dividends = np.zeros_like(portfolio_values)
    cumulative_net_dividends = np.zeros_like(portfolio_values)
    cumulative_dividend_taxes = np.zeros_like(portfolio_values)
    portfolio_values[:, 0] = scenario.initial_capital
    cumulative_contributions[:, 0] = scenario.initial_capital

    schedule = contribution_schedule(
        scenario.monthly_contribution,
        scenario.annual_contribution_increase,
        scenario.horizon_years,
        scenario.frequency,
    )
    period_fee = scenario.annual_fee / periods_per_year(scenario.frequency)
    period_tax_drag = scenario.annual_tax_drag / periods_per_year(scenario.frequency)
    period_dividend_yield = scenario.annual_dividend_yield / periods_per_year(scenario.frequency)
    drag = max(0.0, period_fee + period_tax_drag)

    for period in range(periods):
        contribution = schedule[period]
        cumulative_gross_dividends[:, period + 1] = cumulative_gross_dividends[:, period]
        cumulative_net_dividends[:, period + 1] = cumulative_net_dividends[:, period]
        cumulative_dividend_taxes[:, period + 1] = cumulative_dividend_taxes[:, period]

        values_by_asset += contribution * normalized_weights
        values_by_asset *= 1 + asset_return_paths[:, period, :]
        total = values_by_asset.sum(axis=1)
        total_after_drag = np.maximum(total * (1 - drag), 0)

        if scenario.annual_dividend_yield > 0:
            gross_dividend = total_after_drag * period_dividend_yield
            dividend_tax = gross_dividend * max(0.0, scenario.dividend_tax_rate)
            net_dividend = np.maximum(gross_dividend - dividend_tax, 0)
            cumulative_gross_dividends[:, period + 1] += gross_dividend
            cumulative_net_dividends[:, period + 1] += net_dividend
            cumulative_dividend_taxes[:, period + 1] += dividend_tax

            if scenario.dividend_mode == "reinvest":
                total_after_drag = total_after_drag + net_dividend
            elif scenario.dividend_mode == "withdraw":
                total_after_drag = np.maximum(total_after_drag - gross_dividend, 0)

        scale = np.divide(total_after_drag, total, out=np.zeros_like(total), where=total != 0)
        values_by_asset *= scale[:, None]

        if _is_rebalance_period(period, scenario):
            values_by_asset = total_after_drag[:, None] * normalized_weights

        portfolio_values[:, period + 1] = total_after_drag
        cumulative_contributions[:, period + 1] = cumulative_contributions[:, period] + contribution

    liquidation_values = portfolio_values.copy()
    liquidation_taxes = np.zeros_like(portfolio_values)
    if scenario.tax_mode != "none" and scenario.capital_gains_tax_rate > 0:
        indexed_basis = inflation_indexed_basis_path(
            scenario.initial_capital,
            schedule,
            scenario.annual_inflation,
            scenario.frequency,
        )
        taxable_real_gain = np.maximum(portfolio_values - indexed_basis[None, :], 0)
        liquidation_taxes = taxable_real_gain * scenario.capital_gains_tax_rate
        liquidation_values = np.maximum(portfolio_values - liquidation_taxes, 0)

    dividend_cash_credit = (
        cumulative_net_dividends
        if scenario.dividend_mode in {"track_only", "withdraw"}
        else np.zeros_like(cumulative_net_dividends)
    )
    net_profit_if_sold = liquidation_values + dividend_cash_credit - cumulative_contributions
    cashflows = {
        "gross_portfolio_values": portfolio_values,
        "cumulative_gross_dividends": cumulative_gross_dividends,
        "cumulative_net_dividends": cumulative_net_dividends,
        "cumulative_dividend_taxes": cumulative_dividend_taxes,
        "liquidation_taxes": liquidation_taxes,
        "net_profit_if_sold": net_profit_if_sold,
    }

    return liquidation_values, cumulative_contributions, cashflows


def run_simulation(
    asset_returns: pd.DataFrame,
    scenario: Scenario,
    model: str | None = None,
) -> dict:
    selected_model = model or scenario.model
    return_paths = generate_return_paths(asset_returns, scenario, selected_model)
    portfolio_paths, contributions, cashflows = simulate_paths(return_paths, scenario.weights, scenario)
    summary = summarize_simulation(
        portfolio_paths,
        contributions,
        scenario.target_value,
        scenario.annual_inflation,
        scenario.frequency,
        cashflows,
    )
    return {
        "model": selected_model,
        "paths": portfolio_paths,
        "contributions": contributions,
        "cashflows": cashflows,
        "summary": summary,
    }


def compare_models(asset_returns: pd.DataFrame, scenario: Scenario, models: list[str]) -> pd.DataFrame:
    rows = []
    for model in models:
        result = run_simulation(asset_returns, scenario, model)
        row = {"model": model, **result["summary"]}
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")
