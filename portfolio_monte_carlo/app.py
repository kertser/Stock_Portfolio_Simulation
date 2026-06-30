from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

from portfolio_monte_carlo.charts.plots import (
    correlation_heatmap,
    distribution_overlay,
    drawdown_histogram,
    fan_chart,
    final_value_histogram,
    historical_prices,
    model_comparison_bars,
    rolling_chart,
    sample_trajectories,
)
from portfolio_monte_carlo.core.portfolio import align_weights, correlation_matrix, portfolio_returns
from portfolio_monte_carlo.core.returns import calculate_returns, rolling_returns, rolling_volatility
from portfolio_monte_carlo.core.risk import return_statistics
from portfolio_monte_carlo.core.scenario import Scenario
from portfolio_monte_carlo.core.scenario import periods_per_year
from portfolio_monte_carlo.core.simulation import compare_models, run_simulation
from portfolio_monte_carlo.data.providers import download_yfinance_prices
from portfolio_monte_carlo.data.validation import quality_report_frame, validate_prices


st.set_page_config(
    page_title="Portfolio Monte Carlo Simulator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
      --accent: #176B87;
      --ink: #182026;
      --muted: #5f6f78;
      --surface: #f7f9f8;
    }
    .main .block-container {
      padding-top: 1.5rem;
      padding-bottom: 2rem;
      max-width: 1380px;
    }
    h1, h2, h3 {
      letter-spacing: 0;
    }
    [data-testid="stMetric"] {
      background: var(--surface);
      border: 1px solid #dfe7e4;
      border-radius: 8px;
      padding: 0.85rem 1rem;
    }
    div[data-testid="stSidebar"] {
      border-right: 1px solid #e4ebe8;
    }
    .assumption-note {
      border-left: 4px solid var(--accent);
      padding: 0.65rem 0 0.65rem 1rem;
      color: var(--muted);
      margin: 0.5rem 0 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


MODEL_LABELS = {
    "historical_bootstrap": "Historical bootstrap",
    "block_bootstrap": "Block bootstrap",
    "normal": "Parametric normal",
    "fat_tail": "Fat-tail Student-t",
    "regime": "Regime approximation",
}

CURRENCY_SYMBOLS = {"USD": "$", "ILS": "₪"}

AMOUNT_SCALES = {
    "Units": 1.0,
    "Thousands": 1_000.0,
}

MARKET_PRESETS = {
    "Israel core indices": ["TA35.TA", "^TA125.TA"],
    "Israel large caps": ["LUMI.TA", "POLI.TA", "NICE.TA", "TEVA.TA", "ICL.TA"],
    "Israel banks": ["LUMI.TA", "POLI.TA", "MZTF.TA", "DSCT.TA"],
    "US broad ETFs": ["SPY", "QQQ", "VTI"],
    "Custom": [],
}

TAX_MODE_LABELS = {
    "none": "No tax model",
    "israel_individual": "Israel individual investor, simplified 25%",
    "israel_substantial_shareholder": "Israel substantial shareholder, simplified 30%",
    "custom": "Custom capital gains rate",
}

CHART_TIME_SCALE_LABELS = {
    "years": "Years",
    "months": "Months",
    "periods": "Simulation periods",
}

DIVIDEND_MODE_LABELS = {
    "track_only": "Track only",
    "reinvest": "Reinvest net dividends",
    "withdraw": "Withdraw as cash income",
}


def _parse_list(text: str) -> list[str]:
    return [item.strip().upper() for item in text.replace(";", ",").split(",") if item.strip()]


def _parse_weights(text: str, count: int) -> list[float]:
    raw = [float(item.strip()) for item in text.replace(";", ",").split(",") if item.strip()]
    if len(raw) != count:
        raise ValueError("The number of weights must match the number of tickers.")
    return raw


def _format_money(value: float, currency: str) -> str:
    return f"{CURRENCY_SYMBOLS.get(currency, currency)}{value:,.0f}"


def _format_percent(value: float) -> str:
    return f"{value:.1%}"


def _simulation_x_axis(scenario: Scenario, points: int) -> tuple[pd.Series, str]:
    steps = pd.Series(range(points), dtype=float)
    factor = periods_per_year(scenario.frequency)
    if scenario.chart_time_scale == "years":
        return steps / factor, "Years since start"
    if scenario.chart_time_scale == "months":
        return steps * 12 / factor, "Months since start"
    return steps, f"Simulation periods ({scenario.frequency} steps)"


def _median_hover_metrics(result: dict) -> dict[str, np.ndarray]:
    cashflows = result.get("cashflows", {})
    metrics = {
        "Net profit if sold": cashflows.get("net_profit_if_sold"),
        "Net dividends": cashflows.get("cumulative_net_dividends"),
        "Gross dividends": cashflows.get("cumulative_gross_dividends"),
        "Dividend tax": cashflows.get("cumulative_dividend_taxes"),
        "Capital gains tax": cashflows.get("liquidation_taxes"),
    }
    return {
        label: np.median(values, axis=0)
        for label, values in metrics.items()
        if values is not None
    }


def _amount_input(
    label: str,
    value: float,
    scale: float,
    currency: str,
    step: float,
    help_text: str,
) -> float:
    symbol = CURRENCY_SYMBOLS.get(currency, currency)
    shown = value / scale
    suffix = "" if scale == 1 else " thousand"
    return (
        st.sidebar.number_input(
            f"{label}, {symbol}{suffix}",
            min_value=0.0,
            value=float(shown),
            step=step / scale,
            help=help_text,
        )
        * scale
    )


@st.cache_data(show_spinner=False)
def _load_prices(tickers: tuple[str, ...], start: str, end: str, field: str) -> pd.DataFrame:
    return download_yfinance_prices(list(tickers), start=start, end=end, field=field, use_cache=True)


def _scenario_from_sidebar() -> Scenario:
    uploaded_config = st.sidebar.file_uploader(
        "Load scenario JSON",
        type=["json"],
        help="Import a previously exported scenario. Values from the file become the starting point for the controls below.",
    )
    loaded = {}
    if uploaded_config is not None:
        try:
            loaded = json.loads(uploaded_config.getvalue().decode("utf-8"))
            st.sidebar.success("Scenario loaded.")
        except json.JSONDecodeError:
            st.sidebar.error("The uploaded JSON could not be parsed.")

    base = Scenario.from_dict(loaded) if loaded else Scenario()

    st.sidebar.header("Portfolio")
    preset = st.sidebar.selectbox(
        "Market preset",
        list(MARKET_PRESETS),
        index=0 if not loaded else list(MARKET_PRESETS).index("Custom"),
        help="Quick-start lists. Israeli tickers use Yahoo Finance's common Tel Aviv suffix .TA when available.",
    )
    preset_tickers = MARKET_PRESETS[preset] or base.tickers
    ticker_text = st.sidebar.text_input(
        "Tickers",
        ", ".join(preset_tickers),
        help="Enter 1-5 Yahoo Finance symbols. For Tel Aviv-listed names, try the .TA suffix, for example LUMI.TA or NICE.TA.",
    )
    tickers = _parse_list(ticker_text)[:5]
    weight_default = ", ".join(str(weight) for weight in base.weights[: len(tickers)])
    if len(base.weights) != len(tickers):
        weight_default = ", ".join(["1"] * len(tickers))
    weight_text = st.sidebar.text_input(
        "Weights",
        weight_default,
        help="Portfolio weights for the tickers above. They can be percentages or raw numbers; the app normalizes them automatically.",
    )

    lookback_years = st.sidebar.selectbox(
        "Historical period",
        [5, 10, 15, 20, 30],
        index=[5, 10, 15, 20, 30].index(base.lookback_years) if base.lookback_years in [5, 10, 15, 20, 30] else 3,
        help="How much historical data to request before the custom dates are applied.",
    )
    frequency = st.sidebar.selectbox(
        "Data frequency",
        ["daily", "weekly", "monthly"],
        index=["daily", "weekly", "monthly"].index(base.frequency),
        help="Frequency used for return calculations and simulation periods. Monthly is usually more stable for long horizons.",
    )
    price_field = st.sidebar.selectbox(
        "Price field",
        ["Adj Close", "Close"],
        index=0 if base.price_field == "Adj Close" else 1,
        help="Adjusted close includes corporate actions where the provider supplies them. It is the safer default for return analysis.",
    )

    today = date.today()
    start_default = pd.Timestamp(today) - pd.DateOffset(years=int(lookback_years))
    start_date = st.sidebar.date_input(
        "Start date",
        value=start_default.date(),
        help="First date requested from the market data provider. Actual coverage may start later for newer securities.",
    )
    end_date = st.sidebar.date_input(
        "End date",
        value=today,
        help="Last date requested from the market data provider.",
    )

    st.sidebar.header("Accumulation")
    currency = st.sidebar.selectbox(
        "Display currency",
        ["ILS", "USD"],
        index=["ILS", "USD"].index(base.currency),
        help="Currency used for inputs, outputs, labels, and exports. The app does not perform FX conversion between assets.",
    )
    amount_scale_label = st.sidebar.selectbox(
        "Amount scale",
        list(AMOUNT_SCALES),
        index=0,
        help="Choose Units for exact amounts or Thousands for faster entry of large portfolio values.",
    )
    amount_scale = AMOUNT_SCALES[amount_scale_label]
    initial_capital = _amount_input(
        "Initial capital",
        float(base.initial_capital),
        amount_scale,
        currency,
        1_000.0,
        "Current portfolio value at the beginning of the simulation.",
    )
    monthly_contribution = _amount_input(
        "Monthly contribution",
        float(base.monthly_contribution),
        amount_scale,
        currency,
        100.0,
        "Planned monthly deposit. For weekly or daily simulations it is converted to the matching period amount.",
    )
    annual_contribution_increase = st.sidebar.slider(
        "Annual contribution increase",
        0.0,
        0.10,
        float(base.annual_contribution_increase),
        0.005,
        format="%.3f",
        help="Yearly increase in contributions, for example to approximate salary growth or inflation-linked savings.",
    )
    horizon_unit = st.sidebar.selectbox(
        "Horizon scale",
        ["Years", "Months"],
        index=0,
        help="Choose whether the investment horizon is entered in years or months.",
    )
    if horizon_unit == "Years":
        horizon_length = st.sidebar.slider(
            "Investment horizon",
            1,
            50,
            int(round(base.horizon_years)),
            help="Length of the scenario in years.",
        )
        horizon_years = float(horizon_length)
    else:
        horizon_length = st.sidebar.slider(
            "Investment horizon",
            1,
            600,
            int(round(base.horizon_years * 12)),
            help="Length of the scenario in months. Useful for non-round horizons such as 18 or 30 months.",
        )
        horizon_years = float(horizon_length) / 12
    chart_time_scale = st.sidebar.selectbox(
        "Chart time scale",
        list(CHART_TIME_SCALE_LABELS),
        format_func=CHART_TIME_SCALE_LABELS.get,
        index=list(CHART_TIME_SCALE_LABELS).index(base.chart_time_scale),
        help="Unit used on simulation trajectory charts. This affects only chart labels and scale, not the simulation itself.",
    )
    simulations = st.sidebar.number_input(
        "Number of simulations",
        min_value=500,
        max_value=100_000,
        value=int(base.simulations),
        step=500,
        help="How many random paths to generate. More simulations make percentiles smoother but slower.",
    )
    target_value = _amount_input(
        "Target portfolio value",
        float(base.target_value),
        amount_scale,
        currency,
        10_000.0,
        "Goal threshold used to calculate target probability.",
    )

    st.sidebar.header("Assumptions")
    model = st.sidebar.selectbox(
        "Primary model",
        list(MODEL_LABELS),
        format_func=MODEL_LABELS.get,
        index=list(MODEL_LABELS).index(base.model),
        help="The model used in the Simulation tab. The comparison tab can run several models side by side.",
    )
    rebalancing = st.sidebar.selectbox(
        "Rebalancing",
        ["none", "monthly", "quarterly", "yearly"],
        index=["none", "monthly", "quarterly", "yearly"].index(base.rebalancing),
        help="How often the simulated portfolio is brought back to target weights.",
    )
    block_size_months = st.sidebar.selectbox(
        "Block size",
        [3, 6, 12, 24],
        index=[3, 6, 12, 24].index(base.block_size_months) if base.block_size_months in [3, 6, 12, 24] else 1,
        help="Block length for block bootstrap. Larger blocks preserve longer market episodes but reduce randomness.",
    )
    fat_tail_df = st.sidebar.slider(
        "Fat-tail degrees of freedom",
        3.0,
        30.0,
        float(base.fat_tail_df),
        0.5,
        help="Lower values create heavier tails and more extreme outcomes in the Student-t approximation.",
    )
    annual_fee = st.sidebar.slider(
        "Annual fee / expense ratio",
        0.0,
        0.03,
        float(base.annual_fee),
        0.0005,
        format="%.4f",
        help="Annual drag from management fees, ETF expense ratios, or platform fees.",
    )
    annual_tax_drag = st.sidebar.slider(
        "Other annual tax drag",
        0.0,
        0.05,
        float(base.annual_tax_drag),
        0.001,
        format="%.3f",
        help="Optional extra yearly return drag not captured elsewhere. Dividend taxes and capital gains tax are modeled separately below.",
    )
    annual_dividend_yield = st.sidebar.slider(
        "Estimated annual dividend yield",
        0.0,
        0.12,
        float(base.annual_dividend_yield),
        0.001,
        format="%.3f",
        help="Estimated yearly dividend yield of the whole portfolio. With Adjusted Close data, use Track only unless you intentionally want dividends to affect portfolio cash flows.",
    )
    dividend_tax_rate = st.sidebar.slider(
        "Dividend tax / withholding rate",
        0.0,
        0.50,
        float(base.dividend_tax_rate),
        0.005,
        format="%.3f",
        help="Tax or withholding rate applied to estimated dividends before reinvestment or withdrawal.",
    )
    dividend_mode = st.sidebar.selectbox(
        "Dividend handling",
        list(DIVIDEND_MODE_LABELS),
        format_func=DIVIDEND_MODE_LABELS.get,
        index=list(DIVIDEND_MODE_LABELS).index(base.dividend_mode),
        help="Track only records estimated dividends without changing portfolio value; reinvest adds net dividends; withdraw treats net dividends as cash income outside the portfolio.",
    )
    tax_mode = st.sidebar.selectbox(
        "Capital gains tax model",
        list(TAX_MODE_LABELS),
        format_func=TAX_MODE_LABELS.get,
        index=list(TAX_MODE_LABELS).index(base.tax_mode),
        help="Simplified Israeli tax treatment at final liquidation. It approximates tax on positive real capital gains only.",
    )
    if tax_mode == "israel_individual":
        capital_gains_tax_rate = 0.25
    elif tax_mode == "israel_substantial_shareholder":
        capital_gains_tax_rate = 0.30
    elif tax_mode == "none":
        capital_gains_tax_rate = 0.0
    else:
        capital_gains_tax_rate = st.sidebar.slider(
            "Custom capital gains tax rate",
            0.0,
            0.50,
            float(base.capital_gains_tax_rate),
            0.005,
            format="%.3f",
            help="Custom rate applied to positive real gains at final liquidation.",
        )
    annual_inflation = st.sidebar.slider(
        "Annual inflation adjustment",
        0.0,
        0.10,
        float(base.annual_inflation),
        0.005,
        format="%.3f",
        help="Used for real-return metrics and for indexing the tax basis in the simplified Israeli capital-gains model.",
    )

    try:
        weights = _parse_weights(weight_text, len(tickers))
    except ValueError as exc:
        st.sidebar.error(str(exc))
        weights = [1.0] * max(1, len(tickers))

    return Scenario(
        tickers=tickers,
        weights=weights,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        lookback_years=int(lookback_years),
        frequency=frequency,
        price_field=price_field,
        currency=currency,
        initial_capital=float(initial_capital),
        monthly_contribution=float(monthly_contribution),
        annual_contribution_increase=float(annual_contribution_increase),
        horizon_years=float(horizon_years),
        simulations=int(simulations),
        annual_fee=float(annual_fee),
        annual_tax_drag=float(annual_tax_drag),
        annual_dividend_yield=float(annual_dividend_yield),
        dividend_tax_rate=float(dividend_tax_rate),
        dividend_mode=dividend_mode,
        tax_mode=tax_mode,
        capital_gains_tax_rate=float(capital_gains_tax_rate),
        annual_inflation=float(annual_inflation),
        target_value=float(target_value),
        chart_time_scale=chart_time_scale,
        rebalancing=rebalancing,
        model=model,
        block_size_months=int(block_size_months),
        fat_tail_df=float(fat_tail_df),
        random_seed=42,
    )


def _summary_frame(summary: dict[str, float], currency: str) -> pd.DataFrame:
    labels = {
        "median_final_value": "Median final value",
        "mean_final_value": "Mean final value",
        "p5": "5th percentile",
        "p10": "10th percentile",
        "p25": "25th percentile",
        "p75": "75th percentile",
        "p90": "90th percentile",
        "p95": "95th percentile",
        "probability_reaching_target": "Probability of reaching target",
        "probability_below_contributions": "Probability below total contributions",
        "probability_negative_nominal_return": "Probability of negative nominal return",
        "probability_negative_real_return": "Probability of negative real return",
        "expected_max_drawdown": "Expected max drawdown",
        "median_max_drawdown": "Median max drawdown",
        "worst_5pct_average_outcome": "Worst 5% average outcome",
        "best_5pct_average_outcome": "Best 5% average outcome",
        "total_contributions": "Total contributions",
        "median_gain_over_contributions": "Median gain over contributions",
        "median_net_profit_if_sold": "Median net profit if sold",
        "median_cumulative_gross_dividends": "Median cumulative gross dividends",
        "median_cumulative_net_dividends": "Median cumulative net dividends",
        "median_cumulative_dividend_taxes": "Median cumulative dividend taxes",
        "median_liquidation_tax": "Median capital gains tax if sold",
    }
    money_keys = {
        "median_final_value",
        "mean_final_value",
        "p5",
        "p10",
        "p25",
        "p75",
        "p90",
        "p95",
        "worst_5pct_average_outcome",
        "best_5pct_average_outcome",
        "total_contributions",
        "median_gain_over_contributions",
        "median_net_profit_if_sold",
        "median_cumulative_gross_dividends",
        "median_cumulative_net_dividends",
        "median_cumulative_dividend_taxes",
        "median_liquidation_tax",
    }
    percent_keys = {
        "probability_reaching_target",
        "probability_below_contributions",
        "probability_negative_nominal_return",
        "probability_negative_real_return",
        "expected_max_drawdown",
        "median_max_drawdown",
    }
    rows = []
    for key, value in summary.items():
        if key in money_keys:
            display = _format_money(value, currency)
        elif key in percent_keys:
            display = _format_percent(value)
        else:
            display = value
        rows.append({"metric": labels.get(key, key), "value": value, "display": display})
    return pd.DataFrame(rows)


def main() -> None:
    scenario = _scenario_from_sidebar()
    currency_symbol = CURRENCY_SYMBOLS.get(scenario.currency, scenario.currency)

    st.title("Portfolio Monte Carlo Simulator")
    st.markdown(
        """
        <div class="assumption-note">
        Scenario-based projections for portfolio accumulation. Results are simulated outcome distributions, not predictions.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if scenario.simulations >= 50_000:
        st.warning("50,000+ simulations can be slow on some machines. Reduce the count if the interface feels sluggish.")

    if not scenario.tickers:
        st.error("Enter at least one ticker in the sidebar.")
        return

    try:
        weights = align_weights(scenario.tickers, scenario.weights)
    except ValueError as exc:
        st.error(str(exc))
        return

    try:
        with st.spinner("Downloading and validating historical data..."):
            prices = _load_prices(tuple(scenario.tickers), scenario.start_date, scenario.end_date, scenario.price_field)
            quality = quality_report_frame(validate_prices(prices))
            asset_returns = calculate_returns(prices, scenario.frequency)
            asset_returns = asset_returns.dropna(how="any")
            port_returns = portfolio_returns(asset_returns, list(weights.values))
            combined_returns = asset_returns.copy()
            combined_returns["Portfolio"] = port_returns
            stats = return_statistics(combined_returns, scenario.frequency)
    except Exception as exc:
        st.error(f"Could not prepare market data: {exc}")
        return

    tab_setup, tab_data, tab_sim, tab_compare, tab_risk, tab_export = st.tabs(
        [
            "Portfolio Setup",
            "Historical Data",
            "Simulation",
            "Model Comparison",
            "Risk Analysis",
            "Assumptions & Export",
        ]
    )

    with tab_setup:
        left, right = st.columns([1, 2])
        with left:
            st.subheader("Scenario")
            st.dataframe(weights.to_frame(), use_container_width=True)
            st.metric(
                "Initial capital",
                _format_money(scenario.initial_capital, scenario.currency),
                help="Starting portfolio value before the first simulated return period.",
            )
            st.metric(
                "Monthly contribution",
                _format_money(scenario.monthly_contribution, scenario.currency),
                help="Planned monthly savings amount before annual contribution increases.",
            )
            st.metric(
                "Target probability threshold",
                _format_money(scenario.target_value, scenario.currency),
                help="Target value used to calculate the probability of reaching the goal.",
            )
        with right:
            st.plotly_chart(historical_prices(prices, currency_symbol), use_container_width=True)

    with tab_data:
        st.subheader("Data Quality")
        st.dataframe(quality, use_container_width=True, hide_index=True)
        st.subheader("Historical Statistics")
        st.dataframe(stats, use_container_width=True)
        st.plotly_chart(correlation_heatmap(correlation_matrix(asset_returns)), use_container_width=True)

    with tab_sim:
        run_clicked = st.button(
            "Run primary simulation",
            type="primary",
            help="Generate Monte Carlo paths for the selected primary model and current scenario settings.",
        )
        if run_clicked or "primary_result" not in st.session_state:
            with st.spinner(f"Running {MODEL_LABELS[scenario.model]} simulation..."):
                st.session_state.primary_result = run_simulation(asset_returns, scenario)
                st.session_state.primary_scenario = scenario.to_dict()

        result = st.session_state.primary_result
        summary = result["summary"]
        x_values, x_axis_title = _simulation_x_axis(scenario, result["paths"].shape[1])
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric(
            "Median final value",
            _format_money(summary["median_final_value"], scenario.currency),
            help="The middle final outcome: half of simulations ended above this value and half below.",
        )
        m2.metric(
            "5th percentile",
            _format_money(summary["p5"], scenario.currency),
            help="A downside threshold: 5% of simulations ended below this value.",
        )
        m3.metric(
            "Target probability",
            _format_percent(summary["probability_reaching_target"]),
            help="Share of simulations that ended at or above the target value.",
        )
        m4.metric(
            "Median max drawdown",
            _format_percent(summary["median_max_drawdown"]),
            help="Median worst peak-to-trough decline across simulated paths.",
        )
        m5.metric(
            "Net profit if sold",
            _format_money(summary.get("median_net_profit_if_sold", summary["median_gain_over_contributions"]), scenario.currency),
            help="Median net profit after sale: after-tax liquidation value plus applicable dividend cash income minus total contributions.",
        )

        st.plotly_chart(
            fan_chart(
                result["paths"],
                currency_symbol=currency_symbol,
                x_values=x_values,
                x_axis_title=x_axis_title,
                hover_metrics=_median_hover_metrics(result),
            ),
            use_container_width=True,
        )
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(final_value_histogram(result["paths"], currency_symbol), use_container_width=True)
        with col_b:
            st.plotly_chart(
                sample_trajectories(
                    result["paths"],
                    currency_symbol=currency_symbol,
                    x_values=x_values,
                    x_axis_title=x_axis_title,
                ),
                use_container_width=True,
            )

        st.subheader("Summary Metrics")
        st.dataframe(_summary_frame(summary, scenario.currency), use_container_width=True, hide_index=True)

    with tab_compare:
        st.markdown(
            """
            <div class="assumption-note">
            Different simulation models can produce materially different outcomes. This does not mean one model is objectively correct.
            It shows that long-term projections are highly assumption-dependent.
            </div>
            """,
            unsafe_allow_html=True,
        )
        model_options = st.multiselect(
            "Models to compare",
            list(MODEL_LABELS),
            default=["historical_bootstrap", "block_bootstrap", "normal", "fat_tail"],
            format_func=MODEL_LABELS.get,
            help="Select the models to run on the same portfolio and accumulation assumptions.",
        )
        if st.button(
            "Run model comparison",
            help="Run each selected model using the same scenario so assumption sensitivity is visible.",
        ) and model_options:
            with st.spinner("Running model comparison..."):
                comparison = compare_models(asset_returns, scenario, model_options)
                comparison_results = {model: run_simulation(asset_returns, scenario, model) for model in model_options}
                st.session_state.comparison = comparison
                st.session_state.comparison_results = comparison_results
        if "comparison" in st.session_state:
            comparison = st.session_state.comparison
            st.dataframe(comparison, use_container_width=True)
            st.plotly_chart(model_comparison_bars(comparison, currency_symbol), use_container_width=True)
            st.plotly_chart(distribution_overlay(st.session_state.comparison_results, currency_symbol), use_container_width=True)

    with tab_risk:
        st.subheader("Rolling Risk")
        window = {"daily": 252, "weekly": 52, "monthly": 12}[scenario.frequency]
        st.plotly_chart(rolling_chart(rolling_returns(combined_returns, window), "Rolling Returns", "Return"), use_container_width=True)
        st.plotly_chart(rolling_chart(rolling_volatility(combined_returns, window, scenario.frequency), "Rolling Volatility", "Annualized volatility"), use_container_width=True)
        if "primary_result" in st.session_state:
            st.plotly_chart(drawdown_histogram(st.session_state.primary_result["paths"]), use_container_width=True)

    with tab_export:
        st.subheader("Assumptions")
        st.info(
            "This is not financial advice. Past performance does not guarantee future returns. "
            "Monte Carlo results depend strongly on model assumptions. Historical data may contain errors, "
            "survivorship bias, missing dividends, currency distortions or other limitations. Taxes and fees "
            "are simplified unless explicitly modeled. Results are scenario-based projections, not predictions."
        )
        st.caption(
            "Israeli tax mode is a simplified approximation: it applies the selected capital gains tax rate "
            "to positive real gains at final liquidation, using inflation-indexed contributions as the cost basis. "
            "It does not model every Israeli tax account type, offset, exemption, dividend treatment, or reporting rule."
        )
        scenario_json = json.dumps(scenario.to_dict(), indent=2)
        st.download_button(
            "Download scenario JSON",
            scenario_json,
            "scenario.json",
            "application/json",
            help="Download the complete scenario settings so they can be loaded again later.",
        )
        st.download_button(
            "Download historical statistics CSV",
            stats.to_csv().encode("utf-8"),
            "historical_statistics.csv",
            "text/csv",
            help="Download calculated historical return and risk statistics.",
        )
        if "primary_result" in st.session_state:
            summary_csv = _summary_frame(st.session_state.primary_result["summary"], scenario.currency).to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download summary CSV",
                summary_csv,
                "simulation_summary.csv",
                "text/csv",
                help="Download the calculated summary metrics for the primary simulation.",
            )
            paths_df = pd.DataFrame(st.session_state.primary_result["paths"])
            st.download_button(
                "Download simulation paths CSV",
                paths_df.to_csv(index=False).encode("utf-8"),
                "simulation_paths.csv",
                "text/csv",
                help="Download every simulated portfolio path. This can be a large file for many simulations.",
            )
            cashflows = st.session_state.primary_result["cashflows"]
            final_cashflows = pd.DataFrame(
                {
                    "final_after_tax_liquidation_value": st.session_state.primary_result["paths"][:, -1],
                    "total_contributions": st.session_state.primary_result["contributions"][:, -1],
                    "net_profit_if_sold": cashflows["net_profit_if_sold"][:, -1],
                    "cumulative_gross_dividends": cashflows["cumulative_gross_dividends"][:, -1],
                    "cumulative_net_dividends": cashflows["cumulative_net_dividends"][:, -1],
                    "cumulative_dividend_taxes": cashflows["cumulative_dividend_taxes"][:, -1],
                    "capital_gains_tax_if_sold": cashflows["liquidation_taxes"][:, -1],
                }
            )
            st.download_button(
                "Download final cashflows CSV",
                final_cashflows.to_csv(index=False).encode("utf-8"),
                "simulation_final_cashflows.csv",
                "text/csv",
                help="Download per-simulation final sale profit, dividend totals, and modeled tax amounts.",
            )
            export_x_values, export_x_axis_title = _simulation_x_axis(
                scenario,
                st.session_state.primary_result["paths"].shape[1],
            )
            chart_html = fan_chart(
                st.session_state.primary_result["paths"],
                currency_symbol=currency_symbol,
                x_values=export_x_values,
                x_axis_title=export_x_axis_title,
                hover_metrics=_median_hover_metrics(st.session_state.primary_result),
            ).to_html(include_plotlyjs="cdn")
            st.download_button(
                "Download fan chart HTML",
                chart_html,
                "fan_chart.html",
                "text/html",
                help="Download the interactive fan chart as a standalone HTML file.",
            )


if __name__ == "__main__":
    main()
