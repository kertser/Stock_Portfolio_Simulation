# Portfolio Monte Carlo Simulator

## Task For Coding Agent

You are a senior Python developer with experience in quantitative finance, risk modeling, and data applications.

Build a Python web application called **Portfolio Monte Carlo Simulator** for probabilistic modeling of long-term investment portfolio accumulation.

The application must not claim to predict the market or guarantee profit. It must show a distribution of possible outcomes under user-defined assumptions, historical data, and different simulation models.

## Product Goal

The user should be able to:

1. Enter 1-5 tickers for stocks, ETFs, or indices.
2. Download historical market data from the internet.
3. Inspect data quality and historical return statistics.
4. Define accumulation parameters: initial capital, regular contributions, fees, inflation, investment horizon, and target value.
5. Run Monte Carlo simulations.
6. See a distribution of possible future outcomes rather than a point forecast.
7. Compare several simulation models and understand how strongly results depend on assumptions.
8. Export results and scenario configuration.

## Recommended Technology Stack

Use Python 3.11+.

**Current stack (as of this version):**

- **Dash 4.x** for the primary web interface (`dash_app.py`).
- **dash-bootstrap-components** with FLATLY theme for layout and styling.
- pandas and numpy for calculations.
- scipy for statistical computations.
- Plotly for interactive charts (all charts in `charts/plots.py`).
- yfinance as the primary free market data provider.
- Local disk cache for market data.
- pytest for tests.
- Docker + Docker Compose for containerisation.

The legacy Streamlit interface (`app.py`) is kept for reference but the Dash app is the primary interface. New UI work should target `dash_app.py`.

Alternative libraries are acceptable if they improve quality, but the application must remain easy to run locally and via Docker.

## Architecture

Keep the code modular. **The UI must not contain core financial logic.**

All financial calculations live in `core/`. All chart building lives in `charts/plots.py`. The Dash app (`dash_app.py`) only assembles layout and wires callbacks; it imports from `core/` and `charts/`.

Server-side state for large numpy arrays (simulation paths) is stored in a module-level `_CACHE` dict keyed by UUID. Only the UUID is stored in `dcc.Store` (browser). This keeps the browser payload small.

**Current structure:**

```text
portfolio_monte_carlo/
  dash_app.py           ← primary Dash application
  app.py                ← legacy Streamlit interface (kept for reference)
  assets/
    custom.css          ← Dash CSS overrides (auto-served by Dash)
  data/
    providers.py        ← yfinance download with cache
    cache.py
    validation.py
  core/
    returns.py          ← calculate_returns, rolling_returns, rolling_volatility, annualisation
    portfolio.py        ← align_weights, portfolio_returns
    contributions.py    ← contribution_schedule, inflation_indexed_basis_path
    simulation.py       ← generate_return_paths, simulate_paths, run_simulation, compare_models
    models.py           ← 5 MC models: historical_bootstrap, block_bootstrap, parametric_normal,
                           fat_tail_student_t, regime_switching
    risk.py             ← return_statistics (Sharpe/Sortino/Calmar included), summarize_simulation,
                           drawdown, max_drawdown, historical_var/cvar
    scenario.py         ← Scenario dataclass, serialisation, horizon_periods
  charts/
    plots.py            ← all Plotly figures: fan_chart, correlation_heatmap,
                           contribution_growth_area, model_comparison_bars, etc.
tests/
  test_returns.py
  test_risk.py
  test_simulation.py
Dockerfile
docker-compose.yml
pyproject.toml          ← project metadata and dependencies (uv)
uv.lock                 ← pinned dependency lock file
```
    cache.py
    validation.py
  core/
    returns.py
    portfolio.py
    contributions.py
    simulation.py
    models.py
    risk.py
    scenario.py
  charts/
    plots.py
  tests/
    test_returns.py
    test_risk.py
    test_simulation.py
  README.md
```

The user scenario must be serializable to JSON.

## Data Requirements

Minimum requirements:

- Download historical prices through yfinance.
- Use adjusted close by default.
- Show the actual downloaded date range.
- Check missing values.
- Check zero prices and suspicious price anomalies.
- Detect short or insufficient history.

Nice to have:

- Add local caching.
- Design an extensible interface for additional data providers.
- Optionally support FRED or CPI data for inflation.

API errors must be handled gracefully and shown in human-readable language. Do not expose raw traceback output in the UI.

## Portfolio Requirements

Support:

- One or more assets.
- Asset weights.
- Weight normalization.
- Historical portfolio returns.
- Correlation matrix.
- Rebalancing modes:
  - none;
  - monthly;
  - quarterly;
  - yearly.

For multi-asset portfolios, do not simulate assets independently in models where correlation matters. Preserve or model the joint return distribution.

## Historical Statistics

Calculate for each asset and for the portfolio:

- mean return;
- annualized return;
- annualized volatility;
- CAGR;
- max drawdown;
- skewness;
- kurtosis;
- historical VaR;
- historical CVaR;
- correlation matrix;
- rolling returns;
- rolling volatility.

Annualization factors:

- daily: 252;
- weekly: 52;
- monthly: 12.

## Monte Carlo Models

Implement at least four models.

### 1. Historical Bootstrap

Sample historical returns with replacement.

For multiple assets, sample whole rows of historical returns so historical correlations are preserved.

### 2. Block Bootstrap

Sample consecutive blocks of historical returns.

The user should be able to choose block size:

- 3 months;
- 6 months;
- 12 months;
- custom.

The goal is to partially preserve market serial structure: crises, recoveries, and high-volatility periods.

### 3. Parametric Normal Model

Generate returns from a multivariate normal distribution.

Use:

- historical mean vector;
- covariance matrix.

### 4. Fat-Tail Model

Implement a fat-tail model such as multivariate Student-t or a practical Student-t approximation.

The user should be able to choose:

- conservative;
- balanced;
- aggressive;

or manually set degrees of freedom.

### Optional: Regime Model

If the core application is complete and tested, add a simple regime-switching approximation:

- normal market;
- crisis;
- recovery.

Do not build an academically complex HMM model for the MVP. Prefer a clear and honest approximation with visible parameters.

## Accumulation Model

The user defines:

- initial capital;
- monthly contribution;
- annual contribution increase;
- investment horizon;
- number of simulations;
- annual management fee or ETF expense ratio;
- simplified tax drag;
- inflation adjustment;
- target portfolio value.

Base logic:

```text
V[t+1] = (V[t] + contribution[t]) * (1 + return[t]) - fees[t] - taxes[t]
```

The MVP tax model should be simplified. Clearly state this in the UI.

## Simulation Results

Show a distribution, not a single number.

Metrics:

- median final value;
- mean final value;
- 5th / 10th / 25th / 75th / 90th / 95th percentiles;
- probability of reaching target;
- probability of ending below total contributions;
- probability of negative nominal return;
- probability of negative real return;
- expected max drawdown;
- median max drawdown;
- worst 5% average outcome;
- best 5% average outcome;
- total contributions;
- investment gain over contributions.

Charts:

- fan chart with percentile bands;
- median trajectory;
- random sample trajectories;
- histogram or density chart of final value;
- cumulative contributions vs simulated portfolio value;
- drawdown distribution;
- historical price chart;
- rolling returns;
- rolling volatility;
- correlation heatmap;
- asset statistics table.

## Model Comparison

Model comparison is mandatory.

Run the same scenario through several models:

- historical bootstrap;
- block bootstrap;
- normal;
- fat-tail;
- optional regime model.

Show side-by-side:

- median final value;
- 5th percentile;
- 10th percentile;
- 90th / 95th percentile;
- probability of reaching target;
- probability below total contributions;
- probability of negative real return;
- median max drawdown;
- worst 5% average outcome.

Charts:

- grouped bar chart;
- overlay fan chart;
- final value distribution overlay;
- comparison table;
- target probability by model.

Required explanatory text in the UI:

> Different simulation models can produce materially different outcomes. This does not mean one model is objectively correct. It shows that long-term projections are highly assumption-dependent.

## UI And UX

Build a clean, modern, understandable interface.

Recommended layout:

- sidebar for inputs;
- main area for results;
- tabs:
  1. Portfolio Setup
  2. Historical Data
  3. Simulation
  4. Model Comparison
  5. Risk Analysis
  6. Assumptions & Export

Requirements:

- interactive charts;
- clear labels;
- tooltips for complex parameters;
- progress indicator for long operations;
- human-readable errors;
- no raw technical script feeling in the UI.

Avoid these terms:

- guaranteed return;
- market prediction;
- safe forecast;
- expected profit as certainty.

Prefer these terms:

- simulated outcome distribution;
- scenario-based projection;
- percentile bands;
- downside risk;
- upside scenarios;
- target probability;
- uncertainty range;
- model assumptions.

## Export

Add downloads for:

- summary table as CSV;
- simulation results as CSV or Parquet;
- scenario configuration as JSON;
- scenario configuration import from JSON;
- charts as HTML if convenient.

## Required Disclaimers

Show these disclaimers in the UI:

- This is not financial advice.
- Past performance does not guarantee future returns.
- Monte Carlo results depend strongly on model assumptions.
- Historical data may contain errors, survivorship bias, missing dividends, currency distortions or other limitations.
- Taxes and fees are simplified unless explicitly modeled.
- Results are scenario-based projections, not predictions.

## Performance

Requirements:

- Use numpy vectorization where practical.
- Support 10,000 simulations without noticeable problems.
- For 50,000+ simulations, show a warning or offer an optimized mode.
- Cache market data.
- Avoid re-downloading data unnecessarily.
- Avoid recomputing charts and simulations when inputs have not changed.

## Tests

Add pytest tests for:

- return calculation;
- annualization;
- CAGR;
- max drawdown;
- weighted portfolio returns;
- contribution schedule;
- Monte Carlo output shape;
- percentile calculations;
- model comparison summary.

## README

README must explain:

- what the application does;
- how to install dependencies;
- how to run the application;
- which data sources are used;
- which simulation models are available;
- what model comparison means;
- model limitations;
- an example scenario.

Example scenario:

- Initial capital: 10,000
- Monthly contribution: 500
- Annual contribution increase: 2%
- Horizon: 20 years
- Assets: SPY 70%, QQQ 30%
- Model: historical bootstrap
- Simulations: 10,000
- Target: 500,000

## Definition Of Done: MVP

The MVP is ready when the user can:

1. Run the web application locally.
2. Enter 1-5 tickers.
3. Download historical data from the internet.
4. See data quality checks.
5. See historical return statistics.
6. Configure accumulation parameters.
7. Run a Monte Carlo simulation.
8. See a fan chart.
9. See final value histogram.
10. See summary metrics.
11. Compare at least two models.
12. Download results.

## Definition Of Done: Good Version

The good version is ready when:

- multi-asset portfolio works;
- correlations are preserved correctly;
- historical bootstrap, block bootstrap, normal, and fat-tail models exist;
- full model comparison mode exists;
- the UI is clean and polished (Dash + DBC, fintech dashboard layout);
- caching exists (local disk cache for market data; server-side cache for simulation arrays);
- export exists (CSV, JSON, paths, cashflows);
- README exists and covers Docker;
- basic tests exist (16+ passing);
- code is modular and extensible;
- Sharpe, Sortino, and Calmar ratios are shown in historical statistics;
- correlation heatmap chart is shown in the data tab;
- contribution-vs-growth area chart is shown in the simulation tab;
- the application can be built and run with `docker compose up --build`.

## Containerisation

The application ships with a production-ready `Dockerfile` and `docker-compose.yml`.

Requirements:
- Multi-stage build: builder stage installs deps, runtime stage copies only installed packages.
- Non-root user inside the container.
- Health check against `/_dash-layout`.
- `PORT` and `HOST` configurable via environment variables.
- A named Docker volume persists the yfinance data cache.
- The image exposes port 8050.

## Product Philosophy

Do not build a "future prediction machine".

Build a tool that helps the user reason about risk, uncertainty, drawdowns, model assumptions, and the probability of reaching a target.
