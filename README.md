# Portfolio Monte Carlo Simulator

Portfolio Monte Carlo Simulator is a local Python web application for scenario-based portfolio accumulation analysis. It downloads historical market data, estimates return statistics, runs Monte Carlo simulations, and compares how different model assumptions change the distribution of possible outcomes.

It is not a market prediction tool and it is not financial advice.

## Features

- 1–5 stock, ETF, or index tickers.
- Historical data download through `yfinance` with local disk cache.
- Adjusted close prices by default; data quality checks included.
- Historical statistics: CAGR, annualised return/volatility, max drawdown, Sharpe ratio, Sortino ratio, Calmar ratio, VaR, CVaR, skewness, kurtosis, correlation matrix.
- Multi-asset portfolio weights with automatic normalisation.
- Accumulation scenario: starting capital, monthly contributions, annual contribution increases, fees, tax drag, dividends, simplified capital gains tax, inflation, target value, rebalancing.
- Five Monte Carlo models:
  - Historical bootstrap (preserves correlations)
  - Block bootstrap (preserves serial structure)
  - Parametric normal (multivariate)
  - Fat-tail Student-t
  - Regime approximation
- Model comparison mode: same scenario through all selected models side by side.
- Charts: fan chart, contribution-vs-growth area, final value histogram, random trajectories, income waterfall, target probability gauge, rolling returns/volatility, drawdown distribution, correlation heatmap, model comparison bars, distribution overlay.
- CSV and JSON export of results, paths, cashflows, and scenario configuration.
- Import saved scenarios from JSON.

## Technology Stack

| Layer | Library |
|---|---|
| Web UI | [Dash](https://dash.plotly.com/) 4.x + [dash-bootstrap-components](https://dash-bootstrap-components.opensource.faculty.ai/) (FLATLY theme) |
| Charts | Plotly |
| Data | yfinance, pandas, numpy |
| Maths | scipy |
| Tests | pytest |
| Package manager | [uv](https://docs.astral.sh/uv/) |
| Container | Docker + Docker Compose |

The legacy Streamlit interface (`portfolio_monte_carlo/app.py`) is kept for reference but the Dash app is the primary interface.

## Install (local)

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv sync
```

This creates `.venv` and installs all dependencies from `uv.lock`.

## Run (local)

```bash
uv run monte-carlo
```

Or directly:

```bash
uv run python portfolio_monte_carlo/dash_app.py
```

The app opens at:

```
http://localhost:8050
```

To run the legacy Streamlit interface:

```bash
uv run streamlit run portfolio_monte_carlo/app.py
```

## Run with Docker

```bash
docker compose up --build
```

The app will be available at `http://localhost:8050`.

To stop:

```bash
docker compose down
```

## Test

```bash
uv run pytest
```

## Example Scenario

- Initial capital: 10 000
- Monthly contribution: 1 000
- Annual contribution increase: 2%
- Horizon: 20 years
- Assets: SPY 70%, QQQ 30%
- Primary model: Historical bootstrap
- Simulations: 10 000
- Target: 500 000

## Model Comparison

Different simulation models can produce materially different outcomes. This does not mean one model is objectively correct. It shows that long-term projections are highly assumption-dependent.

The comparison tab runs the same portfolio, contribution schedule, target, horizon, and fee assumptions through selected models, then compares downside percentiles, median outcomes, target probability, drawdowns, and worst-tail outcomes.

## Data Sources

| Source | Usage |
|---|---|
| Yahoo Finance via `yfinance` | Historical daily OHLCV and adjusted close prices |
| Local disk cache | Avoids re-downloading unchanged data |

`yfinance` is convenient but not institutional-grade. Historical data may contain missing dividends, survivorship bias, split issues, currency effects, or provider errors.

## Limitations

- Past performance does not guarantee future returns.
- Monte Carlo results depend strongly on model assumptions.
- Taxes and fees are simplified; the Israeli tax model approximates capital gains tax on positive real gains at final liquidation only.
- Results are scenario-based projections, not predictions.

## Project Structure

```text
portfolio_monte_carlo/
  dash_app.py           ← Dash web application (primary)
  app.py                ← Legacy Streamlit interface
  assets/
    custom.css          ← Dash CSS overrides
  data/
    providers.py        ← yfinance download with cache
    cache.py            ← local disk cache
    validation.py       ← data quality checks
  core/
    returns.py          ← return calculations and annualisation
    portfolio.py        ← weights and portfolio returns
    contributions.py    ← contribution schedule and inflation-indexed basis
    simulation.py       ← path generation and accumulation loop
    models.py           ← five Monte Carlo models
    risk.py             ← risk metrics, Sharpe/Sortino/Calmar, simulation summary
    scenario.py         ← Scenario dataclass + serialisation
  charts/
    plots.py            ← all Plotly figures
tests/
  test_returns.py
  test_risk.py
  test_simulation.py
pyproject.toml          ← project metadata and dependencies (uv)
uv.lock                 ← pinned dependency lock file
Dockerfile
docker-compose.yml
```

