# Portfolio Monte Carlo Simulator

Portfolio Monte Carlo Simulator is a local Python web application for scenario-based portfolio accumulation analysis. It downloads historical market data, estimates return statistics, runs Monte Carlo simulations, and compares how different model assumptions change the distribution of possible outcomes.

It is not a market prediction tool and it is not financial advice.

## Features

- 1-5 stock, ETF, or index tickers.
- Historical data download through `yfinance`.
- Adjusted close prices by default.
- Data quality checks for missing values, short histories, non-positive prices, and suspicious jumps.
- Historical statistics, rolling returns, rolling volatility, and correlation heatmap.
- Multi-asset portfolio weights with normalization.
- Accumulation scenario with starting capital, monthly contributions, annual contribution increases, fees, simplified tax drag, inflation, target value, and rebalancing.
- Monte Carlo models:
  - historical bootstrap;
  - block bootstrap;
  - multivariate normal;
  - fat-tail Student-t approximation;
  - optional simple regime approximation.
- Model comparison mode.
- Fan chart, final value histogram, random trajectories, drawdown distribution, and model comparison charts.
- CSV, JSON, and chart HTML export.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux, activate with:

```bash
source .venv/bin/activate
```

## Run

```bash
streamlit run portfolio_monte_carlo/app.py
```

The app opens in your browser, usually at:

```text
http://localhost:8501
```

## Test

```bash
pytest
```

## Example Scenario

- Initial capital: 10,000
- Monthly contribution: 500
- Annual contribution increase: 2%
- Horizon: 20 years
- Assets: SPY 70%, QQQ 30%
- Model: historical bootstrap
- Simulations: 10,000
- Target: 500,000

## Model Comparison

Different simulation models can produce materially different outcomes. This does not mean one model is objectively correct. It shows that long-term projections are highly assumption-dependent.

The comparison tab runs the same portfolio, contribution schedule, target, horizon, and fee assumptions through selected models, then compares downside percentiles, median outcomes, target probability, drawdowns, and worst-tail outcomes.

## Limitations

- Past performance does not guarantee future returns.
- Monte Carlo results depend strongly on model assumptions.
- Historical data can contain missing dividends, survivorship bias, split issues, currency effects, or provider errors.
- `yfinance` is convenient but not an institutional-grade data source.
- Taxes and fees are simplified unless explicitly modeled.
- Results are scenario-based projections, not predictions.

## Project Structure

```text
portfolio_monte_carlo/
  app.py
  data/
    providers.py
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
```
