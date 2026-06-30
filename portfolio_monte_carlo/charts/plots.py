from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


ACCENT = "#176B87"
INK = "#172026"
MUTED = "#647179"
WARM = "#C7772D"
GRID = "#E6ECEA"


def _finish(fig: go.Figure, yaxis_title: str | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Inter, Segoe UI, Arial, sans-serif", color=INK, size=13),
        title=dict(font=dict(size=19, color=INK), x=0.01, xanchor="left"),
        margin=dict(l=28, r=24, t=62, b=42),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hoverlabel=dict(bgcolor="white", bordercolor=GRID, font_size=12, font_color=INK),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, linecolor=GRID, title=yaxis_title)
    return fig


def _path_x(paths: np.ndarray, x_values: np.ndarray | None) -> np.ndarray:
    if x_values is None:
        return np.arange(paths.shape[1])
    return x_values


def fan_chart(
    paths: np.ndarray,
    title: str = "Simulated Outcome Distribution",
    currency_symbol: str = "$",
    x_values: np.ndarray | None = None,
    x_axis_title: str = "Simulation period",
    hover_metrics: dict[str, np.ndarray] | None = None,
    value_label: str = "After-tax liquidation value",
) -> go.Figure:
    percentiles = {
        "5th": np.percentile(paths, 5, axis=0),
        "10th": np.percentile(paths, 10, axis=0),
        "25th": np.percentile(paths, 25, axis=0),
        "50th": np.percentile(paths, 50, axis=0),
        "75th": np.percentile(paths, 75, axis=0),
        "90th": np.percentile(paths, 90, axis=0),
        "95th": np.percentile(paths, 95, axis=0),
    }
    x = _path_x(paths, x_values)
    customdata = None
    median_hover = f"{x_axis_title}: %{{x:.2f}}<br>{value_label}: {currency_symbol}%{{y:,.0f}}"
    if hover_metrics:
        ordered = list(hover_metrics.items())
        customdata = np.column_stack([values for _, values in ordered])
        extra_lines = [
            f"{label}: {currency_symbol}%{{customdata[{index}]:,.0f}}"
            for index, (label, _) in enumerate(ordered)
        ]
        median_hover = median_hover + "<br>" + "<br>".join(extra_lines)
    fig = go.Figure()
    bands = [("5th", "95th"), ("10th", "90th"), ("25th", "75th")]
    colors = ["rgba(23, 107, 135, 0.10)", "rgba(23, 107, 135, 0.18)", "rgba(23, 107, 135, 0.30)"]
    for (lower, upper), color in zip(bands, colors):
        fig.add_trace(go.Scatter(x=x, y=percentiles[upper], mode="lines", line=dict(width=0), showlegend=False))
        fig.add_trace(
            go.Scatter(
                x=x,
                y=percentiles[lower],
                mode="lines",
                fill="tonexty",
                fillcolor=color,
                line=dict(width=0),
                name=f"{lower}-{upper}",
                hovertemplate=f"{x_axis_title}: %{{x:.2f}}<br>{value_label}: {currency_symbol}%{{y:,.0f}}<extra>{lower}-{upper}</extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=percentiles["50th"],
            mode="lines",
            name="Median",
            line=dict(color=ACCENT, width=3),
            customdata=customdata,
            hovertemplate=median_hover + "<extra>Median</extra>",
        )
    )
    fig.update_layout(title=title, xaxis_title=x_axis_title)
    return _finish(fig, f"{value_label} ({currency_symbol})")


def sample_trajectories(
    paths: np.ndarray,
    sample_size: int = 80,
    currency_symbol: str = "$",
    x_values: np.ndarray | None = None,
    x_axis_title: str = "Simulation period",
) -> go.Figure:
    rng = np.random.default_rng(7)
    indices = rng.choice(paths.shape[0], size=min(sample_size, paths.shape[0]), replace=False)
    x = _path_x(paths, x_values)
    fig = go.Figure()
    for idx in indices:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=paths[idx],
                mode="lines",
                line=dict(color="rgba(23, 107, 135, 0.16)", width=1),
                showlegend=False,
                hovertemplate=f"%{{x:.2f}}<br>{currency_symbol}%{{y:,.0f}}<extra></extra>",
            )
        )
    fig.update_layout(title="Random Sample Trajectories", xaxis_title=x_axis_title)
    return _finish(fig, f"Portfolio value ({currency_symbol})")


def final_value_histogram(paths: np.ndarray, currency_symbol: str = "$") -> go.Figure:
    final_values = paths[:, -1]
    fig = px.histogram(final_values, nbins=60, title="Final Portfolio Value Distribution", color_discrete_sequence=[ACCENT])
    fig.update_traces(marker_line_width=0, opacity=0.86, hovertemplate=f"{currency_symbol}%{{x:,.0f}}<br>%{{y}} simulations<extra></extra>")
    fig.update_layout(xaxis_title=f"Final value ({currency_symbol})", showlegend=False)
    return _finish(fig, "Simulation count")


def drawdown_histogram(paths: np.ndarray) -> go.Figure:
    drawdowns = np.min(paths / np.maximum.accumulate(paths, axis=1) - 1, axis=1)
    fig = px.histogram(drawdowns, nbins=50, title="Max Drawdown Distribution", color_discrete_sequence=[WARM])
    fig.update_traces(marker_line_width=0, opacity=0.86, hovertemplate="%{x:.1%}<br>%{y} simulations<extra></extra>")
    fig.update_layout(xaxis_title="Max drawdown", showlegend=False)
    return _finish(fig, "Simulation count")


def historical_prices(prices: pd.DataFrame, currency_symbol: str = "$") -> go.Figure:
    fig = px.line(prices, title="Historical Prices")
    fig.update_traces(line_width=2.3, hovertemplate="%{x|%Y-%m-%d}<br>" + currency_symbol + "%{y:,.2f}<extra>%{fullData.name}</extra>")
    fig.update_layout(xaxis_title="Date")
    return _finish(fig, f"Adjusted close ({currency_symbol})")


def rolling_chart(data: pd.DataFrame | pd.Series, title: str, yaxis_title: str) -> go.Figure:
    fig = px.line(data, title=title)
    fig.update_traces(line_width=2.1)
    fig.update_layout(xaxis_title="Date")
    return _finish(fig, yaxis_title)


def model_comparison_bars(comparison: pd.DataFrame, currency_symbol: str = "$") -> go.Figure:
    cols = ["p5", "median_final_value", "p95"]
    tidy = comparison[cols].reset_index().melt(id_vars="model", var_name="metric", value_name="value")
    fig = px.bar(
        tidy,
        x="model",
        y="value",
        color="metric",
        barmode="group",
        title="Model Comparison: Downside, Median, Upside",
        color_discrete_sequence=[WARM, ACCENT, "#6C8C5A"],
    )
    fig.update_traces(marker_line_width=0, opacity=0.9)
    fig.update_layout(xaxis_title="Model")
    return _finish(fig, f"Final portfolio value ({currency_symbol})")


def distribution_overlay(results: dict[str, dict], currency_symbol: str = "$") -> go.Figure:
    fig = go.Figure()
    for model, result in results.items():
        fig.add_trace(
            go.Histogram(
                x=result["paths"][:, -1],
                name=model,
                opacity=0.55,
                nbinsx=50,
                histnorm="probability density",
                hovertemplate=f"{currency_symbol}%{{x:,.0f}}<br>Density %{{y:.4f}}<extra>%{{fullData.name}}</extra>",
            )
        )
    fig.update_layout(
        title="Final Value Distribution Overlay",
        xaxis_title=f"Final portfolio value ({currency_symbol})",
        barmode="overlay",
    )
    return _finish(fig, "Density")


def income_waterfall(summary: dict[str, float], currency_symbol: str = "$", title: str = "Income and Tax Breakdown") -> go.Figure:
    contributions = summary.get("total_contributions", 0)
    net_profit = summary.get("median_net_profit_if_sold", summary.get("median_gain_over_contributions", 0))
    net_dividends = summary.get("median_cumulative_net_dividends", 0)
    dividend_tax = -summary.get("median_cumulative_dividend_taxes", 0)
    capital_gains_tax = -summary.get("median_liquidation_tax", 0)
    final_value = summary.get("median_final_value", 0)
    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "relative", "relative", "relative", "total"],
            x=["Contributions", "Net profit", "Net dividends", "Dividend tax", "Capital gains tax", "Liquidation value"],
            y=[contributions, net_profit, net_dividends, dividend_tax, capital_gains_tax, final_value],
            connector={"line": {"color": GRID}},
            increasing={"marker": {"color": ACCENT}},
            decreasing={"marker": {"color": WARM}},
            totals={"marker": {"color": "#6C8C5A"}},
            hovertemplate=f"%{{x}}<br>{currency_symbol}%{{y:,.0f}}<extra></extra>",
        )
    )
    fig.update_layout(title=title, showlegend=False, xaxis_title="")
    return _finish(fig, f"Amount ({currency_symbol})")


def target_probability_gauge(probability: float, title: str = "Target Probability") -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 34, "color": INK}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": GRID},
                "bar": {"color": ACCENT, "thickness": 0.22},
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 35], "color": "rgba(199,119,45,0.18)"},
                    {"range": [35, 70], "color": "rgba(242,180,44,0.18)"},
                    {"range": [70, 100], "color": "rgba(23,107,135,0.16)"},
                ],
            },
            title={"text": title, "font": {"size": 18, "color": INK}},
        )
    )
    fig.update_layout(template="plotly_white", margin=dict(l=24, r=24, t=54, b=24), height=290)
    return fig


def correlation_heatmap(corr: pd.DataFrame, title: str = "Correlation Matrix") -> go.Figure:
    labels = [str(c) for c in corr.columns]
    z = corr.values
    text = [[f"{v:.2f}" for v in row] for row in z]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            colorscale=[[0, WARM], [0.5, "#f7f9f8"], [1, ACCENT]],
            zmid=0,
            zmin=-1,
            zmax=1,
            text=text,
            texttemplate="%{text}",
            textfont={"size": 13},
            hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>",
            showscale=True,
        )
    )
    fig.update_layout(
        title=title,
        height=max(320, len(labels) * 80 + 80),
    )
    return _finish(fig)


def contribution_growth_area(
    paths: np.ndarray,
    contributions: np.ndarray,
    currency_symbol: str = "$",
    x_values: np.ndarray | None = None,
    x_axis_title: str = "Simulation period",
) -> go.Figure:
    x = _path_x(paths, x_values)
    median_portfolio = np.percentile(paths, 50, axis=0)
    median_contributions = np.percentile(contributions, 50, axis=0)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=median_contributions,
            name="Cumulative Contributions",
            fill="tozeroy",
            fillcolor="rgba(23, 107, 135, 0.30)",
            line=dict(color=ACCENT, width=2),
            hovertemplate=f"{x_axis_title}: %{{x:.2f}}<br>Contributions: {currency_symbol}%{{y:,.0f}}<extra>Contributions</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=median_portfolio,
            name="Median Portfolio Value",
            fill="tonexty",
            fillcolor="rgba(108, 140, 90, 0.30)",
            line=dict(color="#4a7c59", width=2.5),
            hovertemplate=f"{x_axis_title}: %{{x:.2f}}<br>Portfolio: {currency_symbol}%{{y:,.0f}}<extra>Portfolio</extra>",
        )
    )
    fig.update_layout(title="Contributions vs Portfolio Growth (Median Path)", xaxis_title=x_axis_title)
    return _finish(fig, f"Value ({currency_symbol})")


def annual_returns_bar(returns_df: pd.DataFrame, title: str = "Annual Returns by Year") -> go.Figure:
    """Bar chart of annual returns grouped by asset.  Positive bars are blue, negative are orange."""
    df = returns_df.dropna(how="all")
    annual = (1 + df).resample("YE").prod() - 1
    _colors = [ACCENT, "#4a7c59", "#9b59b6", "#e74c3c", "#3498db"]
    fig = go.Figure()
    for i, col in enumerate(annual.columns):
        base_rgb = _colors[i % len(_colors)]
        y = annual[col] * 100
        bar_colors = [WARM if v < 0 else base_rgb for v in y]
        fig.add_trace(
            go.Bar(
                x=annual.index.year,
                y=y,
                name=str(col),
                marker_color=bar_colors,
                hovertemplate=(
                    "<b>%{x}</b><br>Annual return: %{y:.1f}%%<extra>" + str(col) + "</extra>"
                ),
            )
        )
    fig.add_hline(y=0, line_color=MUTED, line_width=1, line_dash="dot")
    fig.update_layout(
        title=title,
        barmode="group",
        xaxis=dict(title="Year", tickmode="linear", dtick=1),
    )
    return _finish(fig, "Annual Return (%)")


def returns_distribution(returns_df: pd.DataFrame, title: str = "Periodic Return Distribution") -> go.Figure:
    """Overlapping histogram of periodic (monthly/weekly/daily) returns for each asset."""
    _colors = [ACCENT, "#4a7c59", "#9b59b6", "#e74c3c", "#3498db"]
    fig = go.Figure()
    for i, col in enumerate(returns_df.dropna(how="all").columns):
        data = returns_df[col].dropna() * 100
        fig.add_trace(
            go.Histogram(
                x=data,
                name=str(col),
                opacity=0.68,
                nbinsx=55,
                marker_color=_colors[i % len(_colors)],
                hovertemplate="Return: %{x:.2f}%<br>Count: %{y}<extra>" + str(col) + "</extra>",
            )
        )
    fig.add_vline(x=0, line_color=MUTED, line_width=1.5, line_dash="dot",
                  annotation_text="0%", annotation_position="top right",
                  annotation_font_size=11, annotation_font_color=MUTED)
    fig.update_layout(title=title, barmode="overlay")
    return _finish(fig, "Frequency")
