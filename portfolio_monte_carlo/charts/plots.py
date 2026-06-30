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
            hovertemplate=f"{x_axis_title}: %{{x:.2f}}<br>{currency_symbol}%{{y:,.0f}}<extra>{lower}-{upper}</extra>",
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


def correlation_heatmap(correlation: pd.DataFrame) -> go.Figure:
    fig = px.imshow(
        correlation,
        text_auto=".2f",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        title="Correlation Matrix",
    )
    fig.update_layout(coloraxis_colorbar=dict(title="Correlation"))
    return _finish(fig)


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
