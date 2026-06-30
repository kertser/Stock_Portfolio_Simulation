from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class DataQualityReport:
    ticker: str
    start: str | None
    end: str | None
    observations: int
    missing_values: int
    zero_or_negative_prices: int
    suspicious_jumps: int
    warnings: list[str]


def validate_price_series(series: pd.Series, ticker: str) -> DataQualityReport:
    clean_index = series.dropna().index
    returns = series.pct_change(fill_method=None)
    missing = int(series.isna().sum())
    zero_or_negative = int((series <= 0).sum())
    suspicious = int((returns.abs() > 0.35).sum())
    warnings = []
    if len(clean_index) < 60:
        warnings.append("Very short price history.")
    if missing:
        warnings.append(f"{missing} missing price observations.")
    if zero_or_negative:
        warnings.append(f"{zero_or_negative} non-positive prices detected.")
    if suspicious:
        warnings.append(f"{suspicious} daily moves above 35%; inspect for splits or data errors.")

    return DataQualityReport(
        ticker=ticker,
        start=str(clean_index.min().date()) if len(clean_index) else None,
        end=str(clean_index.max().date()) if len(clean_index) else None,
        observations=int(series.notna().sum()),
        missing_values=missing,
        zero_or_negative_prices=zero_or_negative,
        suspicious_jumps=suspicious,
        warnings=warnings,
    )


def validate_prices(prices: pd.DataFrame) -> list[DataQualityReport]:
    return [validate_price_series(prices[column], str(column)) for column in prices.columns]


def quality_report_frame(reports: list[DataQualityReport]) -> pd.DataFrame:
    rows = []
    for report in reports:
        rows.append(
            {
                "ticker": report.ticker,
                "start": report.start,
                "end": report.end,
                "observations": report.observations,
                "missing_values": report.missing_values,
                "zero_or_negative_prices": report.zero_or_negative_prices,
                "suspicious_jumps": report.suspicious_jumps,
                "warnings": "; ".join(report.warnings) if report.warnings else "OK",
            }
        )
    return pd.DataFrame(rows)
