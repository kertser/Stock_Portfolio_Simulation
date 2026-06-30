from __future__ import annotations

from pathlib import Path

import pandas as pd


CACHE_DIR = Path(".cache") / "market_data"


def cache_path(provider: str, tickers: list[str], start: str | None, end: str | None, field: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    token = "_".join(tickers).replace("/", "-")
    start_token = start or "auto"
    end_token = end or "latest"
    field_token = field.replace(" ", "_").lower()
    return CACHE_DIR / f"{provider}_{token}_{start_token}_{end_token}_{field_token}.parquet"


def read_cached_prices(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_parquet(path)


def write_cached_prices(path: Path, prices: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(path)
