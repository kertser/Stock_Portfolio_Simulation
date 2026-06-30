from __future__ import annotations

from datetime import date

import pandas as pd

from portfolio_monte_carlo.data.cache import cache_path, read_cached_prices, write_cached_prices


def _extract_price_field(raw: pd.DataFrame, tickers: list[str], field: str) -> pd.DataFrame:
    if isinstance(raw.columns, pd.MultiIndex):
        if field in raw.columns.get_level_values(0):
            prices = raw[field].copy()
        elif "Close" in raw.columns.get_level_values(0):
            prices = raw["Close"].copy()
        else:
            raise RuntimeError(f"Downloaded data does not contain {field} or Close prices.")
    else:
        selected_field = field if field in raw.columns else "Close"
        if selected_field not in raw.columns:
            raise RuntimeError(f"Downloaded data does not contain {field} or Close prices.")
        prices = raw[[selected_field]].copy()
        prices.columns = tickers[:1]
    return prices.reindex(columns=tickers)


def _download_raw(yf, tickers: list[str], start: str | None, end: str | None) -> pd.DataFrame:
    return yf.download(
        tickers=" ".join(tickers),
        start=start,
        end=end or date.today().isoformat(),
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=False,
    )


def download_yfinance_prices(
    tickers: list[str],
    start: str | None = None,
    end: str | None = None,
    field: str = "Adj Close",
    use_cache: bool = True,
) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is not installed. Install requirements.txt first.") from exc

    clean_tickers = [ticker.strip().upper() for ticker in tickers if ticker.strip()]
    if not clean_tickers:
        raise ValueError("Enter at least one ticker.")

    path = cache_path("yfinance", clean_tickers, start, end, field)
    if use_cache:
        cached = read_cached_prices(path)
        if cached is not None and not cached.empty:
            return cached

    raw = _download_raw(yf, clean_tickers, start, end)
    if raw.empty:
        raise RuntimeError("No market data was returned. Check tickers and date range.")

    prices = _extract_price_field(raw, clean_tickers, field)
    missing_tickers = [ticker for ticker in clean_tickers if ticker not in prices or prices[ticker].dropna().empty]
    if missing_tickers and len(clean_tickers) > 1:
        recovered = []
        for ticker in clean_tickers:
            single_raw = _download_raw(yf, [ticker], start, end)
            if single_raw.empty:
                continue
            recovered.append(_extract_price_field(single_raw, [ticker], field))
        if recovered:
            prices = pd.concat(recovered, axis=1).reindex(columns=clean_tickers)

    prices = prices.reindex(columns=clean_tickers).dropna(how="all")
    prices.index = pd.to_datetime(prices.index)
    if prices.empty:
        raise RuntimeError("Downloaded prices are empty after cleaning.")

    if use_cache:
        write_cached_prices(path, prices)
    return prices
