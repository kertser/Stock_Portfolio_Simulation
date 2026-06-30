from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal


Frequency = Literal["daily", "weekly", "monthly"]
Rebalancing = Literal["none", "monthly", "quarterly", "yearly"]
Currency = Literal["USD", "ILS"]
TaxMode = Literal["none", "israel_individual", "israel_substantial_shareholder", "custom"]
ChartTimeScale = Literal["years", "months", "periods"]
DividendMode = Literal["track_only", "reinvest", "withdraw"]
SimulationModel = Literal[
    "historical_bootstrap",
    "block_bootstrap",
    "normal",
    "fat_tail",
    "regime",
]


@dataclass(slots=True)
class Scenario:
    tickers: list[str] = field(default_factory=lambda: ["^GSPC"])
    weights: list[float] = field(default_factory=lambda: [1.0])
    start_date: str | None = None
    end_date: str | None = None
    lookback_years: int = 20
    frequency: Frequency = "monthly"
    price_field: str = "Adj Close"
    currency: Currency = "ILS"
    initial_capital: float = 10_000.0
    monthly_contribution: float = 1_000.0
    annual_contribution_increase: float = 0.02
    horizon_years: float = 20.0
    simulations: int = 10_000
    annual_fee: float = 0.001
    annual_tax_drag: float = 0.0
    annual_dividend_yield: float = 0.0
    dividend_tax_rate: float = 0.25
    dividend_mode: DividendMode = "track_only"
    tax_mode: TaxMode = "israel_individual"
    capital_gains_tax_rate: float = 0.25
    annual_inflation: float = 0.02
    target_value: float = 500_000.0
    chart_time_scale: ChartTimeScale = "years"
    rebalancing: Rebalancing = "monthly"
    model: SimulationModel = "historical_bootstrap"
    block_size_months: int = 6
    fat_tail_df: float = 5.0
    random_seed: int | None = 42

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Scenario":
        allowed = set(cls.__dataclass_fields__)
        clean = {key: value for key, value in data.items() if key in allowed}
        return cls(**clean)


def periods_per_year(frequency: Frequency) -> int:
    return {"daily": 252, "weekly": 52, "monthly": 12}[frequency]


def months_to_periods(months: int, frequency: Frequency) -> int:
    if frequency == "daily":
        return max(1, round(months * 252 / 12))
    if frequency == "weekly":
        return max(1, round(months * 52 / 12))
    return max(1, months)


def horizon_periods(horizon_years: float, frequency: Frequency) -> int:
    return max(1, int(round(horizon_years * periods_per_year(frequency))))
