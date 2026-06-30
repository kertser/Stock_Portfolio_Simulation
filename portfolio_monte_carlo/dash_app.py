"""Portfolio Monte Carlo Simulator — Dash application.

Run with:
    python -m portfolio_monte_carlo.dash_app
or:
    python portfolio_monte_carlo/dash_app.py
"""
from __future__ import annotations

import json
import uuid
from datetime import date

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import Input, Output, State, callback, dcc, html, no_update
from dash.dash_table import DataTable

from portfolio_monte_carlo.charts.plots import (
    annual_returns_bar,
    contribution_growth_area,
    distribution_overlay,
    drawdown_histogram,
    fan_chart,
    final_value_histogram,
    historical_prices,
    income_waterfall,
    model_comparison_bars,
    returns_distribution,
    rolling_chart,
    sample_trajectories,
    target_probability_gauge,
)
from portfolio_monte_carlo.core.portfolio import align_weights, portfolio_returns
from portfolio_monte_carlo.core.returns import calculate_returns, rolling_returns, rolling_volatility
from portfolio_monte_carlo.core.risk import return_statistics
from portfolio_monte_carlo.core.scenario import Scenario, periods_per_year
from portfolio_monte_carlo.core.simulation import compare_models, run_simulation
from portfolio_monte_carlo.data.providers import download_yfinance_prices
from portfolio_monte_carlo.data.validation import quality_report_frame, validate_prices

# ── Constants ─────────────────────────────────────────────────────────────────

CURRENCY_SYMBOLS: dict[str, str] = {"USD": "$", "ILS": "₪"}

MARKET_PRESETS: dict[str, list[str]] = {
    "S&P 500 index": ["^GSPC"],
    "S&P 500 ETF (SPY)": ["SPY"],
    "NASDAQ-100 (QQQ)": ["QQQ"],
    "US broad ETFs": ["SPY", "QQQ", "VTI"],
    "Israel core indices": ["TA35.TA", "^TA125.TA"],
    "Israel large caps": ["LUMI.TA", "POLI.TA", "NICE.TA", "TEVA.TA"],
    "Custom": [],
}

DISPLAY_SYMBOLS: dict[str, str] = {
    "^GSPC": "S&P 500",
    "TA35.TA": "TA-35",
    "^TA125.TA": "TA-125",
    "TA90.TA": "TA-90",
}

MODEL_OPTIONS = [
    {"label": "Historical Bootstrap", "value": "historical_bootstrap"},
    {"label": "Block Bootstrap", "value": "block_bootstrap"},
    {"label": "Parametric Normal", "value": "normal"},
    {"label": "Fat-tail Student-t", "value": "fat_tail"},
    {"label": "Regime Approximation", "value": "regime"},
]

TAX_MODES = [
    {"label": "No tax model", "value": "none"},
    {"label": "Israel individual — simplified 25%", "value": "israel_individual"},
    {"label": "Israel substantial shareholder — 30%", "value": "israel_substantial_shareholder"},
    {"label": "Custom capital gains rate", "value": "custom"},
]

DIVIDEND_MODES = [
    {"label": "Track only (no cash-flow effect)", "value": "track_only"},
    {"label": "Reinvest net dividends", "value": "reinvest"},
    {"label": "Withdraw as cash income", "value": "withdraw"},
]

ISRAEL_INDICES = [
    ("TA-35", "TA35.TA", "Large-cap benchmark"),
    ("TA-90", "TA90.TA", "Outside TA-35 within TA-125"),
    ("TA-125", "^TA125.TA", "TA-35 + TA-90 broad benchmark"),
    ("TA-Banks5", "TA-BANKS5.TA", "Five largest banks"),
    ("TA-Technology", "TA-TECH.TA", "Technology sector"),
    ("TA-Finance", "TA-FINANCE.TA", "Financial sector"),
    ("TA-RealEstate", "TA-REALESTATE.TA", "Real estate sector"),
    ("TA-SME60", "TA-SME60.TA", "Small / mid cap"),
    ("TA-AllShare", "TA-ALL.TA", "Broad all-share"),
    ("TA-Biomed", "TA-BIOMED.TA", "Biomed sector"),
]

DISCLAIMER = (
    "This is not financial advice. Past performance does not guarantee future returns. "
    "Monte Carlo results depend strongly on model assumptions. Historical data may contain "
    "errors, survivorship bias, missing dividends, currency distortions or other limitations. "
    "Taxes and fees are simplified unless explicitly modelled. "
    "Results are scenario-based projections, not predictions."
)

TOOLTIP_TEXTS: dict[str, str] = {
    "market-preset": "Quick selection of common index/ETF portfolios. 'Custom' lets you enter any tickers.",
    "tickers-input": "Yahoo Finance symbols, comma-separated (e.g. SPY, QQQ, ^GSPC). TASE stocks end with .TA",
    "weights-input": "Allocation weights per ticker, comma-separated. Auto-normalised to 100%.",
    "lookback-years": "Years of historical data used to estimate return statistics for the simulation.",
    "frequency": "Return calculation period. Monthly is recommended; daily/weekly need more history.",
    "price-field": "Adj Close adjusts for dividends and splits — recommended. Close is raw.",
    "currency": "Display currency for all monetary values in the app.",
    "initial-capital": "Starting portfolio value at the beginning of the simulation horizon.",
    "monthly-contrib": "Fixed amount added to the portfolio each period (monthly by default).",
    "annual-contrib-increase": "Annual growth rate of the monthly contribution — e.g. salary increases.",
    "horizon-years": "Total investment horizon in years.",
    "simulations": "Number of Monte Carlo paths. 10 000 is a good balance of accuracy and speed.",
    "target-value": "Goal portfolio value. The app computes the probability of reaching this target.",
    "primary-model": (
        "Statistical model for return sampling. "
        "Historical Bootstrap: resamples actual return blocks — preserves fat tails and correlations. "
        "Block Bootstrap: resamples multi-period blocks — preserves serial structure. "
        "Normal: multivariate Gaussian — fast, may underestimate tail risk. "
        "Fat-tail: Student-t — heavier tails than normal. "
        "Regime: bull/bear switching model."
    ),
    "rebalancing": "How often to restore asset weights to their targets. 'None' lets weights drift.",
    "block-size": "Block length (months) for Block Bootstrap. Longer = more serial dependence preserved.",
    "fat-tail-df": "Degrees of freedom for the Student-t model. Lower = heavier tails (3 is very fat).",
    "annual-fee": "Annual fund fee / TER (e.g. 0.07% for SPY, 0.2% for typical TA index fund).",
    "tax-drag": "Additional annual tax on returns beyond dividends (e.g. foreign dividend withholding at source).",
    "dividend-yield": "Estimated gross annual dividend yield of the portfolio.",
    "dividend-tax": "Dividend withholding or income tax rate applied to gross dividends.",
    "dividend-mode": (
        "Track only: dividends tracked but not modelled as cash flows. "
        "Reinvest: net dividends added back to portfolio value. "
        "Withdraw: gross dividends removed as cash income."
    ),
    "tax-mode": (
        "Capital gains tax at liquidation. Israeli models apply 25% or 30% only to real gains "
        "(above inflation-indexed cost basis) at the end of the horizon."
    ),
    "annual-inflation": "Expected CPI inflation rate for real-return calculations and indexed cost basis.",
    "start-date": "Start date of the historical data range used to estimate return statistics (YYYY-MM-DD). Longer history captures more market cycles.",
    "end-date": "End date of the historical data range (YYYY-MM-DD). Defaults to today. Recent data better reflects current market structure.",
    "custom-tax-rate": "Custom capital gains tax rate applied to real (inflation-adjusted) gains at final liquidation.",
    "btn-load": "Download historical prices from Yahoo Finance and compute return statistics. Auto-loads on startup with default settings.",
    "btn-run": "Run Monte Carlo simulation with the selected model and parameters. Load market data first.",
    "btn-reset": "Reset all sidebar settings to their factory defaults.",
}

# ── Translations ───────────────────────────────────────────────────────────────

TRANSLATIONS: dict[str, dict[str, str]] = {
    "en": {
        "nav_subtitle": "Scenario-based projections · Not financial advice",
        "tab_setup": "📊 Portfolio Setup",
        "tab_data": "📈 Historical Data",
        "tab_sim": "🎲 Simulation",
        "tab_compare": "⚖️ Model Comparison",
        "tab_risk": "🛡️ Risk Analysis",
        "tab_export": "📤 Export",
        "acc_portfolio": "Portfolio & Data",
        "acc_accumulation": "Accumulation",
        "acc_model": "Model & Assumptions",
        "btn_load": "Load Data",
        "btn_run": "Run Simulation",
        "btn_reset": "Reset to Defaults",
        "tip_market_preset": "Quick selection of common index/ETF portfolios. 'Custom' lets you enter any tickers.",
        "tip_tickers_input": "Yahoo Finance symbols, comma-separated (e.g. SPY, QQQ, ^GSPC). TASE stocks end with .TA",
        "tip_weights_input": "Allocation weights per ticker, comma-separated. Auto-normalised to 100%.",
        "tip_lookback_years": "Years of historical data used to estimate return statistics for the simulation.",
        "tip_frequency": "Return calculation period. Monthly is recommended; daily/weekly need more history.",
        "tip_price_field": "Adj Close adjusts for dividends and splits — recommended. Close is raw.",
        "tip_currency": "Display currency for all monetary values in the app.",
        "tip_start_date": "Start date of the historical data range used to estimate return statistics (YYYY-MM-DD). Longer history captures more market cycles.",
        "tip_end_date": "End date of the historical data range (YYYY-MM-DD). Defaults to today. Recent data better reflects current market structure.",
        "tip_initial_capital": "Starting portfolio value at the beginning of the simulation horizon.",
        "tip_monthly_contrib": "Fixed amount added to the portfolio each period (monthly by default).",
        "tip_annual_contrib_increase": "Annual growth rate of the monthly contribution — e.g. salary increases.",
        "tip_horizon_years": "Total investment horizon in years.",
        "tip_simulations": "Number of Monte Carlo paths. 10 000 is a good balance of accuracy and speed.",
        "tip_target_value": "Goal portfolio value. The app computes the probability of reaching this target.",
        "tip_primary_model": (
            "Statistical model for return sampling. "
            "Historical Bootstrap: resamples actual return blocks — preserves fat tails and correlations. "
            "Block Bootstrap: resamples multi-period blocks — preserves serial structure. "
            "Normal: multivariate Gaussian — fast, may underestimate tail risk. "
            "Fat-tail: Student-t — heavier tails than normal. "
            "Regime: bull/bear switching model."
        ),
        "tip_rebalancing": "How often to restore asset weights to their targets. 'None' lets weights drift.",
        "tip_block_size": "Block length (months) for Block Bootstrap. Longer = more serial dependence preserved.",
        "tip_fat_tail_df": "Degrees of freedom for the Student-t model. Lower = heavier tails (3 is very fat).",
        "tip_annual_fee": "Annual fund fee / TER (e.g. 0.07% for SPY, 0.2% for typical TA index fund).",
        "tip_tax_drag": "Additional annual tax on returns beyond dividends (e.g. foreign dividend withholding at source).",
        "tip_dividend_yield": "Estimated gross annual dividend yield of the portfolio.",
        "tip_dividend_tax": "Dividend withholding or income tax rate applied to gross dividends.",
        "tip_dividend_mode": (
            "Track only: dividends tracked but not modelled as cash flows. "
            "Reinvest: net dividends added back to portfolio value. "
            "Withdraw: gross dividends removed as cash income."
        ),
        "tip_tax_mode": (
            "Capital gains tax at liquidation. Israeli models apply 25% or 30% only to real gains "
            "(above inflation-indexed cost basis) at the end of the horizon."
        ),
        "tip_custom_tax_rate": "Custom capital gains tax rate applied to real (inflation-adjusted) gains at final liquidation.",
        "tip_annual_inflation": "Expected CPI inflation rate for real-return calculations and indexed cost basis.",
        "tip_btn_load": "Download historical prices from Yahoo Finance and compute return statistics. Auto-loads on startup with default settings.",
        "tip_btn_run": "Run Monte Carlo simulation with the selected model and parameters. Load market data first.",
        "tip_btn_reset": "Reset all sidebar settings to their factory defaults.",
    },
    "ru": {
        "nav_subtitle": "Прогнозы на основе сценариев · Не является финансовым советом",
        "tab_setup": "📊 Настройка портфеля",
        "tab_data": "📈 Исторические данные",
        "tab_sim": "🎲 Симуляция",
        "tab_compare": "⚖️ Сравнение моделей",
        "tab_risk": "🛡️ Анализ рисков",
        "tab_export": "📤 Экспорт",
        "acc_portfolio": "Портфель и данные",
        "acc_accumulation": "Накопление",
        "acc_model": "Модель и допущения",
        "btn_load": "Загрузить данные",
        "btn_run": "Запустить симуляцию",
        "btn_reset": "Сбросить настройки",
        "tip_market_preset": "Быстрый выбор распространённых индексов/ETF. 'Custom' позволяет ввести любые тикеры.",
        "tip_tickers_input": "Тикеры Yahoo Finance через запятую (например: SPY, QQQ, ^GSPC). Акции TASE оканчиваются на .TA",
        "tip_weights_input": "Веса активов через запятую. Автоматически нормируются до 100%.",
        "tip_lookback_years": "Количество лет исторических данных для оценки параметров модели.",
        "tip_frequency": "Период расчёта доходности. Рекомендуется ежемесячный; дневной/недельный требует больше истории.",
        "tip_price_field": "Adj Close скорректирован на дивиденды и сплиты — рекомендуется. Close — сырые данные.",
        "tip_currency": "Отображаемая валюта для всех денежных значений в приложении.",
        "tip_start_date": "Дата начала исторических данных для оценки параметров модели (ГГГГ-ММ-ДД).",
        "tip_end_date": "Дата окончания исторических данных (ГГГГ-ММ-ДД). По умолчанию — сегодня.",
        "tip_initial_capital": "Начальная стоимость портфеля на момент начала горизонта симуляции.",
        "tip_monthly_contrib": "Фиксированная сумма, добавляемая в портфель каждый период.",
        "tip_annual_contrib_increase": "Ежегодный темп роста ежемесячного взноса — например, рост зарплаты.",
        "tip_horizon_years": "Общий инвестиционный горизонт в годах.",
        "tip_simulations": "Количество путей Монте-Карло. 10 000 — хороший баланс точности и скорости.",
        "tip_target_value": "Целевая стоимость портфеля. Приложение вычисляет вероятность достижения этой цели.",
        "tip_primary_model": (
            "Статистическая модель для выборки доходностей. "
            "Исторический бутстрэп: ресэмплинг реальных блоков — сохраняет тяжёлые хвосты и корреляции. "
            "Блочный бутстрэп: сохраняет серийную структуру. "
            "Нормальная: многомерное Гауссовское — быстрая, может недооценивать хвостовые риски. "
            "Тяжёлые хвосты: распределение Стьюдента. "
            "Режимы: модель переключения бычий/медвежий рынок."
        ),
        "tip_rebalancing": "Как часто восстанавливать целевые веса активов. 'Нет' — веса свободно дрейфуют.",
        "tip_block_size": "Длина блока (месяцы) для блочного бутстрэпа. Больше = сохраняется больше серийной зависимости.",
        "tip_fat_tail_df": "Степени свободы для модели Стьюдента. Меньше = более тяжёлые хвосты (3 = очень тяжёлые).",
        "tip_annual_fee": "Годовая комиссия фонда / TER (например: 0.07% для SPY).",
        "tip_tax_drag": "Дополнительный годовой налог на доходность помимо дивидендов.",
        "tip_dividend_yield": "Ожидаемая годовая дивидендная доходность портфеля брутто.",
        "tip_dividend_tax": "Налог на дивиденды или ставка удержания.",
        "tip_dividend_mode": (
            "Только отслеживание: дивиденды учитываются, но не влияют на денежные потоки. "
            "Реинвестирование: чистые дивиденды добавляются обратно в стоимость портфеля. "
            "Вывод: дивиденды брутто выводятся как доход наличными."
        ),
        "tip_tax_mode": (
            "Налог на прирост капитала при ликвидации. Израильские модели применяют 25% или 30% "
            "только к реальной прибыли (сверх индексированной базы стоимости) в конце горизонта."
        ),
        "tip_custom_tax_rate": "Пользовательская ставка налога на прирост капитала (реальной прибыли) при ликвидации.",
        "tip_annual_inflation": "Ожидаемый уровень инфляции ИПЦ для расчётов реальной доходности и индексированной базы.",
        "tip_btn_load": "Загрузить исторические цены с Yahoo Finance и рассчитать статистику. Загружается автоматически при запуске.",
        "tip_btn_run": "Запустить симуляцию Монте-Карло с выбранной моделью. Сначала загрузите данные.",
        "tip_btn_reset": "Сбросить все настройки к заводским значениям по умолчанию.",
    },
    "he": {
        "nav_subtitle": "תחזיות מבוססות תרחישים · אינו ייעוץ פיננסי",
        "tab_setup": "📊 הגדרת תיק",
        "tab_data": "📈 נתונים היסטוריים",
        "tab_sim": "🎲 סימולציה",
        "tab_compare": "⚖️ השוואת מודלים",
        "tab_risk": "🛡️ ניתוח סיכונים",
        "tab_export": "📤 ייצוא",
        "acc_portfolio": "תיק ונתונים",
        "acc_accumulation": "צבירה",
        "acc_model": "מודל והנחות",
        "btn_load": "טען נתונים",
        "btn_run": "הפעל סימולציה",
        "btn_reset": "אפס הגדרות",
        "tip_market_preset": "בחירה מהירה של מדדים/קרנות סל נפוצות. 'Custom' מאפשר הזנת כל תיקרים.",
        "tip_tickers_input": "סימולי Yahoo Finance מופרדים בפסיקים (למשל: SPY, QQQ, ^GSPC). מניות TASE מסתיימות ב-.TA",
        "tip_weights_input": "משקלות הקצאה לכל נכס, מופרדים בפסיקים. מנורמלים אוטומטית ל-100%.",
        "tip_lookback_years": "מספר שנות נתונים היסטוריים לאמידת פרמטרי המודל.",
        "tip_frequency": "תקופת חישוב התשואה. חודשי מומלץ; יומי/שבועי דורש יותר היסטוריה.",
        "tip_price_field": "Adj Close מותאם לדיבידנדים ופיצולים — מומלץ. Close הוא נתון גולמי.",
        "tip_currency": "מטבע התצוגה לכל הערכים הכספיים באפליקציה.",
        "tip_start_date": "תאריך תחילת הנתונים ההיסטוריים לאמידת פרמטרי המודל (YYYY-MM-DD).",
        "tip_end_date": "תאריך סיום הנתונים ההיסטוריים (YYYY-MM-DD). ברירת מחדל — היום.",
        "tip_initial_capital": "שווי תיק ההשקעות בתחילת אופק הסימולציה.",
        "tip_monthly_contrib": "סכום קבוע המתווסף לתיק בכל תקופה (חודשי כברירת מחדל).",
        "tip_annual_contrib_increase": "קצב הגידול השנתי של ההפקדה החודשית — למשל, עליות שכר.",
        "tip_horizon_years": "אופק ההשקעה הכולל בשנים.",
        "tip_simulations": "מספר מסלולי מונטה קרלו. 10,000 הוא איזון טוב בין דיוק למהירות.",
        "tip_target_value": "שווי תיק יעד. האפליקציה מחשבת את ההסתברות להגיע ליעד זה.",
        "tip_primary_model": (
            "מודל סטטיסטי לדגימת תשואות. "
            "Bootstrap היסטורי: דגימה מחדש של בלוקים — שומר על זנבות כבדים וקורלציות. "
            "Bootstrap בלוקים: שומר על מבנה סדרתי. "
            "נורמלי: גאוסיאני רב-משתני — מהיר, עלול לזלזל בסיכון הזנבות. "
            "זנבות כבדים: התפלגות t של סטיודנט. "
            "משטרים: מודל מיתוג שוק עולה/יורד."
        ),
        "tip_rebalancing": "באיזו תדירות לשחזר את משקלות היעד. 'ללא' מאפשר סחיפת משקלות.",
        "tip_block_size": "אורך הבלוק (חודשים) ל-Bootstrap בלוקים. ארוך יותר = שמירה טובה יותר על תלות סדרתית.",
        "tip_fat_tail_df": "דרגות חופש למודל t של סטיודנט. נמוך יותר = זנבות כבדים יותר (3 = כבד מאוד).",
        "tip_annual_fee": "דמי ניהול שנתיים / TER (למשל: 0.07% ל-SPY).",
        "tip_tax_drag": "מס שנתי נוסף על התשואות מעבר לדיבידנדים.",
        "tip_dividend_yield": "תשואת דיבידנד שנתית ברוטו משוערת של התיק.",
        "tip_dividend_tax": "שיעור מס על דיבידנדים או ניכוי במקור.",
        "tip_dividend_mode": (
            "מעקב בלבד: דיבידנדים נעקבים ללא השפעה על תזרים מזומנים. "
            "השקעה מחדש: דיבידנדים נטו מתווספים חזרה לשווי התיק. "
            "משיכה: דיבידנדים ברוטו מוסרים כהכנסה."
        ),
        "tip_tax_mode": (
            "מס רווחי הון בפירוק. מודלים ישראלים מחילים 25% או 30% רק על רווח ריאלי "
            "(מעל בסיס עלות צמוד מדד) בסוף האופק."
        ),
        "tip_custom_tax_rate": "שיעור מס רווחי הון מותאם אישית על הרווח הריאלי בפירוק סופי.",
        "tip_annual_inflation": "שיעור אינפלציה CPI צפוי לחישובי תשואה ריאלית ובסיס עלות צמוד.",
        "tip_btn_load": "הורד מחירים היסטוריים מ-Yahoo Finance וחשב סטטיסטיקות. נטען אוטומטית בהפעלה עם הגדרות ברירת מחדל.",
        "tip_btn_run": "הפעל סימולציית מונטה קרלו עם המודל והפרמטרים שנבחרו. טען נתוני שוק תחילה.",
        "tip_btn_reset": "אפס את כל ההגדרות לערכי ברירת המחדל המקוריים.",
    },
}

# ── App initialisation ─────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY, dbc.icons.BOOTSTRAP],
    title="Portfolio Monte Carlo Simulator",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    suppress_callback_exceptions=True,
)
server = app.server  # Expose for production (gunicorn / Docker)

# Server-side cache — keeps large numpy arrays out of the browser
_CACHE: dict[str, dict] = {}

# ── Helpers ────────────────────────────────────────────────────────────────────


def _fmt_money(value: float, currency: str) -> str:
    sym = CURRENCY_SYMBOLS.get(currency, currency)
    sign = "-" if value < 0 else ""
    abs_val = abs(value)
    if abs_val >= 1_000_000:
        return f"{sign}{sym}{abs_val / 1_000_000:.2f}M"
    if abs_val >= 1_000:
        return f"{sign}{sym}{abs_val / 1_000:.1f}K"
    return f"{sign}{sym}{abs_val:,.0f}"


def _fmt_pct(value: float) -> str:
    return f"{value:.1%}"


def _display_symbol(s: str) -> str:
    return DISPLAY_SYMBOLS.get(s, s.lstrip("^"))


def _display_symbols(symbols: list[str]) -> list[str]:
    return [_display_symbol(s) for s in symbols]


def _parse_list(text: str) -> list[str]:
    return [item.strip().upper() for item in text.replace(";", ",").split(",") if item.strip()]


def _parse_weights(text: str, count: int) -> list[float]:
    raw = [float(x.strip()) for x in text.replace(";", ",").split(",") if x.strip()]
    if len(raw) != count:
        raise ValueError(f"Expected {count} weight(s), got {len(raw)}.")
    return raw


def _sim_x_axis(scenario: Scenario, points: int):
    steps = np.arange(points, dtype=float)
    factor = periods_per_year(scenario.frequency)
    return steps / factor, "Years since start"


def _with_display_cols(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [_display_symbol(str(c)) for c in out.columns]
    return out


def _median_hover_metrics(result: dict) -> dict[str, np.ndarray]:
    cf = result.get("cashflows", {})
    metrics = {
        "Net profit if sold": cf.get("net_profit_if_sold"),
        "Net dividends": cf.get("cumulative_net_dividends"),
        "Gross dividends": cf.get("cumulative_gross_dividends"),
        "Dividend tax": cf.get("cumulative_dividend_taxes"),
        "Capital gains tax": cf.get("liquidation_taxes"),
    }
    return {lbl: np.median(v, axis=0) for lbl, v in metrics.items() if v is not None}


# ── Layout helpers ─────────────────────────────────────────────────────────────


def _metric(label: str, value: str, color: str = "primary", icon: str = "bi-graph-up-arrow",
             tooltip_text: str | None = None, tooltip_id: str | None = None) -> dbc.Card:
    info_parts: list = []
    if tooltip_text and tooltip_id:
        info_parts = [
            html.I(
                className="bi bi-question-circle ms-1 text-muted",
                id=tooltip_id,
                style={"fontSize": "0.72rem", "cursor": "help", "opacity": "0.6"},
            ),
            dbc.Tooltip(tooltip_text, target=tooltip_id, placement="top"),
        ]
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [html.I(className=f"bi {icon} metric-icon me-1"), html.Span(value, className="metric-value")]
                    + info_parts,
                    className=f"text-{color}",
                ),
                html.Div(label, className="metric-label"),
            ]
        ),
        className="metric-card h-100",
    )


def _chart_card(fig, style: dict | None = None) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(dcc.Graph(figure=fig, config={"displayModeBar": False, "responsive": True})),
        className="chart-card",
        style=style or {},
    )


def _section(title: str, icon: str = "bi-bar-chart", tooltip: str | None = None) -> html.Div:
    tip_id = f"tip-sec-{title.lower().replace(' ', '-').replace('/', '')[:28]}"
    parts: list = [html.I(className=f"bi {icon} me-2 text-primary"), html.Span(title)]
    if tooltip:
        parts += [
            html.I(
                className="bi bi-question-circle ms-2 text-muted",
                id=tip_id,
                style={"fontSize": "0.72rem", "cursor": "help", "opacity": "0.6"},
            ),
            dbc.Tooltip(tooltip, target=tip_id, placement="top"),
        ]
    return html.Div(parts, className="section-header mt-3")


def _tip(target_id: str) -> dbc.Tooltip | html.Div:
    """Shorthand to create a sidebar tooltip by component ID."""
    text = TOOLTIP_TEXTS.get(target_id, "")
    if not text:
        return html.Div()
    return dbc.Tooltip(text, target=target_id, placement="right", id=f"ttip-{target_id}")


def _empty_state(msg: str = "Load market data to begin.", icon: str = "bi-cloud-arrow-down") -> html.Div:
    return html.Div(
        [
            html.I(className=f"bi {icon} display-4 text-muted d-block mb-3"),
            html.P(msg, className="text-muted"),
        ],
        className="text-center empty-state-wrapper",
    )


def _stats_display_frame(stats: pd.DataFrame, frequency: str) -> pd.DataFrame:
    cols = {
        "annualized_return": "Ann. Return",
        "annualized_volatility": "Ann. Volatility",
        "cagr": "CAGR",
        "max_drawdown": "Max Drawdown",
        "sharpe_ratio": "Sharpe",
        "sortino_ratio": "Sortino",
        "calmar_ratio": "Calmar",
        f"historical_var_5": f"VaR 5% ({frequency})",
        f"historical_cvar_5": f"CVaR 5% ({frequency})",
        "observations": "Obs.",
    }
    avail = [c for c in cols if c in stats.columns]
    # treat historical_var_5 and historical_cvar_5 specially
    rename_map = {}
    for c in avail:
        if c == "historical_var_5":
            rename_map[c] = f"VaR 5% ({frequency})"
        elif c == "historical_cvar_5":
            rename_map[c] = f"CVaR 5% ({frequency})"
        else:
            rename_map[c] = cols[c]
    disp = stats[avail].rename(columns=rename_map).copy()
    pct_cols = ["Ann. Return", "Ann. Volatility", "CAGR", "Max Drawdown",
                f"VaR 5% ({frequency})", f"CVaR 5% ({frequency})"]
    for col in pct_cols:
        if col in disp.columns:
            disp[col] = disp[col].map(lambda v: f"{v:.2%}" if pd.notna(v) else "—")
    for col in ["Sharpe", "Sortino", "Calmar"]:
        if col in disp.columns:
            disp[col] = disp[col].map(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
    if "Obs." in disp.columns:
        disp["Obs."] = disp["Obs."].map(lambda v: int(v) if pd.notna(v) else "—")
    return disp.reset_index()


def _summary_table_rows(summary: dict, currency: str) -> list[dict]:
    money_keys = {
        "median_final_value", "mean_final_value",
        "p5", "p10", "p25", "p75", "p90", "p95",
        "worst_5pct_average_outcome", "best_5pct_average_outcome",
        "total_contributions", "median_gain_over_contributions",
        "median_net_profit_if_sold",
        "median_cumulative_gross_dividends", "median_cumulative_net_dividends",
        "median_cumulative_dividend_taxes", "median_liquidation_tax",
    }
    pct_keys = {
        "probability_reaching_target", "probability_below_contributions",
        "probability_negative_nominal_return", "probability_negative_real_return",
        "expected_max_drawdown", "median_max_drawdown",
    }
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
    rows = []
    for key, val in summary.items():
        if key in money_keys:
            display = _fmt_money(val, currency)
        elif key in pct_keys:
            display = _fmt_pct(val)
        else:
            continue
        rows.append({"Metric": labels.get(key, key), "Value": display})
    return rows


# ── Sidebar ────────────────────────────────────────────────────────────────────


def make_sidebar() -> html.Div:
    today = date.today()
    start_default = f"{today.year - 20}-{today.month:02d}-{today.day:02d}"
    end_default = today.isoformat()

    def _lbl(text: str) -> dbc.Label:
        return dbc.Label(text, className="small fw-semibold mt-2 mb-1")

    return html.Div(
        [
            # ── Scrollable accordion area ──────────────────────────
            html.Div(
                [
                    # ── Import scenario
            dbc.Card(
                dbc.CardBody(
                    [
                        dcc.Upload(
                            id="upload-scenario",
                            children=html.Div(
                                [html.I(className="bi bi-folder2-open me-1"), "Import scenario JSON"],
                                className="small text-secondary",
                            ),
                        ),
                        html.Div(id="upload-status", className="mt-1"),
                    ],
                    className="py-2 px-3",
                ),
                className="mb-2 border-0",
                style={"background": "#f8fafc"},
            ),
            # ── Portfolio & Data accordion
            dbc.Accordion(
                [
                    dbc.AccordionItem(
                        [
                            _lbl("Market Preset"),
                            dcc.Dropdown(
                                id="market-preset",
                                options=[{"label": k, "value": k} for k in MARKET_PRESETS],
                                value=["S&P 500 index"],
                                multi=True,
                                clearable=True,
                                placeholder="Select preset(s) to add tickers…",
                                className="mb-2",
                            ),
                            _tip("market-preset"),
                            _lbl("Tickers"),
                            dbc.Input(
                                id="tickers-input",
                                value="^GSPC",
                                debounce=True,
                                placeholder="SPY, QQQ, ^GSPC",
                                size="sm",
                                className="mb-1",
                            ),
                            _tip("tickers-input"),
                            html.Div(
                                "1–5 Yahoo Finance symbols, comma-separated.",
                                className="text-muted mb-2",
                                style={"fontSize": "0.72rem"},
                            ),
                            _lbl("Weights (%)"),
                            dbc.Input(
                                id="weights-input",
                                value="100",
                                debounce=True,
                                placeholder="70, 30",
                                size="sm",
                                className="mb-1",
                            ),
                            _tip("weights-input"),
                            html.Div(
                                "One per ticker · auto-normalised to 100%.",
                                className="text-muted mb-2",
                                style={"fontSize": "0.72rem"},
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            _lbl("Lookback (yrs)"),
                                            dcc.Dropdown(
                                                id="lookback-years",
                                                options=[{"label": f"{y}y", "value": y} for y in [5, 10, 15, 20, 30]],
                                                value=20,
                                                clearable=False,
                                            ),
                                            _tip("lookback-years"),
                                        ],
                                        width=6,
                                    ),
                                    dbc.Col(
                                        [
                                            _lbl("Frequency"),
                                            dcc.Dropdown(
                                                id="frequency",
                                                options=[
                                                    {"label": "Monthly", "value": "monthly"},
                                                    {"label": "Weekly", "value": "weekly"},
                                                    {"label": "Daily", "value": "daily"},
                                                ],
                                                value="monthly",
                                                clearable=False,
                                            ),
                                            _tip("frequency"),
                                        ],
                                        width=6,
                                    ),
                                ],
                                className="mb-2 g-2",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            _lbl("Start date"),
                                            dbc.Input(
                                                id="start-date",
                                                type="text",
                                                value=start_default,
                                                placeholder="YYYY-MM-DD",
                                                debounce=True,
                                                size="sm",
                                            ),
                                            _tip("start-date"),
                                        ],
                                        width=6,
                                    ),
                                    dbc.Col(
                                        [
                                            _lbl("End date"),
                                            dbc.Input(
                                                id="end-date",
                                                type="text",
                                                value=end_default,
                                                placeholder="YYYY-MM-DD",
                                                debounce=True,
                                                size="sm",
                                            ),
                                            _tip("end-date"),
                                        ],
                                        width=6,
                                    ),
                                ],
                                className="mb-2 g-2",
                            ),
                            _lbl("Price field"),
                            dcc.Dropdown(
                                id="price-field",
                                options=[
                                    {"label": "Adj Close (recommended)", "value": "Adj Close"},
                                    {"label": "Close", "value": "Close"},
                                ],
                                value="Adj Close",
                                clearable=False,
                                className="mb-3",
                            ),
                            _tip("price-field"),
                            # Israel index reference
                            dbc.Accordion(
                                [
                                    dbc.AccordionItem(
                                        [
                                            html.P(
                                                "Reference for TASE symbols. Yahoo availability varies.",
                                                className="text-muted mb-2",
                                                style={"fontSize": "0.72rem"},
                                            ),
                                            html.Div(
                                                dbc.Table(
                                                    [
                                                        html.Thead(
                                                            html.Tr([html.Th("Name"), html.Th("Symbol"), html.Th("Scope")])
                                                        ),
                                                        html.Tbody(
                                                            [
                                                                html.Tr(
                                                                    [
                                                                        html.Td(name, style={"fontSize": "0.73rem"}),
                                                                        html.Td(
                                                                            html.Code(sym, style={"fontSize": "0.72rem"})
                                                                        ),
                                                                        html.Td(scope, style={"fontSize": "0.72rem"}),
                                                                    ]
                                                                )
                                                                for name, sym, scope in ISRAEL_INDICES
                                                            ]
                                                        ),
                                                    ],
                                                    size="sm",
                                                    bordered=False,
                                                    striped=True,
                                                    className="mb-0",
                                                ),
                                                style={"maxHeight": "220px", "overflowY": "auto"},
                                            ),
                                        ],
                                        title=html.Span(
                                            [html.I(className="bi bi-flag me-1"), "🇮🇱 Israel Index Reference"],
                                            style={"fontSize": "0.78rem"},
                                        ),
                                    )
                                ],
                                start_collapsed=True,
                                flush=True,
                                className="mb-2",
                            ),
                        ],
                        title=html.Span(
                            [html.I(className="bi bi-bar-chart-fill me-2 text-primary"), html.Span(id="lbl-acc-portfolio", children="Portfolio & Data")]
                        ),
                    )
                ],
                start_collapsed=False,
                flush=True,
                className="mb-2",
                id="acc-portfolio",
            ),
            # ── Accumulation accordion
            dbc.Accordion(
                [
                    dbc.AccordionItem(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            _lbl("Currency"),
                                            dcc.Dropdown(
                                                id="currency",
                                                options=[
                                                    {"label": "ILS (₪)", "value": "ILS"},
                                                    {"label": "USD ($)", "value": "USD"},
                                                ],
                                                value="ILS",
                                                clearable=False,
                                            ),
                                            _tip("currency"),
                                        ],
                                        width=6,
                                    ),
                                ],
                                className="mb-2 g-2",
                            ),
                            _lbl("Initial Capital"),
                            dbc.InputGroup(
                                [
                                    dbc.InputGroupText(id="ccy-sym-1", children="₪"),
                                    dbc.Input(
                                        id="initial-capital",
                                        type="number",
                                        value=10000,
                                        min=0,
                                        step=1000,
                                        size="sm",
                                    ),
                                ],
                                size="sm",
                                className="mb-2",
                            ),
                            _tip("initial-capital"),
                            _lbl("Monthly Contribution"),
                            dbc.InputGroup(
                                [
                                    dbc.InputGroupText(id="ccy-sym-2", children="₪"),
                                    dbc.Input(
                                        id="monthly-contrib",
                                        type="number",
                                        value=1000,
                                        min=0,
                                        step=100,
                                        size="sm",
                                    ),
                                ],
                                size="sm",
                                className="mb-2",
                            ),
                            _tip("monthly-contrib"),
                            _lbl("Annual Contribution Increase (%)"),
                            dcc.Slider(
                                id="annual-contrib-increase",
                                min=0,
                                max=10,
                                step=0.5,
                                value=2,
                                marks={0: "0%", 5: "5%", 10: "10%"},
                                tooltip={"placement": "bottom", "always_visible": True},
                                className="mb-3",
                            ),
                            _tip("annual-contrib-increase"),
                            _lbl("Investment Horizon (years)"),
                            dcc.Slider(
                                id="horizon-years",
                                min=1,
                                max=50,
                                step=1,
                                value=20,
                                marks={1: "1", 10: "10", 20: "20", 30: "30", 50: "50"},
                                tooltip={"placement": "bottom", "always_visible": True},
                                className="mb-3",
                            ),
                            _tip("horizon-years"),
                            _lbl("Number of Simulations"),
                            dbc.Input(
                                id="simulations",
                                type="number",
                                value=10000,
                                min=500,
                                max=100000,
                                step=500,
                                size="sm",
                                className="mb-2",
                            ),
                            _tip("simulations"),
                            html.Div(id="sim-count-warning", className="mb-1"),
                            _lbl("Target Portfolio Value"),
                            dbc.InputGroup(
                                [
                                    dbc.InputGroupText(id="ccy-sym-3", children="₪"),
                                    dbc.Input(
                                        id="target-value",
                                        type="number",
                                        value=500000,
                                        min=0,
                                        step=10000,
                                        size="sm",
                                    ),
                                ],
                                size="sm",
                                className="mb-2",
                            ),
                            _tip("target-value"),
                        ],
                        title=html.Span(
                            [html.I(className="bi bi-piggy-bank-fill me-2 text-success"), html.Span(id="lbl-acc-accumulation", children="Accumulation")]
                        ),
                    )
                ],
                start_collapsed=False,
                flush=True,
                className="mb-2",
                id="acc-accumulation",
            ),
            # ── Model & Assumptions accordion
            dbc.Accordion(
                [
                    dbc.AccordionItem(
                        [
                            _lbl("Primary Model"),
                            dcc.Dropdown(
                                id="primary-model",
                                options=MODEL_OPTIONS,
                                value="historical_bootstrap",
                                clearable=False,
                                className="mb-2",
                            ),
                            _tip("primary-model"),
                            _lbl("Rebalancing"),
                            dcc.Dropdown(
                                id="rebalancing",
                                options=[
                                    {"label": "None", "value": "none"},
                                    {"label": "Monthly", "value": "monthly"},
                                    {"label": "Quarterly", "value": "quarterly"},
                                    {"label": "Yearly", "value": "yearly"},
                                ],
                                value="monthly",
                                clearable=False,
                                className="mb-2",
                            ),
                            _tip("rebalancing"),
                            _lbl("Block Size (for Block Bootstrap)"),
                            dcc.Dropdown(
                                id="block-size",
                                options=[{"label": f"{m} months", "value": m} for m in [3, 6, 12, 24]],
                                value=6,
                                clearable=False,
                                className="mb-2",
                            ),
                            _tip("block-size"),
                            _lbl("Fat-tail Degrees of Freedom"),
                            dcc.Slider(
                                id="fat-tail-df",
                                min=3,
                                max=30,
                                step=0.5,
                                value=5,
                                marks={3: "3", 10: "10", 20: "20", 30: "30"},
                                tooltip={"placement": "bottom", "always_visible": True},
                                className="mb-3",
                            ),
                            _tip("fat-tail-df"),
                            html.Hr(className="my-2"),
                            _lbl("Annual Fee / Expense Ratio (%)"),
                            dcc.Slider(
                                id="annual-fee",
                                min=0,
                                max=3,
                                step=0.05,
                                value=0.1,
                                marks={0: "0%", 1: "1%", 2: "2%", 3: "3%"},
                                tooltip={"placement": "bottom", "always_visible": True},
                                className="mb-3",
                            ),
                            _tip("annual-fee"),
                            _lbl("Other Annual Tax Drag (%)"),
                            dcc.Slider(
                                id="tax-drag",
                                min=0,
                                max=5,
                                step=0.1,
                                value=0,
                                marks={0: "0%", 2.5: "2.5%", 5: "5%"},
                                tooltip={"placement": "bottom", "always_visible": True},
                                className="mb-3",
                            ),
                            _tip("tax-drag"),
                            _lbl("Estimated Annual Dividend Yield (%)"),
                            dcc.Slider(
                                id="dividend-yield",
                                min=0,
                                max=12,
                                step=0.1,
                                value=0,
                                marks={0: "0%", 4: "4%", 8: "8%", 12: "12%"},
                                tooltip={"placement": "bottom", "always_visible": True},
                                className="mb-3",
                            ),
                            _tip("dividend-yield"),
                            _lbl("Dividend Tax / Withholding (%)"),
                            dcc.Slider(
                                id="dividend-tax",
                                min=0,
                                max=50,
                                step=0.5,
                                value=25,
                                marks={0: "0%", 25: "25%", 50: "50%"},
                                tooltip={"placement": "bottom", "always_visible": True},
                                className="mb-3",
                            ),
                            _tip("dividend-tax"),
                            _lbl("Dividend Handling"),
                            dcc.Dropdown(
                                id="dividend-mode",
                                options=DIVIDEND_MODES,
                                value="track_only",
                                clearable=False,
                                className="mb-2",
                            ),
                            _tip("dividend-mode"),
                            html.Hr(className="my-2"),
                            _lbl("Capital Gains Tax Model"),
                            dcc.Dropdown(
                                id="tax-mode",
                                options=TAX_MODES,
                                value="israel_individual",
                                clearable=False,
                                className="mb-2",
                            ),
                            _tip("tax-mode"),
                            html.Div(
                                id="custom-tax-container",
                                children=[
                                    _lbl("Custom Capital Gains Rate (%)"),
                                    dcc.Slider(
                                        id="custom-tax-rate",
                                        min=0,
                                        max=50,
                                        step=0.5,
                                        value=25,
                                        marks={0: "0%", 25: "25%", 50: "50%"},
                                        tooltip={"placement": "bottom", "always_visible": False},
                                    ),
                                    _tip("custom-tax-rate"),
                                ],
                                style={"display": "none"},
                                className="mb-3",
                            ),
                            _lbl("Annual Inflation (%)"),
                            dcc.Slider(
                                id="annual-inflation",
                                min=0,
                                max=10,
                                step=0.5,
                                value=2,
                                marks={0: "0%", 5: "5%", 10: "10%"},
                                tooltip={"placement": "bottom", "always_visible": True},
                                className="mb-3",
                            ),
                            _tip("annual-inflation"),
                            html.Hr(className="my-2"),
                        ],
                        title=html.Span(
                            [html.I(className="bi bi-gear-fill me-2 text-warning"), html.Span(id="lbl-acc-model", children="Model & Assumptions")]
                        ),
                    )
                ],
                start_collapsed=False,
                flush=True,
                id="acc-model",
            ),
                ],
                className="sidebar-inner",
            ),
            # ── Sticky action footer — always visible ──────────────
            html.Div(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Button(
                                    [html.I(className="bi bi-cloud-download me-1"), html.Span(id="lbl-btn-load", children="Load Data")],
                                    id="load-data-btn",
                                    color="primary",
                                    size="sm",
                                    className="w-100 btn-load",
                                    n_clicks=0,
                                ),
                                width=6,
                            ),
                            dbc.Col(
                                dbc.Button(
                                    [html.I(className="bi bi-play-fill me-1"), html.Span(id="lbl-btn-run", children="Run Simulation")],
                                    id="run-sim-btn",
                                    color="success",
                                    size="sm",
                                    className="w-100 btn-run",
                                    n_clicks=0,
                                ),
                                width=6,
                            ),
                        ],
                        className="g-2 mb-1",
                    ),
                    dbc.Button(
                        [html.I(className="bi bi-arrow-counterclockwise me-1"), html.Span(id="lbl-btn-reset", children="Reset to Defaults")],
                        id="reset-btn",
                        color="outline-secondary",
                        size="sm",
                        className="w-100",
                        n_clicks=0,
                    ),
                    dbc.Tooltip("Reset all settings to defaults", target="reset-btn", placement="top", id="ttip-btn-reset"),
                    dbc.Tooltip(
                        "Download historical prices and compute statistics. "
                        "Auto-loads on startup with default settings.",
                        target="load-data-btn",
                        placement="top",
                        id="ttip-btn-load",
                    ),
                    dbc.Tooltip(
                        "Run Monte Carlo simulation with selected model and parameters. "
                        "Load market data first.",
                        target="run-sim-btn",
                        placement="top",
                        id="ttip-btn-run",
                    ),
                    html.Div(id="load-status", className="mt-1"),
                    html.Div(id="sim-status"),
                ],
                className="sidebar-footer",
            ),
        ],
        className="sidebar-wrapper",
    )


# ── Navbar ─────────────────────────────────────────────────────────────────────


def make_navbar() -> dbc.Navbar:
    return dbc.Navbar(
        dbc.Container(
            [
                # Brand (logo + name)
                html.A(
                    dbc.Row(
                        [
                            dbc.Col(
                                html.Img(src="/assets/logo.svg", height="30px", className="me-2"),
                                width="auto",
                            ),
                            dbc.Col(
                                dbc.NavbarBrand("Portfolio Monte Carlo", className="mb-0"),
                                width="auto",
                            ),
                        ],
                        align="center",
                        className="g-0",
                    ),
                    href="/",
                    style={"textDecoration": "none"},
                ),
                html.Span(
                    id="lbl-nav-subtitle",
                    children="Scenario-based projections · Not financial advice",
                    className="subtitle-text d-none d-xl-inline ms-3",
                ),
                # Status badge + language selector (right side)
                html.Span(
                    id="navbar-data-status",
                    className="navbar-status ms-auto d-none d-md-inline",
                ),
                dbc.ButtonGroup(
                    [
                        dbc.Button("🇺🇸 EN", id="lang-en-btn", color="outline-light", size="sm", n_clicks=0,
                                   className="lang-btn"),
                        dbc.Button("🇷🇺 RU", id="lang-ru-btn", color="outline-light", size="sm", n_clicks=0,
                                   className="lang-btn"),
                        dbc.Button("🇮🇱 HE", id="lang-he-btn", color="outline-light", size="sm", n_clicks=0,
                                   className="lang-btn"),
                    ],
                    className="ms-2",
                    id="lang-btn-group",
                ),
            ],
            fluid=True,
        ),
        className="app-navbar",
        sticky="top",
        style={"backgroundImage": "url('/assets/banner.svg')", "backgroundSize": "cover", "backgroundPosition": "center"},
    )


# ── Main tabs ──────────────────────────────────────────────────────────────────


def make_main() -> html.Div:
    def _tab(content_id: str, label: str, tab_id: str, component_id: str, initial_msg: str = "Load market data to begin.") -> dbc.Tab:
        return dbc.Tab(
            dcc.Loading(
                html.Div(_empty_state(initial_msg), id=content_id),
                type="dot",
                color="#1a56db",
            ),
            id=component_id,
            label=label,
            tab_id=tab_id,
        )

    return html.Div(
        dbc.Tabs(
            [
                _tab("content-setup", "📊 Portfolio Setup", "tab-setup", "lbl-tab-setup", "Click «Load Market Data» in the sidebar to begin."),
                _tab("content-data", "📈 Historical Data", "tab-data", "lbl-tab-data", "Load market data to see historical statistics and charts."),
                _tab("content-sim", "🎲 Simulation", "tab-sim", "lbl-tab-sim", "Load market data, then click «Run Primary Simulation»."),
                dbc.Tab(
                    html.Div(_empty_state("Load data first, then use the Compare tab."), id="content-compare"),
                    id="lbl-tab-compare",
                    label="⚖️ Model Comparison",
                    tab_id="tab-compare",
                ),
                _tab("content-risk", "🛡️ Risk Analysis", "tab-risk", "lbl-tab-risk", "Run a simulation to see risk analysis."),
                dbc.Tab(html.Div(id="content-export"), id="lbl-tab-export", label="📤 Export", tab_id="tab-export"),
            ],
            id="main-tabs",
            active_tab="tab-setup",
        ),
        className="main-col",
    )


# ── App layout ─────────────────────────────────────────────────────────────────

app.layout = html.Div(
    [
        dcc.Store(id="scenario-store"),
        dcc.Store(id="data-key-store"),
        dcc.Store(id="sim-key-store"),
        dcc.Store(id="compare-key-store"),
        dcc.Store(id="lang-store", data="en"),
        # Auto-load on startup after 1.5 s (gives scenario-store time to populate)
        dcc.Interval(id="startup-interval", interval=1500, max_intervals=1),
        make_navbar(),
        dbc.Container(
            dbc.Row(
                [
                    dbc.Col(make_sidebar(), id="sidebar-col", width=3),
                    dbc.Col(make_main(), width=9),
                ],
                className="g-0",
            ),
            fluid=True,
            className="p-0",
        ),
    ],
    style={"backgroundColor": "#f0f4f8"},
)


# ── Scenario store builder (must precede callbacks) ───────────────────────────


def _scenario_inputs():
    return [
        Input("tickers-input", "value"),
        Input("weights-input", "value"),
        Input("lookback-years", "value"),
        Input("frequency", "value"),
        Input("price-field", "value"),
        Input("start-date", "value"),
        Input("end-date", "value"),
        Input("currency", "value"),
        Input("initial-capital", "value"),
        Input("monthly-contrib", "value"),
        Input("annual-contrib-increase", "value"),
        Input("horizon-years", "value"),
        Input("simulations", "value"),
        Input("target-value", "value"),
        Input("primary-model", "value"),
        Input("rebalancing", "value"),
        Input("block-size", "value"),
        Input("fat-tail-df", "value"),
        Input("annual-fee", "value"),
        Input("tax-drag", "value"),
        Input("dividend-yield", "value"),
        Input("dividend-tax", "value"),
        Input("dividend-mode", "value"),
        Input("tax-mode", "value"),
        Input("custom-tax-rate", "value"),
        Input("annual-inflation", "value"),
    ]


def _build_scenario_dict(
    tickers_text, weights_text, lookback_years, frequency, price_field,
    start_date, end_date, currency, initial_capital, monthly_contrib,
    annual_contrib_increase, horizon_years, simulations, target_value,
    primary_model, rebalancing, block_size, fat_tail_df,
    annual_fee, tax_drag, dividend_yield, dividend_tax,
    dividend_mode, tax_mode, custom_tax_rate, annual_inflation,
) -> dict:
    try:
        reverse_display = {v.upper(): k for k, v in DISPLAY_SYMBOLS.items()}
        raw_tickers = _parse_list(tickers_text or "GSPC")[:5]
        tickers = [reverse_display.get(t, t) for t in raw_tickers] or ["^GSPC"]
        try:
            weights = _parse_weights(weights_text or "100", len(tickers))
        except ValueError:
            weights = [1.0] * len(tickers)
        if tax_mode == "israel_individual":
            cg_rate = 0.25
        elif tax_mode == "israel_substantial_shareholder":
            cg_rate = 0.30
        elif tax_mode == "none":
            cg_rate = 0.0
        else:
            cg_rate = float(custom_tax_rate or 25) / 100
        return Scenario(
            tickers=tickers,
            weights=weights,
            lookback_years=int(lookback_years or 20),
            frequency=frequency or "monthly",
            price_field=price_field or "Adj Close",
            start_date=str(start_date or ""),
            end_date=str(end_date or date.today().isoformat()),
            currency=currency or "ILS",
            initial_capital=float(initial_capital or 10000),
            monthly_contribution=float(monthly_contrib or 1000),
            annual_contribution_increase=float(annual_contrib_increase or 2) / 100,
            horizon_years=float(horizon_years or 20),
            simulations=int(simulations or 10000),
            target_value=float(target_value or 500000),
            model=primary_model or "historical_bootstrap",
            rebalancing=rebalancing or "monthly",
            block_size_months=int(block_size or 6),
            fat_tail_df=float(fat_tail_df or 5),
            annual_fee=float(annual_fee or 0.1) / 100,
            annual_tax_drag=float(tax_drag or 0) / 100,
            annual_dividend_yield=float(dividend_yield or 0) / 100,
            dividend_tax_rate=float(dividend_tax or 25) / 100,
            dividend_mode=dividend_mode or "track_only",
            tax_mode=tax_mode or "israel_individual",
            capital_gains_tax_rate=cg_rate,
            annual_inflation=float(annual_inflation or 2) / 100,
            random_seed=42,
        ).to_dict()
    except Exception:
        return Scenario().to_dict()


# ── Callbacks ──────────────────────────────────────────────────────────────────

_SIDEBAR_DEFAULTS = {
    "tickers-input":            "^GSPC",
    "weights-input":            "100",
    "initial-capital":          10000,
    "monthly-contrib":          1000,
    "annual-contrib-increase":  2.0,
    "horizon-years":            20,
    "simulations":              10000,
    "target-value":             500000,
    "primary-model":            "historical_bootstrap",
    "rebalancing":              "monthly",
    "block-size":               6,
    "fat-tail-df":              5.0,
    "annual-fee":               0.1,
    "tax-drag":                 0.0,
    "dividend-yield":           0.0,
    "dividend-tax":             25.0,
    "dividend-mode":            "track_only",
    "tax-mode":                 "israel_individual",
    "custom-tax-rate":          25.0,
    "annual-inflation":         2.0,
    "currency":                 "ILS",
    "market-preset":            ["S&P 500 index"],
    "lookback-years":           20,
    "frequency":                "monthly",
    "price-field":              "Adj Close",
}


@callback(
    Output("tickers-input",           "value", allow_duplicate=True),
    Output("weights-input",           "value", allow_duplicate=True),
    Output("initial-capital",         "value", allow_duplicate=True),
    Output("monthly-contrib",         "value", allow_duplicate=True),
    Output("annual-contrib-increase", "value"),
    Output("horizon-years",           "value"),
    Output("simulations",             "value", allow_duplicate=True),
    Output("target-value",            "value", allow_duplicate=True),
    Output("primary-model",           "value"),
    Output("rebalancing",             "value"),
    Output("block-size",              "value"),
    Output("fat-tail-df",             "value"),
    Output("annual-fee",              "value"),
    Output("tax-drag",                "value"),
    Output("dividend-yield",          "value"),
    Output("dividend-tax",            "value"),
    Output("dividend-mode",           "value"),
    Output("tax-mode",                "value"),
    Output("custom-tax-rate",         "value"),
    Output("annual-inflation",        "value"),
    Output("currency",                "value"),
    Output("market-preset",           "value", allow_duplicate=True),
    Output("lookback-years",          "value"),
    Output("frequency",               "value"),
    Output("price-field",             "value"),
    Input("reset-btn", "n_clicks"),
    prevent_initial_call=True,
)
def reset_settings(_n):
    d = _SIDEBAR_DEFAULTS
    return (
        d["tickers-input"], d["weights-input"], d["initial-capital"],
        d["monthly-contrib"], d["annual-contrib-increase"], d["horizon-years"],
        d["simulations"], d["target-value"], d["primary-model"], d["rebalancing"],
        d["block-size"], d["fat-tail-df"], d["annual-fee"], d["tax-drag"],
        d["dividend-yield"], d["dividend-tax"], d["dividend-mode"], d["tax-mode"],
        d["custom-tax-rate"], d["annual-inflation"], d["currency"],
        d["market-preset"], d["lookback-years"], d["frequency"], d["price-field"],
    )


@callback(Output("custom-tax-container", "style"), Input("tax-mode", "value"))
def toggle_custom_tax(tax_mode: str) -> dict:
    return {"display": "block"} if tax_mode == "custom" else {"display": "none"}


@callback(
    Output("ccy-sym-1", "children"),
    Output("ccy-sym-2", "children"),
    Output("ccy-sym-3", "children"),
    Input("currency", "value"),
)
def update_currency_symbols(currency: str):
    sym = CURRENCY_SYMBOLS.get(currency, currency)
    return sym, sym, sym


@callback(
    Output("sim-count-warning", "children"),
    Input("simulations", "value"),
)
def sim_count_warning(n: int | None):
    if n and n >= 50_000:
        return dbc.Alert(
            "50 000+ simulations may be slow on some machines.",
            color="warning",
            className="py-1 px-2 mb-0",
            style={"fontSize": "0.78rem"},
        )
    return None


@callback(
    Output("tickers-input", "value"),
    Input("market-preset", "value"),
    prevent_initial_call=True,
)
def preset_to_tickers(presets) -> str:
    if not presets:
        return no_update
    if isinstance(presets, str):
        presets = [presets]
    seen: list[str] = []
    for preset in presets:
        for sym in MARKET_PRESETS.get(preset, []):
            if sym not in seen:
                seen.append(sym)
    return ", ".join(_display_symbols(seen)) if seen else no_update


@callback(Output("scenario-store", "data"), _scenario_inputs())
def update_scenario_store(*args):
    return _build_scenario_dict(*args)


@callback(
    Output("upload-status", "children"),
    Output("tickers-input", "value", allow_duplicate=True),
    Output("weights-input", "value", allow_duplicate=True),
    Output("initial-capital", "value", allow_duplicate=True),
    Output("monthly-contrib", "value", allow_duplicate=True),
    Output("target-value", "value", allow_duplicate=True),
    Output("horizon-years", "value", allow_duplicate=True),
    Output("simulations", "value", allow_duplicate=True),
    Input("upload-scenario", "contents"),
    prevent_initial_call=True,
)
def load_uploaded_scenario(contents: str | None):
    empty = [no_update] * 7
    if not contents:
        return [no_update] + empty
    try:
        import base64
        _, encoded = contents.split(",", 1)
        data = json.loads(base64.b64decode(encoded).decode("utf-8"))
        s = Scenario.from_dict(data)
        tickers_display = ", ".join(_display_symbols(s.tickers))
        weights_display = ", ".join(f"{w * 100:.4g}" for w in s.weights)
        return (
            dbc.Alert("Scenario loaded.", color="success", className="py-1 px-2 mb-0", style={"fontSize": "0.78rem"}),
            tickers_display,
            weights_display,
            s.initial_capital,
            s.monthly_contribution,
            s.target_value,
            int(round(s.horizon_years)),
            s.simulations,
        )
    except Exception as exc:
        return (
            dbc.Alert(f"Could not parse scenario: {exc}", color="danger", className="py-1 px-2 mb-0"),
            *empty,
        )


@callback(
    Output("data-key-store", "data"),
    Output("load-status", "children"),
    Output("content-setup", "children"),
    Output("content-data", "children"),
    Output("navbar-data-status", "children"),
    Input("load-data-btn", "n_clicks"),
    Input("startup-interval", "n_intervals"),
    State("scenario-store", "data"),
    prevent_initial_call=True,
)
def load_market_data(n_clicks, n_startup, scenario_dict: dict | None):
    if not (n_clicks or n_startup) or not scenario_dict:
        raise dash.exceptions.PreventUpdate

    try:
        s = Scenario.from_dict(scenario_dict)
        prices = download_yfinance_prices(
            s.tickers, start=s.start_date, end=s.end_date, field=s.price_field, use_cache=True
        )
        quality = quality_report_frame(validate_prices(prices))
        asset_returns = calculate_returns(prices, s.frequency).dropna(how="any")
        port_returns = portfolio_returns(asset_returns, align_weights(s.tickers, s.weights).values)
        combined_returns = asset_returns.copy()
        combined_returns["Portfolio"] = port_returns
        stats = return_statistics(combined_returns, s.frequency)
        stats.index = [_display_symbol(str(i)) for i in stats.index]
        display_prices = _with_display_cols(prices)
        display_returns = _with_display_cols(combined_returns)

        key = str(uuid.uuid4())
        _CACHE[key] = {
            "prices": prices,
            "returns": asset_returns,
            "combined_returns": combined_returns,
            "stats": stats,
            "quality": quality,
            "display_prices": display_prices,
            "display_returns": display_returns,
            "scenario": s,
        }

        setup_content = _render_setup_tab(key, s)
        data_content = _render_data_tab(key, s)
        status = dbc.Alert(
            [html.I(className="bi bi-check-circle me-1"), f"Data loaded: {len(prices)} rows, {len(s.tickers)} ticker(s)."],
            color="success",
            className="py-1 px-2 mb-0",
            style={"fontSize": "0.78rem"},
        )
        nav_badge = dbc.Badge(
            [html.I(className="bi bi-check-circle-fill me-1"), f"{', '.join(_display_symbols(s.tickers))} loaded"],
            color="success",
            className="navbar-badge",
        )
        return key, status, setup_content, data_content, nav_badge

    except Exception as exc:
        err = dbc.Alert(f"Data error: {exc}", color="danger", className="py-1 px-2 mb-0", style={"fontSize": "0.78rem"})
        err_badge = dbc.Badge([html.I(className="bi bi-x-circle-fill me-1"), "Data error"], color="danger", className="navbar-badge")
        return no_update, err, _empty_state(f"Error: {exc}", "bi-exclamation-triangle"), no_update, err_badge


@callback(
    Output("sim-key-store", "data"),
    Output("content-sim", "children"),
    Output("content-risk", "children"),
    Output("sim-status", "children"),
    Output("main-tabs", "active_tab"),
    Input("run-sim-btn", "n_clicks"),
    State("scenario-store", "data"),
    State("data-key-store", "data"),
    prevent_initial_call=True,
)
def run_primary_simulation(n_clicks, scenario_dict: dict | None, data_key: str | None):
    if not n_clicks or not scenario_dict or not data_key or data_key not in _CACHE:
        raise dash.exceptions.PreventUpdate

    try:
        s = Scenario.from_dict(scenario_dict)
        cached = _CACHE[data_key]
        asset_returns = cached["returns"]
        result = run_simulation(asset_returns, s)
        sim_key = str(uuid.uuid4())
        _CACHE[sim_key] = {"result": result, "scenario": s, "data_key": data_key}

        sim_content = _render_sim_tab(sim_key, s, data_key)
        risk_content = _render_risk_tab(sim_key, s, data_key)

        status = dbc.Alert(
            [html.I(className="bi bi-check-circle me-1"), "Simulation complete."],
            color="success",
            className="py-1 px-2 mb-0",
            style={"fontSize": "0.78rem"},
        )
        return sim_key, sim_content, risk_content, status, "tab-sim"

    except Exception as exc:
        err = dbc.Alert(f"Simulation error: {exc}", color="danger", className="py-1 px-2 mb-0")
        return no_update, _empty_state(f"Error: {exc}", "bi-exclamation-triangle"), no_update, err, no_update



@callback(
    Output("content-export", "children"),
    Input("sim-key-store", "data"),
    State("scenario-store", "data"),
)
def render_export_tab(sim_key: str | None, scenario_dict: dict | None):
    s = Scenario.from_dict(scenario_dict) if scenario_dict else Scenario()
    return _render_export_tab(sim_key, s)


# ── Tab renderers (pure Python, called from callbacks) ────────────────────────


def _render_setup_tab(data_key: str, s: Scenario) -> html.Div:
    cached = _CACHE.get(data_key, {})
    display_prices = cached.get("display_prices", pd.DataFrame())
    weights = align_weights(s.tickers, s.weights)
    ccy = CURRENCY_SYMBOLS.get(s.currency, s.currency)

    weight_df = weights.mul(100).round(2).rename("Weight (%)")
    weight_df.index = [_display_symbol(str(i)) for i in weight_df.index]

    return html.Div(
        [
            # Metric summary row
            dbc.Row(
                [
                    dbc.Col(
                        _metric("Initial Capital", _fmt_money(s.initial_capital, s.currency), "primary", "bi-wallet2"),
                        xs=6,
                        lg=3,
                    ),
                    dbc.Col(
                        _metric(
                            "Monthly Contribution",
                            _fmt_money(s.monthly_contribution, s.currency),
                            "success",
                            "bi-arrow-up-circle",
                        ),
                        xs=6,
                        lg=3,
                    ),
                    dbc.Col(
                        _metric(
                            "Target Value",
                            _fmt_money(s.target_value, s.currency),
                            "info",
                            "bi-bullseye",
                        ),
                        xs=6,
                        lg=3,
                    ),
                    dbc.Col(
                        _metric(
                            f"Horizon · {s.frequency}",
                            f"{s.horizon_years:.0f} yrs",
                            "warning",
                            "bi-clock",
                        ),
                        xs=6,
                        lg=3,
                    ),
                ],
                className="g-2 mb-3 metric-row",
            ),
            # Weights table + price chart
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Div("Portfolio Weights", className="section-header"),
                                    DataTable(
                                        data=weight_df.reset_index().rename(columns={"index": "Asset"}).to_dict("records"),
                                        columns=[{"name": c, "id": c} for c in ["Asset", "Weight (%)"]],
                                        style_cell={"fontSize": "0.82rem", "padding": "6px 10px"},
                                        style_header={
                                            "fontWeight": "700",
                                            "backgroundColor": "#f1f5f9",
                                            "fontSize": "0.75rem",
                                            "textTransform": "uppercase",
                                        },
                                        style_data_conditional=[
                                            {"if": {"row_index": "odd"}, "backgroundColor": "#f8fafc"}
                                        ],
                                    ),
                                    html.Div(
                                        "Weights are normalised to 100%.",
                                        className="text-muted mt-2",
                                        style={"fontSize": "0.72rem"},
                                    ),
                                ]
                            ),
                            className="chart-card",
                        ),
                        width=3,
                    ),
                    dbc.Col(
                        _chart_card(
                            historical_prices(display_prices, ccy)
                        ),
                        width=9,
                    ),
                ],
                className="g-2",
            ),
        ]
    )


def _render_data_tab(data_key: str, s: Scenario) -> html.Div:
    cached = _CACHE.get(data_key, {})
    quality = cached.get("quality", pd.DataFrame())
    stats = cached.get("stats", pd.DataFrame())
    display_returns = cached.get("display_returns", pd.DataFrame())

    stats_display = _stats_display_frame(stats, s.frequency) if not stats.empty else pd.DataFrame()

    return html.Div(
        [
            _section("Data Quality", "bi-shield-check", "Shows data coverage, gaps, and quality flags for each downloaded price series."),
            dbc.Card(
                dbc.CardBody(
                    DataTable(
                        data=quality.to_dict("records") if not quality.empty else [],
                        columns=[{"name": c, "id": c} for c in quality.columns] if not quality.empty else [],
                        style_cell={"fontSize": "0.82rem", "padding": "5px 10px"},
                        style_header={
                            "fontWeight": "700",
                            "backgroundColor": "#f1f5f9",
                            "fontSize": "0.75rem",
                            "textTransform": "uppercase",
                        },
                    )
                ),
                className="chart-card mb-3",
            ),
            _section("Historical Statistics", "bi-table", "Annualised metrics from historical data. VaR/CVaR are per-period. Sharpe/Sortino/Calmar use risk-free rate = 0%."),
            html.P(
                f"Annualised metrics · VaR and CVaR are per {s.frequency} period · Sharpe/Sortino/Calmar use risk-free = 0.",
                className="text-muted mb-2",
                style={"fontSize": "0.78rem"},
            ),
            dbc.Card(
                dbc.CardBody(
                    DataTable(
                        data=stats_display.to_dict("records") if not stats_display.empty else [],
                        columns=[{"name": c, "id": c} for c in stats_display.columns] if not stats_display.empty else [],
                        style_cell={"fontSize": "0.82rem", "padding": "5px 10px"},
                        style_header={
                            "fontWeight": "700",
                            "backgroundColor": "#f1f5f9",
                            "fontSize": "0.75rem",
                            "textTransform": "uppercase",
                        },
                        style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#f8fafc"}],
                    )
                ),
                className="chart-card mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        _chart_card(annual_returns_bar(
                            display_returns.drop(columns=["Portfolio"], errors="ignore"),
                            "Annual Returns by Year",
                        )),
                        width=12,
                    ),
                ],
                className="g-2 mb-1",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        _chart_card(returns_distribution(
                            display_returns.drop(columns=["Portfolio"], errors="ignore"),
                            "Periodic Return Distribution",
                        )),
                        width=12,
                    ),
                ],
                className="g-2",
            ),
        ]
    )


def _render_sim_tab(sim_key: str, s: Scenario, data_key: str) -> html.Div:
    cached = _CACHE.get(sim_key, {})
    result = cached.get("result", {})
    summary = result.get("summary", {})
    paths = result.get("paths")
    contributions = result.get("contributions")
    ccy = CURRENCY_SYMBOLS.get(s.currency, s.currency)

    if paths is None:
        return _empty_state("No simulation results yet.")

    x_values, x_axis_title = _sim_x_axis(s, paths.shape[1])

    med = summary.get("median_final_value", 0)
    p5 = summary.get("p5", 0)
    prob_target = summary.get("probability_reaching_target", 0)
    med_dd = summary.get("median_max_drawdown", 0)
    net_profit = summary.get("median_net_profit_if_sold", summary.get("median_gain_over_contributions", 0))
    net_div = summary.get("median_cumulative_net_dividends", 0)

    dd_color = "danger" if abs(med_dd) > 0.3 else "warning" if abs(med_dd) > 0.15 else "success"
    prob_color = "success" if prob_target >= 0.6 else "warning" if prob_target >= 0.3 else "danger"

    return html.Div(
        [
            # 6 metric cards
            dbc.Row(
                [
                    dbc.Col(_metric("Median Final Value", _fmt_money(med, s.currency), "primary", "bi-cash-stack",
                        "The 50th percentile of all simulated after-tax portfolio values at the end of the horizon.",
                        "tip-med-final"), xs=6, xl=2),
                    dbc.Col(_metric("5th Percentile", _fmt_money(p5, s.currency), "warning", "bi-arrow-down-circle",
                        "Downside scenario: 95% of simulations end above this value. Use as a stress-test floor.",
                        "tip-p5"), xs=6, xl=2),
                    dbc.Col(_metric("Target Probability", _fmt_pct(prob_target), prob_color, "bi-bullseye",
                        f"Fraction of simulated paths that reach or exceed the target value of "
                        f"{_fmt_money(s.target_value, s.currency)}.",
                        "tip-prob-target"), xs=6, xl=2),
                    dbc.Col(_metric("Median Max Drawdown", _fmt_pct(med_dd), dd_color, "bi-graph-down-arrow",
                        "Median peak-to-trough decline across all simulated paths. A measure of typical downside volatility.",
                        "tip-med-dd"), xs=6, xl=2),
                    dbc.Col(_metric("Median Net Profit", _fmt_money(net_profit, s.currency), "success", "bi-trophy",
                        "Median profit after capital gains tax and contributions — what you keep if you liquidate at horizon end.",
                        "tip-net-profit"), xs=6, xl=2),
                    dbc.Col(_metric("Median Net Dividends", _fmt_money(net_div, s.currency), "info", "bi-coin",
                        "Median cumulative dividend income (after withholding tax) over the full horizon.",
                        "tip-net-div"), xs=6, xl=2),
                ],
                className="g-2 mb-3 metric-row",
            ),
            # Income waterfall + target gauge
            dbc.Row(
                [
                    dbc.Col(
                        _chart_card(income_waterfall(summary, ccy, "Income & Tax Breakdown")),
                        width=8,
                    ),
                    dbc.Col(
                        _chart_card(target_probability_gauge(prob_target, "Target Probability")),
                        width=4,
                    ),
                ],
                className="g-2 mb-1",
            ),
            # Fan chart (full width)
            _chart_card(
                fan_chart(
                    paths,
                    currency_symbol=ccy,
                    x_values=x_values,
                    x_axis_title=x_axis_title,
                    hover_metrics=_median_hover_metrics(result),
                )
            ),
            # Contribution growth area (full width)
            _chart_card(
                contribution_growth_area(paths, contributions, ccy, x_values, x_axis_title)
            ) if contributions is not None else html.Div(),
            # Histogram + sample trajectories
            dbc.Row(
                [
                    dbc.Col(_chart_card(final_value_histogram(paths, ccy)), width=6),
                    dbc.Col(
                        _chart_card(sample_trajectories(paths, currency_symbol=ccy, x_values=x_values, x_axis_title=x_axis_title)),
                        width=6,
                    ),
                ],
                className="g-2 mb-1",
            ),
            # Summary metrics table
            _section("Full Summary Metrics", "bi-table", "Complete statistics from the Monte Carlo simulation. All values are at the end of the investment horizon unless noted."),
            dbc.Card(
                dbc.CardBody(
                    DataTable(
                        data=_summary_table_rows(summary, s.currency),
                        columns=[{"name": "Metric", "id": "Metric"}, {"name": "Value", "id": "Value"}],
                        style_cell={"fontSize": "0.82rem", "padding": "5px 10px"},
                        style_header={
                            "fontWeight": "700",
                            "backgroundColor": "#f1f5f9",
                            "fontSize": "0.75rem",
                            "textTransform": "uppercase",
                        },
                        style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#f8fafc"}],
                    )
                ),
                className="chart-card",
            ),
        ]
    )


def _render_risk_tab(sim_key: str, s: Scenario, data_key: str) -> html.Div:
    sim_cached = _CACHE.get(sim_key, {})
    data_cached = _CACHE.get(data_key, {})
    result = sim_cached.get("result", {})
    summary = result.get("summary", {})
    paths = result.get("paths")
    stats = data_cached.get("stats", pd.DataFrame())
    display_returns = data_cached.get("display_returns", pd.DataFrame())

    if paths is None:
        return _empty_state("Run simulation first.")

    window = {"daily": 252, "weekly": 52, "monthly": 12}[s.frequency]
    prob_below = summary.get("probability_below_contributions", 0)
    prob_neg_real = summary.get("probability_negative_real_return", 0)
    worst5 = summary.get("worst_5pct_average_outcome", 0)
    exp_dd = summary.get("expected_max_drawdown", 0)

    # historical risk row from stats
    hist_rows = []
    if not stats.empty:
        row = stats.loc["Portfolio"] if "Portfolio" in stats.index else stats.iloc[0]
        hist_rows = [
            {"Metric": "Annual volatility", "Value": f"{row.get('annualized_volatility', 0):.2%}", "Meaning": "Typical yearly return variability"},
            {"Metric": "Max historical drawdown", "Value": f"{row.get('max_drawdown', 0):.2%}", "Meaning": "Worst historical peak-to-trough"},
            {"Metric": f"VaR 5% ({s.frequency})", "Value": f"{row.get('historical_var_5', 0):.2%}", "Meaning": "5% of periods were worse"},
            {"Metric": f"CVaR 5% ({s.frequency})", "Value": f"{row.get('historical_cvar_5', 0):.2%}", "Meaning": "Avg return in worst 5% periods"},
            {"Metric": "Sharpe ratio", "Value": f"{row.get('sharpe_ratio', float('nan')):.2f}" if pd.notna(row.get("sharpe_ratio")) else "—", "Meaning": "Excess return per unit of volatility"},
            {"Metric": "Sortino ratio", "Value": f"{row.get('sortino_ratio', float('nan')):.2f}" if pd.notna(row.get("sortino_ratio")) else "—", "Meaning": "Excess return per unit of downside vol"},
            {"Metric": "Calmar ratio", "Value": f"{row.get('calmar_ratio', float('nan')):.2f}" if pd.notna(row.get("calmar_ratio")) else "—", "Meaning": "CAGR relative to max drawdown"},
        ]

    return html.Div(
        [
            _section("Simulated Downside Risk", "bi-shield-exclamation", "Risk metrics derived from the Monte Carlo simulation paths — not from historical data."),
            dbc.Row(
                [
                    dbc.Col(
                        _metric("P(below contributions)", _fmt_pct(prob_below), "warning", "bi-exclamation-triangle",
                            "Probability that the final portfolio value is less than total money invested (a real loss of capital).",
                            "tip-prob-below"),
                        xs=6, lg=3,
                    ),
                    dbc.Col(
                        _metric("P(negative real return)", _fmt_pct(prob_neg_real), "danger", "bi-arrow-down",
                            "Probability that the inflation-adjusted return is negative — you lost purchasing power.",
                            "tip-prob-real"),
                        xs=6, lg=3,
                    ),
                    dbc.Col(
                        _metric("Worst 5% avg outcome", _fmt_money(worst5, s.currency), "danger", "bi-graph-down",
                            "Average final value across the worst 5% of simulated paths — the expected tail loss scenario.",
                            "tip-worst5"),
                        xs=6, lg=3,
                    ),
                    dbc.Col(
                        _metric("Expected max drawdown", _fmt_pct(exp_dd), "warning", "bi-water",
                            "Mean peak-to-trough decline across all simulated paths — expected worst drop before recovery.",
                            "tip-exp-dd"),
                        xs=6, lg=3,
                    ),
                ],
                className="g-2 mb-3 metric-row",
            ),
            _section("Historical Risk Snapshot", "bi-clock-history", "Risk metrics computed from historical return data. These describe the past, not the simulation."),
            dbc.Card(
                dbc.CardBody(
                    DataTable(
                        data=hist_rows,
                        columns=[{"name": c, "id": c} for c in ["Metric", "Value", "Meaning"]],
                        style_cell={"fontSize": "0.82rem", "padding": "5px 10px"},
                        style_header={
                            "fontWeight": "700",
                            "backgroundColor": "#f1f5f9",
                            "fontSize": "0.75rem",
                            "textTransform": "uppercase",
                        },
                        style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#f8fafc"}],
                    )
                ),
                className="chart-card mb-3",
            ),
            _section("Rolling Risk Charts", "bi-graph-up", "Return and volatility over a rolling window equal to one year (252 daily / 52 weekly / 12 monthly periods)."),
            *(
                [
                    _chart_card(
                        rolling_chart(
                            rolling_returns(display_returns, window), "Rolling Returns", "Return over rolling window"
                        )
                    ),
                    _chart_card(
                        rolling_chart(
                            rolling_volatility(display_returns, window, s.frequency), "Rolling Volatility", "Annualised volatility"
                        )
                    ),
                ]
                if not display_returns.empty
                else []
            ),
            _section("Simulated Drawdown Distribution", "bi-bar-chart-steps", "Distribution of maximum peak-to-trough declines across all simulated paths."),
            _chart_card(drawdown_histogram(paths)),
        ]
    )


def _render_compare_tab_shell() -> html.Div:
    """Static shell for the comparison tab (rendered in layout)."""
    return html.Div(
        [
            html.Div(
                "Different simulation models can produce materially different outcomes. "
                "This does not mean one model is objectively correct. "
                "It shows that long-term projections are highly assumption-dependent.",
                className="disclaimer-box mb-3",
            ),
            _section("Select Models to Compare", "bi-collection", "Each model uses different statistical assumptions to generate return paths. Compare to understand model sensitivity."),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Checklist(
                            id="compare-models-select",
                            options=MODEL_OPTIONS,
                            value=["historical_bootstrap", "block_bootstrap", "normal", "fat_tail"],
                            labelStyle={"display": "flex", "alignItems": "center", "gap": "6px", "marginBottom": "4px"},
                            inputStyle={"marginRight": "4px"},
                            className="small",
                        ),
                        width=6,
                    ),
                    dbc.Col(
                        dbc.Button(
                            [html.I(className="bi bi-play-fill me-2"), "Run Model Comparison"],
                            id="run-compare-btn",
                            color="primary",
                            size="sm",
                        ),
                        width=6,
                        className="d-flex align-items-end",
                    ),
                ],
                className="g-2 mb-3",
            ),
            html.Div(id="compare-results-inner"),
        ]
    )


def _render_compare_tab(ck: str, s: Scenario) -> html.Div:
    cached = _CACHE.get(ck, {})
    comparison_df = cached.get("comparison_df", pd.DataFrame())
    results = cached.get("results", {})
    ccy = CURRENCY_SYMBOLS.get(s.currency, s.currency)

    if comparison_df.empty:
        return html.Div("No comparison data.")

    # Format comparison table
    money_cols = {"p5", "p10", "median_final_value", "p90", "p95", "worst_5pct_average_outcome", "best_5pct_average_outcome"}
    pct_cols = {"probability_reaching_target", "probability_below_contributions", "probability_negative_real_return", "median_max_drawdown"}
    formatted = comparison_df.copy()
    for col in formatted.columns:
        if col in money_cols:
            formatted[col] = formatted[col].map(lambda v: _fmt_money(v, s.currency))
        elif col in pct_cols:
            formatted[col] = formatted[col].map(_fmt_pct)
    formatted = formatted.reset_index()
    formatted.columns = [c.replace("_", " ").title() for c in formatted.columns]

    return html.Div(
        [
            _section("Comparison Results", "bi-table", "Side-by-side comparison of key metrics across all selected Monte Carlo models."),
            dbc.Card(
                dbc.CardBody(
                    DataTable(
                        data=formatted.to_dict("records"),
                        columns=[{"name": c, "id": c} for c in formatted.columns],
                        style_cell={"fontSize": "0.78rem", "padding": "5px 8px"},
                        style_header={
                            "fontWeight": "700",
                            "backgroundColor": "#f1f5f9",
                            "fontSize": "0.7rem",
                            "textTransform": "uppercase",
                            "whiteSpace": "normal",
                        },
                        style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#f8fafc"}],
                    )
                ),
                className="chart-card mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(_chart_card(model_comparison_bars(comparison_df, ccy)), width=12),
                ],
                className="g-2 mb-1",
            ),
            _chart_card(distribution_overlay(results, ccy)),
        ]
    )


def _render_export_tab(sim_key: str | None, s: Scenario) -> html.Div:
    scenario_json = json.dumps(s.to_dict(), indent=2, default=str)
    has_sim = sim_key and sim_key in _CACHE

    download_items = [
        dcc.Download(id="dl-scenario"),
        dbc.Button(
            [html.I(className="bi bi-download me-2"), "Scenario JSON"],
            id="btn-dl-scenario",
            color="outline-primary",
            size="sm",
            className="me-2 mb-2",
        ),
    ]
    if has_sim:
        download_items += [
            dcc.Download(id="dl-summary"),
            dbc.Button(
                [html.I(className="bi bi-download me-2"), "Summary CSV"],
                id="btn-dl-summary",
                color="outline-success",
                size="sm",
                className="me-2 mb-2",
            ),
            dcc.Download(id="dl-paths"),
            dbc.Button(
                [html.I(className="bi bi-download me-2"), "Simulation Paths CSV"],
                id="btn-dl-paths",
                color="outline-secondary",
                size="sm",
                className="me-2 mb-2",
            ),
            dcc.Download(id="dl-cashflows"),
            dbc.Button(
                [html.I(className="bi bi-download me-2"), "Final Cashflows CSV"],
                id="btn-dl-cashflows",
                color="outline-secondary",
                size="sm",
                className="me-2 mb-2",
            ),
        ]

    return html.Div(
        [
            html.Div(DISCLAIMER, className="disclaimer-box mb-3"),
            html.P(
                "Israeli tax mode applies the selected capital gains rate to positive real gains at final liquidation "
                "using inflation-indexed contributions as the cost basis. "
                "It does not model every Israeli tax account type, offset, or exemption.",
                className="text-muted mb-3",
                style={"fontSize": "0.8rem"},
            ),
            _section("Downloads", "bi-download"),
            html.Div(download_items),
            html.Hr(),
            _section("Current Scenario (JSON)", "bi-code-slash"),
            dbc.Card(
                dbc.CardBody(
                    html.Pre(scenario_json, className="mb-0", style={"fontSize": "0.75rem", "maxHeight": "320px", "overflowY": "auto"})
                ),
                className="chart-card",
            ),
        ]
    )


# Wire the comparison tab shell into the layout via a callback
@callback(
    Output("content-compare", "children", allow_duplicate=True),
    Input("main-tabs", "active_tab"),
    State("data-key-store", "data"),
    prevent_initial_call=True,
)
def init_compare_tab(active_tab: str, data_key: str | None):
    if active_tab != "tab-compare":
        raise dash.exceptions.PreventUpdate
    if not data_key:
        return _empty_state("Load market data first, then use Model Comparison.", "bi-collection")
    return _render_compare_tab_shell()


# Wire comparison results into the inner div
@callback(
    Output("compare-results-inner", "children"),
    Output("compare-key-store", "data", allow_duplicate=True),
    Input("run-compare-btn", "n_clicks"),
    State("compare-models-select", "value"),
    State("scenario-store", "data"),
    State("data-key-store", "data"),
    prevent_initial_call=True,
)
def run_comparison_inner(n_clicks, models, scenario_dict, data_key):
    if not n_clicks or not models or not scenario_dict or not data_key or data_key not in _CACHE:
        raise dash.exceptions.PreventUpdate
    try:
        s = Scenario.from_dict(scenario_dict)
        asset_returns = _CACHE[data_key]["returns"]
        comparison_df = compare_models(asset_returns, s, models)
        comparison_results = {m: run_simulation(asset_returns, s, m) for m in models}
        ck = str(uuid.uuid4())
        _CACHE[ck] = {"comparison_df": comparison_df, "results": comparison_results, "scenario": s}
        return _render_compare_tab(ck, s), ck
    except Exception as exc:
        return _empty_state(f"Comparison error: {exc}", "bi-exclamation-triangle"), no_update


# Download callbacks
@callback(
    Output("dl-scenario", "data"),
    Input("btn-dl-scenario", "n_clicks"),
    State("scenario-store", "data"),
    prevent_initial_call=True,
)
def download_scenario(n_clicks, scenario_dict):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    s = Scenario.from_dict(scenario_dict) if scenario_dict else Scenario()
    return dict(content=json.dumps(s.to_dict(), indent=2, default=str), filename="scenario.json")


@callback(
    Output("dl-summary", "data"),
    Input("btn-dl-summary", "n_clicks"),
    State("sim-key-store", "data"),
    State("scenario-store", "data"),
    prevent_initial_call=True,
)
def download_summary(n_clicks, sim_key, scenario_dict):
    if not n_clicks or not sim_key or sim_key not in _CACHE:
        raise dash.exceptions.PreventUpdate
    s = Scenario.from_dict(scenario_dict) if scenario_dict else Scenario()
    summary = _CACHE[sim_key]["result"]["summary"]
    rows = _summary_table_rows(summary, s.currency)
    content = pd.DataFrame(rows).to_csv(index=False)
    return dict(content=content, filename="simulation_summary.csv")


@callback(
    Output("dl-paths", "data"),
    Input("btn-dl-paths", "n_clicks"),
    State("sim-key-store", "data"),
    prevent_initial_call=True,
)
def download_paths(n_clicks, sim_key):
    if not n_clicks or not sim_key or sim_key not in _CACHE:
        raise dash.exceptions.PreventUpdate
    paths = _CACHE[sim_key]["result"]["paths"]
    content = pd.DataFrame(paths).to_csv(index=False)
    return dict(content=content, filename="simulation_paths.csv")


@callback(
    Output("dl-cashflows", "data"),
    Input("btn-dl-cashflows", "n_clicks"),
    State("sim-key-store", "data"),
    prevent_initial_call=True,
)
def download_cashflows(n_clicks, sim_key):
    if not n_clicks or not sim_key or sim_key not in _CACHE:
        raise dash.exceptions.PreventUpdate
    result = _CACHE[sim_key]["result"]
    cf = result["cashflows"]
    df = pd.DataFrame(
        {
            "final_after_tax_liquidation_value": result["paths"][:, -1],
            "total_contributions": result["contributions"][:, -1],
            "net_profit_if_sold": cf["net_profit_if_sold"][:, -1],
            "cumulative_gross_dividends": cf["cumulative_gross_dividends"][:, -1],
            "cumulative_net_dividends": cf["cumulative_net_dividends"][:, -1],
            "cumulative_dividend_taxes": cf["cumulative_dividend_taxes"][:, -1],
            "capital_gains_tax_if_sold": cf["liquidation_taxes"][:, -1],
        }
    )
    return dict(content=df.to_csv(index=False), filename="simulation_final_cashflows.csv")


@callback(
    Output("lang-store", "data"),
    Input("lang-en-btn", "n_clicks"),
    Input("lang-ru-btn", "n_clicks"),
    Input("lang-he-btn", "n_clicks"),
    prevent_initial_call=True,
)
def set_language(n_en, n_ru, n_he):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    btn_id = ctx.triggered[0]["prop_id"].split(".")[0]
    return {"lang-en-btn": "en", "lang-ru-btn": "ru", "lang-he-btn": "he"}.get(btn_id, "en")


@callback(
    Output("lbl-nav-subtitle", "children"),
    Output("lbl-tab-setup", "label"),
    Output("lbl-tab-data", "label"),
    Output("lbl-tab-sim", "label"),
    Output("lbl-tab-compare", "label"),
    Output("lbl-tab-risk", "label"),
    Output("lbl-tab-export", "label"),
    Output("lbl-acc-portfolio", "children"),
    Output("lbl-acc-accumulation", "children"),
    Output("lbl-acc-model", "children"),
    Output("lbl-btn-load", "children"),
    Output("lbl-btn-run", "children"),
    Output("lbl-btn-reset", "children"),
    Output("ttip-market-preset", "children"),
    Output("ttip-tickers-input", "children"),
    Output("ttip-weights-input", "children"),
    Output("ttip-lookback-years", "children"),
    Output("ttip-frequency", "children"),
    Output("ttip-price-field", "children"),
    Output("ttip-currency", "children"),
    Output("ttip-start-date", "children"),
    Output("ttip-end-date", "children"),
    Output("ttip-initial-capital", "children"),
    Output("ttip-monthly-contrib", "children"),
    Output("ttip-annual-contrib-increase", "children"),
    Output("ttip-horizon-years", "children"),
    Output("ttip-simulations", "children"),
    Output("ttip-target-value", "children"),
    Output("ttip-primary-model", "children"),
    Output("ttip-rebalancing", "children"),
    Output("ttip-block-size", "children"),
    Output("ttip-fat-tail-df", "children"),
    Output("ttip-annual-fee", "children"),
    Output("ttip-tax-drag", "children"),
    Output("ttip-dividend-yield", "children"),
    Output("ttip-dividend-tax", "children"),
    Output("ttip-dividend-mode", "children"),
    Output("ttip-tax-mode", "children"),
    Output("ttip-custom-tax-rate", "children"),
    Output("ttip-annual-inflation", "children"),
    Output("ttip-btn-load", "children"),
    Output("ttip-btn-run", "children"),
    Output("ttip-btn-reset", "children"),
    Input("lang-store", "data"),
)
def apply_language(lang: str):
    lang = lang or "en"
    t = TRANSLATIONS.get(lang, TRANSLATIONS["en"])
    return (
        t["nav_subtitle"],
        t["tab_setup"], t["tab_data"], t["tab_sim"], t["tab_compare"], t["tab_risk"], t["tab_export"],
        t["acc_portfolio"], t["acc_accumulation"], t["acc_model"],
        t["btn_load"], t["btn_run"], t["btn_reset"],
        t["tip_market_preset"], t["tip_tickers_input"], t["tip_weights_input"],
        t["tip_lookback_years"], t["tip_frequency"], t["tip_price_field"], t["tip_currency"],
        t["tip_start_date"], t["tip_end_date"],
        t["tip_initial_capital"], t["tip_monthly_contrib"], t["tip_annual_contrib_increase"],
        t["tip_horizon_years"], t["tip_simulations"], t["tip_target_value"],
        t["tip_primary_model"], t["tip_rebalancing"], t["tip_block_size"],
        t["tip_fat_tail_df"], t["tip_annual_fee"], t["tip_tax_drag"],
        t["tip_dividend_yield"], t["tip_dividend_tax"], t["tip_dividend_mode"],
        t["tip_tax_mode"], t["tip_custom_tax_rate"], t["tip_annual_inflation"],
        t["tip_btn_load"], t["tip_btn_run"], t["tip_btn_reset"],
    )


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    """Console script entry point (uv run monte-carlo)."""
    import os

    port = int(os.environ.get("PORT", 8050))
    debug = os.environ.get("DEBUG", "false").lower() == "true"
    app.run(debug=debug, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
