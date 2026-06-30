from __future__ import annotations

import base64
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from portfolio_monte_carlo.charts.plots import (
    distribution_overlay,
    drawdown_histogram,
    fan_chart,
    final_value_histogram,
    historical_prices,
    income_waterfall,
    model_comparison_bars,
    rolling_chart,
    sample_trajectories,
    target_probability_gauge,
)
from portfolio_monte_carlo.core.portfolio import align_weights, portfolio_returns
from portfolio_monte_carlo.core.returns import calculate_returns, rolling_returns, rolling_volatility
from portfolio_monte_carlo.core.risk import return_statistics
from portfolio_monte_carlo.core.scenario import Scenario
from portfolio_monte_carlo.core.scenario import periods_per_year
from portfolio_monte_carlo.core.simulation import compare_models, run_simulation
from portfolio_monte_carlo.data.providers import download_yfinance_prices
from portfolio_monte_carlo.data.validation import quality_report_frame, validate_prices


st.set_page_config(
    page_title="Portfolio Monte Carlo Simulator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


LANGUAGES = {
    "en": "English",
    "ru": "Русский",
    "he": "עברית",
}


I18N = {
    "en": {
        "title": "Portfolio Monte Carlo Simulator",
        "subtitle": "Scenario-based projections for portfolio accumulation. Results are simulated outcome distributions, not predictions.",
        "portfolio": "Portfolio",
        "market_preset": "Market preset",
        "index_reference": "Israeli market index reference",
        "index_reference_note": "Reference list for TASE index families. Yahoo availability can vary by symbol.",
        "tickers": "Tickers / index symbols",
        "weights": "Portfolio weights, % in ticker order",
        "weights_note": "Weights are allocation shares, not money amounts. They are normalized automatically.",
        "historical_lookback": "Historical lookback, years",
        "frequency": "Return frequency / simulation step",
        "price_field": "Price field",
        "start_date": "Start date",
        "end_date": "End date",
        "accumulation": "Accumulation",
        "currency": "Display currency for money amounts",
        "amount_scale": "Money input scale",
        "initial_capital": "Initial capital",
        "monthly_contribution": "Monthly contribution",
        "annual_contribution_increase": "Annual contribution increase, %",
        "horizon_scale": "Horizon scale",
        "investment_horizon": "Investment horizon",
        "chart_time_scale": "Chart time scale",
        "simulations": "Number of simulations, paths",
        "target_value": "Target portfolio value",
        "assumptions": "Assumptions",
        "primary_model": "Primary model",
        "rebalancing": "Rebalancing",
        "block_size": "Block size",
        "fat_tail_df": "Fat-tail degrees of freedom",
        "annual_fee": "Annual fee / expense ratio, %",
        "other_tax_drag": "Other annual tax drag, %",
        "dividend_yield": "Estimated annual dividend yield, %",
        "dividend_tax": "Dividend tax / withholding rate, %",
        "dividend_handling": "Dividend handling",
        "capital_gains_tax_model": "Capital gains tax model",
        "custom_capital_gains_tax": "Custom capital gains tax rate, %",
        "inflation": "Annual inflation adjustment, %",
        "setup": "Portfolio Setup",
        "historical_data": "Historical Data",
        "simulation": "Simulation",
        "comparison": "Model Comparison",
        "risk": "Risk Analysis",
        "export": "Assumptions & Export",
        "scenario": "Scenario",
        "weights_caption": "Weights show the normalized allocation share of each ticker. They always sum to 100%.",
        "data_quality": "Data Quality",
        "historical_statistics": "Historical Statistics",
        "hist_caption": "Brief view with annualized units where relevant. VaR and CVaR are per selected return frequency.",
        "show_raw_stats": "Show full raw statistics",
        "insights": "Insights",
        "run_primary": "Run primary simulation",
        "median_final_value": "Median final value",
        "p5": "5th percentile",
        "target_probability": "Target probability",
        "median_drawdown": "Median max drawdown",
        "net_profit_if_sold": "Net profit if sold",
        "net_dividends": "Net dividends",
        "summary_metrics": "Summary Metrics",
        "income_breakdown": "Income and Tax Breakdown",
        "models_to_compare": "Models to compare",
        "run_comparison": "Run model comparison",
        "risk_snapshot": "Risk Snapshot",
        "risk_caption": "This section separates historical risk from simulated downside risk.",
        "rolling_risk_charts": "Rolling Risk Charts",
        "disclaimer": "This is not financial advice. Past performance does not guarantee future returns. Monte Carlo results depend strongly on model assumptions. Historical data may contain errors, survivorship bias, missing dividends, currency distortions or other limitations. Taxes and fees are simplified unless explicitly modeled. Results are scenario-based projections, not predictions.",
        "download_scenario": "Download scenario JSON",
        "load_scenario": "Load scenario JSON",
        "download_stats": "Download historical statistics CSV",
        "download_summary": "Download summary CSV",
        "download_paths": "Download simulation paths CSV",
        "download_cashflows": "Download final cashflows CSV",
        "download_chart": "Download fan chart HTML",
    },
    "ru": {
        "title": "Симулятор портфеля Монте-Карло",
        "subtitle": "Сценарное моделирование накопления портфеля. Результаты показывают распределение исходов, а не прогноз.",
        "portfolio": "Портфель",
        "market_preset": "Готовый рынок",
        "index_reference": "Справочник индексов Израиля",
        "index_reference_note": "Список семейств индексов TASE. Доступность Yahoo-символов может отличаться.",
        "tickers": "Тикеры / символы индексов",
        "weights": "Веса портфеля, % в порядке тикеров",
        "weights_note": "Веса — это доли распределения, не суммы денег. Они нормализуются автоматически.",
        "historical_lookback": "Исторический период, лет",
        "frequency": "Частота доходности / шаг симуляции",
        "price_field": "Поле цены",
        "start_date": "Дата начала",
        "end_date": "Дата окончания",
        "accumulation": "Накопление",
        "currency": "Валюта отображения сумм",
        "amount_scale": "Масштаб ввода сумм",
        "initial_capital": "Начальный капитал",
        "monthly_contribution": "Ежемесячный взнос",
        "annual_contribution_increase": "Годовой рост взноса, %",
        "horizon_scale": "Масштаб горизонта",
        "investment_horizon": "Инвестиционный горизонт",
        "chart_time_scale": "Масштаб времени на графиках",
        "simulations": "Количество симуляций, траекторий",
        "target_value": "Целевая стоимость портфеля",
        "assumptions": "Допущения",
        "primary_model": "Основная модель",
        "rebalancing": "Ребалансировка",
        "block_size": "Размер блока",
        "fat_tail_df": "Степени свободы fat-tail модели",
        "annual_fee": "Годовая комиссия / expense ratio, %",
        "other_tax_drag": "Прочая годовая налоговая нагрузка, %",
        "dividend_yield": "Ожидаемая дивидендная доходность, %",
        "dividend_tax": "Налог / удержание с дивидендов, %",
        "dividend_handling": "Обработка дивидендов",
        "capital_gains_tax_model": "Модель налога на прирост капитала",
        "custom_capital_gains_tax": "Пользовательский налог на прирост, %",
        "inflation": "Годовая инфляция, %",
        "setup": "Настройка портфеля",
        "historical_data": "Исторические данные",
        "simulation": "Симуляция",
        "comparison": "Сравнение моделей",
        "risk": "Анализ риска",
        "export": "Допущения и экспорт",
        "scenario": "Сценарий",
        "weights_caption": "Веса показывают нормализованную долю каждого тикера. Сумма всегда 100%.",
        "data_quality": "Качество данных",
        "historical_statistics": "Историческая статистика",
        "hist_caption": "Краткий вид с годовыми единицами там, где это уместно. VaR и CVaR указаны на выбранный период доходности.",
        "show_raw_stats": "Показать полную сырую статистику",
        "insights": "Инсайты",
        "run_primary": "Запустить основную симуляцию",
        "median_final_value": "Медианная итоговая стоимость",
        "p5": "5-й перцентиль",
        "target_probability": "Вероятность цели",
        "median_drawdown": "Медианная просадка",
        "net_profit_if_sold": "Чистая прибыль при продаже",
        "net_dividends": "Чистые дивиденды",
        "summary_metrics": "Сводные метрики",
        "income_breakdown": "Доходы и налоги",
        "models_to_compare": "Модели для сравнения",
        "run_comparison": "Запустить сравнение моделей",
        "risk_snapshot": "Снимок риска",
        "risk_caption": "Раздел отделяет исторический риск от смоделированного downside risk.",
        "rolling_risk_charts": "Графики скользящего риска",
        "disclaimer": "Это не финансовая рекомендация. Прошлая доходность не гарантирует будущую. Результаты Монте-Карло сильно зависят от допущений. Исторические данные могут содержать ошибки, survivorship bias, отсутствующие дивиденды, валютные искажения и другие ограничения. Налоги и комиссии упрощены, если явно не указано иное. Результаты являются сценарными проекциями, а не прогнозами.",
        "download_scenario": "Скачать сценарий JSON",
        "load_scenario": "Загрузить сценарий JSON",
        "download_stats": "Скачать историческую статистику CSV",
        "download_summary": "Скачать сводку CSV",
        "download_paths": "Скачать траектории CSV",
        "download_cashflows": "Скачать финальные cashflows CSV",
        "download_chart": "Скачать fan chart HTML",
    },
    "he": {
        "title": "סימולטור מונטה קרלו לתיק השקעות",
        "subtitle": "תחזיות תרחישיות לצבירת תיק. התוצאות מציגות התפלגות תרחישים, לא ניבוי.",
        "portfolio": "תיק השקעות",
        "market_preset": "בחירת שוק מוכנה",
        "index_reference": "רשימת מדדי שוק ישראליים",
        "index_reference_note": "רשימת משפחות מדדי TASE. זמינות סמלי Yahoo עשויה להשתנות.",
        "tickers": "טיקרים / סמלי מדדים",
        "weights": "משקולות התיק, % לפי סדר הטיקרים",
        "weights_note": "משקל הוא שיעור הקצאה, לא סכום כסף. הערכים מנורמלים אוטומטית.",
        "historical_lookback": "תקופה היסטורית, שנים",
        "frequency": "תדירות תשואה / צעד סימולציה",
        "price_field": "שדה מחיר",
        "start_date": "תאריך התחלה",
        "end_date": "תאריך סיום",
        "accumulation": "צבירה",
        "currency": "מטבע להצגת סכומים",
        "amount_scale": "קנה מידה להזנת סכומים",
        "initial_capital": "הון התחלתי",
        "monthly_contribution": "הפקדה חודשית",
        "annual_contribution_increase": "עליית הפקדה שנתית, %",
        "horizon_scale": "קנה מידה לאופק",
        "investment_horizon": "אופק השקעה",
        "chart_time_scale": "קנה מידה לזמן בגרפים",
        "simulations": "מספר סימולציות, מסלולים",
        "target_value": "יעד שווי התיק",
        "assumptions": "הנחות",
        "primary_model": "מודל ראשי",
        "rebalancing": "איזון מחדש",
        "block_size": "גודל בלוק",
        "fat_tail_df": "דרגות חופש במודל זנבות כבדים",
        "annual_fee": "דמי ניהול / expense ratio שנתי, %",
        "other_tax_drag": "גרירת מס שנתית אחרת, %",
        "dividend_yield": "תשואת דיבידנד שנתית מוערכת, %",
        "dividend_tax": "מס / ניכוי במקור על דיבידנד, %",
        "dividend_handling": "טיפול בדיבידנדים",
        "capital_gains_tax_model": "מודל מס רווחי הון",
        "custom_capital_gains_tax": "מס רווח הון מותאם, %",
        "inflation": "התאמת אינפלציה שנתית, %",
        "setup": "הגדרת תיק",
        "historical_data": "נתונים היסטוריים",
        "simulation": "סימולציה",
        "comparison": "השוואת מודלים",
        "risk": "ניתוח סיכון",
        "export": "הנחות וייצוא",
        "scenario": "תרחיש",
        "weights_caption": "המשקולות מציגות את שיעור ההקצאה המנורמל לכל טיקר. הסכום תמיד 100%.",
        "data_quality": "איכות נתונים",
        "historical_statistics": "סטטיסטיקה היסטורית",
        "hist_caption": "תצוגה קצרה עם יחידות שנתיות כשמתאים. VaR ו-CVaR הם לפי תדירות התשואה שנבחרה.",
        "show_raw_stats": "הצגת סטטיסטיקה מלאה",
        "insights": "תובנות",
        "run_primary": "הרצת סימולציה ראשית",
        "median_final_value": "שווי סופי חציוני",
        "p5": "אחוזון 5",
        "target_probability": "הסתברות להגיע ליעד",
        "median_drawdown": "ירידה מקסימלית חציונית",
        "net_profit_if_sold": "רווח נקי במכירה",
        "net_dividends": "דיבידנדים נטו",
        "summary_metrics": "מדדי סיכום",
        "income_breakdown": "פירוט הכנסות ומסים",
        "models_to_compare": "מודלים להשוואה",
        "run_comparison": "הרצת השוואת מודלים",
        "risk_snapshot": "תמונת סיכון",
        "risk_caption": "החלק מפריד בין סיכון היסטורי לבין סיכון downside מדומה.",
        "rolling_risk_charts": "גרפי סיכון מתגלגל",
        "disclaimer": "אין לראות בכך ייעוץ פיננסי. ביצועי עבר אינם מבטיחים תשואה עתידית. תוצאות מונטה קרלו תלויות מאוד בהנחות המודל. נתונים היסטוריים עשויים לכלול שגיאות, הטיית שרידות, דיבידנדים חסרים, עיוותי מטבע ומגבלות אחרות. מסים ודמי ניהול מפושטים אלא אם צוין אחרת. התוצאות הן תחזיות תרחישיות ולא ניבוי.",
        "download_scenario": "הורדת תרחיש JSON",
        "load_scenario": "טעינת תרחיש JSON",
        "download_stats": "הורדת סטטיסטיקה היסטורית CSV",
        "download_summary": "הורדת סיכום CSV",
        "download_paths": "הורדת מסלולים CSV",
        "download_cashflows": "הורדת cashflows סופיים CSV",
        "download_chart": "הורדת fan chart HTML",
    },
}


def t(lang: str, key: str) -> str:
    return I18N.get(lang, I18N["en"]).get(key, I18N["en"].get(key, key))


OPTION_LABELS = {
    "en": {
        "daily": "Daily",
        "weekly": "Weekly",
        "monthly": "Monthly",
        "none": "None",
        "quarterly": "Quarterly",
        "yearly": "Yearly",
        "years": "Years",
        "months": "Months",
        "periods": "Simulation periods",
        "Units": "Units",
        "Thousands": "Thousands",
        "historical_bootstrap": "Historical bootstrap",
        "block_bootstrap": "Block bootstrap",
        "normal": "Parametric normal",
        "fat_tail": "Fat-tail Student-t",
        "regime": "Regime approximation",
        "israel_individual": "Israel individual investor, simplified 25%",
        "israel_substantial_shareholder": "Israel substantial shareholder, simplified 30%",
        "custom": "Custom capital gains rate",
        "track_only": "Track only",
        "reinvest": "Reinvest net dividends",
        "withdraw": "Withdraw as cash income",
    },
    "ru": {
        "daily": "Дни",
        "weekly": "Недели",
        "monthly": "Месяцы",
        "none": "Нет",
        "quarterly": "Ежеквартально",
        "yearly": "Ежегодно",
        "years": "Годы",
        "months": "Месяцы",
        "periods": "Шаги симуляции",
        "Units": "Единицы",
        "Thousands": "Тысячи",
        "historical_bootstrap": "Исторический bootstrap",
        "block_bootstrap": "Блочный bootstrap",
        "normal": "Нормальная модель",
        "fat_tail": "Fat-tail Student-t",
        "regime": "Режимная аппроксимация",
        "israel_individual": "Израиль, физлицо, упрощенно 25%",
        "israel_substantial_shareholder": "Израиль, существенный акционер, упрощенно 30%",
        "custom": "Пользовательская ставка",
        "track_only": "Только отслеживать",
        "reinvest": "Реинвестировать чистые дивиденды",
        "withdraw": "Выводить как денежный доход",
    },
    "he": {
        "daily": "יומי",
        "weekly": "שבועי",
        "monthly": "חודשי",
        "none": "ללא",
        "quarterly": "רבעוני",
        "yearly": "שנתי",
        "years": "שנים",
        "months": "חודשים",
        "periods": "צעדי סימולציה",
        "Units": "יחידות",
        "Thousands": "אלפים",
        "historical_bootstrap": "Bootstrap היסטורי",
        "block_bootstrap": "Bootstrap בבלוקים",
        "normal": "מודל נורמלי",
        "fat_tail": "Student-t עם זנבות כבדים",
        "regime": "קירוב משטרים",
        "israel_individual": "ישראל, יחיד, 25% מפושט",
        "israel_substantial_shareholder": "ישראל, בעל מניות מהותי, 30% מפושט",
        "custom": "שיעור מותאם",
        "track_only": "מעקב בלבד",
        "reinvest": "השקעה מחדש של דיבידנד נטו",
        "withdraw": "משיכה כהכנסה במזומן",
    },
}


def opt(lang: str, value: str) -> str:
    return OPTION_LABELS.get(lang, OPTION_LABELS["en"]).get(value, OPTION_LABELS["en"].get(value, value))

st.markdown(
    """
    <style>
    :root {
      --accent: #176B87;
      --ink: #182026;
      --muted: #5f6f78;
      --surface: #f7f9f8;
    }
    .main .block-container {
      padding-top: 1.5rem;
      padding-bottom: 2rem;
      max-width: 1380px;
    }
    h1, h2, h3 {
      letter-spacing: 0;
    }
    [data-testid="stMetric"] {
      background: var(--surface);
      border: 1px solid #dfe7e4;
      border-radius: 8px;
      padding: 0.85rem 1rem;
    }
    div[data-testid="stSidebar"] {
      border-right: 1px solid #e4ebe8;
    }
    .assumption-note {
      border-left: 4px solid var(--accent);
      padding: 0.65rem 0 0.65rem 1rem;
      color: var(--muted);
      margin: 0.5rem 0 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def _banner_data_uri(path: str) -> str:
    image_path = Path(path)
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _render_global_banner(path: str = "assets/monte_carlo_banner.png") -> None:
    data_uri = _banner_data_uri(path)
    st.markdown(
        f"""
        <style>
        :root {{
            --global-banner-height: clamp(120px, 11vw, 208px);
        }}
        .global-app-banner {{
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            width: 100vw;
            height: var(--global-banner-height);
            z-index: 999999;
            background: #ffffff;
            border-bottom: 1px solid #dfe7e4;
            overflow: hidden;
        }}
        .global-app-banner::before {{
            content: "";
            position: absolute;
            inset: -18px;
            background-image: url("{data_uri}");
            background-size: cover;
            background-position: center center;
            filter: blur(14px) saturate(1.04);
            opacity: 0.72;
            transform: scale(1.04);
        }}
        .global-app-banner img {{
            position: relative;
            z-index: 1;
            width: 100%;
            height: 100%;
            object-fit: contain;
            object-position: center center;
            display: block;
            background: transparent;
        }}
        .main .block-container {{
            padding-top: 0 !important;
        }}
        section[data-testid="stSidebar"] {{
            top: var(--global-banner-height);
            height: calc(100vh - var(--global-banner-height));
        }}
        header[data-testid="stHeader"] {{
            top: var(--global-banner-height);
        }}
        </style>
        <div class="global-app-banner">
            <img src="{data_uri}" alt="Portfolio Monte Carlo Simulator banner" />
        </div>
        """,
        unsafe_allow_html=True,
    )


MODEL_LABELS = {
    "historical_bootstrap": "Historical bootstrap",
    "block_bootstrap": "Block bootstrap",
    "normal": "Parametric normal",
    "fat_tail": "Fat-tail Student-t",
    "regime": "Regime approximation",
}

CURRENCY_SYMBOLS = {"USD": "$", "ILS": "₪"}

AMOUNT_SCALES = {
    "Units": 1.0,
    "Thousands": 1_000.0,
}

MARKET_PRESETS = {
    "S&P 500 index": ["^GSPC"],
    "S&P 500 ETF": ["SPY"],
    "Israel core indices": ["TA35.TA", "^TA125.TA", "TA90.TA"],
    "Israel large caps": ["LUMI.TA", "POLI.TA", "NICE.TA", "TEVA.TA", "ICL.TA"],
    "Israel banks": ["LUMI.TA", "POLI.TA", "MZTF.TA", "DSCT.TA"],
    "US broad ETFs": ["SPY", "QQQ", "VTI"],
    "Custom": [],
}

DISPLAY_SYMBOLS = {
    "^GSPC": "S&P 500",
    "TA35.TA": "TA-35",
    "^TA125.TA": "TA-125",
    "TA90.TA": "TA-90",
}

ISRAEL_INDEX_UNIVERSE = [
    ("TA-35", "TA35.TA", "Large-cap benchmark"),
    ("TA-90", "TA90.TA", "Shares outside TA-35 within TA-125"),
    ("TA-125", "^TA125.TA", "TA-35 plus TA-90 broad benchmark"),
    ("TA-SME60", "TA-SME60.TA", "Small and mid-cap shares"),
    ("TA-AllShare", "TA-ALL.TA", "Broad all-share index"),
    ("TA-Growth", "TA-GROWTH.TA", "Growth-company segment"),
    ("TA-Banks5", "TA-BANKS5.TA", "Five largest banks"),
    ("TA-Finance", "TA-FINANCE.TA", "Financial sector"),
    ("TA-Insurance", "TA-INSURANCE.TA", "Insurance sector"),
    ("TA-RealEstate", "TA-REALESTATE.TA", "Real estate sector"),
    ("TA-Construction", "TA-CONSTRUCTION.TA", "Construction sector"),
    ("TA-Technology", "TA-TECH.TA", "Technology sector"),
    ("TA-Biomed", "TA-BIOMED.TA", "Biomed sector"),
    ("TA-Oil & Gas", "TA-OILGAS.TA", "Oil and gas sector"),
    ("TA-Industrials", "TA-INDUSTRY.TA", "Industrial companies"),
    ("TA-Consumer", "TA-CONSUMER.TA", "Consumer sector"),
]

TAX_MODE_LABELS = {
    "none": "No tax model",
    "israel_individual": "Israel individual investor, simplified 25%",
    "israel_substantial_shareholder": "Israel substantial shareholder, simplified 30%",
    "custom": "Custom capital gains rate",
}

CHART_TIME_SCALE_LABELS = {
    "years": "Years",
    "months": "Months",
    "periods": "Simulation periods",
}

DIVIDEND_MODE_LABELS = {
    "track_only": "Track only",
    "reinvest": "Reinvest net dividends",
    "withdraw": "Withdraw as cash income",
}


def _parse_list(text: str) -> list[str]:
    return [item.strip().upper() for item in text.replace(";", ",").split(",") if item.strip()]


def _display_symbol(symbol: str) -> str:
    return DISPLAY_SYMBOLS.get(symbol, symbol.lstrip("^"))


def _display_symbols(symbols: list[str] | tuple[str, ...]) -> list[str]:
    return [_display_symbol(symbol) for symbol in symbols]


def _with_display_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.copy()
    renamed.columns = [_display_symbol(str(column)) for column in renamed.columns]
    return renamed


def _parse_weights(text: str, count: int) -> list[float]:
    raw = [float(item.strip()) for item in text.replace(";", ",").split(",") if item.strip()]
    if len(raw) != count:
        raise ValueError("The number of weights must match the number of tickers.")
    return raw


def _format_money(value: float, currency: str) -> str:
    return f"{CURRENCY_SYMBOLS.get(currency, currency)}{value:,.0f}"


def _format_percent(value: float) -> str:
    return f"{value:.1%}"


def _simulation_x_axis(scenario: Scenario, points: int) -> tuple[pd.Series, str]:
    steps = pd.Series(range(points), dtype=float)
    factor = periods_per_year(scenario.frequency)
    if scenario.chart_time_scale == "years":
        return steps / factor, "Years since start"
    if scenario.chart_time_scale == "months":
        return steps * 12 / factor, "Months since start"
    return steps, f"Simulation periods ({scenario.frequency} steps)"


def _median_hover_metrics(result: dict) -> dict[str, np.ndarray]:
    cashflows = result.get("cashflows", {})
    metrics = {
        "Net profit if sold": cashflows.get("net_profit_if_sold"),
        "Net dividends": cashflows.get("cumulative_net_dividends"),
        "Gross dividends": cashflows.get("cumulative_gross_dividends"),
        "Dividend tax": cashflows.get("cumulative_dividend_taxes"),
        "Capital gains tax": cashflows.get("liquidation_taxes"),
    }
    return {
        label: np.median(values, axis=0)
        for label, values in metrics.items()
        if values is not None
    }


def _amount_input(
    label: str,
    value: float,
    scale: float,
    currency: str,
    step: float,
    help_text: str,
) -> float:
    symbol = CURRENCY_SYMBOLS.get(currency, currency)
    shown = value / scale
    suffix = "" if scale == 1 else " thousand"
    return (
        st.sidebar.number_input(
            f"{label}, {symbol}{suffix}",
            min_value=0.0,
            value=float(shown),
            step=step / scale,
            help=help_text,
        )
        * scale
    )


@st.cache_data(show_spinner=False)
def _load_prices(tickers: tuple[str, ...], start: str, end: str, field: str) -> pd.DataFrame:
    return download_yfinance_prices(list(tickers), start=start, end=end, field=field, use_cache=True)


def _scenario_from_sidebar() -> tuple[Scenario, str]:
    lang = st.sidebar.selectbox(
        "Language / Язык / שפה",
        list(LANGUAGES),
        format_func=LANGUAGES.get,
        index=0,
        help="Choose the interface language. English is the default.",
    )
    uploaded_config = st.sidebar.file_uploader(
        t(lang, "load_scenario"),
        type=["json"],
        help="Import a previously exported scenario. Values from the file become the starting point for the controls below.",
    )
    loaded = {}
    if uploaded_config is not None:
        try:
            loaded = json.loads(uploaded_config.getvalue().decode("utf-8"))
            st.sidebar.success("Scenario loaded.")
        except json.JSONDecodeError:
            st.sidebar.error("The uploaded JSON could not be parsed.")

    base = Scenario.from_dict(loaded) if loaded else Scenario()

    st.sidebar.header(t(lang, "portfolio"))
    preset = st.sidebar.selectbox(
        t(lang, "market_preset"),
        list(MARKET_PRESETS),
        index=0 if not loaded else list(MARKET_PRESETS).index("Custom"),
        help="Quick-start lists. Default is the S&P 500 index. Israeli symbols use Yahoo Finance notation where available.",
    )
    with st.sidebar.expander(t(lang, "index_reference"), expanded=False):
        st.caption(t(lang, "index_reference_note"))
        st.dataframe(
            pd.DataFrame(
                [
                    (name, _display_symbol(symbol), symbol, scope)
                    for name, symbol, scope in ISRAEL_INDEX_UNIVERSE
                ],
                columns=["Index", "Display", "Yahoo symbol", "Scope"],
            ),
            hide_index=True,
            use_container_width=True,
        )
    preset_tickers = MARKET_PRESETS[preset] or base.tickers
    ticker_text = st.sidebar.text_input(
        t(lang, "tickers"),
        ", ".join(_display_symbols(preset_tickers)),
        help="Enter 1-5 Yahoo Finance symbols. For some indices Yahoo requires a leading ^ internally; the app hides it in labels where possible.",
    )
    reverse_display = {value.upper(): key for key, value in DISPLAY_SYMBOLS.items()}
    tickers = [reverse_display.get(item, item) for item in _parse_list(ticker_text)[:5]]
    shown_weights = base.weights[: len(tickers)]
    if shown_weights and sum(shown_weights) <= 1.5:
        shown_weights = [weight * 100 for weight in shown_weights]
    weight_default = ", ".join(f"{weight:g}" for weight in shown_weights)
    if len(base.weights) != len(tickers):
        weight_default = ", ".join(["1"] * len(tickers))
    weight_text = st.sidebar.text_input(
        t(lang, "weights"),
        weight_default,
        help="Enter one number per ticker in the same order. Example: SPY, QQQ with 70, 30 means 70% SPY and 30% QQQ. Values are normalized to 100%.",
    )
    st.sidebar.caption(t(lang, "weights_note"))

    lookback_years = st.sidebar.selectbox(
        t(lang, "historical_lookback"),
        [5, 10, 15, 20, 30],
        index=[5, 10, 15, 20, 30].index(base.lookback_years) if base.lookback_years in [5, 10, 15, 20, 30] else 3,
        help="Historical data window in calendar years. The actual returned range can be shorter if the symbol has less history.",
    )
    frequency = st.sidebar.selectbox(
        t(lang, "frequency"),
        ["daily", "weekly", "monthly"],
        index=["daily", "weekly", "monthly"].index(base.frequency),
        format_func=lambda value: opt(lang, value),
        help="Unit for historical returns and each Monte Carlo step. Daily uses 252 trading days/year, weekly uses 52, monthly uses 12.",
    )
    price_field = st.sidebar.selectbox(
        t(lang, "price_field"),
        ["Adj Close", "Close"],
        index=0 if base.price_field == "Adj Close" else 1,
        help="Adjusted close includes corporate actions where the provider supplies them. It is the safer default for return analysis.",
    )

    today = date.today()
    start_default = pd.Timestamp(today) - pd.DateOffset(years=int(lookback_years))
    start_date = st.sidebar.date_input(
        t(lang, "start_date"),
        value=start_default.date(),
        help="First date requested from the market data provider. Actual coverage may start later for newer securities.",
    )
    end_date = st.sidebar.date_input(
        t(lang, "end_date"),
        value=today,
        help="Last date requested from the market data provider.",
    )

    st.sidebar.header(t(lang, "accumulation"))
    currency = st.sidebar.selectbox(
        t(lang, "currency"),
        ["ILS", "USD"],
        index=["ILS", "USD"].index(base.currency),
        help="Currency used for inputs, outputs, labels, and exports. The app does not perform FX conversion between assets.",
    )
    amount_scale_label = st.sidebar.selectbox(
        t(lang, "amount_scale"),
        list(AMOUNT_SCALES),
        index=0,
        format_func=lambda value: opt(lang, value),
        help="Choose Units for exact amounts or Thousands for faster entry of large portfolio values.",
    )
    amount_scale = AMOUNT_SCALES[amount_scale_label]
    initial_capital = _amount_input(
        t(lang, "initial_capital"),
        float(base.initial_capital),
        amount_scale,
        currency,
        1_000.0,
        "Current portfolio value at the beginning of the simulation, in the selected display currency.",
    )
    monthly_contribution = _amount_input(
        t(lang, "monthly_contribution"),
        float(base.monthly_contribution),
        amount_scale,
        currency,
        100.0,
        "Planned monthly deposit in the selected display currency. For weekly or daily simulations it is converted to the matching period amount.",
    )
    annual_contribution_increase = st.sidebar.slider(
        t(lang, "annual_contribution_increase"),
        0.0,
        10.0,
        float(base.annual_contribution_increase) * 100,
        0.5,
        format="%.1f%%",
        help="Yearly increase in contributions, for example to approximate salary growth or inflation-linked savings.",
    ) / 100
    horizon_unit = st.sidebar.selectbox(
        t(lang, "horizon_scale"),
        ["years", "months"],
        index=0,
        format_func=lambda value: opt(lang, value),
        help="Choose whether the investment horizon is entered in years or months.",
    )
    if horizon_unit == "years":
        horizon_length = st.sidebar.slider(
            t(lang, "investment_horizon"),
            1,
            50,
            int(round(base.horizon_years)),
            help="Length of the scenario in years.",
        )
        horizon_years = float(horizon_length)
    else:
        horizon_length = st.sidebar.slider(
            t(lang, "investment_horizon"),
            1,
            600,
            int(round(base.horizon_years * 12)),
            help="Length of the scenario in months. Useful for non-round horizons such as 18 or 30 months.",
        )
        horizon_years = float(horizon_length) / 12
    chart_time_scale = st.sidebar.selectbox(
        t(lang, "chart_time_scale"),
        list(CHART_TIME_SCALE_LABELS),
        format_func=lambda value: opt(lang, value),
        index=list(CHART_TIME_SCALE_LABELS).index(base.chart_time_scale),
        help="Unit used on simulation trajectory charts. This affects only chart labels and scale, not the simulation itself.",
    )
    simulations = st.sidebar.number_input(
        t(lang, "simulations"),
        min_value=500,
        max_value=100_000,
        value=int(base.simulations),
        step=500,
        help="How many random paths to generate. More simulations make percentiles smoother but slower.",
    )
    target_value = _amount_input(
        t(lang, "target_value"),
        float(base.target_value),
        amount_scale,
        currency,
        10_000.0,
        "Goal threshold used to calculate target probability.",
    )

    st.sidebar.header(t(lang, "assumptions"))
    model = st.sidebar.selectbox(
        t(lang, "primary_model"),
        list(MODEL_LABELS),
        format_func=lambda value: opt(lang, value),
        index=list(MODEL_LABELS).index(base.model),
        help="The model used in the Simulation tab. The comparison tab can run several models side by side.",
    )
    rebalancing = st.sidebar.selectbox(
        t(lang, "rebalancing"),
        ["none", "monthly", "quarterly", "yearly"],
        index=["none", "monthly", "quarterly", "yearly"].index(base.rebalancing),
        format_func=lambda value: opt(lang, value),
        help="How often the simulated portfolio is brought back to target weights.",
    )
    block_size_months = st.sidebar.selectbox(
        t(lang, "block_size"),
        [3, 6, 12, 24],
        index=[3, 6, 12, 24].index(base.block_size_months) if base.block_size_months in [3, 6, 12, 24] else 1,
        help="Block length for block bootstrap. Larger blocks preserve longer market episodes but reduce randomness.",
    )
    fat_tail_df = st.sidebar.slider(
        t(lang, "fat_tail_df"),
        3.0,
        30.0,
        float(base.fat_tail_df),
        0.5,
        help="Lower values create heavier tails and more extreme outcomes in the Student-t approximation.",
    )
    annual_fee = st.sidebar.slider(
        t(lang, "annual_fee"),
        0.0,
        3.0,
        float(base.annual_fee) * 100,
        0.05,
        format="%.2f%%",
        help="Annual drag from management fees, ETF expense ratios, or platform fees.",
    ) / 100
    annual_tax_drag = st.sidebar.slider(
        t(lang, "other_tax_drag"),
        0.0,
        5.0,
        float(base.annual_tax_drag) * 100,
        0.1,
        format="%.1f%%",
        help="Optional extra yearly return drag not captured elsewhere. Dividend taxes and capital gains tax are modeled separately below.",
    ) / 100
    annual_dividend_yield = st.sidebar.slider(
        t(lang, "dividend_yield"),
        0.0,
        12.0,
        float(base.annual_dividend_yield) * 100,
        0.1,
        format="%.1f%%",
        help="Estimated yearly dividend yield of the whole portfolio. With Adjusted Close data, use Track only unless you intentionally want dividends to affect portfolio cash flows.",
    ) / 100
    dividend_tax_rate = st.sidebar.slider(
        t(lang, "dividend_tax"),
        0.0,
        50.0,
        float(base.dividend_tax_rate) * 100,
        0.5,
        format="%.1f%%",
        help="Tax or withholding rate applied to estimated dividends before reinvestment or withdrawal.",
    ) / 100
    dividend_mode = st.sidebar.selectbox(
        t(lang, "dividend_handling"),
        list(DIVIDEND_MODE_LABELS),
        format_func=lambda value: opt(lang, value),
        index=list(DIVIDEND_MODE_LABELS).index(base.dividend_mode),
        help="Track only records estimated dividends without changing portfolio value; reinvest adds net dividends; withdraw treats net dividends as cash income outside the portfolio.",
    )
    tax_mode = st.sidebar.selectbox(
        t(lang, "capital_gains_tax_model"),
        list(TAX_MODE_LABELS),
        format_func=lambda value: opt(lang, value),
        index=list(TAX_MODE_LABELS).index(base.tax_mode),
        help="Simplified Israeli tax treatment at final liquidation. It approximates tax on positive real capital gains only.",
    )
    if tax_mode == "israel_individual":
        capital_gains_tax_rate = 0.25
    elif tax_mode == "israel_substantial_shareholder":
        capital_gains_tax_rate = 0.30
    elif tax_mode == "none":
        capital_gains_tax_rate = 0.0
    else:
        capital_gains_tax_rate = st.sidebar.slider(
            t(lang, "custom_capital_gains_tax"),
            0.0,
            50.0,
            float(base.capital_gains_tax_rate) * 100,
            0.5,
            format="%.1f%%",
            help="Custom rate applied to positive real gains at final liquidation.",
        ) / 100
    annual_inflation = st.sidebar.slider(
        t(lang, "inflation"),
        0.0,
        10.0,
        float(base.annual_inflation) * 100,
        0.5,
        format="%.1f%%",
        help="Used for real-return metrics and for indexing the tax basis in the simplified Israeli capital-gains model.",
    ) / 100

    try:
        weights = _parse_weights(weight_text, len(tickers))
    except ValueError as exc:
        st.sidebar.error(str(exc))
        weights = [1.0] * max(1, len(tickers))

    scenario = Scenario(
        tickers=tickers,
        weights=weights,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        lookback_years=int(lookback_years),
        frequency=frequency,
        price_field=price_field,
        currency=currency,
        initial_capital=float(initial_capital),
        monthly_contribution=float(monthly_contribution),
        annual_contribution_increase=float(annual_contribution_increase),
        horizon_years=float(horizon_years),
        simulations=int(simulations),
        annual_fee=float(annual_fee),
        annual_tax_drag=float(annual_tax_drag),
        annual_dividend_yield=float(annual_dividend_yield),
        dividend_tax_rate=float(dividend_tax_rate),
        dividend_mode=dividend_mode,
        tax_mode=tax_mode,
        capital_gains_tax_rate=float(capital_gains_tax_rate),
        annual_inflation=float(annual_inflation),
        target_value=float(target_value),
        chart_time_scale=chart_time_scale,
        rebalancing=rebalancing,
        model=model,
        block_size_months=int(block_size_months),
        fat_tail_df=float(fat_tail_df),
        random_seed=42,
    )
    return scenario, lang


def _summary_frame(summary: dict[str, float], currency: str, lang: str = "en") -> pd.DataFrame:
    labels_en = {
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
    labels_ru = {
        "median_final_value": "Медианная итоговая стоимость",
        "mean_final_value": "Средняя итоговая стоимость",
        "p5": "5-й перцентиль",
        "p10": "10-й перцентиль",
        "p25": "25-й перцентиль",
        "p75": "75-й перцентиль",
        "p90": "90-й перцентиль",
        "p95": "95-й перцентиль",
        "probability_reaching_target": "Вероятность достижения цели",
        "probability_below_contributions": "Вероятность ниже взносов",
        "probability_negative_nominal_return": "Вероятность отрицательной номинальной доходности",
        "probability_negative_real_return": "Вероятность отрицательной реальной доходности",
        "expected_max_drawdown": "Ожидаемая максимальная просадка",
        "median_max_drawdown": "Медианная максимальная просадка",
        "worst_5pct_average_outcome": "Средний результат худших 5%",
        "best_5pct_average_outcome": "Средний результат лучших 5%",
        "total_contributions": "Всего взносов",
        "median_gain_over_contributions": "Медианная прибыль сверх взносов",
        "median_net_profit_if_sold": "Медианная чистая прибыль при продаже",
        "median_cumulative_gross_dividends": "Медианные валовые дивиденды",
        "median_cumulative_net_dividends": "Медианные чистые дивиденды",
        "median_cumulative_dividend_taxes": "Медианный налог на дивиденды",
        "median_liquidation_tax": "Медианный налог на прирост при продаже",
    }
    labels_he = {
        "median_final_value": "שווי סופי חציוני",
        "mean_final_value": "שווי סופי ממוצע",
        "p5": "אחוזון 5",
        "p10": "אחוזון 10",
        "p25": "אחוזון 25",
        "p75": "אחוזון 75",
        "p90": "אחוזון 90",
        "p95": "אחוזון 95",
        "probability_reaching_target": "הסתברות להגיע ליעד",
        "probability_below_contributions": "הסתברות מתחת לסך ההפקדות",
        "probability_negative_nominal_return": "הסתברות לתשואה נומינלית שלילית",
        "probability_negative_real_return": "הסתברות לתשואה ריאלית שלילית",
        "expected_max_drawdown": "ירידה מקסימלית צפויה",
        "median_max_drawdown": "ירידה מקסימלית חציונית",
        "worst_5pct_average_outcome": "ממוצע 5% הגרועים",
        "best_5pct_average_outcome": "ממוצע 5% הטובים",
        "total_contributions": "סך ההפקדות",
        "median_gain_over_contributions": "רווח חציוני מעל ההפקדות",
        "median_net_profit_if_sold": "רווח נקי חציוני במכירה",
        "median_cumulative_gross_dividends": "דיבידנדים ברוטו חציוניים",
        "median_cumulative_net_dividends": "דיבידנדים נטו חציוניים",
        "median_cumulative_dividend_taxes": "מס דיבידנדים חציוני",
        "median_liquidation_tax": "מס רווח הון חציוני במכירה",
    }
    labels = {"en": labels_en, "ru": labels_ru, "he": labels_he}.get(lang, labels_en)
    money_keys = {
        "median_final_value",
        "mean_final_value",
        "p5",
        "p10",
        "p25",
        "p75",
        "p90",
        "p95",
        "worst_5pct_average_outcome",
        "best_5pct_average_outcome",
        "total_contributions",
        "median_gain_over_contributions",
        "median_net_profit_if_sold",
        "median_cumulative_gross_dividends",
        "median_cumulative_net_dividends",
        "median_cumulative_dividend_taxes",
        "median_liquidation_tax",
    }
    percent_keys = {
        "probability_reaching_target",
        "probability_below_contributions",
        "probability_negative_nominal_return",
        "probability_negative_real_return",
        "expected_max_drawdown",
        "median_max_drawdown",
    }
    rows = []
    for key, value in summary.items():
        if key in money_keys:
            display = _format_money(value, currency)
        elif key in percent_keys:
            display = _format_percent(value)
        else:
            display = value
        rows.append({"metric": labels.get(key, key), "value": value, "display": display})
    frame = pd.DataFrame(rows)
    if lang == "ru":
        return frame.rename(columns={"metric": "метрика", "value": "значение", "display": "отображение"})
    if lang == "he":
        return frame.rename(columns={"metric": "מדד", "value": "ערך", "display": "תצוגה"})
    return frame


def _historical_stats_display(stats: pd.DataFrame, frequency: str, lang: str = "en") -> pd.DataFrame:
    if lang == "ru":
        columns = {
            "annualized_return": "Годовая доходность, %",
            "annualized_volatility": "Годовая волатильность, %",
            "cagr": "CAGR, %",
            "max_drawdown": "Макс. просадка, %",
            "historical_var_5": f"VaR 5%, за период {frequency}",
            "historical_cvar_5": f"CVaR 5%, за период {frequency}",
            "observations": "Наблюдения, шт.",
        }
    elif lang == "he":
        columns = {
            "annualized_return": "תשואה שנתית, %",
            "annualized_volatility": "תנודתיות שנתית, %",
            "cagr": "CAGR, %",
            "max_drawdown": "ירידה מקסימלית, %",
            "historical_var_5": f"VaR 5%, לתקופת {frequency}",
            "historical_cvar_5": f"CVaR 5%, לתקופת {frequency}",
            "observations": "מספר תצפיות",
        }
    else:
        columns = {
        "annualized_return": "Annual return, %",
        "annualized_volatility": "Annual volatility, %",
        "cagr": "CAGR, %",
        "max_drawdown": "Max drawdown, %",
        "historical_var_5": f"VaR 5%, per {frequency} period",
        "historical_cvar_5": f"CVaR 5%, per {frequency} period",
        "observations": "Observations, count",
        }
    available = [column for column in columns if column in stats.columns]
    display = stats[available].rename(columns=columns).copy()
    percent_like = [column for column in display.columns if "%" in column]
    for column in percent_like:
        display[column] = display[column].map(lambda value: f"{value:.2%}" if pd.notna(value) else "")
    if "Observations, count" in display.columns:
        display["Observations, count"] = display["Observations, count"].astype(int)
    return display


def _risk_snapshot(stats: pd.DataFrame, frequency: str, lang: str = "en") -> pd.DataFrame:
    if "Portfolio" in stats.index:
        row = stats.loc["Portfolio"]
    else:
        row = stats.iloc[0]
    if lang == "ru":
        rows = [
            {"metric": "Годовая волатильность", "value": f"{row['annualized_volatility']:.2%}", "meaning": "Типичная годовая изменчивость доходности."},
            {"metric": "Макс. историческая просадка", "value": f"{row['max_drawdown']:.2%}", "meaning": "Худшее падение от пика до минимума."},
            {"metric": f"VaR 5%, за {frequency}", "value": f"{row['historical_var_5']:.2%}", "meaning": "Только 5% исторических периодов были хуже."},
            {"metric": f"CVaR 5%, за {frequency}", "value": f"{row['historical_cvar_5']:.2%}", "meaning": "Средняя доходность в худших 5% периодов."},
        ]
    elif lang == "he":
        rows = [
            {"metric": "תנודתיות שנתית", "value": f"{row['annualized_volatility']:.2%}", "meaning": "שונות שנתית טיפוסית של תשואות."},
            {"metric": "ירידה היסטורית מקסימלית", "value": f"{row['max_drawdown']:.2%}", "meaning": "הירידה הגרועה ביותר משיא לשפל."},
            {"metric": f"VaR 5%, לתקופת {frequency}", "value": f"{row['historical_var_5']:.2%}", "meaning": "רק 5% מהתקופות ההיסטוריות היו גרועות יותר."},
            {"metric": f"CVaR 5%, לתקופת {frequency}", "value": f"{row['historical_cvar_5']:.2%}", "meaning": "התשואה הממוצעת ב-5% התקופות הגרועות."},
        ]
    else:
        rows = [
            {"metric": "Annual volatility", "value": f"{row['annualized_volatility']:.2%}", "meaning": "Typical yearly variability of returns."},
            {"metric": "Max historical drawdown", "value": f"{row['max_drawdown']:.2%}", "meaning": "Worst historical peak-to-trough decline."},
            {"metric": f"VaR 5%, per {frequency}", "value": f"{row['historical_var_5']:.2%}", "meaning": "Only 5% of historical periods were worse than this."},
            {"metric": f"CVaR 5%, per {frequency}", "value": f"{row['historical_cvar_5']:.2%}", "meaning": "Average return in the worst 5% historical periods."},
        ]
    return pd.DataFrame(rows)


def _scenario_insights(summary: dict[str, float], scenario: Scenario, lang: str) -> list[str]:
    money = lambda value: _format_money(value, scenario.currency)
    if lang == "ru":
        return [
            f"Медианная чистая прибыль при продаже: {money(summary.get('median_net_profit_if_sold', 0))}.",
            f"В неблагоприятном 5-м перцентиле итоговая стоимость около {money(summary['p5'])}.",
            f"Вероятность достичь цели: {_format_percent(summary['probability_reaching_target'])}.",
            f"Ожидаемые чистые дивиденды за горизонт: {money(summary.get('median_cumulative_net_dividends', 0))}.",
        ]
    if lang == "he":
        return [
            f"רווח נקי חציוני במכירה: {money(summary.get('median_net_profit_if_sold', 0))}.",
            f"בתרחיש אחוזון 5 השווי הסופי הוא בערך {money(summary['p5'])}.",
            f"ההסתברות להגיע ליעד: {_format_percent(summary['probability_reaching_target'])}.",
            f"דיבידנדים נטו מוערכים לאורך האופק: {money(summary.get('median_cumulative_net_dividends', 0))}.",
        ]
    return [
        f"Median net profit if sold is {money(summary.get('median_net_profit_if_sold', 0))}.",
        f"The 5th percentile downside final value is about {money(summary['p5'])}.",
        f"Probability of reaching the target is {_format_percent(summary['probability_reaching_target'])}.",
        f"Estimated cumulative net dividends over the horizon are {money(summary.get('median_cumulative_net_dividends', 0))}.",
    ]


def main() -> None:
    scenario, lang = _scenario_from_sidebar()
    currency_symbol = CURRENCY_SYMBOLS.get(scenario.currency, scenario.currency)
    if lang == "he":
        st.markdown("<style>.main, [data-testid='stSidebar'] { direction: rtl; }</style>", unsafe_allow_html=True)

    _render_global_banner()
    st.title(t(lang, "title"))
    st.markdown(
        f"""
        <div class="assumption-note">
        {t(lang, "subtitle")}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if scenario.simulations >= 50_000:
        st.warning("50,000+ simulations can be slow on some machines. Reduce the count if the interface feels sluggish.")

    if not scenario.tickers:
        st.error("Enter at least one ticker in the sidebar.")
        return

    try:
        weights = align_weights(scenario.tickers, scenario.weights)
    except ValueError as exc:
        st.error(str(exc))
        return

    try:
        with st.spinner("Downloading and validating historical data..."):
            prices = _load_prices(tuple(scenario.tickers), scenario.start_date, scenario.end_date, scenario.price_field)
            quality = quality_report_frame(validate_prices(prices))
            quality["ticker"] = quality["ticker"].map(_display_symbol)
            asset_returns = calculate_returns(prices, scenario.frequency)
            asset_returns = asset_returns.dropna(how="any")
            port_returns = portfolio_returns(asset_returns, list(weights.values))
            combined_returns = asset_returns.copy()
            combined_returns["Portfolio"] = port_returns
            stats = return_statistics(combined_returns, scenario.frequency)
            stats.index = [_display_symbol(str(index)) for index in stats.index]
    except Exception as exc:
        st.error(f"Could not prepare market data: {exc}")
        return

    tab_setup, tab_data, tab_sim, tab_compare, tab_risk, tab_export = st.tabs(
        [
            t(lang, "setup"),
            t(lang, "historical_data"),
            t(lang, "simulation"),
            t(lang, "comparison"),
            t(lang, "risk"),
            t(lang, "export"),
        ]
    )

    with tab_setup:
        left, right = st.columns([1, 2])
        with left:
            st.subheader(t(lang, "scenario"))
            display_weights = weights.copy()
            display_weights.index = [_display_symbol(str(index)) for index in display_weights.index]
            weight_display = display_weights.mul(100).round(2).rename({"en": "weight, %", "ru": "вес, %", "he": "משקל, %"}.get(lang, "weight, %")).to_frame()
            st.dataframe(weight_display, use_container_width=True)
            st.caption(t(lang, "weights_caption"))
            st.metric(
                t(lang, "initial_capital"),
                _format_money(scenario.initial_capital, scenario.currency),
                help="Starting portfolio value before the first simulated return period.",
            )
            st.metric(
                t(lang, "monthly_contribution"),
                _format_money(scenario.monthly_contribution, scenario.currency),
                help="Planned monthly savings amount before annual contribution increases.",
            )
            st.metric(
                t(lang, "target_value"),
                _format_money(scenario.target_value, scenario.currency),
                help="Target value used to calculate the probability of reaching the goal.",
            )
        with right:
            st.plotly_chart(historical_prices(_with_display_columns(prices), currency_symbol), use_container_width=True)

    with tab_data:
        st.subheader(t(lang, "data_quality"))
        st.dataframe(quality, use_container_width=True, hide_index=True)
        st.subheader(t(lang, "historical_statistics"))
        st.caption(t(lang, "hist_caption"))
        st.dataframe(_historical_stats_display(stats, scenario.frequency, lang), use_container_width=True)
        with st.expander(t(lang, "show_raw_stats"), expanded=False):
            st.dataframe(stats, use_container_width=True)

    with tab_sim:
        run_clicked = st.button(
            t(lang, "run_primary"),
            type="primary",
            help="Generate Monte Carlo paths for the selected primary model and current scenario settings.",
        )
        if run_clicked or "primary_result" not in st.session_state:
            with st.spinner(f"Running {opt(lang, scenario.model)} simulation..."):
                st.session_state.primary_result = run_simulation(asset_returns, scenario)
                st.session_state.primary_scenario = scenario.to_dict()

        result = st.session_state.primary_result
        summary = result["summary"]
        x_values, x_axis_title = _simulation_x_axis(scenario, result["paths"].shape[1])
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric(
            t(lang, "median_final_value"),
            _format_money(summary["median_final_value"], scenario.currency),
            help="The middle final outcome: half of simulations ended above this value and half below.",
        )
        m2.metric(
            t(lang, "p5"),
            _format_money(summary["p5"], scenario.currency),
            help="A downside threshold: 5% of simulations ended below this value.",
        )
        m3.metric(
            t(lang, "target_probability"),
            _format_percent(summary["probability_reaching_target"]),
            help="Share of simulations that ended at or above the target value.",
        )
        m4.metric(
            t(lang, "median_drawdown"),
            _format_percent(summary["median_max_drawdown"]),
            help="Median worst peak-to-trough decline across simulated paths.",
        )
        m5.metric(
            t(lang, "net_profit_if_sold"),
            _format_money(summary.get("median_net_profit_if_sold", summary["median_gain_over_contributions"]), scenario.currency),
            help="Median net profit after sale: after-tax liquidation value plus applicable dividend cash income minus total contributions.",
        )
        m6.metric(
            t(lang, "net_dividends"),
            _format_money(summary.get("median_cumulative_net_dividends", 0), scenario.currency),
            help="Median cumulative dividends after dividend tax or withholding. Depends on the estimated dividend yield setting.",
        )
        st.subheader(t(lang, "insights"))
        for insight in _scenario_insights(summary, scenario, lang):
            st.markdown(f"- {insight}")
        info_left, info_right = st.columns([1.2, 1])
        with info_left:
            st.plotly_chart(income_waterfall(summary, currency_symbol, t(lang, "income_breakdown")), use_container_width=True)
        with info_right:
            st.plotly_chart(target_probability_gauge(summary["probability_reaching_target"], t(lang, "target_probability")), use_container_width=True)

        st.plotly_chart(
            fan_chart(
                result["paths"],
                currency_symbol=currency_symbol,
                x_values=x_values,
                x_axis_title=x_axis_title,
                hover_metrics=_median_hover_metrics(result),
            ),
            use_container_width=True,
        )
        st.plotly_chart(
            fan_chart(
                result["cashflows"]["net_profit_if_sold"],
                title="Net Profit If Sold Over Time",
                currency_symbol=currency_symbol,
                x_values=x_values,
                x_axis_title=x_axis_title,
                value_label="Net profit if sold",
            ),
            use_container_width=True,
        )
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(final_value_histogram(result["paths"], currency_symbol), use_container_width=True)
        with col_b:
            st.plotly_chart(
                sample_trajectories(
                    result["paths"],
                    currency_symbol=currency_symbol,
                    x_values=x_values,
                    x_axis_title=x_axis_title,
                ),
                use_container_width=True,
            )

        st.subheader(t(lang, "summary_metrics"))
        if lang == "ru":
            income_rows = [
                ("Чистая прибыль при продаже", summary.get("median_net_profit_if_sold", summary["median_gain_over_contributions"])),
                ("Стоимость после налогов при продаже", summary["median_final_value"]),
                ("Всего взносов", summary["total_contributions"]),
                ("Чистые дивиденды", summary.get("median_cumulative_net_dividends", 0)),
                ("Налог на дивиденды", summary.get("median_cumulative_dividend_taxes", 0)),
                ("Налог на прирост при продаже", summary.get("median_liquidation_tax", 0)),
            ]
            item_col, median_col = "показатель", "медианное значение"
        elif lang == "he":
            income_rows = [
                ("רווח נקי במכירה", summary.get("median_net_profit_if_sold", summary["median_gain_over_contributions"])),
                ("שווי לאחר מס במכירה", summary["median_final_value"]),
                ("סך ההפקדות", summary["total_contributions"]),
                ("דיבידנדים נטו", summary.get("median_cumulative_net_dividends", 0)),
                ("מס דיבידנדים", summary.get("median_cumulative_dividend_taxes", 0)),
                ("מס רווח הון במכירה", summary.get("median_liquidation_tax", 0)),
            ]
            item_col, median_col = "פריט", "ערך חציוני"
        else:
            income_rows = [
                ("Net profit if sold", summary.get("median_net_profit_if_sold", summary["median_gain_over_contributions"])),
                ("After-tax liquidation value", summary["median_final_value"]),
                ("Total contributions", summary["total_contributions"]),
                ("Net dividends", summary.get("median_cumulative_net_dividends", 0)),
                ("Dividend tax", summary.get("median_cumulative_dividend_taxes", 0)),
                ("Capital gains tax if sold", summary.get("median_liquidation_tax", 0)),
            ]
            item_col, median_col = "item", "median value"
        st.subheader(t(lang, "income_breakdown"))
        st.dataframe(
            pd.DataFrame(
                [
                    {item_col: label, median_col: _format_money(value, scenario.currency), "raw_value": value}
                    for label, value in income_rows
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.dataframe(_summary_frame(summary, scenario.currency, lang), use_container_width=True, hide_index=True)

    with tab_compare:
        st.markdown(
            """
            <div class="assumption-note">
            Different simulation models can produce materially different outcomes. This does not mean one model is objectively correct.
            It shows that long-term projections are highly assumption-dependent.
            </div>
            """,
            unsafe_allow_html=True,
        )
        model_options = st.multiselect(
            t(lang, "models_to_compare"),
            list(MODEL_LABELS),
            default=["historical_bootstrap", "block_bootstrap", "normal", "fat_tail"],
            format_func=lambda value: opt(lang, value),
            help="Select the models to run on the same portfolio and accumulation assumptions.",
        )
        if st.button(
            t(lang, "run_comparison"),
            help="Run each selected model using the same scenario so assumption sensitivity is visible.",
        ) and model_options:
            with st.spinner("Running model comparison..."):
                comparison = compare_models(asset_returns, scenario, model_options)
                comparison_results = {model: run_simulation(asset_returns, scenario, model) for model in model_options}
                st.session_state.comparison = comparison
                st.session_state.comparison_results = comparison_results
        if "comparison" in st.session_state:
            comparison = st.session_state.comparison
            st.dataframe(comparison, use_container_width=True)
            st.plotly_chart(model_comparison_bars(comparison, currency_symbol), use_container_width=True)
            st.plotly_chart(distribution_overlay(st.session_state.comparison_results, currency_symbol), use_container_width=True)

    with tab_risk:
        st.subheader(t(lang, "risk_snapshot"))
        st.caption(t(lang, "risk_caption"))
        st.dataframe(_risk_snapshot(stats, scenario.frequency, lang), use_container_width=True, hide_index=True)
        if "primary_result" in st.session_state:
            risk_summary = st.session_state.primary_result["summary"]
            r1, r2, r3, r4 = st.columns(4)
            r1.metric(
                "Probability below contributions",
                _format_percent(risk_summary["probability_below_contributions"]),
                help="Share of simulations where final value ended below total contributions.",
            )
            r2.metric(
                "Negative real return probability",
                _format_percent(risk_summary["probability_negative_real_return"]),
                help="Share of simulations where inflation-adjusted final value was below total contributions.",
            )
            r3.metric(
                "Worst 5% average outcome",
                _format_money(risk_summary["worst_5pct_average_outcome"], scenario.currency),
                help="Average final value among the worst 5% of simulated outcomes.",
            )
            r4.metric(
                "Expected max drawdown",
                _format_percent(risk_summary["expected_max_drawdown"]),
                help="Average worst peak-to-trough decline across simulation paths.",
            )

        st.subheader(t(lang, "rolling_risk_charts"))
        window = {"daily": 252, "weekly": 52, "monthly": 12}[scenario.frequency]
        st.caption(f"Rolling window uses one year of selected-frequency data: {window} {scenario.frequency} observations.")
        display_returns = _with_display_columns(combined_returns)
        st.plotly_chart(rolling_chart(rolling_returns(display_returns, window), "Rolling Returns", "Return over rolling window"), use_container_width=True)
        st.plotly_chart(rolling_chart(rolling_volatility(display_returns, window, scenario.frequency), "Rolling Volatility", "Annualized volatility"), use_container_width=True)
        if "primary_result" in st.session_state:
            st.plotly_chart(drawdown_histogram(st.session_state.primary_result["paths"]), use_container_width=True)

    with tab_export:
        st.subheader(t(lang, "assumptions"))
        st.info(
            t(lang, "disclaimer")
        )
        st.caption(
            "Israeli tax mode is a simplified approximation: it applies the selected capital gains tax rate "
            "to positive real gains at final liquidation, using inflation-indexed contributions as the cost basis. "
            "It does not model every Israeli tax account type, offset, exemption, dividend treatment, or reporting rule."
        )
        scenario_json = json.dumps(scenario.to_dict(), indent=2)
        st.download_button(
            t(lang, "download_scenario"),
            scenario_json,
            "scenario.json",
            "application/json",
            help="Download the complete scenario settings so they can be loaded again later.",
        )
        st.download_button(
            t(lang, "download_stats"),
            stats.to_csv().encode("utf-8"),
            "historical_statistics.csv",
            "text/csv",
            help="Download calculated historical return and risk statistics.",
        )
        if "primary_result" in st.session_state:
            summary_csv = _summary_frame(st.session_state.primary_result["summary"], scenario.currency, lang).to_csv(index=False).encode("utf-8")
            st.download_button(
                t(lang, "download_summary"),
                summary_csv,
                "simulation_summary.csv",
                "text/csv",
                help="Download the calculated summary metrics for the primary simulation.",
            )
            paths_df = pd.DataFrame(st.session_state.primary_result["paths"])
            st.download_button(
                t(lang, "download_paths"),
                paths_df.to_csv(index=False).encode("utf-8"),
                "simulation_paths.csv",
                "text/csv",
                help="Download every simulated portfolio path. This can be a large file for many simulations.",
            )
            cashflows = st.session_state.primary_result["cashflows"]
            final_cashflows = pd.DataFrame(
                {
                    "final_after_tax_liquidation_value": st.session_state.primary_result["paths"][:, -1],
                    "total_contributions": st.session_state.primary_result["contributions"][:, -1],
                    "net_profit_if_sold": cashflows["net_profit_if_sold"][:, -1],
                    "cumulative_gross_dividends": cashflows["cumulative_gross_dividends"][:, -1],
                    "cumulative_net_dividends": cashflows["cumulative_net_dividends"][:, -1],
                    "cumulative_dividend_taxes": cashflows["cumulative_dividend_taxes"][:, -1],
                    "capital_gains_tax_if_sold": cashflows["liquidation_taxes"][:, -1],
                }
            )
            st.download_button(
                t(lang, "download_cashflows"),
                final_cashflows.to_csv(index=False).encode("utf-8"),
                "simulation_final_cashflows.csv",
                "text/csv",
                help="Download per-simulation final sale profit, dividend totals, and modeled tax amounts.",
            )
            export_x_values, export_x_axis_title = _simulation_x_axis(
                scenario,
                st.session_state.primary_result["paths"].shape[1],
            )
            chart_html = fan_chart(
                st.session_state.primary_result["paths"],
                currency_symbol=currency_symbol,
                x_values=export_x_values,
                x_axis_title=export_x_axis_title,
                hover_metrics=_median_hover_metrics(st.session_state.primary_result),
            ).to_html(include_plotlyjs="cdn")
            st.download_button(
                t(lang, "download_chart"),
                chart_html,
                "fan_chart.html",
                "text/html",
                help="Download the interactive fan chart as a standalone HTML file.",
            )


if __name__ == "__main__":
    main()
