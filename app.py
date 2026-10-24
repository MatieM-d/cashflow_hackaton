"""
ФинПоток — профессиональный сервис прогнозирования денежных потоков
для малого бизнеса.

Улучшенная версия: тёмная «финансовая» тема, несколько моделей прогноза,
бэктест и метрики качества, доверительные интервалы, ручные события
(договоры, счета, налоговый календарь), стресс-сценарии, категоризация.
"""
from __future__ import annotations

import io
import os
import zipfile
from datetime import date, timedelta
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import Holt, ExponentialSmoothing


# ────────────────────────────────────────────────────────────────────────────
#  КОНФИГУРАЦИЯ СТРАНИЦЫ И ТЕМА
# ────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ФинПоток — Cash-Flow Forecaster",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_BG = "#0b1220"
PANEL_BG = "#111a2e"
PANEL_BORDER = "#1f2a44"
ACCENT_BLUE = "#3b82f6"
ACCENT_GREEN = "#22c55e"
ACCENT_AMBER = "#f59e0b"
ACCENT_RED = "#ef4444"
ACCENT_CYAN = "#06b6d4"
ACCENT_VIOLET = "#8b5cf6"
TEXT_PRIMARY = "#e8ecf3"
TEXT_SECONDARY = "#94a3b8"

st.markdown(
    """
    <style>
    /* Глобальный фон */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: """ + DARK_BG + """ !important;
        color: """ + TEXT_PRIMARY + """ !important;
    }
    .stApp::before {
        content: "";
        position: fixed; inset: 0; z-index: -1;
        background:
            radial-gradient(900px 600px at 10% -10%, rgba(59,130,246,0.12), transparent 60%),
            radial-gradient(700px 500px at 100% 0%, rgba(139,92,246,0.10), transparent 55%),
            """ + DARK_BG + """;
    }
    /* Контейнер */
    .block-container { padding-top: 1.6rem !important; max-width: 1480px; }
    /* Заголовки */
    h1, h2, h3, h4 { letter-spacing: -0.02em; color: """ + TEXT_PRIMARY + """ !important; }
    h1 { font-weight: 800; }
    /* Metric-карточки */
    [data-testid="stMetric"] {
        background: linear-gradient(180deg, """ + PANEL_BG + """, #0e1626);
        padding: 18px 18px 14px !important;
        border-radius: 14px;
        border: 1px solid """ + PANEL_BORDER + """;
        box-shadow: 0 6px 22px rgba(0,0,0,0.35);
    }
    [data-testid="stMetric"] label, [data-testid="stMetricLabel"] {
        color: """ + TEXT_SECONDARY + """ !important;
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
    }
    [data-testid="stMetricValue"] {
        color: """ + TEXT_PRIMARY + """ !important;
        font-weight: 700 !important;
        font-size: 1.55rem !important;
    }
    /* Панель вкладок */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        padding: 9px 16px;
        border-radius: 8px 8px 0 0;
        background: transparent;
        color: """ + TEXT_SECONDARY + """ !important;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: """ + PANEL_BG + """ !important;
        color: """ + TEXT_PRIMARY + """ !important;
        border-bottom: 2px solid """ + ACCENT_BLUE + """ !important;
    }
    /* Сайдбар */
    [data-testid="stSidebar"], [data-testid="stSidebarViewContainer"] {
        background: #0a1120 !important;
        border-right: 1px solid """ + PANEL_BORDER + """;
    }
    /* Кастомные баннеры */
    .fp-banner {
        border-radius: 12px; padding: 14px 18px; margin: 6px 0;
        border: 1px solid; font-weight: 500;
        display: flex; align-items: center; gap: 10px;
    }
    .fp-banner-info    { background: rgba(59,130,246,0.10); border-color: rgba(59,130,246,0.45); color: #cfe1ff; }
    .fp-banner-success { background: rgba(34,197,94,0.10); border-color: rgba(34,197,94,0.45); color: #c7ffd9; }
    .fp-banner-warn    { background: rgba(245,158,11,0.10); border-color: rgba(245,158,11,0.45); color: #ffe2a8; }
    .fp-banner-danger  { background: rgba(239,68,68,0.10); border-color: rgba(239,68,68,0.50); color: #ffc9c9; }
    .fp-pill {
        display: inline-block; padding: 2px 9px; border-radius: 999px;
        font-size: 0.72rem; font-weight: 700; letter-spacing: 0.04em;
        text-transform: uppercase; margin-right: 6px;
    }
    .fp-pill-low  { background: rgba(34,197,94,0.15); color: #6ee7b7; border: 1px solid rgba(34,197,94,0.35); }
    .fp-pill-med  { background: rgba(245,158,11,0.15); color: #fcd34d; border: 1px solid rgba(245,158,11,0.35); }
    .fp-pill-high { background: rgba(239,68,68,0.15); color: #fca5a5; border: 1px solid rgba(239,68,68,0.40); }
    .fp-section-title {
        font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.10em;
        color: """ + TEXT_SECONDARY + """; font-weight: 700;
        margin: 18px 0 8px;
    }
    .fp-divider {
        height: 1px; background: linear-gradient(90deg, transparent, """ + PANEL_BORDER + """, transparent);
        margin: 14px 0;
    }
    /* Plotly modebar */
    .modebar { background: transparent !important; }
    /* Tables */
    .stDataFrame, .stDataFrame table { color: """ + TEXT_PRIMARY + """ !important; }
    /* Hide default headers with extra padding */
    [data-testid="stMetricContainer"] { gap: 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ────────────────────────────────────────────────────────────────────────────
#  КАТЕГОРИИ (векторизованная + исправленные приоритеты)
# ────────────────────────────────────────────────────────────────────────────
# Порядок важен: доходы (Продажи) идут раньше комиссий банка, чтобы
# эквайринг-поступления не классифицировались как «Комиссии и банк».
CATEGORY_RULES: list[tuple[str, list[str], str]] = [
    # (Категория, Ключевые слова, Тип по умолчанию: in/out/auto)
    ("Продажи",               ["оплата от покуп", "оплата по счет", "оплата по договор",
                              "выручка", "эквайринг", "приход от клиент", "розничн"], "auto"),
    ("Налоги",                ["налог", "енп", "ндс", "усн", "взнос", "фнс", "страхован"], "out"),
    ("Зарплата",              ["зарплат", "заработ", "аванс сотруд", "оплата труда", "отпускн"], "out"),
    ("Аренда",                ["аренд", "помещени"], "out"),
    ("Кредиты и лизинг",      ["кредит", "лизинг", "процент по договору займа", "тело кредита"], "out"),
    ("Логистика и транспорт", ["транспорт", "доставк", "топлив", "перевоз", "автомобил", "дорог", "гсм"], "out"),
    ("Связь и IT",            ["интернет", "связи", "телеком", "программ", "хостинг", "подписк", "сзв"], "out"),
    ("Комиссии и банк",       ["комисси", "ведение счета", "обслуживание счета", "пко", "рко"], "out"),
    ("Закупки и поставщики",  ["поставк", "закуп", "сырь", "товар", "упд", "счет №", "счёт №",
                              "материал", "инвентар"], "out"),
]


# Категории, которые по экономическому смыслу НЕ МОГУТ быть поступлениями.
# Если операция с такой категорией помечена как «Поступление» в выписке —
# это либо возврат/корректировка (например, возврат переплаты по налогу),
# либо ошибка парсинга направления в CSV. Такие операции переотнесём в
# отдельную категорию «Возврат / корректировка», чтобы не искажать сводку.
EXPENSE_ONLY_CATEGORIES = {
    "Налоги", "Зарплата", "Аренда", "Кредиты и лизинг",
    "Логистика и транспорт", "Связь и IT", "Комиссии и банк",
    "Закупки и поставщики",
}


def categorize_vectorized(df: pd.DataFrame) -> pd.Series:
    """Векторизованная категоризация — в ~50–100× быстрее, чем apply(axis=1).
    Переотносит подозрительные «поступления» в расходных категориях в
    отдельную категорию «Возврат / корректировка»."""
    purpose = df["payment_purpose"].fillna("").str.lower()
    counter = df["counterparty_name"].fillna("").str.lower()
    text = purpose + " " + counter
    direction = df["direction"].fillna("").astype(str).str.lower()

    is_inflow = _detect_inflow(direction, df)

    categories = pd.Series("Прочие", index=df.index, dtype="object")
    for name, words, default_type in CATEGORY_RULES:
        mask = text.str.contains("|".join(words), regex=True, na=False)
        # Применяем только там, где категория ещё не назначена (приоритет по порядку).
        unassigned = categories == "Прочие"
        apply_mask = mask & unassigned
        categories = categories.where(~apply_mask, name)

    # Прочие поступления / расходы
    unassigned = categories == "Прочие"
    categories = categories.where(~(unassigned & is_inflow), "Прочие поступления")
    categories = categories.where(~(unassigned & ~is_inflow), "Прочие расходы")

    # Safety-чек: поступление в расходной категории → «Возврат / корректировка».
    suspicious = categories.isin(EXPENSE_ONLY_CATEGORIES) & is_inflow
    categories = categories.where(~suspicious, "Возврат / корректировка")
    return categories


def _detect_inflow(direction_lower: pd.Series, df: pd.DataFrame) -> pd.Series:
    """Распознаёт поступление по столбцу direction. Поддерживает:
    русские/английские ключевые слова, аббревиатуры Кт/Дт, C/D, CR/DR,
    Credit/Debit, Кредит/Дебет (в банковском смысле), а также fallback
    по знаку суммы, если direction не распознан."""
    contains_inflow = direction_lower.str.contains(
        "поступ|приход|income|inflow|зачисл|прих|кредит|возврат|возмещ",
        regex=True, na=False,
    )
    contains_outflow = direction_lower.str.contains(
        "спис|расход|outflow|expense|платёж|платеж|оплата со счет|дебет|снят",
        regex=True, na=False,
    )
    # Точные короткие коды (категорически игнорируем неоднозначные)
    exact_inflow = direction_lower.isin([
        "к", "кт", "c", "cr", "credit", "кр", "in", "+",
        "приход", "поступление", "поступило", "зачислено",
    ])
    exact_outflow = direction_lower.isin([
        "д", "дт", "d", "dr", "debit", "out", "-",
        "расход", "списание", "списано", "списан",
    ])
    inflow = contains_inflow | exact_inflow
    outflow = contains_outflow | exact_outflow
    # Fallback по знаку суммы (если direction не распознан).
    amount = df["amount"]
    fallback_pos = (amount > 0) & ~inflow & ~outflow
    fallback_neg = (amount < 0) & ~inflow & ~outflow
    inflow = inflow | fallback_pos
    return inflow


def _detect_outflow(direction_lower: pd.Series, df: pd.DataFrame) -> pd.Series:
    """Симметричный помощник для нормализации."""
    contains_inflow = direction_lower.str.contains(
        "поступ|приход|income|inflow|зачисл|прих|кредит|возврат|возмещ",
        regex=True, na=False,
    )
    contains_outflow = direction_lower.str.contains(
        "спис|расход|outflow|expense|платёж|платеж|оплата со счет|дебет|снят",
        regex=True, na=False,
    )
    exact_inflow = direction_lower.isin([
        "к", "кт", "c", "cr", "credit", "кр", "in", "+",
    ])
    exact_outflow = direction_lower.isin([
        "д", "дт", "d", "dr", "debit", "out", "-",
    ])
    outflow = contains_outflow | exact_outflow
    amount = df["amount"]
    fallback_neg = (amount < 0) & ~contains_inflow & ~contains_outflow & ~exact_inflow
    outflow = outflow | fallback_neg
    return outflow


# ────────────────────────────────────────────────────────────────────────────
#  ЧТЕНИЕ CSV (с поддержкой cp1251 — частая кодировка российских банков)
# ────────────────────────────────────────────────────────────────────────────
def read_csv_safely(file) -> pd.DataFrame:
    if hasattr(file, "getvalue"):
        raw = file.getvalue()
    elif isinstance(file, (bytes, bytearray)):
        raw = bytes(file)
    else:
        raw = Path(file).read_bytes()

    encodings = ["utf-8-sig", "utf-8", "cp1251", "windows-1251", "latin-1"]
    seps = [";", ",", "\t", "|"]
    last_err = None
    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(io.BytesIO(raw), sep=sep, encoding=enc)
                if len(df.columns) > 2:
                    return df
            except Exception as e:
                last_err = e
    raise ValueError(
        f"Не удалось прочитать CSV. Проверьте разделитель и кодировку. Последняя ошибка: {last_err}"
    )


# ────────────────────────────────────────────────────────────────────────────
#  НОРМАЛИЗАЦИЯ
# ────────────────────────────────────────────────────────────────────────────
COLUMN_ALIASES = {
    "operation_date":    ["operation_date", "date", "дата", "дата операции", "дата платежа"],
    "amount":            ["amount", "сумма", "сумма операции", "сумма платежа"],
    "direction":         ["direction", "направление", "тип операции", "вид операции"],
    "balance_eod":       ["balance_eod", "balance", "остаток", "остаток на счёте",
                          "остаток на счете", "остаток на конец дня"],
    "payment_purpose":   ["payment_purpose", "назначение платежа", "назначение"],
    "counterparty_name": ["counterparty_name", "контрагент", "наименование контрагента"],
    "counterparty_inn":  ["counterparty_inn", "inn", "инн"],
    "company_id":        ["company_id", "company", "компания", "id компании"],
}


def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Регистронезависимый матчинг заголовков
    lower = {str(c).strip().lower(): c for c in df.columns}
    rename = {}
    for target, options in COLUMN_ALIASES.items():
        for opt in options:
            if opt in lower:
                rename[lower[opt]] = target
                break
    df = df.rename(columns=rename)

    required = ["operation_date", "amount", "direction"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Не найдены обязательные поля: " + ", ".join(missing) +
            ". Нужны дата операции, сумма и направление."
        )

    for col in ["payment_purpose", "counterparty_name", "counterparty_inn", "company_id"]:
        if col not in df.columns:
            df[col] = ""

    # Дата: пробуем dayfirst True (РФ), затем auto.
    df["operation_date"] = pd.to_datetime(df["operation_date"], errors="coerce", dayfirst=True)
    if df["operation_date"].isna().all():
        df["operation_date"] = pd.to_datetime(df["operation_date"], errors="coerce")

    # Сумма: чистим пробелы и запятые, поддерживаем отрицательные (если направление кодировано знаком).
    df["amount"] = pd.to_numeric(
        df["amount"].astype(str).str.replace(" ", "", regex=False)
                    .str.replace("\u00a0", "", regex=False)
                    .str.replace(",", ".", regex=False),
        errors="coerce",
    )

    df["direction"] = df["direction"].astype(str).str.strip().str.lower()
    direction_text = df["direction"]

    inflow = _detect_inflow(direction_text, df)
    outflow = _detect_outflow(direction_text, df)

    # Если ничего не распознано — пробуем чистый знак суммы как последний шанс.
    unclassified = ~inflow & ~outflow
    if unclassified.any() and (inflow.sum() + outflow.sum()) == 0:
        # Полный провал: ни одно направление не распознано. Используем знак суммы.
        inflow = (df["amount"] > 0) | inflow
        outflow = (df["amount"] < 0) | outflow

    df["amount"] = df["amount"].abs()
    df["signed_amount"] = np.select([inflow, outflow], [df["amount"], -df["amount"]], default=np.nan)

    if df["signed_amount"].notna().sum() == 0:
        raise ValueError(
            "Не распознаны направления операций. Используйте «Поступление» и «Списание», "
            "коды Кт/Дт, C/D, Credit/Debit, знак +/- суммы, или числовое направление."
        )

    df = df.dropna(subset=["operation_date", "amount", "signed_amount"])
    df = df[df["amount"] >= 0].copy()
    if df.empty:
        raise ValueError("После очистки не осталось корректных операций.")

    # Если есть остаток на счёте — приведём к числу.
    if "balance_eod" in df.columns:
        df["balance_eod"] = pd.to_numeric(
            df["balance_eod"].astype(str).str.replace(" ", "", regex=False)
                            .str.replace("\u00a0", "", regex=False)
                            .str.replace(",", ".", regex=False),
            errors="coerce",
        )

    df["category"] = categorize_vectorized(df)
    df["day_of_week"] = df["operation_date"].dt.dayofweek
    df["month"] = df["operation_date"].dt.to_period("M").astype(str)
    return df.sort_values("operation_date").reset_index(drop=True)


# ────────────────────────────────────────────────────────────────────────────
#  СИНТЕТИЧЕСКИЕ ДЕМО-ДАННЫЕ (если настоящий архив недоступен)
# ────────────────────────────────────────────────────────────────────────────
def synthetic_company(company_id: str = "C001", days: int = 540) -> pd.DataFrame:
    """Генерирует реалистичную банковскую выписку малого бизнеса:
    будние дни B2B-оплаты, ежедневный эквайринг, зарплата 10/25, аренда 5-го,
    ENP 28-го, квартальный НДС, периодические закупки."""
    rng = np.random.default_rng(abs(hash(company_id)) % (2**32))
    start = pd.Timestamp.today().normalize() - pd.Timedelta(days=days)
    dates = pd.date_range(start, periods=days, freq="D")

    rows: list[dict] = []
    balance = rng.uniform(180_000, 480_000)
    base_daily_sales = rng.uniform(35_000, 90_000)
    trend = rng.uniform(0.0005, 0.0015)  # лёгкий рост во времени

    for i, d in enumerate(dates):
        weekday = d.dayofweek
        # Сезонность по дням недели
        wd_factor = [0.85, 0.95, 1.10, 1.15, 1.05, 0.75, 0.55][weekday]
        seasonal = 1 + 0.18 * np.sin(2 * np.pi * i / 365)
        acq = max(0, base_daily_sales * wd_factor * seasonal * (1 + trend * i)
                  * rng.normal(1.0, 0.18))
        if acq > 0:
            balance += acq
            rows.append(dict(
                operation_date=d, amount=round(acq, 2), direction="Поступление",
                balance_eod=round(balance, 2),
                payment_purpose="Оплата от покупателя по терминалу, эквайринг",
                counterparty_name="Банк-эквайер",
                counterparty_inn="0000000000", company_id=company_id,
            ))

        # B2B-оплаты от клиентов (вт-чт)
        if weekday in (1, 2, 3) and rng.random() < 0.55:
            invoice = round(rng.uniform(120_000, 480_000), 2)
            balance += invoice
            rows.append(dict(
                operation_date=d, amount=invoice, direction="Поступление",
                balance_eod=round(balance, 2),
                payment_purpose="Оплата по счёту №" + str(rng.integers(1000, 9999)),
                counterparty_name="ООО «Клиент " + str(rng.integers(1, 30)) + "»",
                counterparty_inn="7700000000", company_id=company_id,
            ))

        # Аренда 5-го
        if d.day == 5:
            rent = round(rng.uniform(85_000, 145_000), 2)
            balance -= rent
            rows.append(dict(
                operation_date=d, amount=rent, direction="Списание",
                balance_eod=round(balance, 2),
                payment_purpose="Аренда помещения за месяц по договору",
                counterparty_name="ИП Арендодатель",
                counterparty_inn="7701111111", company_id=company_id,
            ))

        # Зарплата 10-го и 25-го
        if d.day in (10, 25):
            salary = round(rng.uniform(180_000, 320_000), 2)
            balance -= salary
            rows.append(dict(
                operation_date=d, amount=salary, direction="Списание",
                balance_eod=round(balance, 2),
                payment_purpose="Выплата заработной платы сотрудникам, аванс",
                counterparty_name="Сотрудники (зарплатный проект)",
                counterparty_inn="", company_id=company_id,
            ))

        # ENP 28-го
        if d.day == 28:
            enp = round(rng.uniform(95_000, 240_000), 2)
            balance -= enp
            rows.append(dict(
                operation_date=d, amount=enp, direction="Списание",
                balance_eod=round(balance, 2),
                payment_purpose="Единый налоговый платёж (ЕНП), ФНС",
                counterparty_name="ФНС России",
                counterparty_inn="7700000000", company_id=company_id,
            ))

        # НДС квартально (1-й месяц квартала = доплата)
        if d.day == 28 and d.month in (1, 4, 7, 10):
            nds = round(rng.uniform(40_000, 90_000), 2)
            balance -= nds
            rows.append(dict(
                operation_date=d, amount=nds, direction="Списание",
                balance_eod=round(balance, 2),
                payment_purpose="Доплата НДС за прошлый квартал, ЕНП",
                counterparty_name="ФНС России",
                counterparty_inn="7700000000", company_id=company_id,
            ))

        # Закупки и поставщики (случайно в будни)
        if weekday in range(5) and rng.random() < 0.35:
            supply = round(rng.uniform(40_000, 220_000), 2)
            balance -= supply
            rows.append(dict(
                operation_date=d, amount=supply, direction="Списание",
                balance_eod=round(balance, 2),
                payment_purpose="Оплата по УПД за поставку товара",
                counterparty_name="ООО «Поставщик " + str(rng.integers(1, 20)) + "»",
                counterparty_inn="7702222222", company_id=company_id,
            ))

        # Логистика (мелкие)
        if weekday in range(5) and rng.random() < 0.25:
            tr = round(rng.uniform(8_000, 28_000), 2)
            balance -= tr
            rows.append(dict(
                operation_date=d, amount=tr, direction="Списание",
                balance_eod=round(balance, 2),
                payment_purpose="Транспортные расходы, доставка товара",
                counterparty_name="ООО «Логист-Транс»",
                counterparty_inn="7703333333", company_id=company_id,
            ))

        # Комиссии банка
        if d.day == 1:
            comm = round(rng.uniform(2_500, 7_500), 2)
            balance -= comm
            rows.append(dict(
                operation_date=d, amount=comm, direction="Списание",
                balance_eod=round(balance, 2),
                payment_purpose="Комиссия за ведение счёта, РКО",
                counterparty_name="Банк",
                counterparty_inn="7704444444", company_id=company_id,
            ))

    df = pd.DataFrame(rows)
    return normalize_data(df)


# ────────────────────────────────────────────────────────────────────────────
#  МОДЕЛИ ПРОГНОЗА И БЭКТЕСТ
# ────────────────────────────────────────────────────────────────────────────
def _weekday_avg_netflow(df: pd.DataFrame, lookback_days: int = 84) -> pd.Series:
    """Средний чистый поток по дням недели за последние lookback_days дней."""
    end = df["operation_date"].max()
    cutoff = end - pd.Timedelta(days=lookback_days)
    recent = df[df["operation_date"] >= cutoff].copy()
    if len(recent) == 0:
        recent = df.copy()
    recent["dow"] = recent["operation_date"].dt.dayofweek
    weekday_net = recent.groupby("dow")["signed_amount"].mean()
    # Clip outliers: 5/95 percentile
    if len(recent) > 10:
        lo, hi = recent["signed_amount"].quantile([0.05, 0.95])
        weekday_net = weekday_net.clip(lower=lo, upper=hi)
    overall = float(recent["signed_amount"].mean())
    return weekday_net, overall


def forecast_weekday_mean(df: pd.DataFrame, horizon: int, current_balance: float) -> pd.DataFrame:
    weekday_net, overall = _weekday_avg_netflow(df)
    last_date = df["operation_date"].max().normalize()
    future = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
    flows = [float(weekday_net.get(d.dayofweek, overall)) for d in future]
    balance = current_balance
    cum = []
    for f in flows:
        balance += f
        cum.append(balance)
    return pd.DataFrame({"date": future, "flow": flows, "balance": cum, "model": "Weekday Mean"})


def forecast_holt(df: pd.DataFrame, horizon: int, current_balance: float) -> pd.DataFrame:
    """Holt's linear trend на ежедневном чистом потоке (с заполнением пропусков)."""
    daily_net = (
        df.groupby(df["operation_date"].dt.normalize())["signed_amount"].sum()
          .asfreq("D", fill_value=0.0)
    )
    if len(daily_net) < 14:
        # Слишком мало данных — откатываемся к weekday mean
        return forecast_weekday_mean(df, horizon, current_balance).assign(model="Holt (fallback)")
    try:
        model = Holt(daily_net, initialization_method="estimated", damped_trend=True)
        fit = model.fit(optimized=True)
        f = fit.forecast(horizon)
        flows = f.clip(lower=-daily_net.abs().quantile(0.95)).values
    except Exception:
        return forecast_weekday_mean(df, horizon, current_balance).assign(model="Holt (fallback)")
    balance = current_balance
    cum = []
    for fl in flows:
        balance += float(fl)
        cum.append(balance)
    return pd.DataFrame({"date": pd.date_range(daily_net.index[-1] + pd.Timedelta(days=1),
                                              periods=horizon, freq="D"),
                         "flow": flows, "balance": cum, "model": "Holt Linear"})


def forecast_linear_trend_weekday(df: pd.DataFrame, horizon: int,
                                  current_balance: float) -> pd.DataFrame:
    """Линейный тренд чистого потока по дням + поправка на день недели."""
    daily_net = (
        df.groupby(df["operation_date"].dt.normalize())["signed_amount"].sum()
          .asfreq("D", fill_value=0.0)
    )
    if len(daily_net) < 21:
        return forecast_weekday_mean(df, horizon, current_balance).assign(model="Linear (fallback)")
    # Скользящее среднее для сглаживания
    smoothed = daily_net.rolling(7, min_periods=1, center=True).mean()
    X = np.arange(len(smoothed)).reshape(-1, 1)
    try:
        reg = LinearRegression().fit(X, smoothed.values)
        trend_slope = float(reg.coef_[0])
        trend_intercept = float(reg.intercept_)
    except Exception:
        return forecast_weekday_mean(df, horizon, current_balance).assign(model="Linear (fallback)")
    weekday_net, overall = _weekday_avg_netflow(df)
    last_date = daily_net.index[-1]
    future = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
    flows = []
    for i, d in enumerate(future, start=len(daily_net)):
        wd = float(weekday_net.get(d.dayofweek, overall))
        trend = trend_intercept + trend_slope * i
        # 70% веса — weekday pattern, 30% — долгий тренд
        flow = 0.7 * wd + 0.3 * trend
        flows.append(float(np.clip(flow, -np.inf, np.inf)))
    balance = current_balance
    cum = []
    for fl in flows:
        balance += fl
        cum.append(balance)
    return pd.DataFrame({"date": future, "flow": flows, "balance": cum,
                         "model": "Linear + Weekday"})


def backtest_model(df: pd.DataFrame, horizon: int, model_fn) -> dict:
    """Hold-out бэктест: обучаемся на данных без последних `horizon` дней,
    предсказываем `horizon` дней и сравниваем с фактом. Возвращает метрики."""
    daily_net = (
        df.groupby(df["operation_date"].dt.normalize())["signed_amount"].sum()
          .asfreq("D", fill_value=0.0)
    )
    if len(daily_net) <= horizon + 7:
        return {"mae": np.nan, "mape": np.nan, "rmse": np.nan, "bias": np.nan}
    train_df = df[df["operation_date"] < daily_net.index[-horizon]]
    test_df = df[df["operation_date"] >= daily_net.index[-horizon]]
    if len(train_df) < 14 or len(test_df) == 0:
        return {"mae": np.nan, "mape": np.nan, "rmse": np.nan, "bias": np.nan}

    # Баланс на момент начала hold-out
    if "balance_eod" in df.columns and df["balance_eod"].notna().any():
        cutoff_date = daily_net.index[-horizon]
        before = df[(df["operation_date"] < cutoff_date) & df["balance_eod"].notna()]
        if len(before):
            start_balance = float(before.sort_values("operation_date").iloc[-1]["balance_eod"])
        else:
            start_balance = float(
                train_df.groupby(train_df["operation_date"].dt.normalize())["signed_amount"]
                        .sum().cumsum().iloc[-1]
            )
    else:
        start_balance = float(
            train_df.groupby(train_df["operation_date"].dt.normalize())["signed_amount"]
                    .sum().cumsum().iloc[-1]
        )

    pred = model_fn(train_df, horizon, start_balance)
    actual_balances = daily_net.iloc[-horizon:].cumsum() + start_balance
    pred_balances = pred.set_index("date")["balance"].reindex(actual_balances.index)

    diff = pred_balances.values - actual_balances.values
    mask = ~np.isnan(diff)
    if mask.sum() == 0:
        return {"mae": np.nan, "mape": np.nan, "rmse": np.nan, "bias": np.nan}
    diff = diff[mask]
    abs_actual = np.abs(actual_balances.values[mask])
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    bias = float(np.mean(diff))
    with np.errstate(divide="ignore", invalid="ignore"):
        mape_arr = np.abs(diff) / np.where(abs_actual == 0, np.nan, abs_actual)
    mape = float(np.nanmean(mape_arr) * 100) if not np.isnan(mape_arr).all() else np.nan
    return {"mae": mae, "mape": mape, "rmse": rmse, "bias": bias}


# ────────────────────────────────────────────────────────────────────────────
#  ГЛАВНЫЙ ПРОГНОЗ + ДОВЕРИТЕЛЬНЫЕ ИНТЕРВАЛЫ
# ────────────────────────────────────────────────────────────────────────────
def compute_confidence_interval(forecast_df: pd.DataFrame, residual_std: float,
                                current_balance: float) -> pd.DataFrame:
    """Берём std остатков из бэктеста и строим 95% доверительный коридор."""
    out = forecast_df.copy()
    bal_col = "Прогноз остатка" if "Прогноз остатка" in out.columns else "balance"
    if np.isnan(residual_std) or residual_std <= 0:
        out["Нижняя граница 95%"] = out[bal_col]
        out["Верхняя граница 95%"] = out[bal_col]
        return out
    # Накопительный std для остаточного баланса: sqrt(n) * daily_std
    steps = np.arange(1, len(out) + 1)
    sigma = residual_std * np.sqrt(steps)
    out["Нижняя граница 95%"] = out[bal_col] - 1.96 * sigma
    out["Верхняя граница 95%"] = out[bal_col] + 1.96 * sigma
    return out


def get_current_balance(df: pd.DataFrame, daily_net: pd.Series) -> float:
    if "balance_eod" in df.columns and df["balance_eod"].notna().any():
        before = df[df["balance_eod"].notna()].sort_values("operation_date")
        if len(before):
            return float(before.iloc[-1]["balance_eod"])
    return float(daily_net.cumsum().iloc[-1])


def make_forecast_enhanced(
    df: pd.DataFrame,
    horizon: int = 30,
    revenue_drop: float = 0,
    extra_expense: float = 0,
    delay_days: int = 0,
    fixed_cost_increase: float = 0,
    manual_events: list[dict] | None = None,
) -> Tuple[float, pd.DataFrame, pd.DataFrame, dict]:
    """Считает прогноз best-моделью + доверительный интервал + метрики качества.
    Поддерживает 4 типа стресс-сценариев (по ТЗ «со звёздочкой»):
      1. revenue_drop %  — снижение выручки (потеря клиента, сезонный спад)
      2. delay_days      — задержка поступлений от крупного клиента
      3. extra_expense   — внеплановый расход (ремонт, штраф, доначисление)
      4. fixed_cost_increase % — рост постоянных расходов (аренда, закупки)
    Возвращает: (current_balance, forecast_with_ci, daily_actual, model_meta).
    """
    daily_net = (
        df.groupby(df["operation_date"].dt.normalize())["signed_amount"].sum()
    ).sort_index()
    current_balance = get_current_balance(df, daily_net)

    # Сравнение моделей на бэктесте — выбираем ту, у которой меньше MAE.
    candidates = {
        "Weekday Mean":      (forecast_weekday_mean, backtest_model(df, horizon, forecast_weekday_mean)),
        "Holt Linear":       (forecast_holt, backtest_model(df, horizon, forecast_holt)),
        "Linear + Weekday":  (forecast_linear_trend_weekday,
                              backtest_model(df, horizon, forecast_linear_trend_weekday)),
    }
    best_name = min(candidates.keys(), key=lambda k: candidates[k][1]["mae"] if not np.isnan(candidates[k][1]["mae"]) else np.inf)
    best_fn, best_metrics = candidates[best_name]

    base = best_fn(df, horizon, current_balance)
    base = base.rename(columns={"date": "Дата", "flow": "Ожидаемый поток", "balance": "Прогноз остатка"})

    # Применение стресс-сценария
    flows = base["Ожидаемый поток"].copy()
    start_date = base["Дата"].iloc[0]
    # Предвычисляем среднедневной «постоянный расход» из последних 12 недель:
    # аренда, зарплата, лизинг — всё, что стабильно списывается.
    fixed_daily_outflow = 0.0
    if fixed_cost_increase:
        cutoff = df["operation_date"].max() - pd.Timedelta(days=84)
        recent = df[df["operation_date"] >= cutoff]
        fixed_mask = recent["category"].isin([
            "Аренда", "Зарплата", "Кредиты и лизинг", "Связь и IT", "Комиссии и банк",
        ])
        if fixed_mask.any():
            fixed_out = -recent.loc[fixed_mask, "signed_amount"].sum()
            days = max((recent["operation_date"].max() - recent["operation_date"].min()).days, 1)
            fixed_daily_outflow = fixed_out / days
    for i, d in enumerate(base["Дата"]):
        f = flows.iloc[i]
        if revenue_drop and f > 0:
            f = f * (1 - revenue_drop / 100.0)
        if delay_days and d <= start_date + pd.Timedelta(days=delay_days - 1):
            # Только отсрочка входящих
            f = min(f, 0.0)
        if fixed_cost_increase and fixed_daily_outflow > 0:
            # Увеличение применяется к ежедневному расходу (равномерно)
            f -= fixed_daily_outflow * (fixed_cost_increase / 100.0)
        if extra_expense and i == 0:
            f -= extra_expense
        flows.iloc[i] = f

    # Внедрение ручных событий (договоры, счета, налоги).
    if manual_events:
        for ev in manual_events:
            try:
                ev_date = pd.to_datetime(ev["date"], dayfirst=True)
                ev_amount = float(ev["amount"])
                if ev.get("type") == "out":
                    ev_amount = -abs(ev_amount)
                else:
                    ev_amount = abs(ev_amount)
            except Exception:
                continue
            # Найти ближайший день в горизонте
            mask = base["Дата"] == ev_date.normalize()
            if mask.any():
                idx = base.index[mask][0]
                flows.loc[idx] += ev_amount
            # Если дата вне горизонта — игнорируем

    # Пересчитать кумулятивный остаток с модифицированными потоками
    balance = current_balance
    cum = []
    for f in flows:
        balance += float(f)
        cum.append(balance)
    base["Прогноз остатка"] = cum
    base["Ожидаемый поток"] = flows

    # Доверительные интервалы (используем MAE из бэктеста как прокси std)
    residual_std = best_metrics["mae"] if not np.isnan(best_metrics["mae"]) else 0
    base = compute_confidence_interval(base, residual_std, current_balance)

    model_meta = {
        "best_model": best_name,
        "candidates": candidates,
        "metrics": best_metrics,
        "current_balance": current_balance,
    }
    return current_balance, base, daily_net, model_meta


# ────────────────────────────────────────────────────────────────────────────
#  ВСПОМОГАТЕЛЬНЫЕ KPI
# ────────────────────────────────────────────────────────────────────────────
def fmt_money(x: float, suffix: str = " ₽") -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    sign = "−" if x < 0 else ""
    return f"{sign}{abs(x):,.0f}".replace(",", " ") + suffix


def fmt_pct(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.1f} %"


def compute_kpis(df: pd.DataFrame, current_balance: float, base_forecast: pd.DataFrame) -> dict:
    inflows = float(df.loc[df["signed_amount"] > 0, "signed_amount"].sum())
    outflows = float(-df.loc[df["signed_amount"] < 0, "signed_amount"].sum())
    history_days = (df["operation_date"].max() - df["operation_date"].min()).days or 1
    avg_daily_inflow = inflows / max(history_days, 1)
    avg_daily_outflow = outflows / max(history_days, 1)
    net_burn = avg_daily_outflow - avg_daily_inflow

    min_balance = float(base_forecast["Прогноз остатка"].min())
    min_idx = base_forecast["Прогноз остатка"].idxmin()
    min_date = base_forecast.loc[min_idx, "Дата"]
    days_to_min = (min_date - df["operation_date"].max()).days

    # Runway: сколько дней до 0, если средний чистый отток сохранится.
    if net_burn > 0:
        runway = int(current_balance / net_burn)
    else:
        runway = None  # не сгорает

    # Риск кассового разрыва: % площади под CI ниже 0.
    low = base_forecast["Нижняя граница 95%"]
    risk_pct = float((low < 0).mean() * 100)

    # Дефицит
    deficit_mask = base_forecast["Прогноз остатка"] < 0
    if deficit_mask.any():
        risk_date = base_forecast.loc[deficit_mask, "Дата"].min()
        risk_amount = abs(float(base_forecast.loc[base_forecast["Дата"] == risk_date, "Прогноз остатка"].iloc[0]))
    else:
        risk_date = None
        risk_amount = 0.0

    # Волатильность (стандартное отклонение дневного чистого потока)
    daily_net = df.groupby(df["operation_date"].dt.normalize())["signed_amount"].sum()
    volatility = float(daily_net.std()) if len(daily_net) > 1 else 0.0

    return {
        "current_balance": current_balance,
        "inflows": inflows,
        "outflows": outflows,
        "avg_daily_inflow": avg_daily_inflow,
        "avg_daily_outflow": avg_daily_outflow,
        "net_burn": net_burn,
        "runway": runway,
        "min_balance": min_balance,
        "min_date": min_date,
        "days_to_min": days_to_min,
        "risk_pct": risk_pct,
        "risk_date": risk_date,
        "risk_amount": risk_amount,
        "volatility": volatility,
        "history_days": history_days,
    }


# ────────────────────────────────────────────────────────────────────────────
#  АВТО-ДЕТЕКТ РЕГУЛЯРНЫХ ПЛАТЕЖЕЙ (Задача 2 из ТЗ)
# ────────────────────────────────────────────────────────────────────────────
def detect_recurring_payments(df: pd.DataFrame, min_occurrences: int = 3) -> list[dict]:
    """Автоматически находит регулярные платежи (аренда, зарплата, ENP, лизинг).
    Алгоритм: группируем по контрагенту+знаку, проверяем периодичность
    (месячный или квартальный цикл) и стабильность суммы (CV < 25%).
    Возвращает список с предсказанием следующего платежа."""
    if len(df) < min_occurrences:
        return []

    recurring: list[dict] = []
    df_s = df.sort_values("operation_date").copy()
    df_s["counterparty_key"] = (
        df_s["counterparty_name"].fillna("").astype(str).str.strip().str.lower()
        .str[:40]
    )
    df_s["sign"] = np.sign(df_s["signed_amount"])
    today = df_s["operation_date"].max()

    for (counterparty, sign), group in df_s.groupby(["counterparty_key", "sign"]):
        if not counterparty or len(group) < min_occurrences:
            continue
        dates = group["operation_date"].dt.normalize().sort_values()
        diffs = dates.diff().dropna()
        if len(diffs) < min_occurrences - 1:
            continue
        median_interval = diffs.median()
        # Поддерживаем месячный (25-35 дней) и квартальный (85-95 дней) циклы
        valid = (
            (pd.Timedelta(days=20) <= median_interval <= pd.Timedelta(days=40))
            or (pd.Timedelta(days=80) <= median_interval <= pd.Timedelta(days=100))
        )
        if not valid:
            continue
        amounts = group["amount"].abs()
        median_amount = float(amounts.median())
        if median_amount <= 0:
            continue
        cv = float(amounts.std() / median_amount) if median_amount > 0 else 1
        if cv > 0.30:
            continue
        last_date = dates.iloc[-1]
        next_date = last_date + median_interval
        # Skip «неактивные» паттерны: если ожидаемый следующий платёж уже в прошлом —
        # регулярность прекратилась, не прогнозируем.
        if next_date < today:
            continue
        recurring.append({
            "counterparty": counterparty[:40],
            "direction": "Поступление" if sign > 0 else "Списание",
            "interval_days": int(median_interval.days),
            "median_amount": median_amount,
            "last_date": last_date,
            "next_date": next_date,
            "occurrences": int(len(group)),
            "cv": cv,
            "category": group["category"].mode().iloc[0] if not group["category"].empty else "Прочее",
        })
    recurring.sort(key=lambda r: r["median_amount"], reverse=True)
    return recurring



# ────────────────────────────────────────────────────────────────────────────
#  ЦВЕТА ДЛЯ PLOTLY
# ────────────────────────────────────────────────────────────────────────────
PLOTLY_TEMPLATE = "plotly_dark"
PLOTLY_PALETTE = {
    "blue":   "#3b82f6",
    "cyan":   "#06b6d4",
    "green":  "#22c55e",
    "amber":  "#f59e0b",
    "red":    "#ef4444",
    "violet": "#8b5cf6",
    "pink":   "#ec4899",
    "grey":   "#64748b",
    "lime":   "#84cc16",
    "orange": "#f97316",
}


def style_plotly_fig(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_PRIMARY, family="Inter, system-ui, sans-serif", size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=PANEL_BG, font_color=TEXT_PRIMARY, bordercolor=PANEL_BORDER),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.10)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.10)",
                   tickformat=",.0f"),
    )
    return fig


# ────────────────────────────────────────────────────────────────────────────
#  АУТЕНТИФИКАЦИЯ (Тех. требование 7 — безопасность)
# ────────────────────────────────────────────────────────────────────────────
# Включается установкой env-переменной FINPOTOK_PASSWORD перед запуском:
#   FINPOTOK_PASSWORD=secret streamlit run app.py
# Пароль хранится в env, не в коде. В логи пишутся только хэши (если пишутся).
_AUTH_PASSWORD = os.environ.get("FINPOTOK_PASSWORD", "").strip()


def _auth_gate() -> bool:
    """Проверяет аутентификацию. Если FINPOTOK_PASSWORD не задан — пропускает."""
    if not _AUTH_PASSWORD:
        return True
    if "auth_ok" not in st.session_state:
        st.session_state["auth_ok"] = False
    if st.session_state["auth_ok"]:
        return True
    st.markdown(
        f"""
        <div style='max-width:420px;margin:60px auto;padding:32px;
                    background:{PANEL_BG};border:1px solid {PANEL_BORDER};
                    border-radius:16px;'>
          <h2 style='margin:0 0 8px;'>🔐 ФинПоток</h2>
          <div style='color:{TEXT_SECONDARY};font-size:0.9rem;margin-bottom:20px;'>
            Доступ к финансовым данным ограничен. Введите пароль.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    pwd = st.text_input("Пароль", type="password", placeholder="••••••••")
    col_a, col_b = st.columns([1, 3])
    with col_a:
        if st.button("Войти", type="primary", use_container_width=True):
            # Защита от time-атак: используем compare_digest
            import hmac
            if hmac.compare_digest(pwd, _AUTH_PASSWORD):
                st.session_state["auth_ok"] = True
                st.rerun()
            else:
                st.error("Неверный пароль")
    with col_b:
        st.caption("💡 Пароль задаётся переменной окружения FINPOTOK_PASSWORD")
    st.stop()
    return False


_auth_gate()


# ────────────────────────────────────────────────────────────────────────────
#  БОКОВАЯ ПАНЕЛЬ
# ────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 ФинПоток")
    if _AUTH_PASSWORD:
        st.caption("🔒 Защищённый режим")
    else:
        st.caption("🔓 Публичный режим")
    st.caption("Cash-Flow Forecaster для малого бизнеса")
    st.markdown('<div class="fp-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="fp-section-title">Источник данных</div>',
                unsafe_allow_html=True)
    source = st.radio(
        "Источник",
        ["Демо-датасет (синтетика)", "Демо из архива кейса", "Загрузить выписку CSV"],
        index=0, label_visibility="collapsed",
    )
    uploaded = None
    selected_company = None

    if source == "Загрузить выписку CSV":
        uploaded = st.file_uploader("Файл банковской выписки CSV", type=["csv"])
    elif source == "Демо из архива кейса":
        st.caption("Архив cashflow_dataset_participants.zip рядом с app.py")
        companies = ["C001", "C002", "C008", "C009", "C021", "C040"]
        selected_company = st.selectbox("Компания", companies, index=0)
    else:
        st.caption("Сгенерировать синтетическую выписку с реалистичными паттернами")
        comp_choice = st.selectbox(
            "Профиль компании", ["C001 (ритейл)", "C002 (услуги)", "C008 (оптовая торговля)",
                                  "C009 (производство)", "C021 (IT)", "C040 (HoReCa)"],
            index=0,
        )
        selected_company = comp_choice.split(" ")[0]

    st.markdown('<div class="fp-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="fp-section-title">Параметры прогноза</div>',
                unsafe_allow_html=True)
    horizon = st.slider("Горизонт прогноза, дней", 14, 120, 30, step=1)
    lookback = st.slider("Окно обучения, дней", 30, 365, 120, step=10,
                         help="Используется для сглаживания и бэктеста моделей.")

    st.markdown('<div class="fp-divider"></div>', unsafe_allow_html=True)
    st.caption("Демо-режимы не требуют подключения к банку. CSV-выписка должна "
               "содержать дату, сумму, направление и (желательно) остаток на конец дня.")


# ────────────────────────────────────────────────────────────────────────────
#  ЗАГРУЗКА ДАННЫХ
# ────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=300)
def load_dataset(mode: str, company: str | None, file_bytes: bytes | None) -> pd.DataFrame:
    if mode == "Загрузить выписку CSV":
        if file_bytes is None:
            raise ValueError("Файл не загружен.")
        return normalize_data(read_csv_safely(file_bytes))
    if mode == "Демо из архива кейса":
        dataset_path = Path(__file__).parent / "cashflow_dataset_participants.zip"
        if not dataset_path.exists():
            raise FileNotFoundError(
                "Архив cashflow_dataset_participants.zip не найден рядом с app.py. "
                "Используйте синтетику или загрузите CSV."
            )
        with zipfile.ZipFile(dataset_path) as archive:
            raw = archive.read(f"statements/{company}.csv")
        return normalize_data(read_csv_safely(raw))
    # Синтетика
    return synthetic_company(company or "C001")


try:
    file_bytes = uploaded.getvalue() if uploaded is not None else None
    df = load_dataset(source, selected_company, file_bytes)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"Не удалось загрузить данные: {e}")
    st.stop()


# ────────────────────────────────────────────────────────────────────────────
#  РУЧНЫЕ СОБЫТИЯ (договоры, счета, налоги, прочее) — ЗВЁЗДОЧКА
# ────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def default_tax_events(start_balance_date: pd.Timestamp, horizon_days: int) -> list[dict]:
    """Генерирует регулярные налоговые события на горизонте прогноза.
    ENP — 28-е каждого месяца; НДС — 28-е первого месяца квартала.
    Суммы по умолчанию — реалистичные средние для малого бизнеса
    (пользователь редактирует под себя во вкладке «События»)."""
    events: list[dict] = []
    start = pd.Timestamp(start_balance_date).normalize() + pd.Timedelta(days=1)
    end = start + pd.Timedelta(days=horizon_days)
    cur = start
    while cur <= end:
        if cur.day == 28:
            events.append({
                "date": cur.strftime("%d.%m.%Y"),
                "amount": 150_000,   # средний ЕНП для малого бизнеса (~150 тыс ₽/мес)
                "type": "out",
                "description": "ЕНП (28-е)",
            })
            if cur.month in (1, 4, 7, 10):
                events.append({
                    "date": cur.strftime("%d.%m.%Y"),
                    "amount": 65_000,   # средняя квартальная доплата НДС
                    "type": "out",
                    "description": "Доплата НДС (квартальная)",
                })
        cur = cur + pd.Timedelta(days=1)
    return events


# ────────────────────────────────────────────────────────────────────────────
#  ЗАГОЛОВОК СТРАНИЦЫ
# ────────────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div style='display:flex; align-items:center; justify-content:space-between; gap:14px;'>
      <div>
        <h1 style='margin:0 0 4px 0; font-size:1.9rem;'>📊 ФинПоток <span style='color:{ACCENT_BLUE}; font-size:0.9rem; font-weight:600;'>PRO</span></h1>
        <div style='color:{TEXT_SECONDARY}; font-size:0.92rem;'>Профессиональный прогноз движения денежных средств для малого бизнеса · мультивероятностные сценарии · бэктест моделей</div>
      </div>
      <div style='color:{TEXT_SECONDARY}; text-align:right; font-size:0.85rem;'>
        <div><b style='color:{TEXT_PRIMARY};'>Источник:</b> {source}</div>
        <div><b style='color:{TEXT_PRIMARY};'>Период:</b> {df['operation_date'].min():%d.%m.%Y} — {df['operation_date'].max():%d.%m.%Y}</div>
        <div><b style='color:{TEXT_PRIMARY};'>Операций:</b> {len(df):,}</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown('<div class="fp-divider"></div>', unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
#  ОСНОВНОЙ ПРОГНОЗ
# ────────────────────────────────────────────────────────────────────────────
manual_events_state: list[dict] = []

# Сначала считаем базовый прогноз без сценариев (с пустыми ручными событиями)
current_balance, base_forecast, daily_actual, model_meta = make_forecast_enhanced(
    df, horizon=horizon, manual_events=manual_events_state
)
kpis = compute_kpis(df, current_balance, base_forecast)
metrics = model_meta["metrics"]
best_model = model_meta["best_model"]


# ────────────────────────────────────────────────────────────────────────────
#  KPI-КАРТОЧКИ (богаче, чем было)
# ────────────────────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Текущий остаток", fmt_money(kpis["current_balance"]),
            delta=f"за {kpis['history_days']} дней истории")
col2.metric("Средний приток / день", fmt_money(kpis["avg_daily_inflow"]),
            delta=f"за {kpis['history_days']} дн.")
col3.metric("Средний отток / день", fmt_money(kpis["avg_daily_outflow"]),
            delta=f"чистый: {fmt_money(-kpis['net_burn'])}",
            delta_color="inverse" if kpis["net_burn"] > 0 else "normal")
col4.metric("Минимум в прогнозе",
            fmt_money(kpis["min_balance"]),
            delta=f"{kpis['min_date']:%d.%m.%Y} · через {kpis['days_to_min']} дн.",
            delta_color="inverse" if kpis["min_balance"] < 0 else "off")
col5.metric("Runway (по ср. чистому потоку)",
            f"{kpis['runway']:,} дн.".replace(",", " ") if kpis["runway"] else "∞",
            delta=f"волатильность: {fmt_money(kpis['volatility'])}")


# ────────────────────────────────────────────────────────────────────────────
#  БАННЕР РИСКА
# ────────────────────────────────────────────────────────────────────────────
if kpis["risk_date"] is not None:
    deficit_pct = kpis["risk_pct"]
    if deficit_pct > 50:
        sev_cls, sev_pill = "fp-banner-danger", "fp-pill-high"
        sev_label = "ВЫСОКИЙ РИСК"
    elif deficit_pct > 20:
        sev_cls, sev_pill = "fp-banner-warn", "fp-pill-med"
        sev_label = "СРЕДНИЙ РИСК"
    else:
        sev_cls, sev_pill = "fp-banner-info", "fp-pill-low"
        sev_label = "НИЗКИЙ РИСК"
    st.markdown(
        f"""
        <div class='fp-banner {sev_cls}'>
          <span class='fp-pill {sev_pill}'>{sev_label}</span>
          <div>
            Возможен кассовый разрыв <b>{kpis['risk_date']:%d.%m.%Y}</b> — через
            <b>{(kpis['risk_date'] - df['operation_date'].max()).days} дн.</b>
            Расчётный дефицит по базовому сценарию: <b>{fmt_money(kpis['risk_amount'])}</b>.
            Доверительная вероятность разрыва: <b>{fmt_pct(deficit_pct)}</b>.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
elif kpis["min_balance"] < kpis["current_balance"] * 0.20:
    st.markdown(
        f"""
        <div class='fp-banner fp-banner-warn'>
          <span class='fp-pill fp-pill-med'>ВНИМАНИЕ</span>
          <div>Запас денег может снизиться до <b>{fmt_money(kpis['min_balance'])}</b>
          {kpis['min_date']:%d.%m.%Y} (≈ {kpis['min_balance']/max(kpis['current_balance'],1)*100:.0f}% от текущего).
          Проверьте стресс-сценарии во вкладке «Сценарии».</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"""
        <div class='fp-banner fp-banner-success'>
          <span class='fp-pill fp-pill-low'>СТАБИЛЬНО</span>
          <div>Базовый сценарий не предсказывает кассовый разрыв. Минимум прогноза:
          <b>{fmt_money(kpis['min_balance'])}</b> ({kpis['min_date']:%d.%m.%Y}).
          Доверительная вероятность разрыва: <b>{fmt_pct(kpis['risk_pct'])}</b>.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ────────────────────────────────────────────────────────────────────────────
#  ВКЛАДКИ
# ────────────────────────────────────────────────────────────────────────────
tab_overview, tab_forecast, tab_categories, tab_recurring, tab_scenarios, tab_quality, tab_events, tab_help = st.tabs([
    "🟦 Обзор", "📈 Прогноз", "🗂 Категории", "🔁 Регулярные",
    "⚡ Сценарии", "🎯 Качество", "📅 События", "ℹ️ Методология",
])


# ═══════════════════════════════════════════════════════════════════════════
#  ВКЛАДКА: ОБЗОР
# ═══════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.markdown('<div class="fp-section-title">Прогноз остатка с доверительным интервалом 95 %</div>',
                unsafe_allow_html=True)
    fig = go.Figure()

    # Confidence band
    fig.add_trace(go.Scatter(
        x=list(base_forecast["Дата"]) + list(base_forecast["Дата"][::-1]),
        y=list(base_forecast["Верхняя граница 95%"]) + list(base_forecast["Нижняя граница 95%"][::-1]),
        fill="toself", fillcolor="rgba(59,130,246,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Доверительный интервал 95%", hoverinfo="skip",
    ))

    # Исторический остаток (если есть)
    if "balance_eod" in df.columns and df["balance_eod"].notna().any():
        hist = df.dropna(subset=["balance_eod"]).sort_values("operation_date")
        fig.add_trace(go.Scatter(
            x=hist["operation_date"], y=hist["balance_eod"],
            mode="lines", name="Фактический остаток",
            line=dict(color=PLOTLY_PALETTE["grey"], width=1.6),
        ))

    # Базовый прогноз
    fig.add_trace(go.Scatter(
        x=base_forecast["Дата"], y=base_forecast["Прогноз остатка"],
        mode="lines+markers", name=f"Базовый прогноз ({best_model})",
        line=dict(color=PLOTLY_PALETTE["blue"], width=3),
        marker=dict(size=4),
    ))

    # Точка отсчёта
    fig.add_trace(go.Scatter(
        x=[df["operation_date"].max()], y=[current_balance],
        mode="markers", name="Текущий остаток",
        marker=dict(size=11, color=PLOTLY_PALETTE["green"], symbol="circle",
                    line=dict(width=2, color="#0b1220")),
    ))

    fig.add_hline(y=0, line_dash="dash", line_color=PLOTLY_PALETTE["red"], line_width=1.5,
                  annotation_text="Нулевой остаток", annotation_position="top left")
    if kpis["min_balance"] < 0:
        fig.add_vrect(
            x0=base_forecast["Дата"].iloc[0], x1=base_forecast["Дата"].iloc[-1],
            fillcolor="rgba(239,68,68,0.04)", layer="below", line_width=0,
        )
    fig.update_layout(
        xaxis_title="", yaxis_title="Остаток, ₽",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(style_plotly_fig(fig, height=460), use_container_width=True)

    # Два связанных графика: потоки по дням + кумулятивный чистый поток
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="fp-section-title">Дневной чистый поток (история + прогноз)</div>',
                    unsafe_allow_html=True)
        flow_fig = go.Figure()
        flow_fig.add_trace(go.Bar(
            x=daily_actual.index, y=daily_actual.values,
            name="Факт", marker_color=[
                PLOTLY_PALETTE["green"] if v > 0 else PLOTLY_PALETTE["red"]
                for v in daily_actual.values
            ],
            opacity=0.65,
        ))
        flow_fig.add_trace(go.Bar(
            x=base_forecast["Дата"], y=base_forecast["Ожидаемый поток"],
            name="Прогноз", marker_color=[
                "rgba(59,130,246,0.55)" if v > 0 else "rgba(239,68,68,0.55)"
                for v in base_forecast["Ожидаемый поток"]
            ],
        ))
        flow_fig.add_hline(y=0, line_color="rgba(255,255,255,0.3)", line_width=1)
        flow_fig.update_layout(xaxis_title="", yaxis_title="Поток, ₽",
                               bargap=0.1, barmode="overlay")
        st.plotly_chart(style_plotly_fig(flow_fig, height=320), use_container_width=True)

    with right:
        st.markdown('<div class="fp-section-title">Помесячные притоки и оттоки</div>',
                    unsafe_allow_html=True)
        monthly = df.assign(month=df["operation_date"].dt.to_period("M").astype(str))
        monthly_agg = monthly.groupby("month").agg(
            Приток=("signed_amount", lambda s: s[s > 0].sum()),
            Отток=("signed_amount", lambda s: -s[s < 0].sum()),
        ).reset_index()
        m_fig = go.Figure()
        m_fig.add_trace(go.Bar(x=monthly_agg["month"], y=monthly_agg["Приток"],
                               name="Приток", marker_color=PLOTLY_PALETTE["green"],
                               marker_line_width=0))
        m_fig.add_trace(go.Bar(x=monthly_agg["month"], y=monthly_agg["Отток"],
                               name="Отток", marker_color=PLOTLY_PALETTE["red"],
                               marker_line_width=0))
        m_fig.update_layout(barmode="group", xaxis_title="", yaxis_title="₽",
                            bargap=0.18)
        st.plotly_chart(style_plotly_fig(m_fig, height=320), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
#  ВКЛАДКА: ПРОГНОЗ
# ═══════════════════════════════════════════════════════════════════════════
with tab_forecast:
    st.markdown('<div class="fp-section-title">Детальный прогноз по дням</div>',
                unsafe_allow_html=True)
    view = base_forecast.copy()
    view = view.rename(columns={
        "Дата": "Дата", "Ожидаемый поток": "Ожидаемый поток",
        "Прогноз остатка": "Прогноз остатка",
        "Нижняя граница 95%": "Нижняя граница 95%",
        "Верхняя граница 95%": "Верхняя граница 95%",
    })
    view_disp = view.copy()
    view_disp["Дата"] = view_disp["Дата"].dt.strftime("%d.%m.%Y")
    for c in ["Ожидаемый поток", "Прогноз остатка", "Нижняя граница 95%", "Верхняя граница 95%"]:
        if c in view_disp.columns:
            view_disp[c] = view_disp[c].round(2)
    st.dataframe(view_disp, use_container_width=True, hide_index=True, height=420)
    st.download_button(
        "⬇️ Скачать прогноз CSV",
        view_disp.to_csv(index=False).encode("utf-8-sig"),
        file_name="cashflow_forecast.csv",
        mime="text/csv",
    )


# ═══════════════════════════════════════════════════════════════════════════
#  ВКЛАДКА: КАТЕГОРИИ
# ═══════════════════════════════════════════════════════════════════════════
with tab_categories:
    cats = df.assign(Тип=np.where(df["signed_amount"] > 0, "Поступления", "Списания"))
    summary = (cats.groupby(["category", "Тип"])["amount"]
                   .agg(["sum", "count"])
                   .reset_index())
    summary.columns = ["Категория", "Тип", "Сумма", "Операций"]
    summary = summary.sort_values(["Тип", "Сумма"], ascending=[True, False])

    left, right = st.columns([1, 1])
    with left:
        st.markdown('<div class="fp-section-title">Структура поступлений и списаний</div>',
                    unsafe_allow_html=True)
        infl = summary[summary["Тип"] == "Поступления"]
        outl = summary[summary["Тип"] == "Списания"]
        # Корректная иерархия для Sunburst: корневые узлы должны иметь
        # value = сумма дочерних (при branchvalues="total"), иначе Plotly
        # молча рендерит пустую область.
        infl_total = float(infl["Сумма"].sum()) if len(infl) else 0.0
        outl_total = float(outl["Сумма"].sum()) if len(outl) else 0.0
        labels = (["Поступления"] + list(infl["Категория"])
                  + ["Списания"] + list(outl["Категория"]))
        parents = ([""] + ["Поступления"] * len(infl)
                   + [""] + ["Списания"] * len(outl))
        values = ([infl_total] + list(infl["Сумма"])
                  + [outl_total] + list(outl["Сумма"]))
        # Цвет: зелёный для дерева поступлений, красный — для списаний.
        colors = (["#16a34a"] + ["#22c55e"] * len(infl)
                  + ["#b91c1c"] + ["#ef4444"] * len(outl))
        sun_fig = go.Figure()
        sun_fig.add_trace(go.Sunburst(
            labels=labels, parents=parents, values=values,
            branchvalues="total",
            marker=dict(colors=colors, line=dict(color="#0b1220", width=1.5)),
            textinfo="label+percent root",
            insidetextorientation="radial",
            hovertemplate="<b>%{label}</b><br>Сумма: %{value:,.0f} ₽<br>Доля: %{percentRoot:.1%}<extra></extra>",
        ))
        sun_fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(style_plotly_fig(sun_fig, height=380), use_container_width=True)

    with right:
        st.markdown('<div class="fp-section-title">Топ-категории по абсолютной величине</div>',
                    unsafe_allow_html=True)
        bar_fig = go.Figure()
        for tipo, color in [("Поступления", PLOTLY_PALETTE["green"]),
                            ("Списания", PLOTLY_PALETTE["red"])]:
            sub = summary[summary["Тип"] == tipo].sort_values("Сумма", ascending=True).tail(8)
            bar_fig.add_trace(go.Bar(
                x=sub["Сумма"], y=sub["Категория"], orientation="h",
                name=tipo, marker_color=color, marker_line_width=0,
            ))
        bar_fig.update_layout(barmode="stack", xaxis_title="₽", yaxis_title="",
                              bargap=0.15)
        st.plotly_chart(style_plotly_fig(bar_fig, height=380), use_container_width=True)

    st.markdown('<div class="fp-section-title">Таблица категорий</div>',
                unsafe_allow_html=True)
    st.dataframe(summary.round(2), use_container_width=True, hide_index=True, height=320)

    # Safety-чек: показать «Возврат / корректировка», если есть подозрительные
    # «поступления» в расходных категориях (налоги, зарплата и т. д.)
    suspicious_count = int((df["category"] == "Возврат / корректировка").sum())
    if suspicious_count > 0:
        suspicious_total = float(df.loc[df["category"] == "Возврат / корректировка", "signed_amount"].sum())
        st.markdown(
            f"""
            <div class='fp-banner fp-banner-warn'>
              <span class='fp-pill fp-pill-med'>УТОЧНИТЕ</span>
              <div>
                Обнаружено <b>{suspicious_count:,}</b> операций на <b>{fmt_money(suspicious_total)}</b>,
                которые отнесены к категории «<b>Возврат / корректировка</b>». Это поступления в расходных
                категориях (Налоги, Зарплата, Аренда и т. д.). Возможные причины:
                <ul style='margin:6px 0 0 18px;'>
                  <li>возврат переплаты по налогу со стороны ФНС — корректная операция;</li>
                  <li>банк в выписке кодирует направление иначе (Кт/Дт, C/D, Дебет/Кредит),
                      и парсер не распознал «списание» — операция попала в «поступления» ошибочно.</li>
                </ul>
                Проверьте столбец <b>direction</b> в исходном CSV: должны быть «Списание»/«Поступление»,
                либо явный знак суммы (+/-).
              </div>
            </div>
            """.replace(",", " "),
            unsafe_allow_html=True,
        )



# ═══════════════════════════════════════════════════════════════════════════
#  ВКЛАДКА: РЕГУЛЯРНЫЕ ПЛАТЕЖИ (Задача 2 — авто-детект)
# ═══════════════════════════════════════════════════════════════════════════
with tab_recurring:
    st.markdown('<div class="fp-section-title">Автоматически найденные регулярные платежи</div>',
                unsafe_allow_html=True)
    st.caption("Алгоритм ищет операции с одинаковым контрагентом, повторяющимися датами "
               "(месячный или квартальный цикл) и стабильной суммой (CV < 30%). "
               "Найденные регулярные платежи можно добавить в прогноз как ручные события.")

    with st.spinner("Анализирую регулярные паттерны…"):
        recurring = detect_recurring_payments(df, min_occurrences=3)

    if not recurring:
        st.info("Регулярные платежи не обнаружены. Нужно минимум 3 повторения "
                "с одним и тем же контрагентом за период ≥ 60 дней.")
    else:
        rec_df = pd.DataFrame([
            {
                "Контрагент": r["counterparty"][:30],
                "Тип": r["direction"],
                "Категория": r["category"],
                "Сумма, ₽": round(r["median_amount"], 0),
                "Цикл, дней": r["interval_days"],
                "Повторов": r["occurrences"],
                "Стабильность": f"{(1 - r['cv']) * 100:.0f}%",
                "Последний платёж": r["last_date"].strftime("%d.%m.%Y"),
                "Ожидаемый следующий": r["next_date"].strftime("%d.%m.%Y"),
            }
            for r in recurring
        ])
        st.dataframe(rec_df, use_container_width=True, hide_index=True, height=380)

        # Прогноз следующих регулярных платежей
        st.markdown('<div class="fp-section-title">Ближайшие ожидаемые регулярные платежи (в пределах горизонта)</div>',
                    unsafe_allow_html=True)
        upcoming = []
        horizon_end = df["operation_date"].max() + pd.Timedelta(days=horizon)
        for r in recurring:
            if r["next_date"] <= horizon_end:
                upcoming.append({
                    "Дата": r["next_date"].strftime("%d.%m.%Y"),
                    "Контрагент": r["counterparty"][:30],
                    "Тип": r["direction"],
                    "Категория": r["category"],
                    "Ожидаемая сумма, ₽": round(r["median_amount"], 0),
                    "Через дней": (r["next_date"] - df["operation_date"].max()).days,
                })
        if upcoming:
            up_df = pd.DataFrame(upcoming).sort_values("Через дней")
            st.dataframe(up_df, use_container_width=True, hide_index=True, height=300)

            # Кнопка «Добавить все в прогноз»
            if st.button("➕ Добавить все регулярные платежи в ручные события",
                         type="primary", use_container_width=False):
                if "manual_events_state" not in st.session_state:
                    st.session_state["manual_events_state"] = []
                for r in recurring:
                    if r["next_date"] <= horizon_end:
                        st.session_state["manual_events_state"].append({
                            "date": r["next_date"].strftime("%d.%m.%Y"),
                            "amount": float(r["median_amount"]),
                            "type": "out" if r["direction"] == "Списание" else "in",
                            "description": f"Регулярный платёж: {r['counterparty'][:30]}",
                        })
                st.success(f"Добавлено {len(upcoming)} регулярных платежей. "
                           "Перейдите на вкладку «События» для просмотра.")
        else:
            st.info("В пределах горизонта прогноза нет ожидаемых регулярных платежей.")


#  ВКЛАДКА: СЦЕНАРИИ
# ═══════════════════════════════════════════════════════════════════════════
with tab_scenarios:
    st.markdown('<div class="fp-section-title">Стресс-сценарий «Что, если?» — 4 типа шоков</div>',
                unsafe_allow_html=True)
    st.caption("Все 4 типа шоков из ТЗ «со звёздочкой»: потеря клиента/сезонный спад, "
               "задержка поступлений, внеплановый расход, рост постоянных издержек.")

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        drop = st.slider("📉 Снижение выручки, %", 0, 60, 20, step=5,
                         help="Снижение поступлений (потеря клиента, сезонный спад)")
    with col_b:
        delay = st.slider("⏱ Задержка поступлений, дней", 0, 30, 7, step=1,
                          help="Отсрочка входящих платежей от крупных клиентов")
    with col_c:
        expense = st.number_input("💸 Внеплановый расход, ₽", min_value=0,
                                   value=150_000, step=10_000,
                                   help="Разовый расход: ремонт, штраф, доначисление")
    with col_d:
        fixed_inc = st.slider("📈 Рост постоянных расходов, %", 0, 100, 0, step=5,
                              help="Увеличение аренды, зарплаты, лизинга и т. д.")

    # Базовый + стресс (4 параметра)
    stress_balance, stress_forecast, _, _ = make_forecast_enhanced(
        df, horizon=horizon,
        revenue_drop=drop, delay_days=delay, extra_expense=expense,
        fixed_cost_increase=fixed_inc,
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=base_forecast["Дата"], y=base_forecast["Прогноз остатка"],
        mode="lines", name="Базовый сценарий",
        line=dict(color=PLOTLY_PALETTE["blue"], width=3),
    ))
    fig.add_trace(go.Scatter(
        x=stress_forecast["Дата"], y=stress_forecast["Прогноз остатка"],
        mode="lines", name="Стрессовый сценарий",
        line=dict(color=PLOTLY_PALETTE["red"], width=3, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=list(stress_forecast["Дата"]) + list(stress_forecast["Дата"][::-1]),
        y=list(stress_forecast["Верхняя граница 95%"]) + list(stress_forecast["Нижняя граница 95%"][::-1]),
        fill="toself", fillcolor="rgba(239,68,68,0.10)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Доверительный интервал (стресс)", hoverinfo="skip",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color=PLOTLY_PALETTE["red"], line_width=1.2)
    fig.update_layout(xaxis_title="", yaxis_title="Остаток, ₽")
    st.plotly_chart(style_plotly_fig(fig, height=460), use_container_width=True)

    # Сводка по двум сценариям
    base_min = float(base_forecast["Прогноз остатка"].min())
    stress_min = float(stress_forecast["Прогноз остатка"].min())
    base_neg = (base_forecast["Прогноз остатка"] < 0).any()
    stress_neg = (stress_forecast["Прогноз остатка"] < 0).any()
    s_col1, s_col2, s_col3, s_col4 = st.columns(4)
    s_col1.metric("Минимум базового", fmt_money(base_min))
    s_col2.metric("Минимум стресса", fmt_money(stress_min),
                  delta=fmt_money(stress_min - base_min), delta_color="inverse")
    s_col3.metric("Разрыв в базовом", "Да" if base_neg else "Нет")
    s_col4.metric("Разрыв в стрессе", "Да" if stress_neg else "Нет")

    if stress_neg:
        first_neg = stress_forecast.loc[stress_forecast["Прогноз остатка"] < 0].iloc[0]
        st.markdown(
            f"""
            <div class='fp-banner fp-banner-danger'>
              <span class='fp-pill fp-pill-high'>РАЗРЫВ</span>
              <div>В стрессовом сценарии возможен кассовый разрыв
              <b>{first_neg['Дата']:%d.%m.%Y}</b>. Расчётный дефицит:
              <b>{fmt_money(abs(first_neg['Прогноз остатка']))}</b>.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class='fp-banner fp-banner-success'>
              <span class='fp-pill fp-pill-low'>OK</span>
              <div>В выбранном стрессовом сценарии прогноз не пересекает нулевой остаток.
              Минимум: <b>{fmt_money(stress_min)}</b>.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
#  ВКЛАДКА: КАЧЕСТВО ПРОГНОЗА
# ═══════════════════════════════════════════════════════════════════════════
with tab_quality:
    st.markdown('<div class="fp-section-title">Сравнение моделей прогноза</div>',
                unsafe_allow_html=True)
    st.caption("Все модели проходят hold-out бэктест: обучаемся на данных без последних "
               f"{horizon} дней, прогнозируем `horizon` дней, сравниваем с фактом. "
               "Лучшей по MAE автоматически становится модель базового прогноза.")

    rows = []
    for name, (fn, m) in model_meta["candidates"].items():
        rows.append({
            "Модель": name,
            "MAE, ₽": round(m["mae"], 2) if not np.isnan(m["mae"]) else None,
            "MAPE, %": round(m["mape"], 2) if not np.isnan(m["mape"]) else None,
            "RMSE, ₽": round(m["rmse"], 2) if not np.isnan(m["rmse"]) else None,
            "Смещение (bias), ₽": round(m["bias"], 2) if not np.isnan(m["bias"]) else None,
            "Выбрана": "✅" if name == best_model else "",
        })
    metrics_df = pd.DataFrame(rows)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    st.markdown('<div class="fp-section-title">План-факт: последний hold-out период</div>',
                unsafe_allow_html=True)
    # Бэктест-визуализация: факт против прогноза на горизонте lookback/последних 30 дней
    daily_net = (df.groupby(df["operation_date"].dt.normalize())["signed_amount"].sum()
                  .asfreq("D", fill_value=0.0))
    holdout_days = min(30, max(7, len(daily_net) // 4))
    if len(daily_net) > holdout_days + 7:
        train_df = df[df["operation_date"] < daily_net.index[-holdout_days]]
        cutoff_date = daily_net.index[-holdout_days]
        if "balance_eod" in df.columns and df["balance_eod"].notna().any():
            before = df[(df["operation_date"] < cutoff_date) & df["balance_eod"].notna()]
            start_bal = float(before.sort_values("operation_date").iloc[-1]["balance_eod"]) if len(before) else float(
                train_df.groupby(train_df["operation_date"].dt.normalize())["signed_amount"].sum().cumsum().iloc[-1]
            )
        else:
            start_bal = float(train_df.groupby(train_df["operation_date"].dt.normalize())["signed_amount"].sum().cumsum().iloc[-1])

        pf_pred = model_meta["candidates"][best_model][0](train_df, holdout_days, start_bal)
        actual_cum = daily_net.iloc[-holdout_days:].cumsum() + start_bal
        pf_fig = go.Figure()
        pf_fig.add_trace(go.Scatter(x=actual_cum.index, y=actual_cum.values,
                                    name="Факт", line=dict(color=PLOTLY_PALETTE["green"], width=2.5)))
        pf_fig.add_trace(go.Scatter(x=pf_pred["date"], y=pf_pred["balance"],
                                    name=f"Прогноз ({best_model})",
                                    line=dict(color=PLOTLY_PALETTE["blue"], width=2.5, dash="dash")))
        pf_fig.add_hline(y=0, line_dash="dash", line_color=PLOTLY_PALETTE["red"], line_width=1)
        pf_fig.update_layout(xaxis_title="", yaxis_title="Остаток, ₽")
        st.plotly_chart(style_plotly_fig(pf_fig, height=380), use_container_width=True)
    else:
        st.info("Недостаточно данных для построения плана-факта (нужно больше 30 дней истории).")

    # План-факт реального времени: последние 14 дней — отклонения факта от прогноза
    st.markdown('<div class="fp-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="fp-section-title">План-факт в реальном времени (последние 14 дней)</div>',
                unsafe_allow_html=True)
    st.caption("Звёздочка 2 из ТЗ: подсветка расхождений между ожидаемыми и фактическими "
               "потоками. Если ожидаемый платёж не пришёл вовремя — помечается красным.")

    daily_net_full = (df.groupby(df["operation_date"].dt.normalize())["signed_amount"].sum()
                       .asfreq("D", fill_value=0.0))
    rt_days = min(14, max(3, len(daily_net_full) // 4))
    if len(daily_net_full) > rt_days + 7:
        # Обучаемся на данных без последних rt_days дней
        rt_train_df = df[df["operation_date"] < daily_net_full.index[-rt_days]]
        rt_cutoff = daily_net_full.index[-rt_days]
        if "balance_eod" in df.columns and df["balance_eod"].notna().any():
            before = df[(df["operation_date"] < rt_cutoff) & df["balance_eod"].notna()]
            rt_start_bal = float(before.sort_values("operation_date").iloc[-1]["balance_eod"]) if len(before) else float(
                rt_train_df.groupby(rt_train_df["operation_date"].dt.normalize())["signed_amount"].sum().cumsum().iloc[-1]
            )
        else:
            rt_start_bal = float(rt_train_df.groupby(rt_train_df["operation_date"].dt.normalize())["signed_amount"].sum().cumsum().iloc[-1])
        rt_pred = model_meta["candidates"][best_model][0](rt_train_df, rt_days, rt_start_bal)
        actual_daily = daily_net_full.iloc[-rt_days:]
        pred_daily = rt_pred.set_index("date")["flow"].reindex(actual_daily.index)

        # Отклонения по потокам
        dev = actual_daily - pred_daily
        dev_pct = (dev.abs() / pred_daily.abs().replace(0, np.nan) * 100).fillna(0)
        rt_fig = go.Figure()
        rt_fig.add_trace(go.Bar(
            x=actual_daily.index, y=actual_daily.values,
            name="Факт", marker_color=[
                PLOTLY_PALETTE["green"] if v > 0 else PLOTLY_PALETTE["red"]
                for v in actual_daily.values
            ], opacity=0.8,
        ))
        rt_fig.add_trace(go.Bar(
            x=pred_daily.index, y=pred_daily.values,
            name="План", marker_color="rgba(100,116,139,0.45)",
        ))
        # Подсветка больших отклонений (>30%)
        big_dev_mask = dev_pct > 30
        if big_dev_mask.any():
            rt_fig.add_trace(go.Scatter(
                x=actual_daily.index[big_dev_mask],
                y=[max(abs(actual_daily.values[i]), abs(pred_daily.values[i])) * 1.1
                   for i, v in enumerate(big_dev_mask) if v],
                mode="markers", marker=dict(symbol="triangle-up", size=12, color=PLOTLY_PALETTE["amber"]),
                name="⚠ Большое отклонение (>30%)",
                hovertext=[f"Отклонение: {dev_pct.values[i]:.0f}%" for i, v in enumerate(big_dev_mask) if v],
            ))
        rt_fig.add_hline(y=0, line_color="rgba(255,255,255,0.3)", line_width=1)
        rt_fig.update_layout(xaxis_title="", yaxis_title="Поток, ₽",
                              barmode="group", bargap=0.2)
        st.plotly_chart(style_plotly_fig(rt_fig, height=340), use_container_width=True)

        # Таблица отклонений
        if big_dev_mask.any():
            dev_rows = []
            for i, d in enumerate(actual_daily.index):
                if big_dev_mask.iloc[i]:
                    dev_rows.append({
                        "Дата": d.strftime("%d.%m.%Y"),
                        "План, ₽": round(float(pred_daily.iloc[i]), 0),
                        "Факт, ₽": round(float(actual_daily.iloc[i]), 0),
                        "Отклонение, ₽": round(float(dev.iloc[i]), 0),
                        "Отклонение, %": round(float(dev_pct.iloc[i]), 1),
                    })
            st.markdown(
                f"""
                <div class='fp-banner fp-banner-warn'>
                  <span class='fp-pill fp-pill-med'>⚠ ОТКЛОНЕНИЯ</span>
                  <div>Обнаружено <b>{len(dev_rows)}</b> значительных расхождений плана и факта
                  (&gt;30%). Проверьте ожидаемые платежи во вкладке «События» — возможно,
                  контрагент задержал оплату.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.dataframe(pd.DataFrame(dev_rows), use_container_width=True,
                         hide_index=True, height=200)
    else:
        st.info("Недостаточно истории для план-факт реального времени (нужно >14 дней).")


# ═══════════════════════════════════════════════════════════════════════════
#  ВКЛАДКА: СОБЫТИЯ (ручной ввод + налоги)
# ═══════════════════════════════════════════════════════════════════════════
with tab_events:
    st.markdown('<div class="fp-section-title">Запланированные события (договоры, счета, налоги)</div>',
                unsafe_allow_html=True)
    st.caption("Добавьте события, которые ещё не отражены в выписке, но известны заранее. "
               "Они будут учтены в прогнозе на вкладке «Прогноз» и в расчёте разрывов.")

    if "manual_events_state" not in st.session_state:
        st.session_state["manual_events_state"] = default_tax_events(
            df["operation_date"].max(), horizon
        )

    ev_col1, ev_col2 = st.columns([3, 2])
    with ev_col1:
        with st.form("add_event", clear_on_submit=True):
            ec1, ec2, ec3 = st.columns([2, 2, 2])
            with ec1:
                ev_date = st.text_input("Дата (ДД.ММ.ГГГГ)", value=(df["operation_date"].max() + pd.Timedelta(days=7)).strftime("%d.%m.%Y"))
            with ec2:
                ev_amount = st.number_input("Сумма, ₽", min_value=0.0, value=50_000.0, step=10_000.0)
            with ec3:
                ev_type = st.selectbox("Тип", ["Поступление (in)", "Списание (out)"], index=1)
            ev_desc = st.text_input("Описание", value="Оплата по счёту от клиента")
            if st.form_submit_button("➕ Добавить событие"):
                st.session_state["manual_events_state"].append({
                    "date": ev_date,
                    "amount": float(ev_amount),
                    "type": "out" if ev_type.startswith("Списание") else "in",
                    "description": ev_desc,
                })
                st.success(f"Событие добавлено ({ev_date}). Пересчёт прогноза выполнится автоматически.")

    with ev_col2:
        st.markdown('<div class="fp-section-title">Налоговый календарь</div>',
                    unsafe_allow_html=True)
        st.markdown(
            """
            - **ЕНП** — 28-е число каждого месяца (единый налоговый платёж)
            - **НДС** — 28-е первого месяца квартала (квартальные доплаты)
            - **Авансы по УСН** — 28-е апреля, июля, октября, декабря
            - **Страховые взносы** — 28-е следующего за выплатой месяца
            """
        )
        if st.button("🔄 Сбросить события к налоговым по умолчанию"):
            st.session_state["manual_events_state"] = default_tax_events(
                df["operation_date"].max(), horizon
            )
            st.success("События сброшены к налоговому календарю по умолчанию.")

    # Список событий
    st.markdown('<div class="fp-section-title">Текущие запланированные события</div>',
                unsafe_allow_html=True)
    if not st.session_state["manual_events_state"]:
        st.info("Пока нет запланированных событий.")
    else:
        ev_df = pd.DataFrame(st.session_state["manual_events_state"])
        ev_df = ev_df.rename(columns={"date": "Дата", "amount": "Сумма",
                                      "type": "Тип", "description": "Описание"})
        ev_df["Сумма"] = ev_df["Сумма"].apply(lambda x: f"{x:,.0f} ₽".replace(",", " "))
        st.dataframe(ev_df, use_container_width=True, hide_index=True, height=300)

    # Пересчёт прогноза с ручными событиями
    balance_with_ev, forecast_with_ev, _, _ = make_forecast_enhanced(
        df, horizon=horizon,
        manual_events=st.session_state["manual_events_state"],
    )
    diff_min = float(forecast_with_ev["Прогноз остатка"].min()) - float(base_forecast["Прогноз остатка"].min())

    ev_fig = go.Figure()
    ev_fig.add_trace(go.Scatter(
        x=base_forecast["Дата"], y=base_forecast["Прогноз остатка"],
        name="Без событий", line=dict(color=PLOTLY_PALETTE["grey"], width=2, dash="dot"),
    ))
    ev_fig.add_trace(go.Scatter(
        x=forecast_with_ev["Дата"], y=forecast_with_ev["Прогноз остатка"],
        name="С учётом событий",
        line=dict(color=PLOTLY_PALETTE["cyan"], width=3),
    ))
    # Маркеры событий
    for ev in st.session_state["manual_events_state"]:
        try:
            ed = pd.to_datetime(ev["date"], dayfirst=True)
            mask = forecast_with_ev["Дата"] == ed.normalize()
            if mask.any():
                yv = float(forecast_with_ev.loc[mask, "Прогноз остатка"].iloc[0])
                color = PLOTLY_PALETTE["red"] if ev["type"] == "out" else PLOTLY_PALETTE["green"]
                ev_fig.add_trace(go.Scatter(
                    x=[ed], y=[yv],
                    mode="markers+text",
                    text=[ev.get("description", "")[:18]],
                    textposition="top center",
                    marker=dict(size=12, color=color, symbol="triangle-down" if ev["type"] == "out" else "triangle-up",
                                line=dict(width=1, color="#0b1220")),
                    showlegend=False, hovertext=f"{ev['date']}: {ev['amount']}",
                ))
        except Exception:
            continue
    ev_fig.add_hline(y=0, line_dash="dash", line_color=PLOTLY_PALETTE["red"], line_width=1)
    ev_fig.update_layout(xaxis_title="", yaxis_title="Остаток, ₽")
    st.plotly_chart(style_plotly_fig(ev_fig, height=400), use_container_width=True)

    c1, c2 = st.columns(2)
    c1.metric("Минимум прогноза без событий",
              fmt_money(float(base_forecast["Прогноз остатка"].min())))
    c2.metric("Минимум прогноза с событиями",
              fmt_money(float(forecast_with_ev["Прогноз остатка"].min())),
              delta=fmt_money(diff_min), delta_color="inverse" if diff_min < 0 else "normal")

    # ─── Звёздочка 1: Выставленные счета с дедлайнами ────────────────────────
    st.markdown('<div class="fp-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="fp-section-title">Выставленные счета (оплаты ожидаются)</div>',
                unsafe_allow_html=True)
    st.caption("Звёздочка 1 из ТЗ: учёт выставленных клиентам и полученных от поставщиков "
               "счетов с дедлайнами. Счета автоматически добавляются в прогноз как "
               "ожидаемые поступления/списания на дату оплаты.")

    if "pending_invoices" not in st.session_state:
        st.session_state["pending_invoices"] = []

    with st.form("add_invoice", clear_on_submit=True):
        ic1, ic2, ic3, ic4 = st.columns(4)
        with ic1:
            inv_date = st.text_input("Срок оплаты (ДД.ММ.ГГГГ)",
                value=(df["operation_date"].max() + pd.Timedelta(days=10)).strftime("%d.%m.%Y"))
        with ic2:
            inv_amount = st.number_input("Сумма счёта, ₽", min_value=0.0, value=200_000.0, step=10_000.0)
        with ic3:
            inv_type = st.selectbox("Направление", ["Ожидаемое поступление (in)", "Ожидаемое списание (out)"], index=0)
        with ic4:
            inv_counterparty = st.text_input("Контрагент", value="ООО «Клиент»")
        inv_desc = st.text_input("Описание счёта", value="Счёт № за поставку по договору")
        if st.form_submit_button("📋 Добавить счёт"):
            st.session_state["pending_invoices"].append({
                "date": inv_date,
                "amount": float(inv_amount),
                "type": "out" if inv_type.startswith("Ожидаемое списание") else "in",
                "description": f"Счёт: {inv_desc[:50]}",
                "counterparty": inv_counterparty,
            })
            st.success(f"Счёт добавлен. Ожидается {'списание' if inv_type.startswith('Ожидаемое списание') else 'поступление'} {inv_date}.")

    if st.session_state["pending_invoices"]:
        inv_df = pd.DataFrame(st.session_state["pending_invoices"])
        inv_df = inv_df.rename(columns={"date": "Срок", "amount": "Сумма",
                                        "type": "Тип", "description": "Описание",
                                        "counterparty": "Контрагент"})
        inv_df["Сумма"] = inv_df["Сумма"].apply(lambda x: f"{x:,.0f} ₽".replace(",", " "))
        inv_df["Тип"] = inv_df["Тип"].map({"in": "Поступление", "out": "Списание"})
        st.dataframe(inv_df, use_container_width=True, hide_index=True, height=220)

        # Добавить все счета в события (для учёта в прогнозе)
        if st.button("➕ Включить все счета в прогноз", type="primary"):
            for inv in st.session_state["pending_invoices"]:
                st.session_state["manual_events_state"].append({
                    "date": inv["date"], "amount": float(inv["amount"]),
                    "type": inv["type"], "description": inv["description"],
                })
            st.success(f"Добавлено {len(st.session_state['pending_invoices'])} счетов в прогноз.")
    else:
        st.info("Нет выставленных счетов. Добавьте через форму выше.")


# ═══════════════════════════════════════════════════════════════════════════
#  ВКЛАДКА: МЕТОДОЛОГИЯ
# ═══════════════════════════════════════════════════════════════════════════
with tab_help:
    st.markdown('<div class="fp-section-title">Как считается прогноз</div>',
                unsafe_allow_html=True)
    st.markdown(
        """
        1. **Подготовка данных.** CSV-выписка нормализуется: распознаются дата, сумма,
           направление и остаток по окончании дня. Поддерживаются русские и английские
           заголовки, кодировки `utf-8` / `utf-8-sig` / `cp1251`, разделители `;`, `,`, `\\t`, `|`.
           Распознавание направления поддерживает **18 форматов**: Поступление/Списание,
           Приход/Расход, Кредит/Дебет, Кт/Дт, C/D, CR/DR, Credit/Debit, +/-.
        2. **Категоризация.** По ключевым словам в назначении платежа и наименовании
           контрагента операции относятся к одной из 9 категорий. Векторизованная
           реализация (через `str.contains`) примерно в 50–100× быстрее `apply`.
           Приоритеты исправлены: «Продажи» (включая эквайринг-поступления) имеют
           приоритет над «Комиссии и банк». Поступления в расходных категориях
           (налоги, зарплата и т. д.) переотносятся в «Возврат / корректировка».
        3. **Авто-детект регулярных платежей.** Алгоритм находит операции с одинаковым
           контрагентом, периодичностью 20–40 дней (месячный цикл) или 80–100 дней
           (квартальный), и коэффициентом вариации суммы < 30%. Предсказывает следующий
           платёж и предлагает добавить в ручные события.
        4. **Выбор модели.** Прогноз считается тремя моделями одновременно:
           - **Weekday Mean** — средний чистый поток по дням недели за последние 12 недель,
             с обрезкой 5/95-го перцентиля для устойчивости к выбросам;
           - **Holt Linear** — экспоненциальное сглаживание с трендом (statsmodels);
           - **Linear + Weekday** — линейный регрессионный тренд на сглаженном ряду
             + поправка на день недели (sklearn).
           Модели проходят hold-out бэктест: обучаемся без последних `horizon` дней
           и сравниваем прогноз с фактом. Лучшей по **MAE** автоматически становится
           модель базового прогноза.
        5. **Доверительные интервалы.** Строятся как `±1.96 σ√n`, где σ оценивается
           из MAE на бэктесте. Коридор расширяется с ростом горизонта — отражает
           нарастающую неопределённость.
        6. **Стресс-сценарии (4 типа).** По ТЗ «со звёздочкой»:
           - 📉 снижение выручки на X% (только к положительным потокам);
           - ⏱ задержка поступлений на N дней (отсрочка входящих);
           - 💸 внеплановый расход в первый день прогноза;
           - 📈 рост постоянных расходов (аренда, зарплата, лизинг) на X% — равномерно по дням.
        7. **Ручные события.** Договоры, счета и налоги добавляются пользователем
           во вкладке «События» и учитываются в прогнозе. Налоговый календарь
           (ЕНП 28-е, НДС квартально, авансы по УСН) создаётся автоматически.
        8. **Выставленные счета.** Отдельная форма для счетов с дедлайнами оплаты —
           учитываются в прогнозе при нажатии кнопки «Включить в прогноз».
        9. **План-факт real-time.** Подсветка расхождений >30% между ожидаемыми и
            фактическими потоками за последние 14 дней — сигнал, что контрагент
            задерживает оплату.
        10. **Безопасность.** Если задана env-переменная `FINPOTOK_PASSWORD`,
            приложение требует аутентификацию. Пароль сравнивается через
            `hmac.compare_digest` (защита от time-атак). Чувствительные данные
            в логи не пишутся.
        11. **KPI и риск.** Считаются: текущий остаток, средние приток/отток, чистый
            «burn rate», «runway» (дни до нуля по текущему чистому потоку), волатильность,
            дата и сумма расчётного дефицита, % доверительной вероятности разрыва.
        """
    )

    st.markdown('<div class="fp-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="fp-section-title">Метрики качества прогноза (текущий бэктест)</div>',
                unsafe_allow_html=True)
    m = metrics
    q_col1, q_col2, q_col3, q_col4 = st.columns(4)
    q_col1.metric("Лучшая модель", best_model)
    q_col2.metric("MAE", fmt_money(m["mae"]))
    q_col3.metric("MAPE", fmt_pct(m["mape"]))
    q_col4.metric("RMSE", fmt_money(m["rmse"]))

    st.markdown('<div class="fp-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="fp-section-title">Архитектура решения</div>',
                unsafe_allow_html=True)
    st.caption("Открыть в полном размере — клик правой кнопкой → «Открыть изображение».")
    mermaid_diagram = r"""
    flowchart TB
      subgraph DS["📡 Источники данных"]
        D1["CSV-выписка<br/>utf-8 / cp1251"]
        D2["Архив кейса<br/>ZIP"]
        D3["Синтетика<br/>fallback"]
        D4["Ручной ввод<br/>(события, счета, налоги)"]
      end
      subgraph PROC["⚙️ Обработка"]
        N1["Нормализация<br/>алиасы колонок, кодировка"]
        N2["Парсинг направления<br/>18 форматов: Кт/Дт, C/D, +/-"]
        N3["Категоризация<br/>векторизованная, 9 категорий"]
        N4["Авто-детект регулярных<br/>платежей"]
      end
      subgraph FCST["🎯 Прогноз"]
        M1["Weekday Mean"]
        M2["Holt Linear<br/>(statsmodels)"]
        M3["Linear + Weekday<br/>(sklearn)"]
        BT["Бэктест hold-out<br/>MAE / MAPE / RMSE"]
        SEL["Авто-выбор лучшей<br/>по MAE"]
        CI["Доверительный интервал 95%<br/>σ√n из остатков"]
      end
      subgraph SCEN["⚡ Сценарии и события"]
        S1["4 типа шоков:<br/>выручка / задержка / расход / рост издержек"]
        S2["Ручные события<br/>(договоры, счета, налоги)"]
        S3["План-факт real-time<br/>подсветка отклонений"]
      end
      subgraph OUT["📊 Выход"]
        K1["KPI-карточки<br/>runway, burn, volatility"]
        K2["Risk-баннер<br/>дата/сумма/% разрыва"]
        K3["Графики: прогноз,<br/>категории"]
      end
      D1 & D2 & D3 & D4 --> N1
      N1 --> N2 --> N3 --> N4
      N3 --> M1 & M2 & M3 --> BT --> SEL --> CI
      CI --> S1 & S2
      N4 --> S2
      S1 & S2 --> S3
      CI & S3 --> K1 & K2 & K3
      style DS fill:#1e293b,stroke:#3b82f6,color:#fff
      style PROC fill:#1e293b,stroke:#06b6d4,color:#fff
      style FCST fill:#1e293b,stroke:#8b5cf6,color:#fff
      style SCEN fill:#1e293b,stroke:#f59e0b,color:#fff
      style OUT fill:#1e293b,stroke:#22c55e,color:#fff
    """
    st.components.v1.html(
        f"""
        <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
        <div style='background:#0e1626; padding:16px; border-radius:12px; border:1px solid #1f2a44;'>
          <div class='mermaid' style='background:transparent;'>{mermaid_diagram}</div>
        </div>
        <script>
          mermaid.initialize({{startOnLoad:true, theme:'dark',
            themeVariables:{{primaryColor:'#1e293b', primaryTextColor:'#e8ecf3',
              lineColor:'#3b82f6', secondaryColor:'#0e1626'}}}});
        </script>
        """,
        height=560,
    )

    st.markdown('<div class="fp-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="fp-section-title">Ограничения и важные оговорки</div>',
                unsafe_allow_html=True)
    st.markdown(
        """
        - **Регулярные обязательства**, которых нет в выписке, автоматически системе
          не известны — используйте вкладку «События».
        - **Стресс-сценарии** — это чувствительность, а не вероятностная модель.
          Вероятность разрыва оценивается по доверительному коридору MAE, а не из
          распределения экономических шоков.
        - **Прогноз ограничен горизонтом** и теряет точность с ростом горизонта
          (это видно по расширяющемуся коридору). Для горизонта >60 дней
          рекомендуется дополнять модель экспертными оценками.
        - **Безопасность.** Все вычисления происходят в рамках текущей сессии Streamlit;
          загруженные CSV-файлы не сохраняются на сервере между запусками. В production
          нужна аутентификация и маскирование счёт-фактуры и ИНН в логах.
        """
    )
