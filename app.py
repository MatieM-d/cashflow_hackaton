"""
ФинПоток — сервис прогнозирования денежных потоков и предотвращения
кассовых разрывов для малого бизнеса (Хакатон Осень 2026).

Версия с обновлённым UX/UI (Soft White & Violet FinTech Design),
исправленной категоризацией, поддержкой мульти-компаний, математически
корректной оценкой вероятности разрыва и системой рекомендаций.
"""
from __future__ import annotations

import hmac
import io
import math
import os
import zipfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import Holt


# ────────────────────────────────────────────────────────────────────────────
# 1. КОНФИГУРАЦИЯ СТРАНИЦЫ И БЕЛО-ФИОЛЕТОВАЯ ТЕМА (UX/UI)
# ────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ФинПоток — Прогноз денежных потоков",
    page_icon="💜",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Палитра White & Violet (мягкая, не вырвиглазная, высокий комфорт чтения)
BG_MAIN = "#F8F7FC"          # Основной очень светлый лавандово-белый фон
BG_SIDEBAR = "#F2EFFA"       # Мягкий пастельно-фиолетовый фон сайдбара
CARD_BG = "#FFFFFF"          # Чистый белый для карточек и графиков
CARD_BORDER = "#E6E0F4"      # Деликатная фиолетово-серая рамка
VIOLET_DARK = "#4C1D95"      # Глубокий фиолетовый для акцентов и заголовков
VIOLET_PRIMARY = "#6D28D9"   # Основной брендовый фиолетовый
VIOLET_ACCENT = "#7C3AED"    # Яркий, но мягкий фиолетовый для кнопок и линий
VIOLET_LIGHT = "#A78BFA"     # Светло-фиолетовый для второстепенных серий
VIOLET_SOFT_BG = "#EDE9FE"   # Пастельная подложка бейджей и табов
TEXT_PRIMARY = "#1E1B4B"     # Тёмный индиго-графит (не режет глаз как #000)
TEXT_SECONDARY = "#64748B"   # Спокойный серо-синий для подписей
COLOR_INFLOW = "#10B981"     # Мягкий изумрудный (поступления / стабильность)
COLOR_OUTFLOW = "#8B5CF6"    # Фиолетовый (плановые списания в графиках)
COLOR_WARN = "#D97706"       # Мягкий янтарный (внимание)
COLOR_DANGER = "#E11D48"     # Приглушённый малиново-красный (риск разрыва)

st.markdown(
    f"""
    <style>
    /* Глобальные цвета и типографика */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
        background-color: {BG_MAIN} !important;
        color: {TEXT_PRIMARY} !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    [data-testid="stHeader"] {{
        background: rgba(248, 247, 252, 0.85) !important;
        backdrop-filter: blur(8px);
    }}
    .block-container {{
        padding-top: 1.4rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1460px;
    }}
    h1, h2, h3, h4 {{
        color: {TEXT_PRIMARY} !important;
        letter-spacing: -0.02em;
    }}

    /* Сайдбар */
    [data-testid="stSidebar"], [data-testid="stSidebarContent"] {{
        background-color: {BG_SIDEBAR} !important;
        border-right: 1px solid {CARD_BORDER};
    }}
    [data-testid="stSidebar"] * {{
        color: {TEXT_PRIMARY};
    }}

    /* Кнопки Streamlit */
    .stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {{
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.18s ease !important;
        border: 1px solid {CARD_BORDER} !important;
        background-color: {CARD_BG} !important;
        color: {VIOLET_PRIMARY} !important;
        box-shadow: 0 2px 6px rgba(109, 40, 217, 0.05) !important;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        border-color: {VIOLET_ACCENT} !important;
        background-color: {VIOLET_SOFT_BG} !important;
        color: {VIOLET_DARK} !important;
        transform: translateY(-1px);
    }} 
    .stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] > button {{
        background: linear-gradient(135deg, {VIOLET_PRIMARY} 0%, {VIOLET_ACCENT} 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(109, 40, 217, 0.22) !important;
    }}
    .stButton > button[kind="primary"]:hover, [data-testid="stFormSubmitButton"] > button:hover {{
        background: linear-gradient(135deg, {VIOLET_DARK} 0%, {VIOLET_PRIMARY} 100%) !important;
        color: #FFFFFF !important;
    }}

    /* Вкладки (Tabs) — современный pill-дизайн */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background-color: {CARD_BG};
        padding: 6px 8px;
        border-radius: 14px;
        border: 1px solid {CARD_BORDER};
        box-shadow: 0 2px 10px rgba(109, 40, 217, 0.04);
    }}
    .stTabs [data-baseweb="tab"] {{
        padding: 10px 18px;
        border-radius: 10px;
        background: transparent;
        color: {TEXT_SECONDARY} !important;
        font-weight: 600;
        font-size: 0.92rem;
        border: none !important;
    }}
    .stTabs [aria-selected="true"] {{
        background: {VIOLET_SOFT_BG} !important;
        color: {VIOLET_PRIMARY} !important;
        font-weight: 700 !important;
    }}
    .stTabs [data-baseweb="tab-highlight"] {{
        display: none !important;
    }}

    /* Шапка приложения */
    .fp-header-card {{
        background: linear-gradient(135deg, #FFFFFF 0%, #F5F2FF 100%);
        border: 1px solid {CARD_BORDER};
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 16px;
        box-shadow: 0 4px 20px rgba(109, 40, 217, 0.05);
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 16px;
    }}
    .fp-badge {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 12px;
        border-radius: 999px;
        background: {VIOLET_SOFT_BG};
        color: {VIOLET_PRIMARY};
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid #DDD6FE;
    }}

    /* Адаптивные KPI-карточки (без обрезки текста троеточием) */
    /* Адаптивная сетка KPI-карточек */
    .fp-kpi-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 12px;
        margin: 14px 0 18px 0;
    }}
    .fp-kpi-card {{
        background: {CARD_BG};
        border: 1px solid {CARD_BORDER};
        border-radius: 14px;
        padding: 16px 14px;
        box-shadow: 0 4px 14px rgba(109, 40, 217, 0.04);
        flex: 1 1 185px;
        min-width: 180px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }}
    .fp-kpi-value {{
        color: {TEXT_PRIMARY};
        font-size: 1.25rem;
        font-weight: 800;
        line-height: 1.2;
        word-break: break-word;
    }}
    .fp-kpi-card {{
        background: {CARD_BG};
        border: 1px solid {CARD_BORDER};
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 4px 14px rgba(109, 40, 217, 0.04);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }}
    .fp-kpi-title {{
        color: {TEXT_SECONDARY};
        font-size: 0.76rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }}
    .fp-kpi-sub {{
        margin-top: 8px;
        font-size: 0.8rem;
        font-weight: 500;
        color: {TEXT_SECONDARY};
    }}
    .fp-kpi-sub.pos {{ color: #059669; }}
    .fp-kpi-sub.neg {{ color: {COLOR_DANGER}; }}
    .fp-kpi-sub.vio {{ color: {VIOLET_PRIMARY}; }}

    /* Белые контейнеры для графиков и секций */
    .fp-card {{
        background: {CARD_BG};
        border: 1px solid {CARD_BORDER};
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 14px;
        box-shadow: 0 4px 14px rgba(109, 40, 217, 0.04);
    }}
    .fp-section-title {{
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: {VIOLET_PRIMARY};
        font-weight: 700;
        margin: 14px 0 8px 0;
    }}
    .fp-divider {{
        height: 1px;
        background: linear-gradient(90deg, transparent, {CARD_BORDER}, transparent);
        margin: 14px 0;
    }}

    /* Статусные баннеры (мягкие пастельные тона) */
    .fp-banner {{
        border-radius: 14px;
        padding: 16px 20px;
        margin: 10px 0 16px 0;
        border: 1px solid;
        display: flex;
        align-items: flex-start;
        gap: 14px;
        line-height: 1.5;
    }}
    .fp-banner-success {{
        background: #F0FDF4;
        border-color: #BBF7D0;
        color: #14532D;
    }}
    .fp-banner-warn {{
        background: #FFFBEB;
        border-color: #FDE68A;
        color: #78350F;
    }}
    .fp-banner-danger {{
        background: #FFF1F2;
        border-color: #FECDD3;
        color: #881337;
    }}
    .fp-banner-info {{
        background: #F5F3FF;
        border-color: #DDD6FE;
        color: {VIOLET_DARK};
    }}
    .fp-pill {{
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        white-space: nowrap;
    }}
    .fp-pill-low {{ background: #DCFCE7; color: #15803D; }}
    .fp-pill-med {{ background: #FEF3C7; color: #B45309; }}
    .fp-pill-high {{ background: #FFE4E6; color: #BE123C; }}
    .fp-pill-vio {{ background: {VIOLET_SOFT_BG}; color: {VIOLET_PRIMARY}; }}

    /* Стандартные метрики внутри вкладок */
    [data-testid="stMetric"] {{
        background: {CARD_BG};
        padding: 14px 16px !important;
        border-radius: 12px;
        border: 1px solid {CARD_BORDER};
        box-shadow: 0 2px 8px rgba(109, 40, 217, 0.04);
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ────────────────────────────────────────────────────────────────────────────
# 2. КАТЕГОРИЗАЦИЯ ОПЕРАЦИЙ (исправленная логика с разделением приток/отток)
# ────────────────────────────────────────────────────────────────────────────
INFLOW_RULES: list[tuple[str, list[str]]] = [
    ("Возврат / корректировка", [
        "возврат", "возмещ", "корректировк", "переплат", "ошибочн",
    ]),
    ("Кредиты и финансирование", [
        "кредит", "займ", "заем", "субсид", "грант", "транш", "финансирован",
    ]),
    ("Финансовые доходы", [
        "процент на остаток", "депозит", "вклад", "дивиденд", "купон",
    ]),
    ("Продажи и выручка", [
        "оплата", "выручка", "эквайринг", "приход", "розничн", "покуп",
        "клиент", "счет", "счёт", "договор", "товар", "услуг", "работ",
        "упд", "акт", "поставк", "аванс", "реализац", "маркетплейс",
    ]),
]

OUTFLOW_RULES: list[tuple[str, list[str]]] = [
    ("Налоги и взносы", [
        "налог", "енп", "ндс", "усн", "взнос", "фнс", "страхован",
        "пфр", "сфр", "ндфл", "пени", "пеней", "бюджет",
    ]),
    ("Зарплата и персонал", [
        "зарплат", "заработ", "аванс сотруд", "оплата труда", "отпускн",
        "преми", "больничн", "реестр зачислений", "самозанят",
    ]),
    ("Аренда и помещение", [
        "аренд", "помещени", "коммунал", "электроэнерг", "эксплуатац",
    ]),
    ("Кредиты и лизинг", [
        "кредит", "лизинг", "процент по договору", "тело кредита", "погашен", "займ",
    ]),
    ("Логистика и транспорт", [
        "транспорт", "доставк", "топлив", "перевоз", "автомобил", "дорог",
        "гсм", "курьер", "экспедир",
    ]),
    ("Связь, IT и маркетинг", [
        "интернет", "связи", "связь", "телеком", "программ", "хостинг",
        "подписк", "сзв", "реклам", "маркетинг", "лиценз", "1с", " облак",
    ]),
    ("Комиссии и банк", [
        "комисси", "ведение счета", "ведение счёта", "обслуживание счета",
        "пко", "рко", "эквайринг комисс", "банковск",
    ]),
    ("Закупки и поставщики", [
        "поставк", "закуп", "сырь", "товар", "упд", "счет №", "счёт №",
        "материал", "инвентар", "оборудован", "комплектующ", "работ", "услуг",
    ]),
]


def _detect_inflow(direction_lower: pd.Series, df: pd.DataFrame) -> pd.Series:
    contains_inflow = direction_lower.str.contains(
        "поступ|приход|income|inflow|зачисл|прих|кредит|возврат|возмещ",
        regex=True, na=False,
    )
    contains_outflow = direction_lower.str.contains(
        "спис|расход|outflow|expense|платёж|платеж|оплата со счет|дебет|снят",
        regex=True, na=False,
    )
    exact_inflow = direction_lower.isin([
        "к", "кт", "c", "cr", "credit", "кр", "in", "+", "1",
        "приход", "поступление", "поступило", "зачислено",
    ])
    exact_outflow = direction_lower.isin([
        "д", "дт", "d", "dr", "debit", "out", "-", "0", "-1",
        "расход", "списание", "списано", "списан",
    ])
    inflow = (contains_inflow | exact_inflow) & ~exact_outflow
    outflow = contains_outflow | exact_outflow
    amount = df["amount"]
    fallback_pos = (amount > 0) & ~inflow & ~outflow
    return inflow | fallback_pos


def _detect_outflow(direction_lower: pd.Series, df: pd.DataFrame) -> pd.Series:
    contains_inflow = direction_lower.str.contains(
        "поступ|приход|income|inflow|зачисл|прих|кредит|возврат|возмещ",
        regex=True, na=False,
    )
    contains_outflow = direction_lower.str.contains(
        "спис|расход|outflow|expense|платёж|платеж|оплата со счет|дебет|снят",
        regex=True, na=False,
    )
    exact_inflow = direction_lower.isin([
        "к", "кт", "c", "cr", "credit", "кр", "in", "+", "1",
    ])
    exact_outflow = direction_lower.isin([
        "д", "дт", "d", "dr", "debit", "out", "-", "0", "-1",
    ])
    outflow = (contains_outflow | exact_outflow) & ~exact_inflow
    amount = df["amount"]
    fallback_neg = (amount < 0) & ~contains_inflow & ~contains_outflow & ~exact_inflow
    return outflow | fallback_neg


def categorize_vectorized(df: pd.DataFrame) -> pd.Series:
    """Точная векторизованная категоризация с разделением правил для притоков и оттоков."""
    purpose = df["payment_purpose"].fillna("").astype(str).str.lower()
    counter = df["counterparty_name"].fillna("").astype(str).str.lower()
    text = purpose + " " + counter
    direction = df["direction"].fillna("").astype(str).str.lower()

    is_inflow = _detect_inflow(direction, df)
    categories = pd.Series("Прочие", index=df.index, dtype="object")

    # 1. Классификация поступлений
    for name, words in INFLOW_RULES:
        mask = is_inflow & (categories == "Прочие") & text.str.contains("|".join(words), regex=True, na=False)
        categories = categories.where(~mask, name)

    # 2. Классификация списаний
    for name, words in OUTFLOW_RULES:
        mask = (~is_inflow) & (categories == "Прочие") & text.str.contains("|".join(words), regex=True, na=False)
        categories = categories.where(~mask, name)

    # 3. Оставшиеся операции
    categories = categories.where(~((categories == "Прочие") & is_inflow), "Прочие поступления")
    categories = categories.where(~((categories == "Прочие") & (~is_inflow)), "Прочие расходы")
    return categories


# ────────────────────────────────────────────────────────────────────────────
# 3. ЗАГРУЗКА И НОРМАЛИЗАЦИЯ ДАННЫХ (с поддержкой мульти-компаний)
# ────────────────────────────────────────────────────────────────────────────
COLUMN_ALIASES = {
    "operation_date": ["operation_date", "date", "дата", "дата операции", "дата платежа", "дата проводки"],
    "amount": ["amount", "сумма", "сумма операции", "сумма платежа", "сумма в валюте"],
    "direction": ["direction", "направление", "тип операции", "вид операции", "дт/кт"],
    "balance_eod": ["balance_eod", "balance", "остаток", "остаток на счёте", "остаток на счете", "остаток на конец дня"],
    "payment_purpose": ["payment_purpose", "назначение платежа", "назначение", "описание", "комментарий"],
    "counterparty_name": ["counterparty_name", "контрагент", "наименование контрагента", "получатель/плательщик"],
    "counterparty_inn": ["counterparty_inn", "inn", "инн", "инн контрагента"],
    "company_id": ["company_id", "company", "компания", "id компании", "организация"],
}


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
                df = pd.read_csv(io.BytesIO(raw), sep=sep, encoding=enc, low_memory=False)
                if len(df.columns) > 2:
                    return df
            except Exception as e:
                last_err = e
    raise ValueError(f"Не удалось прочитать CSV. Проверьте формат файла. Детали: {last_err}")


def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
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
            ". Нужны: дата операции, сумма и направление."
        )

    for col in ["payment_purpose", "counterparty_name", "counterparty_inn", "company_id"]:
        if col not in df.columns:
            df[col] = ""

    # Парсинг дат (поддержка ISO YYYY-MM-DD и РФ DD.MM.YYYY)
    sample_dates = df["operation_date"].dropna().astype(str).head(20)
    is_iso = sample_dates.str.match(r"^\d{4}-\d{2}-\d{2}").all() if len(sample_dates) else False
    df["operation_date"] = pd.to_datetime(
        df["operation_date"], errors="coerce", dayfirst=not is_iso
    )
    if df["operation_date"].isna().all():
        df["operation_date"] = pd.to_datetime(df["operation_date"], errors="coerce")

    df["amount"] = pd.to_numeric(
        df["amount"].astype(str)
        .str.replace(" ", "", regex=False)
        .str.replace("\u00a0", "", regex=False)
        .str.replace(",", ".", regex=False),
        errors="coerce",
    )

    df["direction"] = df["direction"].astype(str).str.strip().str.lower()
    inflow = _detect_inflow(df["direction"], df)
    outflow = _detect_outflow(df["direction"], df)

    if (inflow.sum() + outflow.sum()) == 0:
        inflow = df["amount"] > 0
        outflow = df["amount"] < 0

    df["amount"] = df["amount"].abs()
    df["signed_amount"] = np.select([inflow, outflow], [df["amount"], -df["amount"]], default=np.nan)

    df = df.dropna(subset=["operation_date", "amount", "signed_amount"])
    df = df[df["amount"] >= 0].copy()
    if df.empty:
        raise ValueError("После очистки не осталось корректных финансовых операций.")

    if "balance_eod" in df.columns:
        df["balance_eod"] = pd.to_numeric(
            df["balance_eod"].astype(str)
            .str.replace(" ", "", regex=False)
            .str.replace("\u00a0", "", regex=False)
            .str.replace(",", ".", regex=False),
            errors="coerce",
        )

    df["category"] = categorize_vectorized(df)
    df["day_of_week"] = df["operation_date"].dt.dayofweek
    df["month"] = df["operation_date"].dt.to_period("M").astype(str)
    return df.sort_values("operation_date").reset_index(drop=True)


def synthetic_company(company_id: str = "C001", days: int = 540) -> pd.DataFrame:
    """Генерирует реалистичную банковскую выписку малого бизнеса."""
    rng = np.random.default_rng(abs(hash(company_id)) % (2**32))
    start = pd.Timestamp.today().normalize() - pd.Timedelta(days=days)
    dates = pd.date_range(start, periods=days, freq="D")

    rows: list[dict] = []
    balance = rng.uniform(250_000, 620_000)
    base_daily_sales = rng.uniform(42_000, 95_000)
    trend = rng.uniform(0.0003, 0.0012)

    for i, d in enumerate(dates):
        weekday = d.dayofweek
        wd_factor = [0.88, 0.96, 1.08, 1.14, 1.06, 0.72, 0.52][weekday]
        seasonal = 1 + 0.16 * np.sin(2 * np.pi * i / 365)
        acq = max(0, base_daily_sales * wd_factor * seasonal * (1 + trend * i) * rng.normal(1.0, 0.16))
        if acq > 0:
            balance += acq
            rows.append(dict(
                operation_date=d, amount=round(acq, 2), direction="Поступление",
                balance_eod=round(balance, 2),
                payment_purpose="Оплата от покупателя по терминалу, эквайринг",
                counterparty_name="Банк-эквайер",
                counterparty_inn="7700000001", company_id=company_id,
            ))

        if weekday in (1, 2, 3) and rng.random() < 0.52:
            invoice = round(rng.uniform(110_000, 420_000), 2)
            balance += invoice
            rows.append(dict(
                operation_date=d, amount=invoice, direction="Поступление",
                balance_eod=round(balance, 2),
                payment_purpose=f"Оплата по счёту №{rng.integers(1000, 9999)} за поставку товара",
                counterparty_name=f"ООО «Клиент {rng.integers(1, 25)}»",
                counterparty_inn="7700000002", company_id=company_id,
            ))

        if d.day == 5:
            rent = round(rng.uniform(95_000, 145_000), 2)
            balance -= rent
            rows.append(dict(
                operation_date=d, amount=rent, direction="Списание",
                balance_eod=round(balance, 2),
                payment_purpose="Аренда помещения за месяц по договору",
                counterparty_name="ИП Арендодатель",
                counterparty_inn="7701111111", company_id=company_id,
            ))

        if d.day in (10, 25):
            salary = round(rng.uniform(190_000, 310_000), 2)
            balance -= salary
            rows.append(dict(
                operation_date=d, amount=salary, direction="Списание",
                balance_eod=round(balance, 2),
                payment_purpose="Выплата заработной платы сотрудникам по зарплатному проекту",
                counterparty_name="Сотрудники (зарплатный проект)",
                counterparty_inn="", company_id=company_id,
            ))

        if d.day == 28:
            enp = round(rng.uniform(110_000, 230_000), 2)
            balance -= enp
            rows.append(dict(
                operation_date=d, amount=enp, direction="Списание",
                balance_eod=round(balance, 2),
                payment_purpose="Единый налоговый платёж (ЕНП), ФНС России",
                counterparty_name="УФК (ФНС России)",
                counterparty_inn="7707329152", company_id=company_id,
            ))

        if d.day == 28 and d.month in (1, 4, 7, 10):
            nds = round(rng.uniform(45_000, 95_000), 2)
            balance -= nds
            rows.append(dict(
                operation_date=d, amount=nds, direction="Списание",
                balance_eod=round(balance, 2),
                payment_purpose="Квартальный платёж по НДС / УСН, ФНС",
                counterparty_name="УФК (ФНС России)",
                counterparty_inn="7707329152", company_id=company_id,
            ))

        if weekday in range(5) and rng.random() < 0.35:
            supply = round(rng.uniform(45_000, 210_000), 2)
            balance -= supply
            rows.append(dict(
                operation_date=d, amount=supply, direction="Списание",
                balance_eod=round(balance, 2),
                payment_purpose="Оплата по УПД за закупку сырья и материалов",
                counterparty_name=f"ООО «Поставщик {rng.integers(1, 18)}»",
                counterparty_inn="7702222222", company_id=company_id,
            ))

        if weekday in range(5) and rng.random() < 0.24:
            tr = round(rng.uniform(9_000, 27_000), 2)
            balance -= tr
            rows.append(dict(
                operation_date=d, amount=tr, direction="Списание",
                balance_eod=round(balance, 2),
                payment_purpose="Транспортные услуги и логистика товара",
                counterparty_name="ООО «Логист-Транс»",
                counterparty_inn="7703333333", company_id=company_id,
            ))

        if d.day == 1:
            comm = round(rng.uniform(2_900, 6_900), 2)
            balance -= comm
            rows.append(dict(
                operation_date=d, amount=comm, direction="Списание",
                balance_eod=round(balance, 2),
                payment_purpose="Комиссия банка за РКО и ведение счёта",
                counterparty_name="АО «Банк»",
                counterparty_inn="7704444444", company_id=company_id,
            ))

    return normalize_data(pd.DataFrame(rows))


# ────────────────────────────────────────────────────────────────────────────
# 4. МОДЕЛИ ПРОГНОЗИРОВАНИЯ, БЭКТЕСТ И ДОВЕРИТЕЛЬНЫЕ ИНТЕРВАЛЫ
# ────────────────────────────────────────────────────────────────────────────
def _daily_net_series(df: pd.DataFrame, lookback_days: int | None = None) -> pd.Series:
    daily = (
        df.groupby(df["operation_date"].dt.normalize())["signed_amount"]
        .sum()
        .asfreq("D", fill_value=0.0)
    )
    if lookback_days and len(daily) > lookback_days:
        daily = daily.iloc[-lookback_days:]
    return daily


def _weekday_avg_netflow(df: pd.DataFrame, lookback_days: int = 90) -> Tuple[pd.Series, float]:
    daily = _daily_net_series(df, lookback_days=lookback_days)
    if len(daily) == 0:
        return pd.Series(0.0, index=range(7)), 0.0
    if len(daily) > 10:
        lo, hi = daily.quantile([0.05, 0.95])
        clipped = daily.clip(lower=lo, upper=hi)
    else:
        clipped = daily
    dow = clipped.index.dayofweek
    weekday_net = clipped.groupby(dow).mean()
    overall = float(clipped.mean())
    return weekday_net, overall


def forecast_weekday_mean(
    df: pd.DataFrame, horizon: int, current_balance: float, lookback_days: int = 90
) -> pd.DataFrame:
    weekday_net, overall = _weekday_avg_netflow(df, lookback_days=lookback_days)
    last_date = df["operation_date"].max().normalize()
    future = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
    flows = [float(weekday_net.get(d.dayofweek, overall)) for d in future]
    cum = np.cumsum(flows) + current_balance
    return pd.DataFrame({"date": future, "flow": flows, "balance": cum, "model": "Weekday Mean"})


def forecast_holt(
    df: pd.DataFrame, horizon: int, current_balance: float, lookback_days: int = 90
) -> pd.DataFrame:
    daily_net = _daily_net_series(df, lookback_days=lookback_days)
    if len(daily_net) < 14:
        return forecast_weekday_mean(df, horizon, current_balance, lookback_days).assign(model="Holt Linear")
    try:
        model = Holt(daily_net, initialization_method="estimated", damped_trend=True)
        fit = model.fit(optimized=True)
        f = fit.forecast(horizon)
        q95 = float(daily_net.abs().quantile(0.95)) if len(daily_net) else np.inf
        flows = np.clip(f.values, -q95, q95)
    except Exception:
        return forecast_weekday_mean(df, horizon, current_balance, lookback_days).assign(model="Holt Linear")

    future = pd.date_range(daily_net.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    cum = np.cumsum(flows) + current_balance
    return pd.DataFrame({"date": future, "flow": flows, "balance": cum, "model": "Holt Linear"})


def forecast_linear_trend_weekday(
    df: pd.DataFrame, horizon: int, current_balance: float, lookback_days: int = 90
) -> pd.DataFrame:
    daily_net = _daily_net_series(df, lookback_days=lookback_days)
    if len(daily_net) < 21:
        return forecast_weekday_mean(df, horizon, current_balance, lookback_days).assign(model="Linear + Weekday")
    smoothed = daily_net.rolling(7, min_periods=1, center=True).mean()
    X = np.arange(len(smoothed)).reshape(-1, 1)
    try:
        reg = LinearRegression().fit(X, smoothed.values)
        slope = float(reg.coef_[0])
        intercept = float(reg.intercept_)
    except Exception:
        return forecast_weekday_mean(df, horizon, current_balance, lookback_days).assign(model="Linear + Weekday")

    weekday_net, overall = _weekday_avg_netflow(df, lookback_days=lookback_days)
    future = pd.date_range(daily_net.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    flows = []
    for i, d in enumerate(future, start=len(daily_net)):
        wd = float(weekday_net.get(d.dayofweek, overall))
        tr = intercept + slope * i
        flows.append(0.72 * wd + 0.28 * tr)
    cum = np.cumsum(flows) + current_balance
    return pd.DataFrame({"date": future, "flow": flows, "balance": cum, "model": "Linear + Weekday"})


def get_current_balance(df: pd.DataFrame, daily_net: pd.Series) -> float:
    if "balance_eod" in df.columns and df["balance_eod"].notna().any():
        valid = df[df["balance_eod"].notna()].sort_values("operation_date")
        if len(valid):
            return float(valid.iloc[-1]["balance_eod"])
    return float(daily_net.cumsum().iloc[-1]) if len(daily_net) else 0.0


def backtest_model(df: pd.DataFrame, horizon: int, model_fn, lookback_days: int = 90) -> dict:
    daily_net = _daily_net_series(df, lookback_days=None)
    bt_horizon = min(horizon, max(7, len(daily_net) // 4))
    if len(daily_net) <= bt_horizon + 14:
        flow_std = float(daily_net.std()) if len(daily_net) > 1 else 0.0
        return {"mae": np.nan, "mape": np.nan, "rmse": np.nan, "bias": np.nan, "flow_std": flow_std}

    cutoff_date = daily_net.index[-bt_horizon]
    train_df = df[df["operation_date"] < cutoff_date]
    if len(train_df) < 14:
        flow_std = float(daily_net.std())
        return {"mae": np.nan, "mape": np.nan, "rmse": np.nan, "bias": np.nan, "flow_std": flow_std}

    train_daily = _daily_net_series(train_df, lookback_days=None)
    start_balance = get_current_balance(train_df, train_daily)

    pred = model_fn(train_df, bt_horizon, start_balance, lookback_days=lookback_days)
    actual_flows = daily_net.iloc[-bt_horizon:]
    actual_balances = actual_flows.cumsum() + start_balance

    pred_indexed = pred.set_index("date").reindex(actual_balances.index)
    pred_balances = pred_indexed["balance"]
    pred_flows = pred_indexed["flow"]

    diff_bal = (pred_balances - actual_balances).dropna().values
    diff_flow = (pred_flows - actual_flows).dropna().values

    if len(diff_bal) == 0:
        return {"mae": np.nan, "mape": np.nan, "rmse": np.nan, "bias": np.nan, "flow_std": float(daily_net.std())}

    mae = float(np.mean(np.abs(diff_bal)))
    rmse = float(np.sqrt(np.mean(diff_bal ** 2)))
    bias = float(np.mean(diff_bal))

    abs_act = np.abs(actual_balances.values[:len(diff_bal)])
    denom = np.maximum(abs_act, np.mean(abs_act) * 0.1 + 1.0)
    mape = float(np.mean(np.abs(diff_bal) / denom) * 100.0)

    # Стандартное отклонение дневной ошибки потока (для корректного 95% ДИ)
    flow_std = float(np.std(diff_flow)) if len(diff_flow) > 1 else float(daily_net.std())
    return {"mae": mae, "mape": mape, "rmse": rmse, "bias": bias, "flow_std": flow_std}


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def compute_confidence_interval(forecast_df: pd.DataFrame, daily_flow_std: float) -> pd.DataFrame:
    """Строит 95% доверительный интервал и считает вероятность разрыва P(Balance_t < 0)."""
    out = forecast_df.copy()
    bal_col = "Прогноз остатка" if "Прогноз остатка" in out.columns else "balance"
    if np.isnan(daily_flow_std) or daily_flow_std <= 0:
        out["Нижняя граница 95%"] = out[bal_col]
        out["Верхняя граница 95%"] = out[bal_col]
        out["Вероятность разрыва, %"] = np.where(out[bal_col] < 0, 100.0, 0.0)
        return out

    steps = np.arange(1, len(out) + 1)
    sigma_t = daily_flow_std * np.sqrt(steps)
    out["Нижняя граница 95%"] = out[bal_col] - 1.96 * sigma_t
    out["Верхняя граница 95%"] = out[bal_col] + 1.96 * sigma_t

    probs = []
    for bal, sig in zip(out[bal_col].values, sigma_t):
        if sig <= 0:
            probs.append(100.0 if bal < 0 else 0.0)
        else:
            z = (0.0 - float(bal)) / float(sig)
            probs.append(round(_normal_cdf(z) * 100.0, 1))
    out["Вероятность разрыва, %"] = probs
    return out


def make_forecast_enhanced(
    df: pd.DataFrame,
    horizon: int = 30,
    lookback_days: int = 120,
    selected_model: str = "Авто (лучшая по MAE)",
    revenue_drop: float = 0.0,
    extra_expense: float = 0.0,
    delay_days: int = 0,
    fixed_cost_increase: float = 0.0,
    manual_events: list[dict] | None = None,
) -> Tuple[float, pd.DataFrame, pd.Series, dict]:
    daily_net = (
        df.groupby(df["operation_date"].dt.normalize())["signed_amount"].sum().sort_index()
    )
    current_balance = get_current_balance(df, daily_net)

    candidates = {
        "Weekday Mean": (
            forecast_weekday_mean,
            backtest_model(df, horizon, forecast_weekday_mean, lookback_days),
        ),
        "Holt Linear": (
            forecast_holt,
            backtest_model(df, horizon, forecast_holt, lookback_days),
        ),
        "Linear + Weekday": (
            forecast_linear_trend_weekday,
            backtest_model(df, horizon, forecast_linear_trend_weekday, lookback_days),
        ),
    }

    if selected_model in candidates:
        best_name = selected_model
    else:
        best_name = min(
            candidates.keys(),
            key=lambda k: candidates[k][1]["mae"] if not np.isnan(candidates[k][1]["mae"]) else np.inf,
        )
    best_fn, best_metrics = candidates[best_name]

    base = best_fn(df, horizon, current_balance, lookback_days=lookback_days)
    base = base.rename(columns={"date": "Дата", "flow": "Ожидаемый поток", "balance": "Прогноз остатка"})
    base["Плановые события"] = 0.0
    base["Описание событий"] = ""

    flows = base["Ожидаемый поток"].astype(float).copy()
    start_date = base["Дата"].iloc[0]

    # Расчёт среднедневных постоянных издержек для стресс-сценария
    fixed_daily_outflow = 0.0
    if fixed_cost_increase > 0:
        cutoff = df["operation_date"].max() - pd.Timedelta(days=lookback_days)
        recent = df[df["operation_date"] >= cutoff]
        fixed_mask = recent["category"].isin([
            "Аренда и помещение", "Зарплата и персонал", "Кредиты и лизинг",
            "Связь, IT и маркетинг", "Комиссии и банк",
        ])
        if fixed_mask.any():
            fixed_out = -recent.loc[fixed_mask, "signed_amount"].sum()
            span_days = max((recent["operation_date"].max() - recent["operation_date"].min()).days, 1)
            fixed_daily_outflow = fixed_out / span_days

    # Среднедневной приток и отток для реалистичного моделирования шоков
    recent_cut = df[df["operation_date"] >= (df["operation_date"].max() - pd.Timedelta(days=lookback_days))]
    span_d = max((recent_cut["operation_date"].max() - recent_cut["operation_date"].min()).days, 1)
    avg_inflow_day = float(recent_cut.loc[recent_cut["signed_amount"] > 0, "signed_amount"].sum()) / span_d

    delayed_pool = 0.0
    for i, d in enumerate(base["Дата"]):
        f = float(flows.iloc[i])
        if revenue_drop > 0:
            # Снижаем дневную выручку пропорционально среднему притоку
            drop_val = max(f, avg_inflow_day * 0.65) * (revenue_drop / 100.0)
            f -= drop_val
        if delay_days > 0:
            if i < delay_days:
                held = max(0.0, avg_inflow_day * 0.55)
                f -= held
                delayed_pool += held
            elif i == delay_days and delayed_pool > 0:
                f += delayed_pool * 0.85  # Клиент оплачивает задержанный счёт после отсрочки
        if fixed_cost_increase > 0 and fixed_daily_outflow > 0:
            f -= fixed_daily_outflow * (fixed_cost_increase / 100.0)
        if extra_expense > 0 and i == 2:
            f -= extra_expense
        flows.iloc[i] = f

    # Инкрементальное наложение плановых событий (договоры, счета, налоги, регулярные)
    if manual_events:
        for ev in manual_events:
            try:
                ev_date = pd.to_datetime(ev["date"], dayfirst=True).normalize()
                ev_amount = float(ev["amount"])
                ev_signed = -abs(ev_amount) if ev.get("type") == "out" else abs(ev_amount)
            except Exception:
                continue
            mask = base["Дата"] == ev_date
            if mask.any():
                idx = base.index[mask][0]
                flows.loc[idx] += ev_signed
                base.loc[idx, "Плановые события"] += ev_signed
                prev_desc = base.loc[idx, "Описание событий"]
                label = f"{ev.get('description', 'Событие')} ({ev.get('source', 'План')})"
                base.loc[idx, "Описание событий"] = f"{prev_desc}; {label}".strip("; ")

    base["Ожидаемый поток"] = flows
    base["Прогноз остатка"] = np.cumsum(flows.values) + current_balance

    daily_flow_std = best_metrics.get("flow_std", 0.0)
    base = compute_confidence_interval(base, daily_flow_std)

    model_meta = {
        "best_model": best_name,
        "candidates": candidates,
        "metrics": best_metrics,
        "current_balance": current_balance,
    }
    return current_balance, base, daily_net, model_meta


# ────────────────────────────────────────────────────────────────────────────
# 5. ФОРМАТИРОВАНИЕ, KPI И СИСТЕМА РЕКОМЕНДАЦИЙ (Задача из Цели PDF)
# ────────────────────────────────────────────────────────────────────────────
def fmt_money(x: float | None, suffix: str = " ₽") -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    sign = "−" if x < 0 else ""
    return f"{sign}{abs(x):,.0f}".replace(",", " ") + suffix


def fmt_compact(x: float | None, suffix: str = " ₽") -> str:
    """Компактная запись для крупных сумм, чтобы карточки никогда не обрезались."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    sign = "−" if x < 0 else ""
    ax = abs(x)
    if ax >= 1_000_000_000:
        return f"{sign}{ax / 1_000_000_000:.2f} млрд{suffix}".replace(".", ",")
    if ax >= 10_000_000:
        return f"{sign}{ax / 1_000_000:.2f} млн{suffix}".replace(".", ",")
    return fmt_money(x, suffix)


def fmt_pct(x: float | None) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.1f} %"


def compute_kpis(df: pd.DataFrame, current_balance: float, base_forecast: pd.DataFrame) -> dict:
    inflows = float(df.loc[df["signed_amount"] > 0, "signed_amount"].sum())
    outflows = float(-df.loc[df["signed_amount"] < 0, "signed_amount"].sum())
    history_days = max((df["operation_date"].max() - df["operation_date"].min()).days, 1)
    avg_daily_inflow = inflows / history_days
    avg_daily_outflow = outflows / history_days
    net_burn = avg_daily_outflow - avg_daily_inflow

    min_balance = float(base_forecast["Прогноз остатка"].min())
    min_idx = base_forecast["Прогноз остатка"].idxmin()
    min_date = base_forecast.loc[min_idx, "Дата"]
    days_to_min = int((min_date - df["operation_date"].max()).days)

    runway = int(max(0, current_balance) / net_burn) if net_burn > 0 else None

    # Максимальная дневная вероятность ухода в минус на горизонте прогноза
    risk_pct = float(base_forecast["Вероятность разрыва, %"].max())

    deficit_mask = base_forecast["Прогноз остатка"] < 0
    if deficit_mask.any():
        first_neg_row = base_forecast.loc[deficit_mask].iloc[0]
        risk_date = first_neg_row["Дата"]
        risk_amount = abs(float(min_balance))
        gap_days_count = int(deficit_mask.sum())
    else:
        risk_date = None
        risk_amount = 0.0
        gap_days_count = 0

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
        "gap_days_count": gap_days_count,
        "volatility": volatility,
        "history_days": history_days,
    }


def generate_recommendations(
    df: pd.DataFrame, base_forecast: pd.DataFrame, kpis: dict
) -> list[str]:
    """Формирует понятные предпринимателю объяснения и рекомендации (Требование Цели PDF)."""
    recs: list[str] = []
    safety_buffer = kpis["avg_daily_outflow"] * 7  # Недельный запас прочности

    # Находим дни наибольших списаний в прогнозе
    worst_days = base_forecast.nsmallest(2, "Ожидаемый поток")
    reasons = []
    for _, row in worst_days.iterrows():
        if row["Ожидаемый поток"] < 0:
            desc = row["Описание событий"] if row["Описание событий"] else "плановые списания"
            reasons.append(f"{row['Дата']:%d.%m.%Y} ({fmt_money(row['Ожидаемый поток'])}, {desc})")

    if kpis["risk_date"] is not None:
        need_sum = kpis["risk_amount"] + safety_buffer
        recs.append(
            f"🚨 <b>Причина кассового разрыва:</b> к {kpis['risk_date']:%d.%m.%Y} накопленные списания превысят "
            f"остаток на счёте. Пиковые оттоки ожидаются: {', '.join(reasons) if reasons else 'по регулярным расходам'}."
        )
        recs.append(
            f"💡 <b>Действие 1 (Дебиторка / предоплата):</b> ускорьте поступления от клиентов или запросите аванс "
            f"на сумму минимум <b>{fmt_money(kpis['risk_amount'])}</b> (оптимально с буфером — <b>{fmt_money(need_sum)}</b>) до {kpis['risk_date']:%d.%m.%Y}."
        )
        recs.append(
            "💡 <b>Действие 2 (Перенос некритичных оплат):</b> согласуйте с поставщиками отсрочку платежей по категории "
            "«Закупки и поставщики» на 7–14 дней после прохождения даты налоговых и зарплатных выплат (25–28 числа)."
        )
        recs.append(
            f"💡 <b>Действие 3 (Кредитное плечо):</b> при невозможности сдвинуть платежи заранее оформите овердрафт "
            f"или возобновляемую кредитную линию на <b>{fmt_money(kpis['risk_amount'] * 1.15)}</b> на срок {max(7, kpis['gap_days_count'] + 5)} дн."
        )
    elif kpis["min_balance"] < safety_buffer or kpis["risk_pct"] >= 25.0:
        shortfall = max(0.0, safety_buffer - kpis["min_balance"])
        recs.append(
            f"⚠️ <b>Снижение подушки безопасности:</b> около {kpis['min_date']:%d.%m.%Y} остаток опустится до "
            f"<b>{fmt_money(kpis['min_balance'])}</b>, что меньше 7-дневного запаса расходов ({fmt_money(safety_buffer)})."
        )
        if reasons:
            recs.append(f"📌 <b>Факторы нагрузки:</b> крупнейшие ожидаемые оттоки — {', '.join(reasons)}.")
        recs.append(
            f"💡 <b>Рекомендация:</b> зарезервируйте на счёте дополнительно <b>{fmt_money(shortfall)}</b> к 25–28 числу "
            "под зарплату и ЕНП и не назначайте крупных разовых закупок на эту неделю."
        )
    else:
        surplus = max(0.0, kpis["min_balance"] - safety_buffer * 1.5)
        recs.append(
            f"✅ <b>Финансовая устойчивость:</b> на всём горизонте прогноза сохраняется профицит ликвидности. "
            f"Минимальный остаток ({fmt_money(kpis['min_balance'])} на {kpis['min_date']:%d.%m.%Y}) уверенно перекрывает обязательные платежи."
        )
        if surplus > 100_000:
            recs.append(
                f"💜 <b>Управление свободным остатком:</b> среднесрочный неснижаемый остаток ~<b>{fmt_money(surplus)}</b> "
                "можно разместить на бизнес-овернайт или краткосрочный депозит без риска для операционных платежей."
            )
    return recs


# ────────────────────────────────────────────────────────────────────────────
# 6. АВТО-ДЕТЕКТ РЕГУЛЯРНЫХ ПЛАТЕЖЕЙ И НАЛОГОВЫЙ КАЛЕНДАРЬ
# ────────────────────────────────────────────────────────────────────────────
def detect_recurring_payments(df: pd.DataFrame, min_occurrences: int = 3) -> list[dict]:
    if len(df) < min_occurrences:
        return []

    recurring: list[dict] = []
    df_s = df.sort_values("operation_date").copy()
    df_s["counterparty_key"] = (
        df_s["counterparty_name"].fillna("").astype(str).str.strip()
        .replace("", np.nan)
        .fillna(df_s["category"])
        .str.slice(0, 42)
    )
    df_s["sign"] = np.sign(df_s["signed_amount"])
    today = df_s["operation_date"].max().normalize()

    for (counterparty, sign), group in df_s.groupby(["counterparty_key", "sign"]):
        if not counterparty or len(group) < min_occurrences:
            continue
        # Агрегируем операции одного дня с одним контрагентом
        daily_grp = group.groupby(group["operation_date"].dt.normalize())["amount"].sum()
        if len(daily_grp) < min_occurrences:
            continue
        dates = pd.Series(daily_grp.index).sort_values()
        diffs = dates.diff().dropna()
        median_interval = diffs.median()

        valid_cycle = (
            (pd.Timedelta(days=12) <= median_interval <= pd.Timedelta(days=18))  # Аванс/зарплата (2 раза в мес)
            or (pd.Timedelta(days=24) <= median_interval <= pd.Timedelta(days=36))  # Ежемесячный
            or (pd.Timedelta(days=80) <= median_interval <= pd.Timedelta(days=100))  # Квартальный
        )
        if not valid_cycle:
            continue

        amounts = daily_grp.abs()
        median_amount = float(amounts.median())
        if median_amount <= 0:
            continue
        cv = float(amounts.std() / median_amount) if len(amounts) > 1 else 0.0
        if cv > 0.38:
            continue

        last_date = dates.iloc[-1]
        next_date = last_date + median_interval
        while next_date <= today:
            next_date += median_interval

        if (next_date - last_date).days > 120:
            continue

        recurring.append({
            "counterparty": str(counterparty),
            "direction": "Поступление" if sign > 0 else "Списание",
            "interval_days": int(median_interval.days),
            "median_amount": median_amount,
            "last_date": last_date,
            "next_date": next_date,
            "occurrences": int(len(daily_grp)),
            "cv": cv,
            "category": group["category"].mode().iloc[0] if not group["category"].empty else "Прочее",
        })

    recurring.sort(key=lambda r: r["median_amount"], reverse=True)
    return recurring


def default_tax_events(df: pd.DataFrame, horizon_days: int) -> list[dict]:
    """Формирует налоговый календарь с адаптацией сумм под масштаб выбранной компании."""
    start_date = df["operation_date"].max().normalize()
    tax_rows = df[df["category"] == "Налоги и взносы"]
    if len(tax_rows) >= 2:
        typical_enp = float(round(tax_rows["amount"].median(), -3))
    else:
        monthly_out = float(-df.loc[df["signed_amount"] < 0, "signed_amount"].sum()) / max(
            (df["operation_date"].max() - df["operation_date"].min()).days / 30.0, 1.0
        )
        typical_enp = float(round(max(25_000.0, monthly_out * 0.12), -3))
    typical_nds = float(round(typical_enp * 0.45, -3))

    events: list[dict] = []
    cur = start_date + pd.Timedelta(days=1)
    end = start_date + pd.Timedelta(days=horizon_days)
    while cur <= end:
        if cur.day == 28:
            events.append({
                "date": cur.strftime("%d.%m.%Y"),
                "amount": typical_enp,
                "type": "out",
                "description": "Единый налоговый платёж (ЕНП до 28-го)",
                "source": "Налоговый календарь ФНС",
            })
            if cur.month in (1, 4, 7, 10):
                events.append({
                    "date": cur.strftime("%d.%m.%Y"),
                    "amount": typical_nds,
                    "type": "out",
                    "description": "Квартальный платёж НДС / аванс УСН",
                    "source": "Налоговый календарь ФНС",
                })
        cur += pd.Timedelta(days=1)
    return events


# ────────────────────────────────────────────────────────────────────────────
# 7. СТИЛИЗАЦИЯ ГРАФИКОВ PLOTLY (Бело-фиолетовая тема)
# ────────────────────────────────────────────────────────────────────────────
def style_plotly_fig(fig: go.Figure, height: int = 410) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=12, r=12, t=34, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        font=dict(color=TEXT_PRIMARY, family="Inter, system-ui, sans-serif", size=13),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            bgcolor="rgba(255,255,255,0.85)",
        ),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#FFFFFF", font_color=TEXT_PRIMARY, bordercolor=CARD_BORDER),
        xaxis=dict(
            gridcolor="#F1EEF9",
            linecolor=CARD_BORDER,
            zerolinecolor=CARD_BORDER,
        ),
        yaxis=dict(
            gridcolor="#F1EEF9",
            linecolor=CARD_BORDER,
            zerolinecolor="#DDD6FE",
            tickformat=",.0f",
        ),
    )
    return fig


# ────────────────────────────────────────────────────────────────────────────
# 8. АУТЕНТИФИКАЦИЯ (Техническое требование №7 — Безопасность)
# ────────────────────────────────────────────────────────────────────────────
_AUTH_PASSWORD = os.environ.get("FINPOTOK_PASSWORD", "").strip()


def _auth_gate() -> bool:
    if not _AUTH_PASSWORD:
        return True
    if st.session_state.get("auth_ok", False):
        return True
    st.markdown(
        f"""
        <div class='fp-card' style='max-width:440px;margin:60px auto;padding:30px;'>
          <h2 style='margin:0 0 8px;color:{VIOLET_PRIMARY};'>🔐 ФинПоток · Защищённый вход</h2>
          <div style='color:{TEXT_SECONDARY};font-size:0.92rem;'>
            Доступ к финансовым данным компании защищён паролем.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    pwd = st.text_input("Пароль доступа", type="password", placeholder="••••••••")
    if st.button("Войти в систему", type="primary"):
        if hmac.compare_digest(pwd, _AUTH_PASSWORD):
            st.session_state["auth_ok"] = True
            st.rerun()
        else:
            st.error("Неверный пароль.")
    st.stop()
    return False


_auth_gate()


# ────────────────────────────────────────────────────────────────────────────
# 9. БОКОВАЯ ПАНЕЛЬ (Логичное управление источником, компанией и параметрами)
# ────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=300)
def load_raw_dataset(mode: str, archive_company: str | None, file_bytes: bytes | None) -> pd.DataFrame:
    if mode == "Загрузить выписку CSV":
        if file_bytes is None:
            raise ValueError("Загрузите CSV-файл банковской выписки в боковой панели.")
        return normalize_data(read_csv_safely(file_bytes))
    if mode == "Демо из архива кейса":
        dataset_path = Path(__file__).parent / "cashflow_dataset_participants.zip"
        if not dataset_path.exists():
            raise FileNotFoundError(
                "Архив cashflow_dataset_participants.zip не найден рядом с app.py. "
                "Выберите «Демо-датасет (синтетика)» или загрузите свой CSV."
            )
        with zipfile.ZipFile(dataset_path) as archive:
            csv_files = [n for n in archive.namelist() if n.endswith(".csv") and "all_operations" not in n]
            target = f"statements/{archive_company}.csv"
            if target not in archive.namelist() and csv_files:
                target = csv_files[0]
            raw = archive.read(target)
        return normalize_data(read_csv_safely(raw))
    return synthetic_company(archive_company or "C001")


with st.sidebar:
    st.markdown(
        f"""
        <div style='display:flex;align-items:center;gap:10px;margin-bottom:4px;'>
          <div style='width:36px;height:36px;border-radius:10px;background:linear-gradient(135deg,{VIOLET_PRIMARY},{VIOLET_ACCENT});
                      display:flex;align-items:center;justify-content:center;color:#fff;font-weight:800;font-size:1.1rem;'>
            ФП
          </div>
          <div>
            <div style='font-weight:800;font-size:1.15rem;color:{TEXT_PRIMARY};line-height:1.1;'>ФинПоток</div>
            <div style='font-size:0.76rem;color:{VIOLET_PRIMARY};font-weight:600;'>Cash-Flow Forecaster</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("🔒 Защищённый режим" if _AUTH_PASSWORD else "🔓 Публичный демо-режим")
    st.markdown('<div class="fp-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="fp-section-title">1. Источник данных</div>', unsafe_allow_html=True)
    source = st.radio(
        "Источник",
        ["Демо-датасет (синтетика)", "Демо из архива кейса", "Загрузить выписку CSV"],
        index=0,
        label_visibility="collapsed",
    )

    uploaded = None
    selected_company = "C001"
    if source == "Загрузить выписку CSV":
        uploaded = st.file_uploader("Файл банковской выписки (CSV)", type=["csv"])
    elif source == "Демо из архива кейса":
        selected_company = st.selectbox(
            "Компания из архива",
            ["C001", "C002", "C008", "C009", "C021", "C040"],
            index=0,
        )
    else:
        comp_choice = st.selectbox(
            "Профиль бизнеса",
            [
                "C001 — Розничная торговля",
                "C002 — Сфера услуг (B2B)",
                "C008 — Оптовые поставки",
                "C009 — Производство",
                "C021 — IT и разработка",
                "C040 — Кафе и HoReCa",
            ],
            index=0,
        )
        selected_company = comp_choice.split(" ")[0]

    # Загружаем датасет, чтобы проверить наличие нескольких компаний внутри CSV
    try:
        file_bytes = uploaded.getvalue() if uploaded is not None else None
        raw_df = load_raw_dataset(source, selected_company, file_bytes)
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Фильтр по company_id (критично для all_operations.csv, где десятки компаний!)
    unique_companies = [
        str(c) for c in raw_df["company_id"].dropna().unique() if str(c).strip() != ""
    ]
    if len(unique_companies) > 1:
        st.markdown('<div class="fp-section-title">Фильтр компании в CSV</div>', unsafe_allow_html=True)
        chosen_cid = st.selectbox(
            "Выберите компанию из выписки",
            sorted(unique_companies) + ["Все компании (сводно)"],
            index=0,
            help="В загруженном файле обнаружено несколько компаний. Выберите одну для точного расчёта остатка.",
        )
        if chosen_cid != "Все компании (сводно)":
            df = raw_df[raw_df["company_id"].astype(str) == chosen_cid].copy().reset_index(drop=True)
            active_company_label = chosen_cid
        else:
            df = raw_df
            active_company_label = f"Все ({len(unique_companies)} комп.)"
    else:
        df = raw_df
        active_company_label = unique_companies[0] if unique_companies else selected_company

    st.markdown('<div class="fp-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="fp-section-title">2. Параметры прогноза</div>', unsafe_allow_html=True)
    horizon = st.slider("Горизонт прогноза, дней", 14, 120, 30, step=1)
    lookback = st.slider(
        "Окно обучения, дней", 30, 365, 120, step=10,
        help="Глубина исторических данных для обучения моделей и расчёта средних потоков.",
    )
    model_choice = st.selectbox(
        "Модель прогнозирования",
        ["Авто (лучшая по MAE)", "Weekday Mean", "Holt Linear", "Linear + Weekday"],
        index=0,
    )
    include_events_toggle = st.toggle(
        "Учитывать налоги, счета и плановые события",
        value=True,
        help="Включает события из вкладки «Планировщик» прямо в основной прогноз и оценку риска.",
    )


# ────────────────────────────────────────────────────────────────────────────
# 10. ИНИЦИАЛИЗАЦИЯ СОБЫТИЙ И РАСЧЁТ ОСНОВНОГО ПРОГНОЗА
# ────────────────────────────────────────────────────────────────────────────
context_key = f"{source}_{active_company_label}_{len(df)}"
if st.session_state.get("_last_context_key") != context_key:
    st.session_state["_last_context_key"] = context_key
    st.session_state["manual_events_state"] = default_tax_events(df, horizon)
    st.session_state["pending_invoices"] = []

active_events = st.session_state.get("manual_events_state", []) if include_events_toggle else []

current_balance, base_forecast, daily_actual, model_meta = make_forecast_enhanced(
    df,
    horizon=horizon,
    lookback_days=lookback,
    selected_model=model_choice,
    manual_events=active_events,
)
kpis = compute_kpis(df, current_balance, base_forecast)
metrics = model_meta["metrics"]
best_model = model_meta["best_model"]
recommendations = generate_recommendations(df, base_forecast, kpis)

# Кнопка быстрого скачивания прогноза в сайдбаре
with st.sidebar:
    st.markdown('<div class="fp-divider"></div>', unsafe_allow_html=True)
    export_df = base_forecast.copy()
    export_df["Дата"] = export_df["Дата"].dt.strftime("%d.%m.%Y")
    st.download_button(
        "📥 Скачать прогноз (CSV)",
        export_df.round(2).to_csv(index=False).encode("utf-8-sig"),
        file_name=f"finpotok_forecast_{active_company_label}.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ────────────────────────────────────────────────────────────────────────────
# 11. ШАПКА, БАННЕР РИСКА И НЕОБРЕЗАЕМЫЕ KPI-КАРТОЧКИ
# ────────────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class='fp-header-card'>
      <div>
        <div style='display:flex;align-items:center;gap:10px;flex-wrap:wrap;'>
          <h1 style='margin:0;font-size:1.75rem;font-weight:800;'>ФинПоток</h1>
          <span class='fp-badge'>💜 Бизнес: {active_company_label}</span>
          <span class='fp-badge'>🤖 Модель: {best_model}</span>
          <span class='fp-badge'>📅 Горизонт: {horizon} дн.</span>
        </div>
        <div style='color:{TEXT_SECONDARY};font-size:0.92rem;margin-top:6px;'>
          Интеллектуальное прогнозирование движения денежных средств, контроль кассовых разрывов и сценарное планирование
        </div>
      </div>
      <div style='text-align:right;font-size:0.84rem;color:{TEXT_SECONDARY};'>
        <div><b style='color:{TEXT_PRIMARY};'>Период выписки:</b> {df['operation_date'].min():%d.%m.%Y} — {df['operation_date'].max():%d.%m.%Y}</div>
        <div><b style='color:{TEXT_PRIMARY};'>Операций обработано:</b> {len(df):,} ({kpis['history_days']} дн.)</div>
      </div>
    </div>
    """.replace(",", " "),
    unsafe_allow_html=True,
)

# Баннер статуса кассового разрыва
if kpis["risk_date"] is not None:
    st.markdown(
        f"""
        <div class='fp-banner fp-banner-danger'>
          <span class='fp-pill fp-pill-high'>РИСК РАЗРЫВА</span>
          <div>
            <b>Внимание! Прогнозируется кассовый разрыв {kpis['risk_date']:%d.%m.%Y}</b>
            (через {(kpis['risk_date'] - df['operation_date'].max()).days} дн.).
            Максимальная нехватка средств составит <b>{fmt_money(kpis['risk_amount'])}</b>
            (дней в минусе: {kpis['gap_days_count']}).
            Расчётная вероятность дефицита с учётом волатильности: <b>{fmt_pct(kpis['risk_pct'])}</b>.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
elif kpis["min_balance"] < kpis["avg_daily_outflow"] * 7 or kpis["risk_pct"] >= 25.0:
    st.markdown(
        f"""
        <div class='fp-banner fp-banner-warn'>
          <span class='fp-pill fp-pill-med'>ВНИМАНИЕ</span>
          <div>
            Кассовый разрыв в базовом сценарии не ожидается, но <b>{kpis['min_date']:%d.%m.%Y}</b>
            остаток снизится до <b>{fmt_money(kpis['min_balance'])}</b>.
            Вероятность ухода в минус при задержке оплат: <b>{fmt_pct(kpis['risk_pct'])}</b>.
            Рекомендуется проверить вкладку «Сценарии».
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"""
        <div class='fp-banner fp-banner-success'>
          <span class='fp-pill fp-pill-low'>СТАБИЛЬНО</span>
          <div>
            <b>Финансовая ситуация устойчива:</b> кассовых разрывов на горизонте {horizon} дней не прогнозируется.
            Минимальный ожидаемый остаток: <b>{fmt_money(kpis['min_balance'])}</b> ({kpis['min_date']:%d.%m.%Y}).
            Вероятность возникновения дефицита: <b>{fmt_pct(kpis['risk_pct'])}</b>.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# 5 адаптивных KPI-карточек (никогда не обрезаются троеточием)
net_flow_daily = kpis["avg_daily_inflow"] - kpis["avg_daily_outflow"]
net_cls = "pos" if net_flow_daily >= 0 else "neg"
min_cls = "neg" if kpis["min_balance"] < 0 else "vio"
runway_str = f"{kpis['runway']} дн." if kpis["runway"] is not None else "Профицит (∞)"

st.markdown(
    f"""
    <div class='fp-kpi-grid'>
      <div class='fp-kpi-card'>
        <div class='fp-kpi-title'>Текущий остаток на счёте</div>
        <div class='fp-kpi-value'>{fmt_compact(kpis['current_balance'])}</div>
        <div class='fp-kpi-sub vio'>Точно: {fmt_money(kpis['current_balance'])}</div>
      </div>
      <div class='fp-kpi-card'>
        <div class='fp-kpi-title'>Средний приток / день</div>
        <div class='fp-kpi-value'>{fmt_compact(kpis['avg_daily_inflow'])}</div>
        <div class='fp-kpi-sub pos'>Всего: {fmt_compact(kpis['inflows'])} за {kpis['history_days']} дн.</div>
      </div>
      <div class='fp-kpi-card'>
        <div class='fp-kpi-title'>Средний отток / день</div>
        <div class='fp-kpi-value'>{fmt_compact(kpis['avg_daily_outflow'])}</div>
        <div class='fp-kpi-sub {net_cls}'>Чистый поток: {fmt_compact(net_flow_daily)}/день</div>
      </div>
      <div class='fp-kpi-card'>
        <div class='fp-kpi-title'>Минимум в прогнозе ({horizon} дн.)</div>
        <div class='fp-kpi-value'>{fmt_compact(kpis['min_balance'])}</div>
        <div class='fp-kpi-sub {min_cls}'>Дата: {kpis['min_date']:%d.%m.%Y} (через {kpis['days_to_min']} дн.)</div>
      </div>
      <div class='fp-kpi-card'>
        <div class='fp-kpi-title'>Запас прочности (Runway)</div>
        <div class='fp-kpi-value'>{runway_str}</div>
        <div class='fp-kpi-sub'>Волатильность: ±{fmt_compact(kpis['volatility'])}/дн.</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ────────────────────────────────────────────────────────────────────────────
# 12. ОСНОВНЫЕ ВКЛАДКИ ИНТЕРФЕЙСА (5 логичных рабочих пространств)
# ────────────────────────────────────────────────────────────────────────────
tab_dashboard, tab_scenarios, tab_planner, tab_categories, tab_quality = st.tabs([
    "📊 Дашборд и Прогноз",
    "⚡ Сценарии «Что если?»",
    "📅 Планировщик (Счета, Налоги, Регулярные)",
    "🗂 Категории и Выписка",
    "🎯 Качество, План-Факт и Архитектура",
])


# ═══════════════════════════════════════════════════════════════════════════
# ВКЛАДКА 1: ДАШБОРД И ПРОГНОЗ
# ═══════════════════════════════════════════════════════════════════════════
with tab_dashboard:
    # Панель быстрых переключателей прямо над графиком
    ctrl_c1, ctrl_c2, ctrl_c3 = st.columns([2, 2, 2])
    with ctrl_c1:
        show_history_days = st.selectbox(
            "Показать историю на графике",
            [30, 60, 90, 180, "Вся история", "Только прогноз"],
            index=1,
        )
    with ctrl_c2:
        show_ci = st.checkbox("Показать 95% доверительный коридор", value=True)
    with ctrl_c3:
        show_event_markers = st.checkbox("Отметить налоги и плановые события", value=True)

    fig = go.Figure()

    # 1. Доверительный интервал (мягкая фиолетовая заливка)
    if show_ci:
        fig.add_trace(go.Scatter(
            x=list(base_forecast["Дата"]) + list(base_forecast["Дата"][::-1]),
            y=list(base_forecast["Верхняя граница 95%"]) + list(base_forecast["Нижняя граница 95%"][::-1]),
            fill="toself",
            fillcolor="rgba(124, 58, 237, 0.11)",
            line=dict(color="rgba(124, 58, 237, 0.2)", width=1),
            name="Доверительный коридор 95%",
            hoverinfo="skip",
        ))

    # 2. Исторический остаток
    if show_history_days != "Только прогноз" and "balance_eod" in df.columns and df["balance_eod"].notna().any():
        hist = (
            df.dropna(subset=["balance_eod"])
            .sort_values("operation_date")
            .groupby(df["operation_date"].dt.normalize())["balance_eod"]
            .last()
            .reset_index()
        )
        if isinstance(show_history_days, int):
            cutoff_h = df["operation_date"].max() - pd.Timedelta(days=show_history_days)
            hist = hist[hist["operation_date"] >= cutoff_h]
        fig.add_trace(go.Scatter(
            x=hist["operation_date"],
            y=hist["balance_eod"],
            mode="lines",
            name="Фактический остаток",
            line=dict(color="#94A3B8", width=2.2),
        ))

    # 3. Линия прогноза остатка (фирменный фиолетовый)
    fig.add_trace(go.Scatter(
        x=base_forecast["Дата"],
        y=base_forecast["Прогноз остатка"],
        mode="lines+markers",
        name=f"Прогноз остатка ({best_model})",
        line=dict(color=VIOLET_PRIMARY, width=3.2),
        marker=dict(size=5, color=VIOLET_ACCENT),
    ))

    # 4. Точка текущего баланса
    fig.add_trace(go.Scatter(
        x=[df["operation_date"].max().normalize()],
        y=[current_balance],
        mode="markers",
        name="Текущий остаток",
        marker=dict(size=11, color=COLOR_INFLOW, line=dict(width=2, color="#FFFFFF")),
    ))

    # 5. Маркеры плановых событий и налогов
    if show_event_markers:
        ev_rows = base_forecast[base_forecast["Плановые события"] != 0]
        if not ev_rows.empty:
            fig.add_trace(go.Scatter(
                x=ev_rows["Дата"],
                y=ev_rows["Прогноз остатка"],
                mode="markers",
                name="Налоги / Плановые события",
                marker=dict(
                    size=11,
                    symbol="diamond",
                    color=[COLOR_INFLOW if v > 0 else COLOR_DANGER for v in ev_rows["Плановые события"]],
                    line=dict(width=1.5, color="#FFFFFF"),
                ),
                hovertext=[
                    f"{d}: {fmt_money(v)}" for d, v in zip(ev_rows["Описание событий"], ev_rows["Плановые события"])
                ],
            ))

    fig.add_hline(
        y=0, line_dash="dash", line_color=COLOR_DANGER, line_width=1.6,
        annotation_text="Граница кассового разрыва (0 ₽)", annotation_position="top left",
    )
    st.plotly_chart(style_plotly_fig(fig, height=430), use_container_width=True)

    # Блок рекомендаций для предпринимателя (Цель из PDF кейса)
    st.markdown('<div class="fp-section-title">Рекомендации для принятия решений</div>', unsafe_allow_html=True)
    rec_html = "".join(f"<li style='margin-bottom:6px;'>{r}</li>" for r in recommendations)
    st.markdown(
        f"""
        <div class='fp-card' style='background:#FAF8FF;border-color:#DDD6FE;'>
          <ul style='margin:0;padding-left:18px;color:{TEXT_PRIMARY};font-size:0.93rem;line-height:1.55;'>
            {rec_html}
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Два графика рядом: Дневной чистый поток и Помесячная динамика
    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown('<div class="fp-section-title">Ожидаемые дневные потоки на горизонте прогноза</div>', unsafe_allow_html=True)
        flow_fig = go.Figure()
        flow_fig.add_trace(go.Bar(
            x=base_forecast["Дата"],
            y=base_forecast["Ожидаемый поток"],
            name="Дневной чистый поток",
            marker_color=[
                COLOR_INFLOW if v >= 0 else COLOR_OUTFLOW for v in base_forecast["Ожидаемый поток"]
            ],
        ))
        flow_fig.add_hline(y=0, line_color=CARD_BORDER, line_width=1)
        flow_fig.update_layout(xaxis_title="", yaxis_title="Поток за день, ₽", showlegend=False)
        st.plotly_chart(style_plotly_fig(flow_fig, height=320), use_container_width=True)

    with c_right:
        st.markdown('<div class="fp-section-title">Исторические притоки и оттоки по месяцам</div>', unsafe_allow_html=True)
        monthly_agg = df.groupby("month").agg(
            Поступления=("signed_amount", lambda s: s[s > 0].sum()),
            Списания=("signed_amount", lambda s: -s[s < 0].sum()),
        ).reset_index().tail(12)
        m_fig = go.Figure()
        m_fig.add_trace(go.Bar(
            x=monthly_agg["month"], y=monthly_agg["Поступления"],
            name="Поступления", marker_color=COLOR_INFLOW,
        ))
        m_fig.add_trace(go.Bar(
            x=monthly_agg["month"], y=monthly_agg["Списания"],
            name="Списания", marker_color=VIOLET_ACCENT,
        ))
        m_fig.update_layout(barmode="group", xaxis_title="", yaxis_title="Сумма, ₽", bargap=0.22)
        st.plotly_chart(style_plotly_fig(m_fig, height=320), use_container_width=True)

    # Детальная таблица прогноза по дням
    with st.expander("📋 Открыть детальную таблицу прогноза остатка по дням", expanded=False):
        view_disp = base_forecast.copy()
        view_disp["Дата"] = view_disp["Дата"].dt.strftime("%d.%m.%Y")
        num_cols = ["Ожидаемый поток", "Плановые события", "Прогноз остатка", "Нижняя граница 95%", "Верхняя граница 95%"]
        for c in num_cols:
            if c in view_disp.columns:
                view_disp[c] = view_disp[c].round(0)
        st.dataframe(view_disp, use_container_width=True, hide_index=True, height=360)


# ═══════════════════════════════════════════════════════════════════════════
# ВКЛАДКА 2: СЦЕНАРИИ «ЧТО ЕСЛИ?» (Звёздочка №3 — с быстрыми пресетами)
# ═══════════════════════════════════════════════════════════════════════════
with tab_scenarios:
    st.markdown('<div class="fp-section-title">Быстрые пресеты стресс-сценариев в 1 клик</div>', unsafe_allow_html=True)
    if "sc_drop" not in st.session_state:
        st.session_state["sc_drop"] = 20
        st.session_state["sc_delay"] = 7
        st.session_state["sc_exp"] = int(round(kpis["avg_daily_outflow"] * 3, -3))
        st.session_state["sc_fixed"] = 0

    p1, p2, p3, p4, p5 = st.columns(5)
    if p1.button("🔄 Сбросить (0%)", use_container_width=True):
        st.session_state.update(sc_drop=0, sc_delay=0, sc_exp=0, sc_fixed=0)
    if p2.button("⏳ Задержка клиента (14 дн.)", use_container_width=True):
        st.session_state.update(sc_drop=0, sc_delay=14, sc_exp=0, sc_fixed=0)
    if p3.button("📉 Сезонный спад (-25%)", use_container_width=True):
        st.session_state.update(sc_drop=25, sc_delay=0, sc_exp=0, sc_fixed=0)
    if p4.button("🛠 Внеплановый ремонт", use_container_width=True):
        st.session_state.update(
            sc_drop=0, sc_delay=0,
            sc_exp=int(round(max(150_000, kpis["avg_daily_outflow"] * 5), -3)),
            sc_fixed=10,
        )
    if p5.button("⚡ Жёсткий стресс (Комбо)", type="primary", use_container_width=True):
        st.session_state.update(
            sc_drop=30, sc_delay=14,
            sc_exp=int(round(max(200_000, kpis["avg_daily_outflow"] * 4), -3)),
            sc_fixed=20,
        )

    st.markdown('<div class="fp-section-title">Ручная настройка параметров сценария</div>', unsafe_allow_html=True)
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        drop = st.slider("📉 Снижение выручки, %", 0, 60, key="sc_drop", step=5)
    with col_b:
        delay = st.slider("⏱ Задержка оплат клиентов, дней", 0, 30, key="sc_delay", step=1)
    with col_c:
        expense = st.number_input("💸 Разовый внеплановый расход, ₽", min_value=0, step=25_000, key="sc_exp")
    with col_d:
        fixed_inc = st.slider("📈 Рост постоянных расходов, %", 0, 100, key="sc_fixed", step=5)

    _, stress_forecast, _, _ = make_forecast_enhanced(
        df,
        horizon=horizon,
        lookback_days=lookback,
        selected_model=model_choice,
        revenue_drop=drop,
        delay_days=delay,
        extra_expense=expense,
        fixed_cost_increase=fixed_inc,
        manual_events=active_events,
    )

    sc_fig = go.Figure()
    sc_fig.add_trace(go.Scatter(
        x=base_forecast["Дата"], y=base_forecast["Прогноз остатка"],
        mode="lines", name="Базовый прогноз",
        line=dict(color=VIOLET_PRIMARY, width=3.2),
    ))
    sc_fig.add_trace(go.Scatter(
        x=stress_forecast["Дата"], y=stress_forecast["Прогноз остатка"],
        mode="lines", name="Стрессовый сценарий «Что если?»",
        line=dict(color=COLOR_DANGER, width=3, dash="dash"),
    ))
    sc_fig.add_trace(go.Scatter(
        x=list(stress_forecast["Дата"]) + list(stress_forecast["Дата"][::-1]),
        y=list(stress_forecast["Верхняя граница 95%"]) + list(stress_forecast["Нижняя граница 95%"][::-1]),
        fill="toself", fillcolor="rgba(225, 29, 72, 0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Доверительный коридор (стресс)", hoverinfo="skip",
    ))
    sc_fig.add_hline(y=0, line_dash="dash", line_color=COLOR_DANGER, line_width=1.5)
    st.plotly_chart(style_plotly_fig(sc_fig, height=420), use_container_width=True)

    base_min = float(base_forecast["Прогноз остатка"].min())
    stress_min = float(stress_forecast["Прогноз остатка"].min())
    stress_neg = (stress_forecast["Прогноз остатка"] < 0).any()

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Минимум базового", fmt_compact(base_min))
    sc2.metric("Минимум в стрессе", fmt_compact(stress_min), delta=fmt_compact(stress_min - base_min), delta_color="inverse")
    sc3.metric("Макс. вероятность разрыва", fmt_pct(float(stress_forecast["Вероятность разрыва, %"].max())))
    sc4.metric("Кассовый разрыв в стрессе", "Да (Требуется резерв)" if stress_neg else "Нет (Устойчиво)")

    if stress_neg:
        first_neg = stress_forecast.loc[stress_forecast["Прогноз остатка"] < 0].iloc[0]
        st.markdown(
            f"""
            <div class='fp-banner fp-banner-danger'>
              <span class='fp-pill fp-pill-high'>СТРЕСС-ТЕСТ НЕ ПРОЙДЕН</span>
              <div>
                При заданных параметрах шока кассовый разрыв наступит <b>{first_neg['Дата']:%d.%m.%Y}</b>.
                Максимальная глубина дефицита составит <b>{fmt_money(abs(stress_min))}</b>.
                Для защиты бизнеса необходим резервный лимит (овердрафт или стабфонд) не менее <b>{fmt_money(abs(stress_min) * 1.1)}</b>.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
# ВКЛАДКА 3: ПЛАНИРОВЩИК (События, Счета, Налоги, Регулярные платежи и Интеграции)
# ═══════════════════════════════════════════════════════════════════════════
with tab_planner:
    st.markdown(
        """
        <div class='fp-banner fp-banner-info'>
          <span class='fp-pill fp-pill-vio'>ЕДИНЫЙ ПЛАНИРОВЩИК</span>
          <div>
            Все добавленные здесь договоры, выставленные счета, регулярные платежи и налоги
            <b>автоматически учитываются в главном прогнозе</b> на вкладке «Дашборд и Прогноз», а их источник сохраняется.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Верхняя панель быстрых действий планировщика
    act1, act2, act3 = st.columns(3)
    with act1:
        if st.button("🔄 Обновить налоговый календарь по умолчанию", use_container_width=True):
            st.session_state["manual_events_state"] = default_tax_events(df, horizon)
            st.rerun()
    with act2:
        if st.button("🔗 Импорт счетов из 1С / ЭДО (демо-интеграция)", use_container_width=True):
            base_d = df["operation_date"].max().normalize()
            demo_1c = [
                {
                    "date": (base_d + pd.Timedelta(days=6)).strftime("%d.%m.%Y"),
                    "amount": float(round(kpis["avg_daily_inflow"] * 2.2, -3)),
                    "type": "in",
                    "description": "Оплата по договору поставки №114/26 (ООО «Альфа»)",
                    "source": "Интеграция 1С:Бухгалтерия",
                },
                {
                    "date": (base_d + pd.Timedelta(days=12)).strftime("%d.%m.%Y"),
                    "amount": float(round(kpis["avg_daily_outflow"] * 1.8, -3)),
                    "type": "out",
                    "description": "Счёт на оплату сырья №892 (ООО «ТехСнаб»)",
                    "source": "ЭДО Контур.Диадок",
                },
            ]
            st.session_state["manual_events_state"].extend(demo_1c)
            st.rerun()
    with act3:
        if st.button("🗑 Очистить все плановые события", use_container_width=True):
            st.session_state["manual_events_state"] = []
            st.rerun()

    pl_left, pl_right = st.columns([3, 2])

    with pl_left:
        st.markdown('<div class="fp-section-title">Добавить событие, счёт или договор вручную</div>', unsafe_allow_html=True)
        with st.form("unified_add_event", clear_on_submit=True):
            f1, f2, f3 = st.columns(3)
            with f1:
                ev_date = st.text_input(
                    "Дата (ДД.ММ.ГГГГ)",
                    value=(df["operation_date"].max() + pd.Timedelta(days=7)).strftime("%d.%m.%Y"),
                )
            with f2:
                ev_amount = st.number_input(
                    "Сумма, ₽",
                    min_value=0.0,
                    value=float(round(max(50_000.0, kpis["avg_daily_outflow"]), -3)),
                    step=10_000.0,
                )
            with f3:
                ev_type = st.selectbox("Направление", ["Ожидаемое списание (out)", "Ожидаемое поступление (in)"])

            f4, f5 = st.columns([2, 1])
            with f4:
                ev_desc = st.text_input("Назначение / Контрагент", value="Счёт №45 от клиента за поставку")
            with f5:
                ev_src = st.selectbox("Источник данных", ["Ручной ввод", "Выставленный счёт", "Договор", "Банк (платёжка)"])

            if st.form_submit_button("➕ Добавить в прогноз", use_container_width=True):
                st.session_state["manual_events_state"].append({
                    "date": ev_date,
                    "amount": float(ev_amount),
                    "type": "out" if "out" in ev_type else "in",
                    "description": ev_desc,
                    "source": ev_src,
                })
                st.rerun()

    with pl_right:
        st.markdown('<div class="fp-section-title">Активные события в прогнозе</div>', unsafe_allow_html=True)
        if not st.session_state.get("manual_events_state"):
            st.info("Список плановых событий пуст.")
        else:
            ev_df = pd.DataFrame(st.session_state["manual_events_state"])
            if "source" not in ev_df.columns:
                ev_df["source"] = "План"
            ev_df_disp = pd.DataFrame({
                "Дата": ev_df["date"],
                "Тип": ev_df["type"].map({"in": "🟢 Приток", "out": "🟣 Отток"}),
                "Сумма": ev_df["amount"].apply(fmt_money),
                "Описание": ev_df["description"],
                "Источник": ev_df["source"],
            })
            st.dataframe(ev_df_disp, use_container_width=True, hide_index=True, height=255)

    st.markdown('<div class="fp-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="fp-section-title">Автоматически обнаруженные регулярные платежи из истории</div>', unsafe_allow_html=True)
    recurring = detect_recurring_payments(df, min_occurrences=3)
    if not recurring:
        st.info("Регулярные платежи с устойчивым интервалом в истории не обнаружены.")
    else:
        horizon_end = df["operation_date"].max().normalize() + pd.Timedelta(days=horizon)
        upcoming_rec = [r for r in recurring if r["next_date"] <= horizon_end]

        rc_top1, rc_top2 = st.columns([3, 2])
        with rc_top1:
            st.caption(
                f"Найдено регулярных паттернов: {len(recurring)} (из них {len(upcoming_rec)} попадают в текущий горизонт {horizon} дней)."
            )
        with rc_top2:
            if upcoming_rec and st.button(
                f"➕ Включить {len(upcoming_rec)} предстоящих регулярных платежей в прогноз",
                type="primary",
                use_container_width=True,
            ):
                for r in upcoming_rec:
                    st.session_state["manual_events_state"].append({
                        "date": r["next_date"].strftime("%d.%m.%Y"),
                        "amount": float(round(r["median_amount"], 0)),
                        "type": "out" if r["direction"] == "Списание" else "in",
                        "description": f"Регулярный: {r['counterparty']}",
                        "source": "Авто-детект регулярных",
                    })
                st.rerun()

        rec_table = pd.DataFrame([
            {
                "Контрагент / Шаблон": r["counterparty"],
                "Направление": r["direction"],
                "Категория": r["category"],
                "Медианная сумма": fmt_money(r["median_amount"]),
                "Цикл, дн.": r["interval_days"],
                "Повторений": r["occurrences"],
                "Стабильность": f"{(1 - r['cv']) * 100:.0f}%",
                "Следующий платёж": r["next_date"].strftime("%d.%m.%Y"),
            }
            for r in recurring
        ])
        st.dataframe(rec_table, use_container_width=True, hide_index=True, height=280)


# ═══════════════════════════════════════════════════════════════════════════
# ВКЛАДКА 4: КАТЕГОРИИ И ВЫПИСКА
# ═══════════════════════════════════════════════════════════════════════════
with tab_categories:
    cats = df.assign(Тип=np.where(df["signed_amount"] > 0, "Поступления", "Списания"))
    summary = (
        cats.groupby(["category", "Тип"])["amount"]
        .agg(["sum", "count"])
        .reset_index()
        .rename(columns={"category": "Категория", "sum": "Сумма", "count": "Операций"})
        .sort_values(["Тип", "Сумма"], ascending=[True, False])
    )

    cat_l, cat_r = st.columns([1, 1])
    with cat_l:
        st.markdown('<div class="fp-section-title">Структура денежных потоков по категориям</div>', unsafe_allow_html=True)
        infl = summary[summary["Тип"] == "Поступления"]
        outl = summary[summary["Тип"] == "Списания"]
        infl_total = float(infl["Сумма"].sum()) if len(infl) else 0.0
        outl_total = float(outl["Сумма"].sum()) if len(outl) else 0.0

        labels = ["Поступления"] + list(infl["Категория"]) + ["Списания"] + list(outl["Категория"])
        parents = [""] + ["Поступления"] * len(infl) + [""] + ["Списания"] * len(outl)
        values = [infl_total] + list(infl["Сумма"]) + [outl_total] + list(outl["Сумма"])

        # Мягкая бело-фиолетовая и изумрудная палитра вместо кислотного красно-зелёного
        colors = (
            ["#10B981"] + ["#34D399"] * len(infl)
            + [VIOLET_PRIMARY] + [VIOLET_LIGHT] * len(outl)
        )
        sun_fig = go.Figure(go.Sunburst(
            labels=labels, parents=parents, values=values,
            branchvalues="total",
            marker=dict(colors=colors, line=dict(color="#FFFFFF", width=2)),
            textinfo="label+percent root",
            hovertemplate="<b>%{label}</b><br>Сумма: %{value:,.0f} ₽<br>Доля: %{percentRoot:.1%}<extra></extra>",
        ))
        sun_fig.update_layout(margin=dict(l=8, r=8, t=8, b=8))
        st.plotly_chart(style_plotly_fig(sun_fig, height=370), use_container_width=True)

    with cat_r:
        st.markdown('<div class="fp-section-title">Топ-категории по объёму средств</div>', unsafe_allow_html=True)
        bar_fig = go.Figure()
        for tipo, color in [("Поступления", COLOR_INFLOW), ("Списания", VIOLET_ACCENT)]:
            sub = summary[summary["Тип"] == tipo].sort_values("Сумма", ascending=True).tail(8)
            bar_fig.add_trace(go.Bar(
                x=sub["Сумма"], y=sub["Категория"], orientation="h",
                name=tipo, marker_color=color,
            ))
        bar_fig.update_layout(barmode="group", xaxis_title="Сумма, ₽", yaxis_title="")
        st.plotly_chart(style_plotly_fig(bar_fig, height=370), use_container_width=True)

    # Удобный фильтр по таблице категорий и операциям
    f_col1, f_col2 = st.columns([1, 2])
    with f_col1:
        type_filter = st.radio(
            "Фильтр таблицы", ["Все", "Поступления", "Списания"], horizontal=True
        )
    with f_col2:
        search_q = st.text_input("🔍 Поиск по контрагенту или назначению платежа", placeholder="Например: ФНС, аренда, эквайринг...")

    filtered_summary = summary if type_filter == "Все" else summary[summary["Тип"] == type_filter]
    disp_sum = filtered_summary.copy()
    disp_sum["Доля в типе, %"] = (
        disp_sum["Сумма"] / disp_sum.groupby("Тип")["Сумма"].transform("sum") * 100
    ).round(1)
    disp_sum["Сумма"] = disp_sum["Сумма"].round(2)
    st.dataframe(disp_sum, use_container_width=True, hide_index=True, height=280)

    if search_q.strip():
        q = search_q.strip().lower()
        ops = df[
            df["payment_purpose"].astype(str).str.lower().str.contains(q, na=False)
            | df["counterparty_name"].astype(str).str.lower().str.contains(q, na=False)
        ].copy()
        st.caption(f"Найдено операций по запросу «{search_q}»: {len(ops)}")
        st.dataframe(
            ops[["operation_date", "counterparty_name", "payment_purpose", "category", "signed_amount"]].tail(200),
            use_container_width=True, hide_index=True, height=250,
        )


# ═══════════════════════════════════════════════════════════════════════════
# ВКЛАДКА 5: КАЧЕСТВО, ПЛАН-ФАКТ И АРХИТЕКТУРА
# ═══════════════════════════════════════════════════════════════════════════
with tab_quality:
    st.markdown('<div class="fp-section-title">Сравнение моделей прогнозирования на Hold-Out бэктесте</div>', unsafe_allow_html=True)
    rows = []
    for name, (_, m) in model_meta["candidates"].items():
        rows.append({
            "Модель": name,
            "MAE (ср. ошибка остатка), ₽": round(m["mae"], 0) if not np.isnan(m["mae"]) else None,
            "MAPE, %": round(m["mape"], 2) if not np.isnan(m["mape"]) else None,
            "RMSE, ₽": round(m["rmse"], 0) if not np.isnan(m["rmse"]) else None,
            "Смещение (Bias), ₽": round(m["bias"], 0) if not np.isnan(m["bias"]) else None,
            "Статус": "💜 Активна" if name == best_model else "Доступна",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    q_left, q_right = st.columns(2)
    daily_net_full = _daily_net_series(df, lookback_days=None)
    holdout_days = min(30, max(7, len(daily_net_full) // 4))

    with q_left:
        st.markdown('<div class="fp-section-title">Бэктест остатка: План vs Факт (Hold-Out)</div>', unsafe_allow_html=True)
        if len(daily_net_full) > holdout_days + 14:
            cutoff_date = daily_net_full.index[-holdout_days]
            train_df = df[df["operation_date"] < cutoff_date]
            start_bal = get_current_balance(train_df, _daily_net_series(train_df))
            pf_pred = model_meta["candidates"][best_model][0](
                train_df, holdout_days, start_bal, lookback_days=lookback
            )
            actual_cum = daily_net_full.iloc[-holdout_days:].cumsum() + start_bal

            pf_fig = go.Figure()
            pf_fig.add_trace(go.Scatter(
                x=actual_cum.index, y=actual_cum.values,
                name="Фактический остаток", line=dict(color=COLOR_INFLOW, width=2.8),
            ))
            pf_fig.add_trace(go.Scatter(
                x=pf_pred["date"], y=pf_pred["balance"],
                name=f"Прогноз модели ({best_model})",
                line=dict(color=VIOLET_PRIMARY, width=2.8, dash="dash"),
            ))
            st.plotly_chart(style_plotly_fig(pf_fig, height=340), use_container_width=True)
        else:
            st.info("Недостаточно исторических дней для Hold-Out графика.")

    with q_right:
        st.markdown('<div class="fp-section-title">План-Факт потоков за последние 14 дней (контроль задержек)</div>', unsafe_allow_html=True)
        rt_days = min(14, max(5, len(daily_net_full) // 5))
        if len(daily_net_full) > rt_days + 14:
            rt_cut = daily_net_full.index[-rt_days]
            rt_train = df[df["operation_date"] < rt_cut]
            rt_start = get_current_balance(rt_train, _daily_net_series(rt_train))
            rt_pred = model_meta["candidates"][best_model][0](
                rt_train, rt_days, rt_start, lookback_days=lookback
            )
            act_d = daily_net_full.iloc[-rt_days:]
            prd_d = rt_pred.set_index("date")["flow"].reindex(act_d.index)

            rt_fig = go.Figure()
            rt_fig.add_trace(go.Bar(
                x=act_d.index, y=act_d.values,
                name="Факт", marker_color=VIOLET_PRIMARY,
            ))
            rt_fig.add_trace(go.Bar(
                x=prd_d.index, y=prd_d.values,
                name="План (ожидание)", marker_color="#CBD5E1",
            ))
            rt_fig.update_layout(barmode="group")
            st.plotly_chart(style_plotly_fig(rt_fig, height=340), use_container_width=True)

    st.markdown('<div class="fp-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="fp-section-title">Схема архитектуры решения (Бело-фиолетовая тема)</div>', unsafe_allow_html=True)
    mermaid_diagram = r"""
    flowchart LR
      subgraph DS["📥 Источники данных"]
        D1["CSV-выписка банка"]
        D2["Архив кейса / Мульти-компании"]
        D3["Планировщик: Счета, Договоры, 1С"]
        D4["Налоговый календарь ФНС"]
      end
      subgraph PROC["⚙️ Обработка и аналитика"]
        N1["Нормализация и очистка"]
        N2["Раздельная категоризация (Приток/Отток)"]
        N3["Авто-детект регулярных платежей"]
      end
      subgraph FCST["🤖 Ядро прогнозирования"]
        M1["Weekday Mean / Holt / Linear+WD"]
        BT["Hold-Out Бэктест (MAE, MAPE, RMSE)"]
        CI["Доверительный коридор 95% и P(Разрыва)"]
      end
      subgraph OUT["💜 UX/UI Дашборд предпринимателя"]
        O1["График остатка и План-Факт"]
        O2["Стресс-сценарии Что-Если"]
        O3["Рекомендации по предотвращению разрыва"]
      end
      D1 & D2 --> N1 --> N2 --> N3
      N2 --> M1 --> BT --> CI
      D3 & D4 & N3 --> CI
      CI --> O1 & O2 & O3
    """
    st.components.v1.html(
        f"""
        <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
        <div style='background:#FFFFFF; padding:18px; border-radius:14px; border:1px solid {CARD_BORDER};'>
          <div class='mermaid' style='background:transparent;'>{mermaid_diagram}</div>
        </div>
        <script>
          mermaid.initialize({{
            startOnLoad: true,
            theme: 'base',
            themeVariables: {{
              primaryColor: '#F3E8FF',
              primaryTextColor: '#1E1B4B',
              primaryBorderColor: '#7C3AED',
              lineColor: '#6D28D9',
              secondaryColor: '#F8F7FC',
              tertiaryColor: '#FFFFFF'
            }}
          }});
        </script>
        """,
        height=310,
    )