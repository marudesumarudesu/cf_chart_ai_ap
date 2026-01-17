from __future__ import annotations

import logging

import io
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf


JPX_LIST_URL = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"


# A pragmatic set of indices/benchmarks that work well on Yahoo Finance.
#
# NOTE:
# - TOPIX is available on Yahoo Finance as ^TOPX (and on Yahoo Japan as 998405.T).
#   On some environments, one may work while the other doesn't, so we keep fallbacks.
INDEX_TICKERS = {
    "日経平均 (Nikkei 225)": "^N225",
    "TOPIX": "^TOPX",
    "米ドル/円": "JPY=X",
    "S&P 500": "^GSPC",
    "NASDAQ 総合": "^IXIC",
    "NYダウ": "^DJI",
    "WTI 原油": "CL=F",
    "金 (Gold)": "GC=F",
}

# Per-index ticker fallbacks (queried in a single batched yfinance call on the app side).
INDEX_TICKER_CANDIDATES: Dict[str, List[str]] = {
    # Prefer ^TOPX, fallback to Yahoo Japan's 998405.T, then liquid TOPIX ETFs as last resort.
    "TOPIX": ["^TOPX", "998405.T", "1306.T", "1475.T"],
}


@dataclass(frozen=True)
class PriceMove:
    last: float
    change: float
    change_pct: float


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _download_bytes(url: str, timeout: int = 25) -> bytes:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def download_jpx_list() -> pd.DataFrame:
    """Download and parse the JPX (TSE-listed issues) Excel.

    Source is JPX's published Excel file. It is typically updated monthly.
    """
    raw = _download_bytes(JPX_LIST_URL)
    df = pd.read_excel(io.BytesIO(raw), sheet_name=0)

    # Normalize column names (JPX changes can happen; we keep it resilient).
    df.columns = [str(c).strip() for c in df.columns]

    # Expected columns include: 'コード', '銘柄名', '市場・商品区分', '33業種区分', etc.
    if "コード" not in df.columns or "銘柄名" not in df.columns:
        raise ValueError("JPX Excel format changed: required columns not found.")

    df["コード"] = df["コード"].astype(str).str.zfill(4)
    df["銘柄名"] = df["銘柄名"].astype(str).str.strip()

    # Most Japanese equities on Yahoo Finance use the Tokyo Stock Exchange suffix '.T'.
    df["yfinance"] = df["コード"] + ".T"

    # A compact label for UI search.
    df["label"] = df["yfinance"] + "  —  " + df["銘柄名"]

    # Sort for nicer UX.
    df = df.sort_values(["コード"]).reset_index(drop=True)
    return df


@st.cache_data(ttl=15 * 60, show_spinner=False)
def fetch_ohlcv(
    tickers: Iterable[str],
    period_days: int = 420,
    interval: str = "1d",
) -> Dict[str, pd.DataFrame]:
    """Fetch OHLCV for one or more tickers via yfinance with caching.

    Notes for rate limits:
      - We fetch via yf.download in a single call for all tickers.
      - Streamlit cache reduces repeat requests.
      - Keep tickers count reasonable (UI enforces a soft limit).
    """
    tickers = [t.strip() for t in tickers if str(t).strip()]
    if not tickers:
        return {}

    period_days = int(np.clip(period_days, 30, 3650))
    period = f"{period_days}d"

    # yfinance is chatty; avoid spamming logs on Streamlit Cloud.
    yf_logger = logging.getLogger("yfinance")
    yf_logger.setLevel(logging.CRITICAL)

    data = yf.download(
        tickers=" ".join(tickers),
        period=period,
        interval=interval,
        group_by="ticker",
        auto_adjust=False,
        actions=False,
        threads=False,
        progress=False,
    )

    out: Dict[str, pd.DataFrame] = {}
    if data is None or len(data) == 0:
        return out

    if isinstance(data.columns, pd.MultiIndex):
        # Multi-ticker case
        for t in tickers:
            if t not in data.columns.get_level_values(0):
                continue
            df = data[t].copy()
            df = _sanitize_ohlcv(df)
            if len(df):
                out[t] = df
    else:
        # Single ticker case
        out[tickers[0]] = _sanitize_ohlcv(data.copy())

    return out


def _sanitize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Standardize columns when yfinance changes casing
    df.columns = [str(c).title() for c in df.columns]
    # Keep common set
    keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    df = df[keep]
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    df = df.dropna(subset=[c for c in ["Open", "High", "Low", "Close"] if c in df.columns], how="all")
    return df


def last_close_and_change(df: pd.DataFrame) -> PriceMove | None:
    if df is None or df.empty or "Close" not in df.columns:
        return None
    close = df["Close"].dropna()
    if len(close) < 2:
        return None
    last = _safe_float(close.iloc[-1])
    prev = _safe_float(close.iloc[-2])
    change = last - prev
    change_pct = (change / prev * 100.0) if np.isfinite(prev) and prev != 0 else float("nan")
    return PriceMove(last=last, change=change, change_pct=change_pct)


def normalize_equal_weight_index(
    price_dict: Dict[str, pd.DataFrame],
    base: float = 100.0,
) -> pd.DataFrame:
    """Build an equal-weight (average) index from multiple tickers."""
    if not price_dict:
        return pd.DataFrame()

    closes = {}
    for t, df in price_dict.items():
        if df is None or df.empty or "Close" not in df.columns:
            continue
        s = df["Close"].astype(float)
        closes[t] = s

    if not closes:
        return pd.DataFrame()

    close_df = pd.concat(closes, axis=1).sort_index()
    norm_prices = close_df / close_df.iloc[0] * base

    rets = close_df.pct_change()
    ew_ret = rets.mean(axis=1, skipna=True).fillna(0.0)
    ew_index = (1.0 + ew_ret).cumprod() * base

    out = pd.DataFrame({"EW_INDEX": ew_index})
    for t in norm_prices.columns:
        out[f"{t}_NORM"] = norm_prices[t]

    return out
