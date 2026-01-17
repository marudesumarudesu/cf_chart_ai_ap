from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf


JPX_LIST_URL = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"


# Indices/benchmarks on Yahoo Finance
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
# Some environments can fetch TOPIX as ^TOPX, others as 998405.T, so we keep fallbacks.
INDEX_TICKER_CANDIDATES = {
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
    """Download and parse the JPX (TSE-listed issues) Excel."""
    raw = _download_bytes(JPX_LIST_URL)
    df = pd.read_excel(io.BytesIO(raw), sheet_name=0)
    df.columns = [str(c).strip() for c in df.columns]

    if "コード" not in df.columns or "銘柄名" not in df.columns:
        raise ValueError("JPX Excel format changed: required columns not found.")

    df["コード"] = df["コード"].astype(str).str.zfill(4)
    df["銘柄名"] = df["銘柄名"].astype(str).str.strip()

    df["yfinance"] = df["コード"] + ".T"
    df["label"] = df["yfinance"] + "  —  " + df["銘柄名"]

    df = df.sort_values(["コード"]).reset_index(drop=True)
    return df


@st.cache_data(ttl=15 * 60, show_spinner=False)
def fetch_ohlcv(
    tickers: Iterable[str],
    period_days: int = 420,
    interval: str = "1d",
) -> Dict[str, pd.DataFrame]:
    """Fetch OHLCV for tickers via yfinance with caching."""
    tickers = [str(t).strip() for t in tickers if str(t).strip()]
    if not tickers:
        return {}

    period_days = int(np.clip(period_days, 30, 3650))
    period = f"{period_days}d"

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
        for t in tickers:
            if t not in data.columns.get_level_values(0):
                continue
            df = data[t].copy()
            df = _sanitize_ohlcv(df)
            if len(df):
                out[t] = df
    else:
        out[tickers[0]] = _sanitize_ohlcv(data.copy())

    return out


def _sanitize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).title() for c in df.columns]
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
    """Build an equal-weight index (EW) from multiple tickers.

    Fix for mixed calendars (JP + US):
      - Build union date index (already via concat outer join)
      - Start from the latest first-valid date across series (so all series exist)
      - Forward-fill closes on missing dates to avoid broken lines
    """
    if not price_dict:
        return pd.DataFrame()

    closes: Dict[str, pd.Series] = {}
    first_valids: List[pd.Timestamp] = []

    for t, df in price_dict.items():
        if df is None or df.empty or "Close" not in df.columns:
            continue
        s = df["Close"].astype(float).copy()
        s = s[~s.index.duplicated(keep="last")].sort_index()
        if s.notna().any():
            fv = s.first_valid_index()
            if fv is not None:
                first_valids.append(pd.to_datetime(fv))
            closes[t] = s

    if len(closes) < 2:
        return pd.DataFrame()

    # Union of trading dates across all series
    close_df = pd.concat(closes, axis=1).sort_index()

    # Start where all series have begun (prevents filling "from the future")
    if first_valids:
        start = max(first_valids)
        close_df = close_df.loc[close_df.index >= start]

    # Forward-fill missing closes (calendar mismatch => continuous lines)
    close_df = close_df.ffill()

    # Drop rows that are still all-NaN (should be rare after start+ffill)
    close_df = close_df.dropna(how="all")
    if close_df.empty:
        return pd.DataFrame()

    # Normalize each series to base
    norm_prices = close_df / close_df.iloc[0] * base

    # Equal-weight index computed from average daily returns
    rets = close_df.pct_change()
    ew_ret = rets.mean(axis=1, skipna=True).fillna(0.0)
    ew_index = (1.0 + ew_ret).cumprod() * base

    out = pd.DataFrame({"EW_INDEX": ew_index}, index=close_df.index)

    for t in norm_prices.columns:
        out[f"{t}_NORM"] = norm_prices[t]

    return out
