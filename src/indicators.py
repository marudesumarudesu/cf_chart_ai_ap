from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IndicatorLine:
    name: str
    series: pd.Series
    kind: str = "line"  # 'line' or 'bar'


@dataclass(frozen=True)
class IndicatorPanel:
    title: str
    lines: List[IndicatorLine]
    y_ref_lines: List[float] | None = None


def sma(close: pd.Series, period: int) -> pd.Series:
    return close.rolling(period).mean()


def ema(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def bollinger(close: pd.Series, period: int = 20, std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = close.rolling(period).mean()
    sd = close.rolling(period).std(ddof=0)
    upper = mid + std * sd
    lower = mid - std * sd
    return lower, mid, upper


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k: int = 14, d: int = 3, smooth: int = 3) -> Tuple[pd.Series, pd.Series]:
    ll = low.rolling(k).min()
    hh = high.rolling(k).max()
    raw_k = (close - ll) / (hh - ll).replace(0.0, np.nan) * 100.0
    sm_k = raw_k.rolling(smooth).mean()
    sm_d = sm_k.rolling(d).mean()
    return sm_k, sm_d


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = true_range(high, low, close)
    atr_ = tr.ewm(alpha=1 / period, adjust=False).mean()

    plus_di = 100.0 * pd.Series(plus_dm, index=high.index).ewm(alpha=1 / period, adjust=False).mean() / atr_.replace(0.0, np.nan)
    minus_di = 100.0 * pd.Series(minus_dm, index=high.index).ewm(alpha=1 / period, adjust=False).mean() / atr_.replace(0.0, np.nan)

    dx = (100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)).fillna(0.0)
    adx_ = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx_, plus_di, minus_di


def ichimoku(high: pd.Series, low: pd.Series, close: pd.Series, tenkan: int = 9, kijun: int = 26, senkou_b: int = 52) -> Dict[str, pd.Series]:
    tenkan_sen = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2.0
    kijun_sen = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2.0
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2.0).shift(kijun)
    senkou_span_b = ((high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2.0).shift(kijun)
    chikou_span = close.shift(-kijun)
    return {
        "Tenkan": tenkan_sen,
        "Kijun": kijun_sen,
        "SenkouA": senkou_span_a,
        "SenkouB": senkou_span_b,
        "Chikou": chikou_span,
    }


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume.fillna(0.0)).cumsum()


def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    tp = (high + low + close) / 3.0
    ma = tp.rolling(period).mean()
    md = (tp - ma).abs().rolling(period).mean()
    return (tp - ma) / (0.015 * md.replace(0.0, np.nan))


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    return -100.0 * (hh - close) / (hh - ll).replace(0.0, np.nan)


def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    tp = (high + low + close) / 3.0
    pv = tp * volume.fillna(0.0)
    return pv.cumsum() / volume.fillna(0.0).cumsum().replace(0.0, np.nan)


def parabolic_sar(high: pd.Series, low: pd.Series, step: float = 0.02, max_step: float = 0.2) -> pd.Series:
    # Simplified PSAR implementation
    high = high.astype(float)
    low = low.astype(float)

    sar = pd.Series(index=high.index, dtype=float)
    bull = True
    af = step
    ep = high.iloc[0]
    sar.iloc[0] = low.iloc[0]

    for i in range(1, len(high)):
        prev_sar = sar.iloc[i - 1]
        if bull:
            sar_i = prev_sar + af * (ep - prev_sar)
            sar_i = min(sar_i, low.iloc[i - 1], low.iloc[i])
            if low.iloc[i] < sar_i:
                bull = False
                sar_i = ep
                ep = low.iloc[i]
                af = step
            else:
                if high.iloc[i] > ep:
                    ep = high.iloc[i]
                    af = min(af + step, max_step)
        else:
            sar_i = prev_sar + af * (ep - prev_sar)
            sar_i = max(sar_i, high.iloc[i - 1], high.iloc[i])
            if high.iloc[i] > sar_i:
                bull = True
                sar_i = ep
                ep = high.iloc[i]
                af = step
            else:
                if low.iloc[i] < ep:
                    ep = low.iloc[i]
                    af = min(af + step, max_step)

        sar.iloc[i] = sar_i

    return sar


def supertrend(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 10, mult: float = 3.0) -> Tuple[pd.Series, pd.Series]:
    atr_ = atr(high, low, close, period)
    hl2 = (high + low) / 2.0
    upper = hl2 + mult * atr_
    lower = hl2 - mult * atr_

    st = pd.Series(index=close.index, dtype=float)
    direction = pd.Series(index=close.index, dtype=int)

    for i in range(len(close)):
        if i == 0:
            st.iloc[i] = upper.iloc[i]
            direction.iloc[i] = -1
            continue

        prev_st = st.iloc[i - 1]
        prev_dir = direction.iloc[i - 1]

        # Final upper/lower bands
        final_upper = upper.iloc[i] if (upper.iloc[i] < upper.iloc[i - 1]) or (close.iloc[i - 1] > upper.iloc[i - 1]) else upper.iloc[i - 1]
        final_lower = lower.iloc[i] if (lower.iloc[i] > lower.iloc[i - 1]) or (close.iloc[i - 1] < lower.iloc[i - 1]) else lower.iloc[i - 1]

        if prev_dir == -1:
            if close.iloc[i] <= final_upper:
                st.iloc[i] = final_upper
                direction.iloc[i] = -1
            else:
                st.iloc[i] = final_lower
                direction.iloc[i] = 1
        else:
            if close.iloc[i] >= final_lower:
                st.iloc[i] = final_lower
                direction.iloc[i] = 1
            else:
                st.iloc[i] = final_upper
                direction.iloc[i] = -1

    return st, direction


# ---- Public builders (used by plotting layer) ----


def build_indicator_overlays(
    df: pd.DataFrame,
    selected: List[str],
    params: Dict,
) -> List[IndicatorLine]:
    """Indicators drawn on the price panel (candlestick)."""
    if df is None or df.empty:
        return []

    high = df.get("High")
    low = df.get("Low")
    close = df.get("Close")
    vol = df.get("Volume")

    out: List[IndicatorLine] = []

    if "SMA" in selected:
        p = int(params.get("sma_period", 20))
        out.append(IndicatorLine(f"SMA({p})", sma(close, p)))

    if "EMA" in selected:
        p = int(params.get("ema_period", 20))
        out.append(IndicatorLine(f"EMA({p})", ema(close, p)))

    if "Bollinger" in selected:
        p = int(params.get("bb_period", 20))
        s = float(params.get("bb_std", 2.0))
        lo, mid, up = bollinger(close, p, s)
        out.append(IndicatorLine(f"BB Lower({p},{s})", lo))
        out.append(IndicatorLine(f"BB Mid({p})", mid))
        out.append(IndicatorLine(f"BB Upper({p},{s})", up))

    if "Ichimoku" in selected and high is not None and low is not None:
        tenkan = int(params.get("ichi_tenkan", 9))
        kijun = int(params.get("ichi_kijun", 26))
        senkou_b = int(params.get("ichi_senkou_b", 52))
        lines = ichimoku(high, low, close, tenkan, kijun, senkou_b)
        out.append(IndicatorLine("Tenkan", lines["Tenkan"]))
        out.append(IndicatorLine("Kijun", lines["Kijun"]))
        out.append(IndicatorLine("SenkouA", lines["SenkouA"]))
        out.append(IndicatorLine("SenkouB", lines["SenkouB"]))

    if "VWAP" in selected and vol is not None and high is not None and low is not None:
        out.append(IndicatorLine("VWAP", vwap(high, low, close, vol)))

    if "ParabolicSAR" in selected and high is not None and low is not None:
        step = float(params.get("psar_step", 0.02))
        mstep = float(params.get("psar_max_step", 0.2))
        out.append(IndicatorLine("Parabolic SAR", parabolic_sar(high, low, step, mstep)))

    if "Supertrend" in selected and high is not None and low is not None:
        p = int(params.get("supertrend_period", 10))
        m = float(params.get("supertrend_mult", 3.0))
        st, _ = supertrend(high, low, close, p, m)
        out.append(IndicatorLine(f"Supertrend({p},{m})", st))

    return out


def build_indicator_panels(
    df: pd.DataFrame,
    selected: List[str],
    params: Dict,
) -> List[IndicatorPanel]:
    """Indicators drawn as separate panels."""
    if df is None or df.empty:
        return []

    high = df.get("High")
    low = df.get("Low")
    close = df.get("Close")
    vol = df.get("Volume")

    panels: List[IndicatorPanel] = []

    if "RSI" in selected:
        p = int(params.get("rsi_period", 14))
        panels.append(
            IndicatorPanel(
                title=f"RSI({p})",
                lines=[IndicatorLine("RSI", rsi(close, p))],
                y_ref_lines=[30, 70],
            )
        )

    if "MACD" in selected:
        fast = int(params.get("macd_fast", 12))
        slow = int(params.get("macd_slow", 26))
        sig = int(params.get("macd_signal", 9))
        m, s, h = macd(close, fast, slow, sig)
        panels.append(
            IndicatorPanel(
                title=f"MACD({fast},{slow},{sig})",
                lines=[
                    IndicatorLine("MACD", m),
                    IndicatorLine("Signal", s),
                    IndicatorLine("Histogram", h, kind="bar"),
                ],
            )
        )

    if "Stochastic" in selected and high is not None and low is not None:
        k = int(params.get("stoch_k", 14))
        d = int(params.get("stoch_d", 3))
        smooth = int(params.get("stoch_smooth", 3))
        k_line, d_line = stochastic(high, low, close, k, d, smooth)
        panels.append(
            IndicatorPanel(
                title=f"Stochastic({k},{d},{smooth})",
                lines=[IndicatorLine("%K", k_line), IndicatorLine("%D", d_line)],
                y_ref_lines=[20, 80],
            )
        )

    if "ATR" in selected and high is not None and low is not None:
        p = int(params.get("atr_period", 14))
        panels.append(IndicatorPanel(title=f"ATR({p})", lines=[IndicatorLine("ATR", atr(high, low, close, p))]))

    if "ADX" in selected and high is not None and low is not None:
        p = int(params.get("adx_period", 14))
        a, pdi, mdi = adx(high, low, close, p)
        panels.append(
            IndicatorPanel(
                title=f"ADX({p})",
                lines=[IndicatorLine("ADX", a), IndicatorLine("+DI", pdi), IndicatorLine("-DI", mdi)],
                y_ref_lines=[20],
            )
        )

    if "OBV" in selected and vol is not None:
        panels.append(IndicatorPanel(title="OBV", lines=[IndicatorLine("OBV", obv(close, vol))]))

    if "CCI" in selected and high is not None and low is not None:
        p = int(params.get("cci_period", 20))
        panels.append(
            IndicatorPanel(
                title=f"CCI({p})",
                lines=[IndicatorLine("CCI", cci(high, low, close, p))],
                y_ref_lines=[-100, 100],
            )
        )

    if "Williams%R" in selected and high is not None and low is not None:
        p = int(params.get("willr_period", 14))
        panels.append(
            IndicatorPanel(
                title=f"Williams %R({p})",
                lines=[IndicatorLine("%R", williams_r(high, low, close, p))],
                y_ref_lines=[-80, -20],
            )
        )

    return panels
