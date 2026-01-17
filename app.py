from __future__ import annotations

import html
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

from src.data import (
    INDEX_TICKERS,
    INDEX_TICKER_CANDIDATES,
    download_jpx_list,
    fetch_ohlcv,
    last_close_and_change,
    normalize_equal_weight_index,
)
from src.indicators import build_indicator_overlays, build_indicator_panels
from src.plotting import equal_weight_index_chart, focus_chart, multi_candlestick_subplots
from src.style import inject_css


PLOTLY_CONFIG = {
    "scrollZoom": True,
    "displaylogo": False,
    "responsive": True,
    "displayModeBar": True,
}

st.set_page_config(
    page_title="JP Market Canvas",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

st.markdown(
    """
# 📈 JP Market Canvas
**yfinance × Streamlit** で作る、日本株チャートダッシュボードです。

> ※ 本アプリは情報提供を目的としており、投資助言・売買推奨ではありません。
"""
)

with st.sidebar:
    st.subheader("銘柄選択")
    universe_df = download_jpx_list()

    colf1, colf2 = st.columns(2)
    with colf1:
        only_stocks = st.checkbox(
            "株式中心", value=True,
            help="ETF/REIT等が混じる場合があるので、株式中心に絞ります（完全には保証できません）"
        )
    with colf2:
        max_select = st.number_input("最大選択数", min_value=1, max_value=30, value=12, step=1)

    work_df = universe_df.copy()
    if only_stocks:
        market_col = None
        for c in ["市場・商品区分", "市場・商品区分（市場区分）", "市場区分"]:
            if c in work_df.columns:
                market_col = c
                break
        if market_col:
            work_df = work_df[work_df[market_col].astype(str).str.contains("株式", na=False)]

    query = st.text_input("検索（コード・銘柄名）", value="")
    if query.strip():
        q = query.strip().lower()
        work_df = work_df[
            work_df["コード"].astype(str).str.contains(q, na=False)
            | work_df["銘柄名"].astype(str).str.lower().str.contains(q, na=False)
            | work_df["yfinance"].astype(str).str.lower().str.contains(q, na=False)
        ]

    options = work_df["label"].tolist()

    default_labels = []
    default_candidates = ["7203.T", "6758.T", "9984.T", "8306.T"]
    for t in default_candidates:
        m = universe_df.loc[universe_df["yfinance"] == t, "label"]
        if len(m):
            default_labels.append(m.iloc[0])

    selected_labels = st.multiselect(
        "表示する銘柄（複数選択OK）",
        options=options,
        default=default_labels,
        help="まずは 3〜8 銘柄が快適です（増やすほど重くなります）。",
    )

    manual = st.text_input(
        "手動で追加（yfinanceティッカーをカンマ区切り）",
        value="",
        help="例：9432.T, 6501.T, ^N225 など",
    )

    st.divider()
    st.subheader("チャート設定")
    candles = st.slider("取得期間（ローソク足本数 / 最大90）", min_value=20, max_value=90, value=90, step=5)
    show_volume = st.checkbox("出来高を表示（詳細分析）", value=True)

    st.caption("API制限対策：取得結果はキャッシュされます。銘柄数を増やしすぎると取得が遅くなる場合があります。")


def _extract_tickers(selected_labels: List[str], manual_text: str, max_n: int) -> List[str]:
    tickers = []
    for label in selected_labels:
        t = label.split("—", 1)[0].strip()
        if t:
            tickers.append(t)

    if manual_text.strip():
        extra = [x.strip() for x in manual_text.split(",") if x.strip()]
        tickers.extend(extra)

    seen = set()
    uniq = []
    for t in tickers:
        if t not in seen:
            uniq.append(t)
            seen.add(t)

    return uniq[: max_n]


selected_tickers = _extract_tickers(selected_labels, manual, int(max_select))
if not selected_tickers:
    st.info("左のサイドバーから銘柄を選択してください。")
    st.stop()


def _suggest_fetch_days(display_bars: int) -> int:
    display_bars = int(np.clip(display_bars, 20, 90))
    # indicator stability buffer
    days = display_bars * 4 + 120
    return int(np.clip(days, 180, 720))


def _dedup_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _resolve_index_data(period_days: int) -> tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
    candidates_by_name: Dict[str, List[str]] = {}
    all_candidates: List[str] = []

    for name, primary in INDEX_TICKERS.items():
        cands = INDEX_TICKER_CANDIDATES.get(name, [primary])
        if primary not in cands:
            cands = [primary] + cands
        cands = _dedup_keep_order([c for c in cands if str(c).strip()])
        candidates_by_name[name] = cands
        all_candidates.extend(cands)

    prices = fetch_ohlcv(_dedup_keep_order(all_candidates), period_days=int(period_days), interval="1d")

    resolved: Dict[str, pd.DataFrame] = {}
    used: Dict[str, str] = {}
    for name, cands in candidates_by_name.items():
        for t in cands:
            df = prices.get(t)
            if df is not None and not df.empty:
                resolved[name] = df
                used[name] = t
                break

    return resolved, used


fetch_days = _suggest_fetch_days(int(candles))
price_dict = fetch_ohlcv(selected_tickers, period_days=int(fetch_days), interval="1d")

missing = [t for t in selected_tickers if t not in price_dict]
if missing:
    st.warning(
        "取得できない銘柄がありました：" + ", ".join(missing) + "\n\n"
        "（yfinance側でデータが存在しない、またはティッカー表記が異なる可能性があります）"
    )

available_tickers = [t for t in selected_tickers if t in price_dict]
if not available_tickers:
    st.error("選択した銘柄のデータを取得できませんでした。別の銘柄でお試しください。")
    st.stop()

name_map: Dict[str, str] = {}
try:
    if "yfinance" in universe_df.columns and "銘柄名" in universe_df.columns:
        name_map = dict(zip(universe_df["yfinance"].astype(str), universe_df["銘柄名"].astype(str)))
except Exception:
    name_map = {}


def _pretty_label(t: str) -> str:
    n = name_map.get(t)
    return f"{t}  {n}" if n else t


selected_pretty = [_pretty_label(t) for t in available_tickers]
chips_html = "".join([f'<span class="ticker-chip">{html.escape(text)}</span>' for text in selected_pretty])

st.markdown(
    f"""
<div class="ticker-bar">
  <div class="ticker-bar-title">選択中</div>
  <div class="ticker-chips">{chips_html}</div>
</div>
""",
    unsafe_allow_html=True,
)
st.caption(
    f"表示は『{int(candles)}本』。指標安定化のため、内部では最大 {int(fetch_days)} 日ぶん取得して必要な範囲だけ描画します。"
)

index_period_days = int(max(60, fetch_days))
index_data, index_used_ticker = _resolve_index_data(period_days=index_period_days)

tab0, tab1, tab2, tab3 = st.tabs(["📌 ダッシュボード", "🧩 マルチ銘柄", "🔎 詳細分析", "🧮 平均インデックス"])

with tab0:
    st.subheader("重要指数")
    cols = st.columns(4)
    items = list(INDEX_TICKERS.keys())

    for i, name in enumerate(items[:8]):
        df = index_data.get(name)
        move = last_close_and_change(df) if df is not None else None
        with cols[i % 4]:
            if move is None:
                st.metric(name, value="取得できません", delta="-")
            else:
                st.metric(
                    name,
                    value=f"{move.last:,.2f}",
                    delta=f"{move.change:,.2f} ({move.change_pct:+.2f}%)",
                )

    st.markdown("### 指数チャート（選択して表示）")
    available_indices = [n for n in INDEX_TICKERS.keys() if n in index_data]
    if not available_indices:
        st.info("指数データを取得できませんでした（yfinance側の制限や一時的な障害の可能性があります）。")
    else:
        chosen_index = st.radio(
            "表示する指数",
            options=available_indices,
            horizontal=True,
            label_visibility="collapsed",
        )
        df_idx = index_data[chosen_index]
        t_used = index_used_ticker.get(chosen_index, INDEX_TICKERS.get(chosen_index, chosen_index))
        if df_idx is not None and not df_idx.empty and len(df_idx) > int(candles):
            df_idx = df_idx.iloc[-int(candles):]
        fig_idx = focus_chart(
            ticker=f"{chosen_index} ({t_used})",
            df=df_idx,
            overlays=[],
            panels=[],
            candles=int(candles),
            show_volume=False,
        )
        st.plotly_chart(fig_idx, use_container_width=True, config=PLOTLY_CONFIG)

with tab1:
    st.subheader("複数銘柄をローソク足で同時表示（グリッド）")
    c1, c2 = st.columns([1, 2])
    with c1:
        n_cols = st.slider("列数（1〜4）", min_value=1, max_value=4, value=2, step=1)
    with c2:
        st.caption("列数を増やすほど一覧性が上がります（その分、1枚あたりは小さく見えます）。")

    fig_multi = multi_candlestick_subplots(price_dict, available_tickers, candles=int(candles), n_cols=int(n_cols))
    st.plotly_chart(fig_multi, use_container_width=True, config=PLOTLY_CONFIG)

with tab2:
    st.subheader("選択した銘柄を徹底的にいじる")
    focus_ticker = st.selectbox("分析する銘柄", options=available_tickers, index=0)

    st.markdown("### インジケーター（10種類以上から選択）")
    indicator_options = [
        "SMA", "EMA", "Bollinger", "Ichimoku", "VWAP", "ParabolicSAR", "Supertrend",
        "RSI", "MACD", "Stochastic", "ATR", "ADX", "OBV", "CCI", "Williams%R",
    ]
    selected_indicators = st.multiselect(
        "追加するインジケーター",
        options=indicator_options,
        default=["SMA", "Bollinger", "RSI"],
    )

    with st.expander("パラメータ（必要なものだけ触ればOK）", expanded=False):
        p1, p2, p3 = st.columns(3)
        with p1:
            sma_period = st.number_input("SMA period", 2, 300, 20)
            ema_period = st.number_input("EMA period", 2, 300, 20)
            rsi_period = st.number_input("RSI period", 2, 100, 14)
            atr_period = st.number_input("ATR period", 2, 100, 14)
            adx_period = st.number_input("ADX period", 2, 100, 14)
        with p2:
            bb_period = st.number_input("BB period", 2, 200, 20)
            bb_std = st.number_input("BB std", 0.5, 5.0, 2.0)
            macd_fast = st.number_input("MACD fast", 2, 60, 12)
            macd_slow = st.number_input("MACD slow", 2, 200, 26)
            macd_signal = st.number_input("MACD signal", 2, 60, 9)
        with p3:
            stoch_k = st.number_input("Stoch %K", 2, 60, 14)
            stoch_d = st.number_input("Stoch %D", 2, 30, 3)
            stoch_smooth = st.number_input("Stoch smooth", 1, 30, 3)
            psar_step = st.number_input("PSAR step", 0.01, 0.2, 0.02)
            psar_max_step = st.number_input("PSAR max", 0.05, 0.5, 0.2)

        ichi_tenkan = st.number_input("Ichimoku Tenkan", 2, 30, 9)
        ichi_kijun = st.number_input("Ichimoku Kijun", 5, 60, 26)
        ichi_senkou_b = st.number_input("Ichimoku SenkouB", 10, 120, 52)
        supertrend_period = st.number_input("Supertrend period", 2, 60, 10)
        supertrend_mult = st.number_input("Supertrend multiplier", 1.0, 10.0, 3.0)

    params = {
        "sma_period": int(sma_period),
        "ema_period": int(ema_period),
        "bb_period": int(bb_period),
        "bb_std": float(bb_std),
        "rsi_period": int(rsi_period),
        "macd_fast": int(macd_fast),
        "macd_slow": int(macd_slow),
        "macd_signal": int(macd_signal),
        "stoch_k": int(stoch_k),
        "stoch_d": int(stoch_d),
        "stoch_smooth": int(stoch_smooth),
        "atr_period": int(atr_period),
        "adx_period": int(adx_period),
        "cci_period": 20,
        "willr_period": 14,
        "ichi_tenkan": int(ichi_tenkan),
        "ichi_kijun": int(ichi_kijun),
        "ichi_senkou_b": int(ichi_senkou_b),
        "psar_step": float(psar_step),
        "psar_max_step": float(psar_max_step),
        "supertrend_period": int(supertrend_period),
        "supertrend_mult": float(supertrend_mult),
    }

    df_focus = price_dict[focus_ticker]
    overlays = build_indicator_overlays(df_focus, selected_indicators, params)
    panels = build_indicator_panels(df_focus, selected_indicators, params)

    fig_focus = focus_chart(
        ticker=focus_ticker,
        df=df_focus,
        overlays=overlays,
        panels=panels,
        candles=int(candles),
        show_volume=bool(show_volume),
    )
    st.plotly_chart(fig_focus, use_container_width=True, config=PLOTLY_CONFIG)

with tab3:
    st.subheader("選択銘柄の『平均インデックス（100基準）』")

    add_indices = st.multiselect(
        "平均インデックスに追加する指数（任意）",
        options=list(INDEX_TICKERS.keys()),
        default=[],
        help="日米混在でも線が途切れないように、営業日ズレは補正します（休場日は横ばいになります）。",
    )

    calc_price_dict: Dict[str, pd.DataFrame] = {t: price_dict[t] for t in available_tickers if t in price_dict}

    index_used: List[str] = []
    index_label: Dict[str, str] = {}

    for n in add_indices:
        df = index_data.get(n)
        t_used = index_used_ticker.get(n, INDEX_TICKERS.get(n, n))
        if df is None or df.empty:
            continue
        calc_price_dict[t_used] = df
        index_used.append(t_used)
        index_label[t_used] = n

    if len(calc_price_dict) < 2:
        st.info("平均インデックスは2銘柄以上で作成できます。")
    else:
        index_df = normalize_equal_weight_index(calc_price_dict)

        # Display window
        if index_df is not None and not index_df.empty and len(index_df) > int(candles):
            index_df = index_df.iloc[-int(candles):]

        # Normalize to 100 at start for all series
        if index_df is not None and not index_df.empty:
            if "EW_INDEX" in index_df.columns and index_df["EW_INDEX"].iloc[0] != 0:
                index_df["EW_INDEX"] = index_df["EW_INDEX"] / float(index_df["EW_INDEX"].iloc[0]) * 100.0

            for c in [c for c in index_df.columns if c.endswith("_NORM")]:
                v0 = float(index_df[c].iloc[0]) if len(index_df[c]) else None
                if v0 and v0 != 0:
                    index_df[c] = index_df[c] / v0 * 100.0

        fig_index = equal_weight_index_chart(
            index_df=index_df,
            stock_tickers=available_tickers,
            index_tickers=index_used,
            ticker_label=index_label,
        )
        st.plotly_chart(fig_index, use_container_width=True, config=PLOTLY_CONFIG)

st.divider()
st.caption("データ取得：Yahoo Finance / yfinance。JPXの銘柄一覧Excelを読み込み、検索・選択UIを作っています。")
