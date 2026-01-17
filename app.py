from __future__ import annotations

import textwrap
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

from src.data import INDEX_TICKERS, download_jpx_list, fetch_ohlcv, last_close_and_change, normalize_equal_weight_index
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

# -------------------------
# Sidebar: Universe + Controls
# -------------------------
with st.sidebar:
    st.subheader("銘柄選択")
    universe_df = download_jpx_list()

    # Filters
    colf1, colf2 = st.columns(2)
    with colf1:
        only_stocks = st.checkbox("株式中心", value=True, help="ETF/REIT等が混じる場合があるので、株式中心に絞ります（完全には保証できません）")
    with colf2:
        max_select = st.number_input("最大選択数", min_value=1, max_value=30, value=12, step=1)

    work_df = universe_df.copy()
    if only_stocks:
        # A gentle filter that keeps '内国株式' style rows. Column name differs by file version, so do best-effort.
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
        help="ここは検索できます。多すぎると表示が重くなるので、まずは 3〜8 銘柄がおすすめです。",
    )

    manual = st.text_input(
        "手動で追加（yfinanceティッカーをカンマ区切り）",
        value="",
        help="例：998407.O, 9432.T, 6501.T, ^N225 など",
    )

    st.divider()

    st.subheader("チャート設定")
    # "取得期間" はユーザーが見たいローソク足の本数と一致させる（最大90）
    candles = st.slider("取得期間（ローソク足本数 / 最大90）", min_value=20, max_value=90, value=90, step=5)
    show_volume = st.checkbox("出来高を表示（詳細分析）", value=True)

    st.caption(
        "API制限対策：取得結果はキャッシュされます。銘柄数を増やしすぎると取得が遅くなる場合があります。"
    )


def _extract_tickers(selected_labels: List[str], manual_text: str, max_n: int) -> List[str]:
    tickers = []
    for label in selected_labels:
        t = label.split("—", 1)[0].strip()
        if t:
            tickers.append(t)

    if manual_text.strip():
        extra = [x.strip() for x in manual_text.split(",") if x.strip()]
        tickers.extend(extra)

    # unique while preserving order
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
    """yfinance 取得期間（カレンダー日数）の目安。

    表示本数は最大 90 本なので、指標計算のための余裕を持たせつつ、
    取りすぎで重くならないように上限も設ける。
    """
    display_bars = int(np.clip(display_bars, 20, 90))
    # 例: 90本 -> 480日、20本 -> 200日
    days = display_bars * 4 + 120
    return int(np.clip(days, 180, 720))


# -------------------------
# Data fetch (cached)
# -------------------------
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


# -------------------------
# Selected tickers summary (shown away from the selector)
# -------------------------
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
st.markdown(
    """
<div class="ticker-bar">
  <div class="ticker-bar-title">選択中</div>
  <div class="ticker-chips">{chips}</div>
</div>
""".format(
        chips="".join([f"<span class=\"ticker-chip\">{text}</span>" for text in selected_pretty])
    ),
    unsafe_allow_html=True,
)
st.caption(
    f"表示は『{int(candles)}本』。指標安定化のため、内部では最大 {int(fetch_days)} 日ぶん取得して必要な範囲だけ描画します。"
)


# -------------------------
# Tabs
# -------------------------
tab0, tab1, tab2, tab3 = st.tabs(["📌 ダッシュボード", "🧩 マルチ銘柄", "🔎 詳細分析", "🧮 平均インデックス"])


with tab0:
    st.subheader("重要指数")

    idx_prices = fetch_ohlcv(list(INDEX_TICKERS.values()), period_days=60, interval="1d")

    cols = st.columns(4)
    items = list(INDEX_TICKERS.items())
    for i, (name, ticker) in enumerate(items[:8]):
        df = idx_prices.get(ticker)
        move = last_close_and_change(df) if df is not None else None
        with cols[i % 4]:
            if move is None:
                st.metric(name, value="-", delta="-")
            else:
                st.metric(
                    name,
                    value=f"{move.last:,.2f}",
                    delta=f"{move.change:,.2f} ({move.change_pct:+.2f}%)",
                )

    st.divider()
    st.subheader("今日のチェック（あなたのウォッチ）")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.write("選択中：", ", ".join(available_tickers))
        st.caption("ヒント：次のタブでローソク足を並べる / 詳細分析でインジケーターを重ねられます。")
    with c2:
        st.info(
            "表示が重い場合：\n"
            "- 銘柄数を減らす\n"
            "- 取得期間を短くする\n"
            "- ローソク本数を減らす\n"
        )


with tab1:
    st.subheader("複数銘柄をローソク足で同時表示")
    fig_multi = multi_candlestick_subplots(price_dict, available_tickers, candles=int(candles))
    st.plotly_chart(
        fig_multi,
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )


with tab2:
    st.subheader("選択した銘柄を徹底的にいじる")

    focus_ticker = st.selectbox("分析する銘柄", options=available_tickers, index=0)

    st.markdown("### インジケーター（10種類以上から選択）")
    indicator_options = [
        "SMA",
        "EMA",
        "Bollinger",
        "Ichimoku",
        "VWAP",
        "ParabolicSAR",
        "Supertrend",
        "RSI",
        "MACD",
        "Stochastic",
        "ATR",
        "ADX",
        "OBV",
        "CCI",
        "Williams%R",
    ]

    selected_indicators = st.multiselect(
        "追加するインジケーター",
        options=indicator_options,
        default=["SMA", "Bollinger", "RSI"],
        help="重ねすぎると読みにくくなるので、まずは 2〜4 個がおすすめです。",
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

    st.plotly_chart(
        fig_focus,
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )


with tab3:
    st.subheader("選択銘柄の『平均インデックス』")

    if len(available_tickers) < 2:
        st.info("平均インデックスは2銘柄以上で作成できます。左のサイドバーで銘柄数を増やしてください。")
    else:
        index_df = normalize_equal_weight_index(price_dict)
        # 表示本数に合わせて見える範囲も揃える
        if index_df is not None and not index_df.empty and len(index_df) > int(candles):
            index_df = index_df.iloc[-int(candles) :]
        # 見えている範囲の先頭を 100 にそろえて比較しやすくする
        if index_df is not None and not index_df.empty:
            base_v = float(index_df["EW_INDEX"].iloc[0]) if "EW_INDEX" in index_df.columns else None
            if base_v and base_v != 0:
                index_df["EW_INDEX"] = index_df["EW_INDEX"] / base_v * 100.0
            for c in [c for c in index_df.columns if c.endswith("_NORM")]:
                bv = float(index_df[c].iloc[0]) if len(index_df[c]) else None
                if bv and bv != 0:
                    index_df[c] = index_df[c] / bv * 100.0
        fig_index = equal_weight_index_chart(index_df, available_tickers)
        st.plotly_chart(
            fig_index,
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )

        st.caption(
            "作り方：日次リターンを銘柄ごとに計算し、その平均を積み上げた等金額（Equal-Weight）指数です。"
        )


st.divider()
st.caption(
    "データ取得：Yahoo Finance / yfinance。JPXの銘柄一覧Excelを読み込み、検索・選択UIを作っています。"
)
