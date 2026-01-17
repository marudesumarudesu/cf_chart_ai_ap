from __future__ import annotations

from math import ceil
from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.indicators import IndicatorLine, IndicatorPanel


def _base_layout(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(
        title=title,
        margin=dict(l=18, r=18, t=55, b=18),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
        template="plotly_dark",
        dragmode="pan",
        font=dict(size=14, color="#F8FAFC"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.14)",
        tickfont=dict(color="rgba(226, 232, 240, 0.92)"),
        zeroline=False,
        rangeslider_visible=False,
    )
    fig.update_yaxes(
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.14)",
        tickfont=dict(color="rgba(226, 232, 240, 0.92)"),
        zeroline=False,
    )
    return fig


def _constrain_all_xaxes(fig: go.Figure, x_min, x_max) -> go.Figure:
    """Constrain zoom/pan so viewport cannot go beyond data bounds (all x-axes)."""
    if x_min is None or x_max is None:
        return fig
    fig.update_xaxes(
        range=[x_min, x_max],
        minallowed=x_min,
        maxallowed=x_max,
        rangeslider_visible=False,
    )
    return fig


def multi_candlestick_subplots(
    price_dict: Dict[str, pd.DataFrame],
    tickers: List[str],
    candles: int = 90,
    n_cols: int = 1,
) -> go.Figure:
    """Multi-ticker candlestick in a grid layout (1-4 columns)."""
    tickers = [t for t in tickers if t in price_dict]
    if not tickers:
        return _base_layout(go.Figure(), title="マルチ銘柄ローソク足")

    n_cols = int(max(1, min(4, n_cols)))
    n_rows = int(ceil(len(tickers) / n_cols))

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        shared_xaxes=False,
        vertical_spacing=0.06,
        horizontal_spacing=0.06,
        subplot_titles=tickers,
    )

    # Compute overall x-range from the displayed window only
    x_min = None
    x_max = None

    for idx, t in enumerate(tickers):
        r = idx // n_cols + 1
        c = idx % n_cols + 1

        df = price_dict[t].copy()
        if len(df) > candles:
            df = df.iloc[-candles:]

        if not df.empty:
            mn = df.index.min()
            mx = df.index.max()
            x_min = mn if x_min is None else min(x_min, mn)
            x_max = mx if x_max is None else max(x_max, mx)

        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df.get("Open"),
                high=df.get("High"),
                low=df.get("Low"),
                close=df.get("Close"),
                name=t,
                showlegend=False,
            ),
            row=r,
            col=c,
        )

    height = max(520, 320 * n_rows)
    fig.update_layout(height=height)
    fig = _base_layout(fig, title="マルチ銘柄ローソク足（グリッド）")
    fig.update_layout(uirevision="|".join(tickers))
    fig = _constrain_all_xaxes(fig, x_min, x_max)
    return fig


def focus_chart(
    ticker: str,
    df: pd.DataFrame,
    overlays: List[IndicatorLine],
    panels: List[IndicatorPanel],
    candles: int = 90,
    show_volume: bool = True,
) -> go.Figure:
    df = df.copy()
    if len(df) > candles:
        df = df.iloc[-candles:]

    view_index = df.index
    n_panels = len(panels)
    rows = 1 + (1 if show_volume else 0) + n_panels

    row_heights: List[float] = []
    row_heights.append(0.60)
    if show_volume:
        row_heights.append(0.15)
    for _ in range(n_panels):
        row_heights.append(max(0.25 / max(1, n_panels), 0.12))

    s = sum(row_heights)
    row_heights = [h / s for h in row_heights]

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=row_heights,
        subplot_titles=["Price"]
        + (["Volume"] if show_volume else [])
        + [p.title for p in panels],
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df.get("Open"),
            high=df.get("High"),
            low=df.get("Low"),
            close=df.get("Close"),
            name=f"{ticker} OHLC",
        ),
        row=1,
        col=1,
    )

    for line in overlays:
        s2 = line.series.reindex(view_index)
        fig.add_trace(
            go.Scatter(
                x=view_index,
                y=s2.values,
                mode="lines",
                name=line.name,
            ),
            row=1,
            col=1,
        )

    next_row = 2
    if show_volume and "Volume" in df.columns:
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["Volume"],
                name="Volume",
                showlegend=False,
            ),
            row=next_row,
            col=1,
        )
        next_row += 1

    for p in panels:
        for line in p.lines:
            s2 = line.series.reindex(view_index)
            if line.kind == "bar":
                fig.add_trace(
                    go.Bar(x=view_index, y=s2.values, name=line.name),
                    row=next_row,
                    col=1,
                )
            else:
                fig.add_trace(
                    go.Scatter(x=view_index, y=s2.values, mode="lines", name=line.name),
                    row=next_row,
                    col=1,
                )

        if p.y_ref_lines:
            for y in p.y_ref_lines:
                fig.add_hline(y=y, line_width=1, opacity=0.35, row=next_row, col=1)

        next_row += 1

    fig.update_layout(height=max(650, 220 * rows))
    fig = _base_layout(fig, title=f"{ticker} — 詳細チャート")
    fig.update_layout(uirevision=ticker)
    fig = _constrain_all_xaxes(fig, view_index.min(), view_index.max())
    return fig


def equal_weight_index_chart(
    index_df: pd.DataFrame,
    stock_tickers: List[str],
    index_tickers: List[str],
    ticker_label: Optional[Dict[str, str]] = None,
) -> go.Figure:
    """EW index chart with clear line distinctions:
       - EW_INDEX: thick solid
       - Indices: medium dotted
       - Individual stocks: thin + low opacity
    """
    fig = go.Figure()
    ticker_label = ticker_label or {}

    if index_df is None or index_df.empty:
        return _base_layout(fig, title="平均インデックス")

    # EW index (dominant)
    if "EW_INDEX" in index_df.columns:
        fig.add_trace(
            go.Scatter(
                x=index_df.index,
                y=index_df["EW_INDEX"],
                mode="lines",
                name="平均インデックス（EW）",
                line=dict(width=4),
            )
        )

    # Indices (clear but secondary)
    for t in index_tickers:
        col = f"{t}_NORM"
        if col not in index_df.columns:
            continue
        name = ticker_label.get(t, t)
        fig.add_trace(
            go.Scatter(
                x=index_df.index,
                y=index_df[col],
                mode="lines",
                name=f"指数：{name}",
                line=dict(width=2.6, dash="dot"),
                opacity=0.95,
            )
        )

    # Individual stocks (background context)
    for t in stock_tickers:
        col = f"{t}_NORM"
        if col not in index_df.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=index_df.index,
                y=index_df[col],
                mode="lines",
                name=t,
                line=dict(width=1.2),
                opacity=0.30,
                showlegend=False,  # avoid legend clutter; hover still shows series
            )
        )

    fig = _base_layout(fig, title="平均インデックス（100基準）")
    fig.update_layout(height=560, uirevision="ew-index")
    fig = _constrain_all_xaxes(fig, index_df.index.min(), index_df.index.max())
    return fig                              
