from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.indicators import IndicatorLine, IndicatorPanel


def _base_layout(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(
        title=title,
        margin=dict(l=20, r=20, t=55, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
        template="plotly_dark",
        dragmode="pan",
        font=dict(size=13),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        showline=False,
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.14)",
        zeroline=False,
    )
    fig.update_yaxes(
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        showline=False,
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.14)",
        zeroline=False,
    )
    return fig


def _constrain_xaxis(fig: go.Figure, x_min, x_max) -> go.Figure:
    """Constrain zoom/pan so the viewport cannot go beyond the data bounds.

    Plotly >= 5.17 supports `minallowed` / `maxallowed` for this.
    """
    if x_min is None or x_max is None:
        return fig

    # Avoid pandas Timestamp serialization quirks by keeping them as-is.
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
) -> go.Figure:
    tickers = [t for t in tickers if t in price_dict]
    rows = max(1, len(tickers))

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[1.0 / rows] * rows,
    )

    for i, t in enumerate(tickers, start=1):
        df = price_dict[t].copy()
        if len(df) > candles:
            df = df.iloc[-candles:]

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
            row=i,
            col=1,
        )
        fig.update_yaxes(title_text=t, row=i, col=1)

    # Shared x-axis bounds (prevent panning beyond the latest candle)
    x_min = None
    x_max = None
    for t in tickers:
        df = price_dict[t]
        if df is None or df.empty:
            continue
        view = df.iloc[-candles:] if len(df) > candles else df
        if view.empty:
            continue
        mn = view.index.min()
        mx = view.index.max()
        x_min = mn if x_min is None else min(x_min, mn)
        x_max = mx if x_max is None else max(x_max, mx)

    fig.update_layout(height=max(450, 220 * rows))
    fig = _base_layout(fig, title="マルチ銘柄ローソク足")
    fig.update_layout(uirevision="|".join(tickers))
    fig = _constrain_xaxis(fig, x_min, x_max)
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

    # Normalize heights to sum=1
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

    # --- Price row ---
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

    # overlays (SMA/EMA/etc)
    for line in overlays:
        s = line.series.reindex(view_index)
        fig.add_trace(
            go.Scatter(
                x=view_index,
                y=s.values,
                mode="lines",
                name=line.name,
            ),
            row=1,
            col=1,
        )

    # --- Volume row ---
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
        fig.update_yaxes(title_text="Vol", row=next_row, col=1)
        next_row += 1

    # --- Indicator panels ---
    for p in panels:
        for line in p.lines:
            s = line.series.reindex(view_index)
            if line.kind == "bar":
                fig.add_trace(
                    go.Bar(x=view_index, y=s.values, name=line.name),
                    row=next_row,
                    col=1,
                )
            else:
                fig.add_trace(
                    go.Scatter(x=view_index, y=s.values, mode="lines", name=line.name),
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
    fig = _constrain_xaxis(fig, view_index.min(), view_index.max())
    return fig


def equal_weight_index_chart(index_df: pd.DataFrame, tickers: List[str]) -> go.Figure:
    fig = go.Figure()

    if index_df is None or index_df.empty:
        return _base_layout(fig, title="平均インデックス")

    fig.add_trace(go.Scatter(x=index_df.index, y=index_df["EW_INDEX"], mode="lines", name="EW_INDEX"))

    # Show up to 6 underlying normalized series (too many makes it unreadable)
    shown = 0
    for t in tickers:
        col = f"{t}_NORM"
        if col in index_df.columns:
            fig.add_trace(go.Scatter(x=index_df.index, y=index_df[col], mode="lines", name=t, opacity=0.55))
            shown += 1
        if shown >= 6:
            break

    fig = _base_layout(fig, title="選択銘柄から作る『平均インデックス』")
    fig.update_layout(height=520, uirevision="ew-index")
    if not index_df.empty:
        fig = _constrain_xaxis(fig, index_df.index.min(), index_df.index.max())
    return fig
