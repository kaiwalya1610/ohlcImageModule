"""Rendering helpers for deterministic candlestick image creation."""
from __future__ import annotations

import io
import math
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
from PIL import Image

from .config import RenderConfig

try:  # Pillow 10+ exposes the Resampling enum
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
except AttributeError:  # pragma: no cover - compatibility path
    RESAMPLE_BICUBIC = Image.BICUBIC


def _prepare_volume(series: pd.Series) -> pd.Series:
    """Min-max normalise volume data for the plotting panel."""

    min_val = series.min()
    max_val = series.max()
    if math.isclose(max_val, min_val):
        return pd.Series(np.zeros(len(series)), index=series.index, name=series.name)
    return (series - min_val) / (max_val - min_val + 1e-8)


def _with_alpha(color: str, alpha: float) -> str:
    """Apply transparency to a matplotlib-compatible colour string."""

    rgba = mcolors.to_rgba(color, alpha=alpha)
    return mcolors.to_hex(rgba, keep_alpha=True)


def render_candlestick(df_window: pd.DataFrame, cfg: RenderConfig) -> Image.Image:
    """Render a candlestick chart into a PIL image."""

    plot_df = df_window.copy()
    include_volume = cfg.include_volume and "Volume" in plot_df.columns
    if include_volume:
        plot_df["Volume"] = _prepare_volume(plot_df["Volume"])

    wick_colors = (
        {"up": cfg.up_color, "down": cfg.down_color}
        if cfg.include_wicks
        else {"up": cfg.bg, "down": cfg.bg}
    )
    edge_colors = {"up": cfg.up_color, "down": cfg.down_color}
    volume_color = _with_alpha(cfg.up_color, 0.5)
    market_colors = mpf.make_marketcolors(
        up=cfg.up_color,
        down=cfg.down_color,
        edge=edge_colors,
        wick=wick_colors,
        volume={"up": volume_color, "down": volume_color},
        ohlc=edge_colors,
    )

    rc = {
        "axes.facecolor": cfg.bg,
        "axes.edgecolor": cfg.bg,
        "figure.facecolor": cfg.bg,
        "savefig.facecolor": cfg.bg,
        "axes.grid": False,
        "xtick.color": cfg.bg,
        "ytick.color": cfg.bg,
    }

    style = mpf.make_mpf_style(marketcolors=market_colors, rc=rc)
    chart_type = "candle"
    wick_width = cfg.wick_width if cfg.include_wicks else 0.0

    fig, axes = mpf.plot(
        plot_df,
        type=chart_type,
        volume=include_volume,
        style=style,
        figsize=(cfg.img_size / cfg.dpi, cfg.img_size / cfg.dpi),
        tight_layout=True,
        update_width_config={
            "candle_linewidth": cfg.line_width,
            "wick_linewidth": wick_width,
            "volume_linewidth": cfg.line_width,
        },
        axisoff=True,
        returnfig=True,
        datetime_format="",
        xrotation=0,
    )

    axes_iter: Sequence = axes if isinstance(axes, Sequence) else [axes]
    for ax in axes_iter:
        ax.set_axis_off()

    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=cfg.dpi,
        bbox_inches="tight",
        pad_inches=cfg.tight_layout_pad,
    )
    plt.close(fig)
    buf.seek(0)
    image = Image.open(buf).convert("RGB")
    image = image.resize((cfg.img_size, cfg.img_size), RESAMPLE_BICUBIC)
    return image
