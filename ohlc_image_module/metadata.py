"""Metadata helpers for OHLC image generation."""
from __future__ import annotations

import json
import os
from typing import Dict

import pandas as pd

from .config import PRICE_COLUMNS, RenderConfig


def build_metadata_row(
    ticker: str,
    interval: str,
    df_window: pd.DataFrame,
    img_path: str,
    n_bars: int,
    window: int,
    stride: int,
    idx_start: int,
    idx_end: int,
    normalize: str,
    cfg: RenderConfig,
) -> Dict[str, object]:
    """Construct the metadata dictionary for a rendered image."""

    ohlc_stats: Dict[str, Dict[str, float]] = {}
    for col in PRICE_COLUMNS:
        series = df_window[col]
        ohlc_stats[col] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "std": float(series.std(ddof=0)),
        }

    start_ts = df_window.index[0]
    end_ts = df_window.index[-1]
    ohlc_stats["start_iso"] = start_ts.isoformat()
    ohlc_stats["end_iso"] = end_ts.isoformat()

    return {
        "ticker": ticker,
        "interval": interval,
        "start_ts": start_ts.isoformat(),
        "end_ts": end_ts.isoformat(),
        "img_path": img_path,
        "n_bars": n_bars,
        "window": window,
        "stride": stride,
        "index_start": idx_start,
        "index_end": idx_end,
        "normalize": normalize,
        "bg": cfg.bg,
        "up_color": cfg.up_color,
        "down_color": cfg.down_color,
        "line_width": cfg.line_width,
        "wick_width": cfg.wick_width,
        "include_wicks": cfg.include_wicks,
        "has_volume": cfg.include_volume,
        "img_size": cfg.img_size,
        "dpi": cfg.dpi,
        "seed": int(os.environ.get("PYTHONHASHSEED", "0")),
        "ohlc_stats_json": json.dumps(ohlc_stats, sort_keys=True),
    }
