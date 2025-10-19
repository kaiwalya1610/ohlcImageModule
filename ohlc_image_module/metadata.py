"""Metadata helpers for OHLC image generation."""
from __future__ import annotations

import json
from typing import Dict, Optional, Any

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
    segment_id: Optional[int] = None,
    segmentation_mode: Optional[str] = None,
    segmentation_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, object]:
    """Construct the metadata dictionary for a rendered image.
    
    Args:
        ticker: Stock ticker symbol
        interval: Time interval (e.g., "1d", "1h")
        df_window: DataFrame window/segment
        img_path: Path to saved image
        n_bars: Number of bars in window
        window: Window size (for fixed mode)
        stride: Stride (for fixed mode)
        idx_start: Starting index in original series
        idx_end: Ending index in original series
        normalize: Normalization mode
        cfg: Render configuration
        segment_id: Segment identifier (for both dynamic and fixed modes)
        segmentation_mode: "dynamic" or "fixed"
        segmentation_config: Segmentation configuration object
        
    Returns:
        Metadata dictionary
    """

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

    # Build base metadata
    metadata = {
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
        "ohlc_stats_json": json.dumps(ohlc_stats, sort_keys=True),
    }
    
    # Add segmentation metadata
    if segment_id is not None:
        metadata["segment_id"] = segment_id
    
    if segmentation_mode is not None:
        metadata["segmentation_mode"] = segmentation_mode
    else:
        # Default to "fixed" for backward compatibility
        metadata["segmentation_mode"] = "fixed"
    
    if segmentation_config is not None:
        seg_params = {
            "mode": segmentation_config.get("mode", "dynamic"),
            "model": segmentation_config.get("model", "l2"),
            "penalty": segmentation_config.get("penalty", 3.0),
            "min_segment_length": segmentation_config.get("min_segment_length", 10),
            "max_segment_length": segmentation_config.get("max_segment_length", 200),
            "jump": segmentation_config.get("jump", 5),
        }
        metadata["segmentation_params_json"] = json.dumps(seg_params, sort_keys=True)
    
    return metadata
