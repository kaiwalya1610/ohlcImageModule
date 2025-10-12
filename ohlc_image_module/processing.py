"""Windowing and normalisation utilities for OHLC image generation."""
from __future__ import annotations

from typing import Generator, Tuple, Dict, Any

import pandas as pd

from .config import PRICE_COLUMNS
from .segmentation import create_segmenter


def iter_windows(
    df: pd.DataFrame, window: int, stride: int
) -> Generator[Tuple[pd.DataFrame, int, int], None, None]:
    """Yield rolling windows of the dataframe."""

    n = len(df)
    if window <= 0 or window >= n:
        yield df, 0, n - 1
        return

    step = stride if stride > 0 else window
    for start_idx in range(0, n - window + 1, step):
        end_idx = start_idx + window
        yield df.iloc[start_idx:end_idx], start_idx, end_idx - 1


def iter_segments(
    df: pd.DataFrame, config: Dict[str, Any]
) -> Generator[Tuple[pd.DataFrame, int, int, int], None, None]:
    """Yield segments of the dataframe based on segmentation configuration.
    
    This function provides a unified interface for both dynamic (change-point based)
    and fixed-window segmentation modes.
    
    Args:
        df: DataFrame with OHLC data and DatetimeIndex
        config: Segmentation configuration
        
    Yields:
        Tuples of (segment_df, start_idx, end_idx, segment_id) where:
            - segment_df: DataFrame slice for the segment
            - start_idx: Starting index in original df (inclusive)
            - end_idx: Ending index in original df (inclusive)
            - segment_id: Sequential segment identifier (0-indexed)
    """
    n = len(df)
    if n == 0:
        return
    
    # Create appropriate segmenter
    try:
        segmenter = create_segmenter(config)
        segments = segmenter.segment(df)
    except ImportError:
        # Fallback to fixed windowing if ruptures not available in dynamic mode
        import warnings
        warnings.warn(
            "Dynamic segmentation requested but ruptures not installed. "
            "Falling back to fixed-window mode.",
            UserWarning
        )
        min_seg_len = config.get("min_segment_length", 10)
        fallback_config = {
            "mode": "fixed",
            "window": min_seg_len * 2,
            "stride": min_seg_len
        }
        segmenter = create_segmenter(fallback_config)
        segments = segmenter.segment(df)
    
    # Yield segments
    for segment_id, (start_idx, end_idx) in enumerate(segments):
        segment_df = df.iloc[start_idx:end_idx + 1]
        yield segment_df, start_idx, end_idx, segment_id


def normalize_ohlc(df_window: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Normalise OHLC columns in the provided window."""

    df_norm = df_window.copy()
    if mode == "none":
        return df_norm

    price_values = df_window[PRICE_COLUMNS]
    if mode == "zscore":
        mean = price_values.mean()
        std = price_values.std(ddof=0)
        std = std.mask(std < 1e-12, 1.0)
        df_norm[PRICE_COLUMNS] = (price_values - mean) / (std + 1e-8)
    elif mode == "minmax":
        min_val = price_values.min()
        max_val = price_values.max()
        denom = (max_val - min_val).mask((max_val - min_val) < 1e-12, 1.0)
        df_norm[PRICE_COLUMNS] = (price_values - min_val) / (denom + 1e-8)
        df_norm[PRICE_COLUMNS] = df_norm[PRICE_COLUMNS].clip(0.0, 1.0)
    else:
        raise ValueError(f"Unsupported normalization mode: {mode}")

    return df_norm
