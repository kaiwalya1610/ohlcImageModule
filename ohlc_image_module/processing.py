"""Windowing and normalisation utilities for OHLC image generation."""
from __future__ import annotations

from typing import Generator, Tuple

import pandas as pd

from .config import PRICE_COLUMNS


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
