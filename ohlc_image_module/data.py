"""Data acquisition and validation helpers for OHLC image generation."""
from __future__ import annotations

import re
from typing import Optional

import pandas as pd
from dateutil import tz
import yfinance as yf

from .config import REQUIRED_COLUMNS


def fetch_ohlcv(ticker: str, start: str, end: str, interval: str) -> pd.DataFrame:
    """Fetch OHLCV data from Yahoo Finance and normalise the index."""

    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,
            prepost=False,
            progress=False,
        )
    except Exception as exc:  # pragma: no cover - defensive path
        raise RuntimeError(f"Failed to download data for {ticker!r}: {exc}") from exc

    # Flatten MultiIndex columns (yfinance wraps cols for multi-ticker support)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty:
        raise ValueError(f"No data returned for ticker {ticker!r} in the specified range.")

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Downloaded data missing required columns: {missing_cols}")

    df = df.loc[:, REQUIRED_COLUMNS].copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Expected DatetimeIndex from yfinance download.")

    # Convert to IST (Indian Standard Time)
    if df.index.tz is None:
        df.index = df.index.tz_localize(tz.UTC).tz_convert("Asia/Kolkata")
    else:
        df.index = df.index.tz_convert("Asia/Kolkata")
    df.index = df.index.tz_localize(None)  # Remove timezone for cleaner display
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()
    df = df.dropna(subset=["Open", "High", "Low", "Close"])

    return df


def _parse_interval_seconds(interval: str) -> Optional[float]:
    """Convert an interval string to seconds for gap validation."""

    match = re.fullmatch(r"(\d+)([a-zA-Z]+)", interval.strip())
    if not match:
        return None
    value = int(match.group(1))
    unit = match.group(2).lower()
    unit_seconds = {
        "m": 60,
        "min": 60,
        "h": 3600,
        "d": 86400,
        "wk": 604800,
        "w": 604800,
        "mo": 2592000,
        "month": 2592000,
    }.get(unit)
    if unit_seconds is None:
        return None
    return value * unit_seconds


def validate_df(df: pd.DataFrame, interval: str, fail_on_gaps: bool) -> None:
    """Validate the downloaded dataframe for ordering and temporal gaps."""

    if df.empty:
        raise ValueError("Dataframe is empty after preprocessing.")

    if not df.index.is_monotonic_increasing:
        raise ValueError("Datetime index must be monotonic increasing.")

    if fail_on_gaps:
        seconds = _parse_interval_seconds(interval)
        if seconds is None:
            raise ValueError(
                "Cannot validate gaps for unsupported interval format: %s" % interval
            )
        diffs = df.index.to_series().diff().dropna().dt.total_seconds()
        if not diffs.empty:
            max_gap = diffs.max()
            if max_gap > 5 * seconds:
                raise ValueError(
                    f"Detected temporal gap of {max_gap} seconds which exceeds tolerance "
                    f"for interval {interval}."
                )
