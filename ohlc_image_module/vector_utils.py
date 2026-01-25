"""Time-series vectorization utilities for enriched OHLC embeddings.

This module provides functions to generate fixed-length "DNA" vectors from
OHLC data for hybrid retrieval alongside visual embeddings. The vectors
capture multiple channels of information:
- Normalized price series
- RSI (Relative Strength Index)
- Log returns
- Relative volume
"""
from __future__ import annotations

import numpy as np
import talib
from scipy.interpolate import interp1d

# Constants
DEFAULT_TARGET_LEN = 128
RSI_DEFAULT_PERIOD = 14
RELATIVE_VOL_PERIOD = 20
NUM_CHANNELS = 4


def interpolate_series(data: np.ndarray, target_len: int = DEFAULT_TARGET_LEN) -> np.ndarray:
    """Interpolate a time series to a fixed length using linear interpolation.

    Args:
        data: Input array of values
        target_len: Target output length (default: 128)

    Returns:
        np.float32 array of shape (target_len,)

    Edge cases:
        - If len(data) < 2: returns zero-vector
        - NaN/Inf values are replaced with 0.0 before interpolation
    """
    data = np.asarray(data, dtype=np.float64)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    if len(data) < 2:
        return np.zeros(target_len, dtype=np.float32)

    # Create interpolation function
    x_original = np.linspace(0, 1, len(data))
    x_target = np.linspace(0, 1, target_len)

    interp_func = interp1d(x_original, data, kind='linear', fill_value='extrapolate')
    result = interp_func(x_target)

    return np.asarray(result, dtype=np.float32)


def standard_scale_series(data: np.ndarray) -> np.ndarray:
    """Apply z-score standardization to a series.

    Computes (x - mean) / std for each element.

    Args:
        data: Input array of values

    Returns:
        np.float32 array with z-score standardized values

    Edge cases:
        - If std < 1e-10 (flat-line): returns zeros
        - NaN/Inf values are replaced with 0.0 before computation
    """
    data = np.asarray(data, dtype=np.float64)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    mean = np.mean(data)
    std = np.std(data)

    if std < 1e-10:
        return np.zeros(len(data), dtype=np.float32)

    result = (data - mean) / std
    return np.asarray(result, dtype=np.float32)


def compute_rsi(prices: np.ndarray, period: int = RSI_DEFAULT_PERIOD) -> np.ndarray:
    """Compute RSI (Relative Strength Index) using TA-Lib.

    Uses TA-Lib's RSI implementation which applies Wilder's smoothing method.

    Args:
        prices: Array of closing prices
        period: RSI period (default: 14)

    Returns:
        np.float32 array with RSI values (0-100 range)

    Edge cases:
        - If len(prices) < 2: returns array of 50s (neutral)
        - NaN values from TA-Lib (initial period) are filled with 50 (neutral)
    """
    prices = np.asarray(prices, dtype=np.float64)
    prices = np.nan_to_num(prices, nan=0.0, posinf=0.0, neginf=0.0)

    n = len(prices)
    if n < 2:
        return np.full(n if n > 0 else 1, 50.0, dtype=np.float32)

    # Use TA-Lib RSI
    rsi = talib.RSI(prices, timeperiod=period)

    # Fill NaN values (initial period) with neutral RSI of 50
    rsi = np.nan_to_num(rsi, nan=50.0)

    return rsi.astype(np.float32)


def compute_log_returns(prices: np.ndarray) -> np.ndarray:
    """Compute log returns: ln(P_t / P_{t-1}).

    Args:
        prices: Array of prices

    Returns:
        np.float32 array of log returns (same length as input)

    Edge cases:
        - First element is set to 0.0
        - Invalid ratios (<=0) are replaced with 0.0
    """
    prices = np.asarray(prices, dtype=np.float64)
    prices = np.nan_to_num(prices, nan=0.0, posinf=0.0, neginf=0.0)

    n = len(prices)
    if n < 2:
        return np.zeros(n if n > 0 else 1, dtype=np.float32)

    log_returns = np.zeros(n, dtype=np.float64)

    for i in range(1, n):
        if prices[i - 1] > 0 and prices[i] > 0:
            log_returns[i] = np.log(prices[i] / prices[i - 1])
        else:
            log_returns[i] = 0.0

    return log_returns.astype(np.float32)


def compute_relative_volume(volume: np.ndarray, period: int = RELATIVE_VOL_PERIOD) -> np.ndarray:
    """Compute relative volume: Volume / SMA(Volume, period).

    Args:
        volume: Array of volume values
        period: Lookback period for SMA (default: 20)

    Returns:
        np.float32 array of relative volume values

    Edge cases:
        - Early bars use cumulative average instead of fixed period
        - Zero SMA returns 1.0 (neutral)
    """
    volume = np.asarray(volume, dtype=np.float64)
    volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)

    n = len(volume)
    if n == 0:
        return np.array([1.0], dtype=np.float32)

    rel_vol = np.ones(n, dtype=np.float64)

    for i in range(n):
        if i == 0:
            # First bar: relative volume is 1.0 (no history)
            rel_vol[i] = 1.0
        elif i < period:
            # Use cumulative average for early bars
            avg = np.mean(volume[:i])
            if avg < 1e-10:
                rel_vol[i] = 1.0
            else:
                rel_vol[i] = volume[i] / avg
        else:
            # Use rolling window average
            avg = np.mean(volume[i - period:i])
            if avg < 1e-10:
                rel_vol[i] = 1.0
            else:
                rel_vol[i] = volume[i] / avg

    return rel_vol.astype(np.float32)


def generate_enriched_vector(
    close: np.ndarray,
    volume: np.ndarray,
    target_len: int = DEFAULT_TARGET_LEN
) -> np.ndarray:
    """Generate a multi-channel enriched vector from OHLC data.

    Creates a 4-channel vector by computing and concatenating:
    - Ch1: Interpolated close prices, z-score normalized
    - Ch2: RSI, interpolated and z-score normalized
    - Ch3: Log returns, interpolated and z-score normalized
    - Ch4: Relative volume, interpolated and z-score normalized

    Args:
        close: Array of closing prices
        volume: Array of volume values
        target_len: Length per channel (default: 128)

    Returns:
        np.float32 array of shape (4 * target_len,) = (512,) by default
    """
    close = np.asarray(close, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)

    # Channel 1: Normalized price
    ch1_interp = interpolate_series(close, target_len)
    ch1 = standard_scale_series(ch1_interp)

    # Channel 2: RSI
    rsi = compute_rsi(close)
    ch2_interp = interpolate_series(rsi, target_len)
    ch2 = standard_scale_series(ch2_interp)

    # Channel 3: Log returns
    log_ret = compute_log_returns(close)
    ch3_interp = interpolate_series(log_ret, target_len)
    ch3 = standard_scale_series(ch3_interp)

    # Channel 4: Relative volume
    rel_vol = compute_relative_volume(volume)
    ch4_interp = interpolate_series(rel_vol, target_len)
    ch4 = standard_scale_series(ch4_interp)

    # Concatenate all channels
    result = np.concatenate([ch1, ch2, ch3, ch4])

    return result.astype(np.float32)
