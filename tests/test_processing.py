import numpy as np
import pandas as pd
import pytest

from ohlc_image_module.processing import iter_windows, normalize_ohlc


def make_sample_df(length: int = 6) -> pd.DataFrame:
    index = pd.date_range("2023-01-01", periods=length, freq="D")
    data = {
        "Open": np.linspace(1.0, 6.0, length),
        "High": np.linspace(2.0, 7.0, length),
        "Low": np.linspace(0.5, 5.5, length),
        "Close": np.linspace(1.5, 6.5, length),
        "Volume": np.linspace(100, 600, length),
    }
    return pd.DataFrame(data, index=index)


def test_iter_windows_full_series_when_window_not_positive():
    df = make_sample_df()
    windows = list(iter_windows(df, window=0, stride=0))
    assert len(windows) == 1
    window_df, start_idx, end_idx = windows[0]
    pd.testing.assert_frame_equal(window_df, df)
    assert start_idx == 0
    assert end_idx == len(df) - 1


def test_iter_windows_sliding_windows_with_default_stride():
    df = make_sample_df(length=8)
    windows = list(iter_windows(df, window=4, stride=0))
    # Expect two non-overlapping windows because stride defaults to window size.
    assert len(windows) == 2
    first_df, first_start, first_end = windows[0]
    second_df, second_start, second_end = windows[1]

    assert first_start == 0
    assert first_end == 3
    assert second_start == 4
    assert second_end == 7
    pd.testing.assert_frame_equal(first_df, df.iloc[0:4])
    pd.testing.assert_frame_equal(second_df, df.iloc[4:8])


def test_normalize_ohlc_zscore():
    df = make_sample_df(length=3)
    norm_df = normalize_ohlc(df, mode="zscore")

    for column in ["Open", "High", "Low", "Close"]:
        series = norm_df[column]
        # z-score should have zero mean (within numerical tolerance) and unit variance.
        assert series.mean() == pytest.approx(0.0, abs=1e-8)
        assert series.std(ddof=0) == pytest.approx(1.0, abs=1e-6)

    # Volume should remain untouched.
    pd.testing.assert_series_equal(norm_df["Volume"], df["Volume"])


def test_normalize_ohlc_minmax():
    df = make_sample_df(length=4)
    norm_df = normalize_ohlc(df, mode="minmax")
    for column in ["Open", "High", "Low", "Close"]:
        series = norm_df[column]
        assert series.min() == pytest.approx(0.0)
        assert series.max() == pytest.approx(1.0)

    pd.testing.assert_series_equal(norm_df["Volume"], df["Volume"])
