import json

import pandas as pd
import pytest

from ohlc_image_module.config import RenderConfig
from ohlc_image_module.data import validate_df
from ohlc_image_module.metadata import build_metadata_row


def make_df_with_gap():
    index = pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-10"])
    data = {
        "Open": [1.0, 1.1, 1.2],
        "High": [1.2, 1.3, 1.4],
        "Low": [0.9, 1.0, 1.1],
        "Close": [1.05, 1.15, 1.25],
        "Volume": [1000, 1100, 1200],
    }
    return pd.DataFrame(data, index=index)


def make_valid_df():
    index = pd.date_range("2023-01-01", periods=5, freq="D")
    data = {
        "Open": [1, 2, 3, 4, 5],
        "High": [2, 3, 4, 5, 6],
        "Low": [0.5, 1.5, 2.5, 3.5, 4.5],
        "Close": [1.5, 2.5, 3.5, 4.5, 5.5],
        "Volume": [100, 120, 140, 160, 180],
    }
    return pd.DataFrame(data, index=index)


def test_validate_df_allows_monotonic_without_gaps():
    df = make_valid_df()
    # Should not raise when gaps are allowed.
    validate_df(df, interval="1d", fail_on_gaps=False)


def test_validate_df_raises_on_large_gap_when_requested():
    df = make_df_with_gap()
    with pytest.raises(ValueError):
        validate_df(df, interval="1d", fail_on_gaps=True)


def test_validate_df_rejects_unknown_interval_when_failing_on_gaps():
    df = make_valid_df()
    with pytest.raises(ValueError):
        validate_df(df, interval="weird", fail_on_gaps=True)


def test_build_metadata_row_serialises_expected_fields(tmp_path, monkeypatch):
    df = make_valid_df()
    cfg = RenderConfig(
        bg="white",
        up_color="black",
        down_color="gray",
        line_width=0.8,
        wick_width=0.6,
        include_wicks=True,
        include_volume=True,
        img_size=256,
        dpi=128,
        tight_layout_pad=0.0,
    )

    monkeypatch.setenv("PYTHONHASHSEED", "123")
    row = build_metadata_row(
        ticker="TEST",
        interval="1d",
        df_window=df,
        img_path=str(tmp_path / "image.png"),
        n_bars=len(df),
        window=5,
        stride=5,
        idx_start=0,
        idx_end=len(df) - 1,
        normalize="zscore",
        cfg=cfg,
    )

    assert row["ticker"] == "TEST"
    assert row["interval"] == "1d"
    assert row["seed"] == 123
    assert row["has_volume"] is True

    stats = json.loads(row["ohlc_stats_json"])
    for column in ["Open", "High", "Low", "Close"]:
        assert column in stats
        assert {"min", "max", "mean", "std"}.issubset(stats[column].keys())
    assert stats["start_iso"].startswith("2023-01")
    assert stats["end_iso"].startswith("2023-01")
