"""Tests for dynamic segmentation functionality."""
from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from ohlc_image_module.segmentation import (
    RupturesSegmenter,
    FixedWindowSegmenter,
    create_segmenter,
)
from ohlc_image_module.processing import iter_segments


# Test fixtures
@pytest.fixture
def simple_ohlc_df():
    """Create a simple OHLC DataFrame for testing."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="1D")
    np.random.seed(42)
    
    close = 100 + np.cumsum(np.random.randn(100) * 0.5)
    
    df = pd.DataFrame({
        "Open": close + np.random.randn(100) * 0.2,
        "High": close + np.abs(np.random.randn(100) * 0.5),
        "Low": close - np.abs(np.random.randn(100) * 0.5),
        "Close": close,
        "Volume": np.random.randint(1000, 10000, 100),
    }, index=dates)
    
    return df


@pytest.fixture
def regime_change_df():
    """Create a DataFrame with clear regime changes."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="1D")
    
    # Create three distinct regimes
    regime1 = np.ones(30) * 100  # Flat at 100
    regime2 = np.linspace(100, 120, 40)  # Uptrend
    regime3 = np.ones(30) * 120  # Flat at 120
    
    close = np.concatenate([regime1, regime2, regime3])
    close += np.random.randn(100) * 0.5  # Add small noise
    
    df = pd.DataFrame({
        "Open": close + np.random.randn(100) * 0.2,
        "High": close + np.abs(np.random.randn(100) * 0.3),
        "Low": close - np.abs(np.random.randn(100) * 0.3),
        "Close": close,
        "Volume": np.random.randint(1000, 10000, 100),
    }, index=dates)
    
    return df


@pytest.fixture
def volatile_df():
    """Create a DataFrame with high volatility."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="1D")
    np.random.seed(42)
    
    close = 100 + np.cumsum(np.random.randn(100) * 5)  # High volatility
    
    df = pd.DataFrame({
        "Open": close + np.random.randn(100) * 2,
        "High": close + np.abs(np.random.randn(100) * 3),
        "Low": close - np.abs(np.random.randn(100) * 3),
        "Close": close,
        "Volume": np.random.randint(1000, 10000, 100),
    }, index=dates)
    
    return df


# Config dict tests (removed validation - keeping simple)
class TestSegmentationConfig:
    """Test segmentation config dicts."""
    
    def test_default_config(self):
        """Test default configuration."""
        from ohlc_image_module.segmentation import DEFAULT_CONFIG
        assert DEFAULT_CONFIG["mode"] == "dynamic"
        assert DEFAULT_CONFIG["model"] == "l2"
        assert DEFAULT_CONFIG["penalty"] == 3.0
        assert DEFAULT_CONFIG["min_segment_length"] == 10
        assert DEFAULT_CONFIG["max_segment_length"] == 200
        assert DEFAULT_CONFIG["jump"] == 5
    
    def test_fixed_mode_config(self):
        """Test fixed mode configuration."""
        config = {"mode": "fixed", "window": 50, "stride": 25}
        assert config["mode"] == "fixed"
        assert config["window"] == 50
        assert config["stride"] == 25


# RupturesSegmenter tests
class TestRupturesSegmenter:
    """Test RupturesSegmenter functionality."""
    
    def test_basic_segmentation(self, simple_ohlc_df):
        """Test basic segmentation on simple data."""
        pytest.importorskip("ruptures")
        
        config = {
            "mode": "dynamic",
            "penalty": 5.0,
            "min_segment_length": 10,
        }
        segmenter = RupturesSegmenter(config)
        segments = segmenter.segment(simple_ohlc_df)
        
        assert len(segments) > 0
        assert all(isinstance(seg, tuple) and len(seg) == 2 for seg in segments)
    
    def test_regime_change_detection(self, regime_change_df):
        """Test detection of regime changes."""
        pytest.importorskip("ruptures")
        
        config = {
            "mode": "dynamic",
            "penalty": 10.0,
            "min_segment_length": 10,
        }
        segmenter = RupturesSegmenter(config)
        segments = segmenter.segment(regime_change_df)
        
        # Should detect at least 2 segments (regime changes)
        assert len(segments) >= 2
    
    def test_segment_coverage(self, simple_ohlc_df):
        """Test that segments cover the entire series without gaps or overlaps."""
        pytest.importorskip("ruptures")
        
        config = {"penalty": 3.0}
        segmenter = RupturesSegmenter(config)
        segments = segmenter.segment(simple_ohlc_df)
        
        n = len(simple_ohlc_df)
        covered = set()
        
        for start, end in segments:
            segment_indices = set(range(start, end + 1))
            # Check no overlaps
            assert not (covered & segment_indices), "Segments overlap"
            covered.update(segment_indices)
        
        # Check full coverage
        assert covered == set(range(n)), "Segments don't cover full series"
    
    def test_min_segment_length_enforcement(self, simple_ohlc_df):
        """Test minimum segment length is enforced."""
        pytest.importorskip("ruptures")
        
        min_length = 15
        config = SegmentationConfig(
            penalty=1.0,  # Low penalty to encourage more segments
            min_segment_length=min_length,
        )
        segmenter = RupturesSegmenter(config)
        segments = segmenter.segment(simple_ohlc_df)
        
        for start, end in segments:
            length = end - start + 1
            assert length >= min_length, f"Segment length {length} < min {min_length}"
    
    def test_max_segment_length_enforcement(self, simple_ohlc_df):
        """Test maximum segment length is enforced."""
        pytest.importorskip("ruptures")
        
        max_length = 30
        config = SegmentationConfig(
            penalty=50.0,  # High penalty to encourage fewer segments
            max_segment_length=max_length,
        )
        segmenter = RupturesSegmenter(config)
        segments = segmenter.segment(simple_ohlc_df)
        
        for start, end in segments:
            length = end - start + 1
            assert length <= max_length, f"Segment length {length} > max {max_length}"
    
    def test_very_short_series(self):
        """Test handling of very short series."""
        pytest.importorskip("ruptures")
        
        dates = pd.date_range(start="2023-01-01", periods=5, freq="1D")
        df = pd.DataFrame({
            "Open": [100, 101, 102, 103, 104],
            "High": [101, 102, 103, 104, 105],
            "Low": [99, 100, 101, 102, 103],
            "Close": [100, 101, 102, 103, 104],
            "Volume": [1000] * 5,
        }, index=dates)
        
        config = {"min_segment_length": 10}
        segmenter = RupturesSegmenter(config)
        
        with pytest.warns(UserWarning, match="Series too short"):
            segments = segmenter.segment(df)
        
        assert len(segments) == 1
        assert segments[0] == (0, 4)
    
    def test_different_models(self, simple_ohlc_df):
        """Test different ruptures cost models."""
        pytest.importorskip("ruptures")
        
        for model in ["l2", "rbf"]:
            config = {"model": model}
            segmenter = RupturesSegmenter(config)
            segments = segmenter.segment(simple_ohlc_df)
            
            assert len(segments) > 0, f"Model {model} produced no segments"


# FixedWindowSegmenter tests
class TestFixedWindowSegmenter:
    """Test FixedWindowSegmenter functionality."""
    
    def test_fixed_window_basic(self, simple_ohlc_df):
        """Test basic fixed window segmentation."""
        config = {"mode": "fixed", "window": 20, "stride": 10}
        segmenter = FixedWindowSegmenter(config)
        segments = segmenter.segment(simple_ohlc_df)
        
        assert len(segments) > 0
        # Check window size
        for i, (start, end) in enumerate(segments[:-1]):  # Exclude last which might be shorter
            assert end - start + 1 == 20
    
    def test_fixed_window_stride(self, simple_ohlc_df):
        """Test stride behavior."""
        config = {"mode": "fixed", "window": 20, "stride": 20}
        segmenter = FixedWindowSegmenter(config)
        segments = segmenter.segment(simple_ohlc_df)
        
        # Non-overlapping windows
        for i in range(len(segments) - 1):
            _, end1 = segments[i]
            start2, _ = segments[i + 1]
            assert start2 == end1 + 1
    
    def test_full_series_mode(self, simple_ohlc_df):
        """Test full series mode (window=0)."""
        config = {"mode": "fixed", "window": 0}
        segmenter = FixedWindowSegmenter(config)
        segments = segmenter.segment(simple_ohlc_df)
        
        assert len(segments) == 1
        assert segments[0] == (0, len(simple_ohlc_df) - 1)


# Factory function tests
class TestCreateSegmenter:
    """Test create_segmenter factory function."""
    
    def test_create_dynamic_segmenter(self):
        """Test creating dynamic segmenter."""
        pytest.importorskip("ruptures")
        
        config = SegmentationConfig(mode="dynamic")
        segmenter = create_segmenter(config)
        
        assert isinstance(segmenter, RupturesSegmenter)
    
    def test_create_fixed_segmenter(self):
        """Test creating fixed segmenter."""
        config = {"mode": "fixed"}
        segmenter = create_segmenter(config)
        
        assert isinstance(segmenter, FixedWindowSegmenter)
    
    def test_invalid_mode(self):
        """Test invalid mode raises error."""
        # Mode validation now happens in create_segmenter
        with pytest.raises(ValueError):
            create_segmenter({"mode": "invalid"})


# Integration tests with iter_segments
class TestIterSegments:
    """Test iter_segments integration function."""
    
    def test_iter_segments_dynamic(self, simple_ohlc_df):
        """Test iter_segments with dynamic mode."""
        pytest.importorskip("ruptures")
        
        config = {"mode": "dynamic", "penalty": 5.0}
        segments = list(iter_segments(simple_ohlc_df, config))
        
        assert len(segments) > 0
        
        for segment_df, start, end, seg_id in segments:
            assert isinstance(segment_df, pd.DataFrame)
            assert len(segment_df) == end - start + 1
            assert seg_id >= 0
    
    def test_iter_segments_fixed(self, simple_ohlc_df):
        """Test iter_segments with fixed mode."""
        config = {"mode": "fixed", "window": 20, "stride": 10}
        segments = list(iter_segments(simple_ohlc_df, config))
        
        assert len(segments) > 0
        
        for segment_df, start, end, seg_id in segments:
            assert isinstance(segment_df, pd.DataFrame)
            assert len(segment_df) <= 20
    
    def test_iter_segments_sequential_ids(self, simple_ohlc_df):
        """Test that segment IDs are sequential."""
        pytest.importorskip("ruptures")
        
        config = SegmentationConfig(mode="dynamic")
        segments = list(iter_segments(simple_ohlc_df, config))
        
        for i, (_, _, _, seg_id) in enumerate(segments):
            assert seg_id == i
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame({
            "Open": [],
            "High": [],
            "Low": [],
            "Close": [],
            "Volume": [],
        })
        
        config = {}
        segments = list(iter_segments(df, config))
        
        assert len(segments) == 0


# Property-based tests (if hypothesis is available)
try:
    from hypothesis import given, strategies as st
    
    class TestPropertyBased:
        """Property-based tests for segmentation."""
        
        @given(
            n=st.integers(min_value=20, max_value=200),
            penalty=st.floats(min_value=1.0, max_value=20.0),
        )
        def test_segment_coverage_property(self, n, penalty):
            """Property: segments always cover the full series."""
            pytest.importorskip("ruptures")
            
            dates = pd.date_range(start="2023-01-01", periods=n, freq="1D")
            close = 100 + np.cumsum(np.random.randn(n))
            
            df = pd.DataFrame({
                "Open": close,
                "High": close + 1,
                "Low": close - 1,
                "Close": close,
                "Volume": np.ones(n) * 1000,
            }, index=dates)
            
            config = {"penalty": penalty, "min_segment_length": 5}
            segmenter = RupturesSegmenter(config)
            segments = segmenter.segment(df)
            
            # Check coverage
            covered = set()
            for start, end in segments:
                covered.update(range(start, end + 1))
            
            assert covered == set(range(n))

except ImportError:
    # Hypothesis not available, skip property-based tests
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



