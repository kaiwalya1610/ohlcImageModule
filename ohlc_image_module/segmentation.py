"""Dynamic time series segmentation for OHLC data.

This module provides adaptive segmentation capabilities to identify regime changes
in financial time series, replacing fixed-length windowing with data-driven boundary
detection using change-point detection algorithms.
"""
from __future__ import annotations

from typing import List, Protocol, Tuple, Dict, Any
import warnings

import numpy as np
import pandas as pd


# Default config values
DEFAULT_CONFIG = {
    "mode": "dynamic",
    "window": 0,
    "stride": 0,
    "model": "l2",
    "penalty": 3.0,
    "min_segment_length": 10,
    "max_segment_length": 200,
    "jump": 2,
}


class Segmenter(Protocol):
    """Protocol for time series segmentation implementations."""
    
    def segment(self, df: pd.DataFrame) -> List[Tuple[int, int]]:
        """Segment the dataframe into coherent regimes.
        
        Args:
            df: DataFrame with OHLC data and DatetimeIndex
            
        Returns:
            List of (start_idx, end_idx) tuples representing segments.
            Indices are inclusive on both ends.
        """
        ...


class RupturesSegmenter:
    """Change-point detection segmenter using the ruptures library.
    
    This segmenter uses statistical change-point detection to identify regime
    changes in price series. It supports multiple cost functions and constraints.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the segmenter.
        
        Args:
            config: Segmentation configuration dict
            
        Raises:
            ImportError: If ruptures library is not installed
        """
        self.config = {**DEFAULT_CONFIG, **config}
        
        try:
            import ruptures as rpt
            self._rpt = rpt
        except ImportError as exc:
            raise ImportError(
                "ruptures library is required for dynamic segmentation. "
                "Install it with: pip install ruptures>=1.1.0"
            ) from exc
    
    def segment(self, df: pd.DataFrame) -> List[Tuple[int, int]]:
        """Detect change-points and return segment boundaries.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            List of (start_idx, end_idx) tuples for each segment
        """
        n = len(df)
        
        # Edge case: very short series
        if n < self.config["min_segment_length"]:
            warnings.warn(
                f"Series too short ({n} bars) for segmentation. "
                f"Returning single segment.",
                UserWarning
            )
            return [(0, n - 1)]
        
        # Extract Close prices for change-point detection
        signal = df["Close"].values.reshape(-1, 1)
        
        # Apply change-point detection
        try:
            algo = self._rpt.Pelt(
                model=self.config["model"],
                min_size=self.config["min_segment_length"],
                jump=self.config["jump"]
            )
            algo.fit(signal)
            
            # Get breakpoints (these are indices just after segment ends)
            breakpoints = algo.predict(pen=self.config["penalty"])
            
        except Exception as exc:
            warnings.warn(
                f"Change-point detection failed: {exc}. "
                f"Returning single segment.",
                UserWarning
            )
            return [(0, n - 1)]
        
        # Convert breakpoints to segment (start, end) tuples
        segments = self._breakpoints_to_segments(breakpoints, n)
        
        # Enforce maximum segment length if specified
        if self.config["max_segment_length"] > 0:
            segments = self._enforce_max_length(segments, self.config["max_segment_length"])
        
        return segments
    
    def _breakpoints_to_segments(
        self, breakpoints: List[int], n: int
    ) -> List[Tuple[int, int]]:
        """Convert ruptures breakpoint format to (start, end) tuples.
        
        Args:
            breakpoints: Breakpoint indices from ruptures (1-indexed, exclusive end)
            n: Total length of series
            
        Returns:
            List of (start_idx, end_idx) tuples (0-indexed, inclusive)
        """
        segments = []
        start_idx = 0
        
        for bp in breakpoints:
            if bp > n:  # Ruptures sometimes returns n+1 as final breakpoint
                bp = n
            
            end_idx = bp - 1  # Convert to inclusive end index
            
            if end_idx >= start_idx:
                segments.append((start_idx, end_idx))
            
            start_idx = bp
        
        # Handle case where last breakpoint isn't n
        if segments and segments[-1][1] < n - 1:
            segments.append((segments[-1][1] + 1, n - 1))
        elif not segments:
            segments.append((0, n - 1))
        
        return segments
    
    def _enforce_max_length(
        self, segments: List[Tuple[int, int]], max_length: int
    ) -> List[Tuple[int, int]]:
        """Split segments that exceed maximum length.
        
        Args:
            segments: Original segments
            max_length: Maximum allowed segment length
            
        Returns:
            Segments with length constraint enforced
        """
        refined_segments = []
        
        for start_idx, end_idx in segments:
            segment_length = end_idx - start_idx + 1
            
            if segment_length <= max_length:
                refined_segments.append((start_idx, end_idx))
            else:
                # Split long segment into equal-length chunks
                num_chunks = int(np.ceil(segment_length / max_length))
                chunk_size = segment_length // num_chunks
                
                for i in range(num_chunks):
                    chunk_start = start_idx + i * chunk_size
                    chunk_end = min(start_idx + (i + 1) * chunk_size - 1, end_idx)
                    
                    # Last chunk gets any remainder
                    if i == num_chunks - 1:
                        chunk_end = end_idx
                    
                    refined_segments.append((chunk_start, chunk_end))
        
        return refined_segments


class FixedWindowSegmenter:
    """Fixed-length window segmenter for backward compatibility.
    
    This segmenter wraps the existing fixed-window logic to provide a unified
    interface with dynamic segmentation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the segmenter.
        
        Args:
            config: Segmentation configuration dict (uses window and stride fields)
        """
        self.config = {**DEFAULT_CONFIG, **config}
    
    def segment(self, df: pd.DataFrame) -> List[Tuple[int, int]]:
        """Generate fixed-length windows.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            List of (start_idx, end_idx) tuples for each window
        """
        n = len(df)
        window = self.config["window"]
        stride = self.config["stride"] if self.config["stride"] > 0 else window
        
        segments = []
        
        # Handle full-series case
        if window <= 0 or window >= n:
            return [(0, n - 1)]
        
        # Generate fixed windows
        for start_idx in range(0, n - window + 1, stride):
            end_idx = start_idx + window - 1
            segments.append((start_idx, end_idx))
        
        return segments


def create_segmenter(config: Dict[str, Any]) -> Segmenter:
    """Factory function to create appropriate segmenter based on config.
    
    Args:
        config: Segmentation configuration dict
        
    Returns:
        Segmenter instance (RupturesSegmenter or FixedWindowSegmenter)
        
    Raises:
        ValueError: If mode is invalid
    """
    full_config = {**DEFAULT_CONFIG, **config}
    
    if full_config["mode"] == "dynamic":
        return RupturesSegmenter(config)
    elif full_config["mode"] == "fixed":
        return FixedWindowSegmenter(config)
    else:
        raise ValueError(f"Unknown segmentation mode: {full_config['mode']}")



