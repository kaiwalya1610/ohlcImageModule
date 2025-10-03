"""Configuration objects and shared constants for OHLC image generation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final, List

PRICE_COLUMNS: Final[List[str]] = ["Open", "High", "Low", "Close"]
REQUIRED_COLUMNS: Final[List[str]] = PRICE_COLUMNS + ["Volume"]


@dataclass(frozen=True)
class RenderConfig:
    """Container for rendering related configuration."""

    bg: str
    up_color: str
    down_color: str
    line_width: float
    wick_width: float
    include_wicks: bool
    include_volume: bool
    img_size: int
    dpi: int
    tight_layout_pad: float
