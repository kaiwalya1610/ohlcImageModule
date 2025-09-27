"""Modular helpers for generating embedding-ready OHLC candlestick images."""
from .config import PRICE_COLUMNS, REQUIRED_COLUMNS, RenderConfig
from .data import fetch_ohlcv, validate_df
from .determinism import set_determinism
from .io_utils import ensure_outdirs, save_image, write_metadata
from .metadata import build_metadata_row
from .processing import iter_windows, normalize_ohlc
from .render import render_candlestick

__all__ = [
    "PRICE_COLUMNS",
    "REQUIRED_COLUMNS",
    "RenderConfig",
    "fetch_ohlcv",
    "validate_df",
    "set_determinism",
    "ensure_outdirs",
    "save_image",
    "write_metadata",
    "build_metadata_row",
    "iter_windows",
    "normalize_ohlc",
    "render_candlestick",
]
