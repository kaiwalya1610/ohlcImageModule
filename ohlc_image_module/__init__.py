"""Modular helpers for generating embedding-ready OHLC candlestick images."""
from .config import PRICE_COLUMNS, REQUIRED_COLUMNS, RenderConfig
from .data import fetch_ohlcv, validate_df
from .embedding_verification import (
    bundle_to_dict,
    generate_visualisations,
    geometry_sanity_check,
    linear_probe_walk_forward,
    load_embedding_array,
    load_embeddings,
    retrieval_probe,
    run_verification_protocol,
    save_report,
)
from .metadata import build_metadata_row
from .processing import iter_windows, normalize_ohlc
from .render import render_candlestick

__all__ = [
    "PRICE_COLUMNS",
    "REQUIRED_COLUMNS",
    "RenderConfig",
    "fetch_ohlcv",
    "validate_df",
    "geometry_sanity_check",
    "linear_probe_walk_forward",
    "retrieval_probe",
    "run_verification_protocol",
    "generate_visualisations",
    "load_embeddings",
    "load_embedding_array",
    "bundle_to_dict",
    "save_report",
    "build_metadata_row",
    "iter_windows",
    "normalize_ohlc",
    "render_candlestick",
]
