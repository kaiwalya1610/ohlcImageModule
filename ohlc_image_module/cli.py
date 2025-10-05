"""Command line interface for generating embedding-ready OHLC images.

Example usage
-------------

* Full range chart for a daily equity series::

    python make_ohlc_images.py --ticker AAPL --start 2023-01-01 --end 2023-12-31 --interval 1d --out_dir ./out

* Rolling windows for embedding pipelines::

    python make_ohlc_images.py --ticker MSFT --start 2022-01-01 --end 2023-12-31 --interval 1h --window 128 --stride 0 --normalize zscore --img_size 256 --no_volume --out_dir ./out
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
from typing import List, Optional

import pandas as pd

from .config import RenderConfig
from .data import fetch_ohlcv, validate_df
from .metadata import build_metadata_row
from .processing import iter_windows, normalize_ohlc
from .render import render_candlestick


LOGGER_NAME = "make_ohlc_images"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Generate embedding-ready OHLC images.")
    parser.add_argument("--ticker", required=True, help="Ticker symbol (e.g. AAPL).")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD, inclusive).")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD, inclusive).")
    parser.add_argument(
        "--interval",
        default="1d",
        help="Sampling interval supported by Yahoo Finance (e.g. 1d, 1h, 5m).",
    )
    parser.add_argument("--out_dir", default="./out", help="Output directory root.")
    parser.add_argument(
        "--img_size", type=int, default=256, help="Square output size for the PNG (pixels)."
    )
    parser.add_argument("--dpi", type=int, default=128, help="Figure DPI before resizing.")
    parser.add_argument(
        "--no_volume",
        action="store_true",
        help="Disable rendering of the volume subplot.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=0,
        help="Window size (number of bars). If zero, renders the full series.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=0,
        help="Stride between windows. Defaults to window length for non-overlapping windows.",
    )
    parser.add_argument(
        "--normalize",
        choices=["zscore", "minmax", "none"],
        default="zscore",
        help="Normalization mode applied to OHLC prices.",
    )
    parser.add_argument("--bg", default="none", help="Background colour.")
    parser.add_argument("--up_color", default="green", help="Colour for up candles.")
    parser.add_argument("--down_color", default="red", help="Colour for down candles.")
    parser.add_argument(
        "--line_width", type=float, default=0.8, help="Outline line width for candles."
    )
    parser.add_argument(
        "--wick_width", type=float, default=0.6, help="Line width for candle wicks."
    )
    parser.add_argument(
        "--include_wicks",
        dest="include_wicks",
        action="store_true",
        default=True,
        help="Render wick lines on candlesticks (default).",
    )
    parser.add_argument(
        "--no_include_wicks",
        dest="include_wicks",
        action="store_false",
        help="Disable wick rendering for candlesticks.",
    )
    parser.add_argument(
        "--tight_layout_pad",
        type=float,
        default=0.0,
        help="Padding passed to matplotlib savefig.",
    )
    parser.add_argument(
        "--save_metadata_csv",
        dest="save_metadata_csv",
        action="store_true",
        default=True,
        help="Persist metadata.csv (default).",
    )
    parser.add_argument(
        "--no_save_metadata_csv",
        dest="save_metadata_csv",
        action="store_false",
        help="Skip writing metadata CSV.",
    )
    parser.add_argument(
        "--fail_on_gaps",
        action="store_true",
        help="Raise an error if large gaps are detected in the series.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit the number of generated images (useful for smoke tests).",
    )

    return parser.parse_args(argv)


def _render_windows(
    df: pd.DataFrame,
    args: argparse.Namespace,
    render_cfg: RenderConfig,
    out_dir: str,
    logger: logging.Logger,
) -> List[Dict[str, object]]:
    """Iterate over windows, render images, and collect metadata rows."""

    metadata_rows: List[Dict[str, object]] = []
    effective_stride = args.stride if args.stride > 0 else (args.window if args.window > 0 else 0)

    for window_df, idx_start, idx_end in iter_windows(df, args.window, args.stride):
        if len(window_df) < 5:
            logger.info(
                "Skipping window [%d:%d] due to insufficient bars (%d).",
                idx_start,
                idx_end,
                len(window_df),
            )
            continue

        norm_df = normalize_ohlc(window_df, args.normalize)

        image = render_candlestick(norm_df, render_cfg)

        start_label = window_df.index[0].strftime("%Y%m%d")
        end_label = window_df.index[-1].strftime("%Y%m%d")
        suffix = (
            f"win{args.window}-stride{effective_stride}-idx{idx_start}" if args.window > 0 else "FULL"
        )
        filename = f"{args.ticker}_{args.interval}_{start_label}_{end_label}_{suffix}.png"
        img_path = os.path.join(out_dir, "images", filename)
        
        # Save image
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        image.save(img_path, format="PNG")

        logger.info("Saved image %s", img_path)

        metadata_rows.append(
            build_metadata_row(
                ticker=args.ticker,
                interval=args.interval,
                df_window=norm_df,
                img_path=img_path,
                n_bars=len(window_df),
                window=args.window,
                stride=effective_stride,
                idx_start=idx_start,
                idx_end=idx_end,
                normalize=args.normalize,
                cfg=render_cfg,
            )
        )

        if args.limit > 0 and len(metadata_rows) >= args.limit:
            logger.info("Limit reached (%d images).", args.limit)
            break

    return metadata_rows


def main(argv: Optional[List[str]] = None) -> None:
    """Entry point for the CLI utility."""

    args = parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logger = logging.getLogger(LOGGER_NAME)

    try:
        start_dt = dt.datetime.fromisoformat(args.start)
        end_dt = dt.datetime.fromisoformat(args.end)
    except ValueError as exc:
        raise SystemExit(f"Invalid date provided: {exc}")

    if end_dt < start_dt:
        raise SystemExit("End date must be greater than or equal to start date.")

    logger.info(
        "Fetching data for %s from %s to %s at interval %s",
        args.ticker,
        args.start,
        args.end,
        args.interval,
    )

    df = fetch_ohlcv(args.ticker, args.start, args.end, args.interval)
    logger.info("Downloaded %d rows of data.", len(df))

    validate_df(df, args.interval, args.fail_on_gaps)

    # Build render config
    render_cfg = RenderConfig(
        bg=args.bg,
        up_color=args.up_color,
        down_color=args.down_color,
        line_width=args.line_width,
        wick_width=args.wick_width,
        include_wicks=args.include_wicks,
        include_volume=not args.no_volume,
        img_size=args.img_size,
        dpi=args.dpi,
        tight_layout_pad=args.tight_layout_pad,
    )
    
    # Create output directories
    os.makedirs(args.out_dir, exist_ok=True)
    
    metadata_rows = _render_windows(df, args, render_cfg, args.out_dir, logger)

    if not metadata_rows:
        logger.warning("No images were generated.")
        return

    if args.save_metadata_csv:
        metadata_path = os.path.join(args.out_dir, "metadata.csv")
        df_meta = pd.DataFrame(metadata_rows)
        df_meta.to_csv(metadata_path, index=False)
        logger.info("Metadata written to %s", metadata_path)

    logger.info("Generated %d image(s).", len(metadata_rows))
