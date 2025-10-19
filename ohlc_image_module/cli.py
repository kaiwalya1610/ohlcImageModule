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
from typing import Dict, List, Optional

import pandas as pd

from .config import RenderConfig
from .data import fetch_ohlcv, validate_df
from .metadata import build_metadata_row
from .processing import iter_windows, iter_segments, normalize_ohlc
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
    
    # Segmentation options
    seg_group = parser.add_argument_group("Segmentation Options")
    seg_group.add_argument(
        "--fixed-window",
        action="store_true",
        help="Use fixed-length windowing (legacy mode). Default is dynamic segmentation.",
    )
    seg_group.add_argument(
        "--segmentation-model",
        choices=["l2", "rbf", "linear", "normal", "ar"],
        default="l2",
        help="Ruptures cost function for dynamic segmentation (default: l2 for mean shifts).",
    )
    seg_group.add_argument(
        "--segmentation-penalty",
        type=float,
        default=3.0,
        help="Detection sensitivity (higher = fewer segments, default: 3.0).",
    )
    seg_group.add_argument(
        "--min-segment-length",
        type=int,
        default=10,
        help="Minimum bars per segment (default: 10).",
    )
    seg_group.add_argument(
        "--max-segment-length",
        type=int,
        default=200,
        help="Maximum bars per segment (0 = unlimited, default: 200).",
    )
    seg_group.add_argument(
        "--segmentation-jump",
        type=int,
        default=5,
        help="Computational optimization - consider every Nth point as breakpoint (default: 5).",
    )

    return parser.parse_args(argv)


def _render_segments(
    df: pd.DataFrame,
    args: argparse.Namespace,
    render_cfg: RenderConfig,
    seg_cfg: Dict[str, object],
    out_dir: str,
    logger: logging.Logger,
) -> List[Dict[str, object]]:
    """Iterate over segments, render images, and collect metadata rows."""

    metadata_rows: List[Dict[str, object]] = []

    for segment_df, idx_start, idx_end, segment_id in iter_segments(df, seg_cfg):
        if len(segment_df) < 5:
            logger.info(
                "Skipping segment %d [%d:%d] due to insufficient bars (%d).",
                segment_id,
                idx_start,
                idx_end,
                len(segment_df),
            )
            continue

        norm_df = normalize_ohlc(segment_df, args.normalize)

        image = render_candlestick(norm_df, render_cfg)

        start_label = segment_df.index[0].strftime("%Y%m%d")
        end_label = segment_df.index[-1].strftime("%Y%m%d")
        
        # Generate filename based on segmentation mode
        if seg_cfg["mode"] == "dynamic":
            suffix = f"seg{segment_id:04d}-idx{idx_start}"
        else:
            effective_stride = args.stride if args.stride > 0 else (args.window if args.window > 0 else 0)
            suffix = (
                f"win{args.window}-stride{effective_stride}-idx{idx_start}" 
                if args.window > 0 else "FULL"
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
                n_bars=len(segment_df),
                window=args.window,
                stride=args.stride if args.stride > 0 else (args.window if args.window > 0 else 0),
                idx_start=idx_start,
                idx_end=idx_end,
                normalize=args.normalize,
                cfg=render_cfg,
                segment_id=segment_id,
                segmentation_mode=seg_cfg["mode"],
                segmentation_config=seg_cfg,
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
    
    # Build segmentation config
    if args.fixed_window:
        seg_cfg = {
            "mode": "fixed",
            "window": args.window,
            "stride": args.stride,
        }
        logger.info("Using fixed-window segmentation (legacy mode)")
    else:
        seg_cfg = {
            "mode": "dynamic",
            "model": args.segmentation_model,
            "penalty": args.segmentation_penalty,
            "min_segment_length": args.min_segment_length,
            "max_segment_length": args.max_segment_length,
            "jump": args.segmentation_jump,
        }
        logger.info(
            "Using dynamic segmentation (model=%s, penalty=%.2f, min=%d, max=%d)",
            args.segmentation_model,
            args.segmentation_penalty,
            args.min_segment_length,
            args.max_segment_length,
        )
    
    # Create output directories
    os.makedirs(args.out_dir, exist_ok=True)
    
    metadata_rows = _render_segments(df, args, render_cfg, seg_cfg, args.out_dir, logger)

    if not metadata_rows:
        logger.warning("No images were generated.")
        return

    if args.save_metadata_csv:
        metadata_path = os.path.join(args.out_dir, "metadata.csv")
        df_meta = pd.DataFrame(metadata_rows)
        df_meta.to_csv(metadata_path, index=False)
        logger.info("Metadata written to %s", metadata_path)

    logger.info("Generated %d image(s).", len(metadata_rows))
