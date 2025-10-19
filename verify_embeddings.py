"""Command line driver for the embedding verification protocol."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from ohlc_image_module.embedding_verification import (
    bundle_to_dict,
    load_embedding_array,
    load_embeddings,
    run_verification_protocol,
    save_report,
)


def _parse_date(value: Optional[str]) -> Optional[dt.datetime]:
    if value is None:
        return None
    return dt.datetime.fromisoformat(value)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify DINOv3 candlestick embeddings via a 7-step protocol."
    )
    parser.add_argument("--embeddings", required=True, help="Path to embeddings matrix.")
    parser.add_argument("--metadata", required=True, help="Path to metadata file (CSV/Parquet).")
    parser.add_argument(
        "--timestamp-col",
        default="timestamp",
        help="Metadata column containing the window timestamp.",
    )
    parser.add_argument(
        "--return-label-col",
        default=None,
        help="Metadata column with next-horizon return sign labels (0/1).",
    )
    parser.add_argument(
        "--volatility-col",
        default=None,
        help="Metadata column with volatility regime labels.",
    )
    parser.add_argument(
        "--style-embeddings",
        default=None,
        help="Optional path to style-preserving augmented embeddings for invariance checks.",
    )
    parser.add_argument(
        "--structure-embeddings",
        default=None,
        help="Optional path to structure-breaking augmented embeddings.",
    )
    parser.add_argument(
        "--linear-probe-splits",
        type=int,
        default=5,
        help="Number of purged walk-forward splits for the linear probe.",
    )
    parser.add_argument(
        "--linear-probe-gap",
        type=int,
        default=5,
        help="Gap (purge) between train/test windows in walk-forward evaluation.",
    )
    parser.add_argument(
        "--clustering-ks",
        default="3,4,5",
        help="Comma separated list of k values for k-means clustering.",
    )
    parser.add_argument(
        "--visualisation-dir",
        default="verification_outputs",
        help="Directory to store generated plots.",
    )
    parser.add_argument(
        "--report-path",
        default=None,
        help="Optional path for the JSON summary report (defaults to output dir/report.json).",
    )
    parser.add_argument(
        "--stress-train-end",
        default=None,
        help="ISO date marking the end of the training sample for the stress test.",
    )
    parser.add_argument(
        "--stress-test-start",
        default=None,
        help="ISO date for the start of the evaluation window in the stress test.",
    )
    parser.add_argument(
        "--stress-test-end",
        default=None,
        help="ISO date for the end of the evaluation window in the stress test.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity level.",
    )

    return parser.parse_args(argv)


def _load_optional_embeddings(path: Optional[str]) -> Optional[np.ndarray]:
    if not path:
        return None
    return load_embedding_array(path)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level))
    logger = logging.getLogger("verify_embeddings")

    embeddings, metadata = load_embeddings(
        args.embeddings, args.metadata, timestamp_col=args.timestamp_col
    )

    style_embeddings = _load_optional_embeddings(args.style_embeddings)
    structure_embeddings = _load_optional_embeddings(args.structure_embeddings)

    clustering_ks = tuple(int(k.strip()) for k in args.clustering_ks.split(",") if k.strip())

    bundle = run_verification_protocol(
        embeddings=embeddings,
        metadata=metadata,
        timestamp_col=args.timestamp_col,
        return_label_col=args.return_label_col,
        volatility_col=args.volatility_col,
        style_embeddings=style_embeddings,
        structure_embeddings=structure_embeddings,
        linear_probe_splits=args.linear_probe_splits,
        linear_probe_gap=args.linear_probe_gap,
        clustering_ks=clustering_ks,
        visualisation_dir=args.visualisation_dir,
        stress_train_end=_parse_date(args.stress_train_end),
        stress_test_start=_parse_date(args.stress_test_start),
        stress_test_end=_parse_date(args.stress_test_end),
    )

    output_dir = Path(args.visualisation_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report_path) if args.report_path else output_dir / "report.json"
    save_report(bundle, report_path)

    logger.info("Verification report written to %s", report_path)
    logger.info(json.dumps(bundle_to_dict(bundle), indent=2))

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

