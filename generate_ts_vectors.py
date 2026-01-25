#!/usr/bin/env python3
"""Generate time-series vectors for OHLC segments.

Batch processes a metadata CSV to generate enriched "DNA" vectors from OHLC data
for hybrid retrieval alongside visual embeddings.
"""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ohlc_image_module.data import fetch_ohlcv
from ohlc_image_module.vector_utils import (
    generate_enriched_vector,
    DEFAULT_TARGET_LEN,
)


# =============================================================================
# CONFIGURATION - Edit these values directly
# =============================================================================
METADATA_PATH = "dinov3_nifty50_dataset/metadata/dataset-TRAIN.csv"
OUT_DIR = "dinov3_nifty50_dataset"
INTERVAL = "5m"
TARGET_LEN = DEFAULT_TARGET_LEN  # 128 per channel, 512 total
# =============================================================================


def get_vector_filename(image_name: str) -> str:
    """Convert image filename to vector filename.

    Example: ADANIENT_NS_seg_0000.png -> ADANIENT_NS_seg_0000.npy
    """
    return image_name.rsplit('.', 1)[0] + '.npy'


def fetch_ticker_data(
    ticker: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    interval: str,
) -> Optional[pd.DataFrame]:
    """Fetch OHLC data for a ticker with buffer for edge segments.

    Args:
        ticker: Stock symbol
        start_date: Earliest segment start time
        end_date: Latest segment end time
        interval: OHLC interval

    Returns:
        DataFrame with OHLC data, or None if fetch fails
    """
    # Add buffer to ensure we capture all data
    buffer = timedelta(days=2)
    start_str = (start_date - buffer).strftime("%Y-%m-%d")
    end_str = (end_date + buffer).strftime("%Y-%m-%d")

    try:
        df = fetch_ohlcv(ticker, start_str, end_str, interval)
        return df
    except Exception as e:
        print(f"  ⚠️  Failed to fetch {ticker}: {e}")
        return None


def slice_segment(
    df: pd.DataFrame,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> Optional[pd.DataFrame]:
    """Slice a segment from the full OHLC DataFrame by timestamp range.

    Args:
        df: Full OHLC DataFrame with DatetimeIndex
        start_time: Segment start timestamp
        end_time: Segment end timestamp

    Returns:
        Sliced DataFrame, or None if no data in range
    """
    mask = (df.index >= start_time) & (df.index <= end_time)
    segment = df.loc[mask]

    if len(segment) < 2:
        return None

    return segment


def process_ticker_segments(
    ticker: str,
    segments: List[Dict],
    df: pd.DataFrame,
    vectors_dir: Path,
    target_len: int,
) -> List[Dict]:
    """Process all segments for a single ticker.

    Args:
        ticker: Stock symbol
        segments: List of segment metadata dicts
        df: Full OHLC DataFrame for the ticker
        vectors_dir: Output directory for vector files
        target_len: Length per channel in the vector

    Returns:
        List of enriched metadata dicts with vector info
    """
    results = []

    for seg in segments:
        result = seg.copy()

        # Parse timestamps
        start_time = pd.Timestamp(seg['segment_start_time'])
        end_time = pd.Timestamp(seg['segment_end_time'])

        # Slice the segment
        segment_df = slice_segment(df, start_time, end_time)

        if segment_df is None:
            result['vector_path'] = None
            result['vector_status'] = 'no_data'
            results.append(result)
            continue

        # Extract close and volume
        close = segment_df['Close'].values
        volume = segment_df['Volume'].values

        # Generate vector
        try:
            vector = generate_enriched_vector(close, volume, target_len)

            # Save vector
            vector_filename = get_vector_filename(seg['image_name'])
            vector_path = vectors_dir / vector_filename
            np.save(vector_path, vector)

            result['vector_path'] = f"vectors/{vector_filename}"
            result['vector_status'] = 'success'
        except Exception as e:
            result['vector_path'] = None
            result['vector_status'] = f'error: {str(e)}'

        results.append(result)

    return results


def generate_vectors(
    metadata_path: str = METADATA_PATH,
    out_dir: str = OUT_DIR,
    interval: str = INTERVAL,
    target_len: int = TARGET_LEN,
) -> pd.DataFrame:
    """Generate time-series vectors for all segments in metadata.

    Args:
        metadata_path: Path to metadata CSV file
        out_dir: Output directory
        interval: OHLC interval
        target_len: Length per channel in the vector

    Returns:
        DataFrame with enriched metadata including vector paths
    """
    # Load metadata
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df_meta = pd.read_csv(metadata_path)
    print(f"📂 Loaded metadata: {len(df_meta)} segments")

    # Setup output directories
    out_dir = Path(out_dir)
    vectors_dir = out_dir / "vectors"
    vectors_dir.mkdir(parents=True, exist_ok=True)

    # Group by ticker (class_id)
    grouped = df_meta.groupby('class_id')
    tickers = list(grouped.groups.keys())

    print(f"📊 Processing {len(tickers)} tickers")
    print(f"📏 Vector config: {target_len} per channel, {4 * target_len} total")
    print("=" * 60)

    # Process each ticker
    all_results = []
    success_count = 0
    fail_count = 0

    for i, ticker in enumerate(tickers, 1):
        ticker_segments = grouped.get_group(ticker).to_dict('records')

        print(f"\n[{i}/{len(tickers)}] {ticker} ({len(ticker_segments)} segments)")

        # Get date range for this ticker
        start_times = [pd.Timestamp(s['segment_start_time']) for s in ticker_segments]
        end_times = [pd.Timestamp(s['segment_end_time']) for s in ticker_segments]
        min_start = min(start_times)
        max_end = max(end_times)

        # Fetch data once for this ticker
        df = fetch_ticker_data(ticker, min_start, max_end, interval)

        if df is None:
            # Mark all segments as failed
            for seg in ticker_segments:
                result = seg.copy()
                result['vector_path'] = None
                result['vector_status'] = 'fetch_failed'
                all_results.append(result)
                fail_count += 1
            continue

        # Process all segments for this ticker
        results = process_ticker_segments(
            ticker, ticker_segments, df, vectors_dir, target_len
        )

        # Count successes/failures
        for r in results:
            if r['vector_status'] == 'success':
                success_count += 1
            else:
                fail_count += 1

        all_results.extend(results)

        ticker_success = sum(1 for r in results if r['vector_status'] == 'success')
        print(f"  ✅ Generated {ticker_success}/{len(ticker_segments)} vectors")

    # Save enriched metadata
    df_enriched = pd.DataFrame(all_results)
    enriched_path = out_dir / "metadata_enriched.csv"
    df_enriched.to_csv(enriched_path, index=False)

    print("\n" + "=" * 60)
    print("🎉 Vector Generation Complete!")
    print(f"   ✅ Success: {success_count}")
    print(f"   ❌ Failed: {fail_count}")
    print(f"   📁 Vectors: {vectors_dir}")
    print(f"   📄 Metadata: {enriched_path}")

    return df_enriched


def main():
    """Main entry point using hardcoded configuration."""
    generate_vectors(
        metadata_path=METADATA_PATH,
        out_dir=OUT_DIR,
        interval=INTERVAL,
        target_len=TARGET_LEN,
    )


if __name__ == "__main__":
    main()
