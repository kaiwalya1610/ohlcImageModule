#!/usr/bin/env python3
"""Generate DINOv3-compatible dataset from Nifty 50 stocks.

Creates sliding-window candlestick images for all Nifty 50 stocks with proper
folder structure and metadata files required by DINOv3 training protocol.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from PIL import Image

from ohlc_image_module.config import RenderConfig
from ohlc_image_module.data import fetch_ohlcv
from ohlc_image_module.render import render_candlestick
from ohlc_image_module.processing import iter_segments
from ohlc_image_module.vector_utils import (
    generate_enriched_vector,
    DEFAULT_TARGET_LEN,
)


# Nifty 50 stock symbols (as of Oct 2025 - adjust if needed)
# NIFTY_50_SYMBOLS = [
#     "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
#     "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BPCL.NS", "BHARTIARTL.NS",
#     "BRITANNIA.NS", "CIPLA.NS", "COALINDIA.NS", "DIVISLAB.NS", "DRREDDY.NS",
#     "EICHERMOT.NS", "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS",
#     "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "ITC.NS",
#     "INDUSINDBK.NS", "INFY.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS",
#     "M&M.NS", "MARUTI.NS", "NTPC.NS", "NESTLEIND.NS", "ONGC.NS",
#     "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS", "SUNPHARMA.NS",
#     "TCS.NS", "TATACONSUM.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "TECHM.NS",
#     "TITAN.NS", "ULTRACEMCO.NS", "UPL.NS", "WIPRO.NS", "ZOMATO.NS"
# ]
NIFTY_50_SYMBOLS = [
    "ADANIENT.NS", "ADANIPORTS.NS"]


def get_date_range_for_past_week() -> Tuple[str, str]:
    """Calculate date range for past 1 week (trading days)."""
    end_date = datetime.now()
    # Go back 2 weeks to ensure we get at least 5 trading days
    start_date = end_date - timedelta(days=7)
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")


def generate_images_for_stock(
    ticker: str,
    start: str,
    end: str,
    interval: str,
    render_cfg: RenderConfig,
    seg_cfg: Dict[str, object],
    output_dir: Path,
    vectors_dir: Path,
    class_index: int,
    target_len: int = DEFAULT_TARGET_LEN,
) -> List[Dict[str, object]]:
    """Generate segmented images and time-series vectors for a single stock using dynamic segmentation.

    Args:
        ticker: Stock symbol
        start: Start date
        end: End date
        interval: Time interval
        render_cfg: Rendering configuration
        seg_cfg: Segmentation configuration
        output_dir: Output directory for images
        vectors_dir: Output directory for time-series vectors
        class_index: Class index for this stock
        target_len: Length per channel in the time-series vector (default: 128, total: 512)

    Returns:
        List of metadata dicts for each generated image and vector
    """
    # Fetch data
    try:
        df = fetch_ohlcv(ticker, start, end, interval)
    except Exception as e:
        print(f"⚠️  Skipping {ticker}: {e}")
        return []
    
    # Need enough data for at least one segment
    min_len = seg_cfg.get("min_segment_length", 10)
    if len(df) < min_len:
        print(f"⚠️  {ticker}: Not enough data ({len(df)} < {min_len})")
        return []
    
    # Ensure output directory exists (unified folder for all images)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate segments using dynamic segmentation
    segments = list(iter_segments(df, seg_cfg))
    
    print(f"📊 {ticker}: {len(df)} candles → {len(segments)} segments (mode: {seg_cfg.get('mode', 'dynamic')})")
    
    # Collect metadata for each segment
    metadata_list = []
    
    # Render each segment and capture metadata
    for segment_df, idx_start, idx_end, segment_id in segments:
        if len(segment_df) < 5:
            continue  # Skip very short segments

        img = render_candlestick(segment_df, render_cfg)

        # Save with consistent naming: TICKER_seg_XXXX.png
        img_name = f"{ticker.replace('.', '_')}_seg_{segment_id:04d}.png"
        img_path = output_dir / img_name
        img.save(img_path)

        # Generate time-series vector
        vector_filename = img_name.rsplit('.', 1)[0] + '.npy'
        vector_path_rel = f"vectors/{vector_filename}"
        vector_status = 'success'

        try:
            # Extract close and volume from segment
            close = segment_df['Close'].values
            volume = segment_df['Volume'].values

            # Generate enriched vector
            vector = generate_enriched_vector(close, volume, target_len)

            # Save vector
            vector_path = vectors_dir / vector_filename
            np.save(vector_path, vector)
        except Exception as e:
            vector_path_rel = None
            vector_status = f'error: {str(e)}'

        # Capture segment metadata
        metadata = {
            'image_name': img_name,
            'class_id': ticker,
            'class_index': class_index,
            'segment_start_time': df.index[idx_start],
            'segment_end_time': df.index[idx_end],
            'segment_start_price': float(segment_df.iloc[0]['Open']),
            'segment_end_price': float(segment_df.iloc[-1]['Close']),
            'next_segment_close_price': None,  # Will be filled in next step
            'vector_path': vector_path_rel,
            'vector_status': vector_status,
        }
        metadata_list.append(metadata)
    
    # Add lookahead feature: next segment's first close price
    for i in range(len(metadata_list)):
        if i < len(metadata_list) - 1:
            # Get the next segment's start price (first close)
            next_metadata = metadata_list[i + 1]
            metadata_list[i]['next_segment_close_price'] = next_metadata['segment_start_price']
        # Last segment keeps None/NaN
    
    return metadata_list


def generate_dinov3_metadata(
    output_root: Path,
    all_metadata: List[Dict[str, object]],
    split: str = "TRAIN"
) -> None:
    """Generate metadata files for DINOv3 with rich retrieval features.

    Creates:
        - dataset-TRAIN.csv: Rich CSV with temporal, price, and vector features
        - class-ids-TRAIN.txt: List of stock symbols (backward compatibility)
        - class-names-TRAIN.txt: List of stock names (backward compatibility)

    Args:
        output_root: Root directory for dataset
        all_metadata: List of metadata dicts from all stocks (includes vector info)
        split: Dataset split name (default: "TRAIN")
    """
    metadata_dir = output_root / "metadata"
    metadata_dir.mkdir(exist_ok=True)
    
    if not all_metadata:
        raise ValueError("No metadata provided!")
    
    print(f"\n📁 Processing {len(all_metadata)} images")
    
    # Extract unique class info for backward compatibility files
    unique_classes = sorted(set(m['class_id'] for m in all_metadata))
    class_ids = unique_classes
    class_names = class_ids  # For stocks, ID = name
    
    # Save rich metadata as CSV
    dataset_path = metadata_dir / f"dataset-{split}.csv"
    
    # Convert to DataFrame for easy CSV writing
    df_metadata = pd.DataFrame(all_metadata)
    
    # Reorder columns for readability
    column_order = [
        'image_name',
        'class_id',
        'class_index',
        'segment_start_time',
        'segment_end_time',
        'segment_start_price',
        'segment_end_price',
        'next_segment_close_price',
        'vector_path',
        'vector_status'
    ]
    df_metadata = df_metadata[column_order]
    
    # Write CSV
    df_metadata.to_csv(dataset_path, index=False)
    
    # Save backward compatibility files
    class_ids_path = metadata_dir / f"class-ids-{split}.txt"
    class_names_path = metadata_dir / f"class-names-{split}.txt"
    
    # Write class IDs (one per line)
    with open(class_ids_path, 'w', encoding='utf-8') as f:
        for class_id in class_ids:
            f.write(f"{class_id}\n")
    
    # Write class names (one per line)
    with open(class_names_path, 'w', encoding='utf-8') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")
    
    print(f"✅ Metadata saved:")
    print(f"   - {dataset_path} ({len(all_metadata)} rows, includes image + vector metadata)")
    print(f"   - {class_ids_path} ({len(class_ids)} classes)")
    print(f"   - {class_names_path}")


def main():
    """Main entry point."""
    # Configuration
    OUTPUT_ROOT = Path("dinov3_nifty50_dataset")
    INTERVAL = "5m"
    TARGET_LEN = DEFAULT_TARGET_LEN  # 128 per channel, 512 total

    # Segmentation config - using dynamic segmentation by default
    seg_cfg = {
        "mode": "dynamic",
        "model": "l2",
        "penalty": 3.0,
        "min_segment_length": 10,
        "max_segment_length": 100,
        "jump": 5,
    }

    # Render config (using defaults from your cli.py)
    render_cfg = RenderConfig(
        bg="#000000",
        up_color="#00ff00",
        down_color="#ff0000",
        line_width=0.8,
        wick_width=0.5,
        include_wicks=True,
        include_volume=True,
        img_size=224,
        dpi=100,
        tight_layout_pad=0.05,
    )

    # Setup directories
    images_dir = OUTPUT_ROOT / "images"
    vectors_dir = OUTPUT_ROOT / "vectors"
    images_dir.mkdir(parents=True, exist_ok=True)
    vectors_dir.mkdir(parents=True, exist_ok=True)
    
    # Get date range
    start_date, end_date = get_date_range_for_past_week()
    
    print("=" * 70)
    print("🚀 DINOv3 Nifty 50 Dataset Generator (Images + Time-Series Vectors)")
    print("=" * 70)
    print(f"📅 Date Range: {start_date} to {end_date}")
    print(f"📊 Interval: {INTERVAL}")
    print(f"🔍 Segmentation: {seg_cfg['mode']} (model={seg_cfg['model']}, penalty={seg_cfg['penalty']})")
    print(f"📏 Segment Length: min={seg_cfg['min_segment_length']}, max={seg_cfg['max_segment_length']}")
    print(f"🎨 Image Size: {render_cfg.img_size}x{render_cfg.img_size}")
    print(f"📐 Vector Config: {TARGET_LEN} per channel, {4 * TARGET_LEN} total")
    print(f"💾 Output: {OUTPUT_ROOT}")
    print("=" * 70)
    
    # Process each stock and collect metadata
    all_metadata = []
    successful_stocks = 0
    
    for class_idx, ticker in enumerate(NIFTY_50_SYMBOLS):
        print(f"\n[{class_idx + 1}/{len(NIFTY_50_SYMBOLS)}] Processing {ticker}...")
        
        stock_metadata = generate_images_for_stock(
            ticker=ticker,
            start=start_date,
            end=end_date,
            interval=INTERVAL,
            render_cfg=render_cfg,
            seg_cfg=seg_cfg,
            output_dir=images_dir,
            vectors_dir=vectors_dir,
            class_index=class_idx,
            target_len=TARGET_LEN,
        )
        
        if stock_metadata:
            all_metadata.extend(stock_metadata)
            successful_stocks += 1
    
    total_images = len(all_metadata)
    
    print("\n" + "=" * 70)
    print(f"✅ Image & Vector Generation Complete!")
    print(f"   - Successful Stocks: {successful_stocks}/{len(NIFTY_50_SYMBOLS)}")
    print(f"   - Total Images: {total_images}")
    print(f"   - Total Vectors: {total_images}")
    print("=" * 70)
    
    # Generate DINOv3 metadata
    if total_images > 0:
        print("\n📝 Generating metadata files (images + vectors)...")
        generate_dinov3_metadata(OUTPUT_ROOT, all_metadata)
        
        print("\n" + "=" * 70)
        print("🎉 Dataset Ready for DINOv3 Training!")
        print("=" * 70)
        print(f"\n📂 Dataset Structure:")
        print(f"   {OUTPUT_ROOT}/")
        print(f"   ├── images/")
        print(f"   │   ├── RELIANCE_NS_seg_0000.png")
        print(f"   │   ├── RELIANCE_NS_seg_0001.png")
        print(f"   │   ├── TCS_NS_seg_0000.png")
        print(f"   │   └── ...")
        print(f"   ├── vectors/")
        print(f"   │   ├── RELIANCE_NS_seg_0000.npy  # (512,) float32 array")
        print(f"   │   ├── RELIANCE_NS_seg_0001.npy")
        print(f"   │   ├── TCS_NS_seg_0000.npy")
        print(f"   │   └── ...")
        print(f"   └── metadata/")
        print(f"       ├── dataset-TRAIN.csv  # includes vector_path and vector_status")
        print(f"       ├── class-ids-TRAIN.txt")
        print(f"       └── class-names-TRAIN.txt")
    else:
        print("\n⚠️  No images generated. Check your data sources!")


if __name__ == "__main__":
    main()

