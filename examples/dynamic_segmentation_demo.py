#!/usr/bin/env python3
"""
Dynamic Segmentation Demo

This script demonstrates the difference between dynamic and fixed segmentation
by generating side-by-side visualizations and embedding similarity analyses.

Usage:
    python examples/dynamic_segmentation_demo.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

from ohlc_image_module.data import fetch_ohlcv
from ohlc_image_module.processing import iter_segments
from ohlc_image_module.config import RenderConfig
from ohlc_image_module.render import render_candlestick


def plot_segmentation_comparison(ticker: str, start: str, end: str, interval: str = "1d"):
    """Generate side-by-side comparison of dynamic vs fixed segmentation."""
    
    print(f"\n{'='*70}")
    print(f"📊 Segmentation Comparison: {ticker}")
    print(f"{'='*70}\n")
    
    # Fetch data
    print(f"📥 Fetching data for {ticker}...")
    df = fetch_ohlcv(ticker, start, end, interval)
    print(f"✅ Downloaded {len(df)} bars\n")
    
    # Configure segmentations
    configs = {
        "Dynamic (penalty=2.0)": {
            "mode": "dynamic",
            "penalty": 2.0,
            "min_segment_length": 10,
            "max_segment_length": 100,
        },
        "Dynamic (penalty=5.0)": {
            "mode": "dynamic",
            "penalty": 5.0,
            "min_segment_length": 10,
            "max_segment_length": 100,
        },
        "Fixed (window=30)": {
            "mode": "fixed",
            "window": 30,
            "stride": 15,
        },
        "Fixed (window=50)": {
            "mode": "fixed",
            "window": 50,
            "stride": 25,
        },
    }
    
    # Generate segments
    all_segments = {}
    for label, config in configs.items():
        try:
            segments = list(iter_segments(df, config))
            all_segments[label] = segments
            print(f"  {label:30s}: {len(segments)} segments")
        except ImportError:
            if "Dynamic" in label:
                print(f"  {label:30s}: Skipped (ruptures not installed)")
    
    if not all_segments:
        print("\n⚠️  No segments generated. Install ruptures: pip install ruptures")
        return
    
    # Plot comparison
    fig, axes = plt.subplots(len(all_segments), 1, figsize=(16, 3 * len(all_segments)))
    if len(all_segments) == 1:
        axes = [axes]
    
    fig.suptitle(f"Segmentation Comparison: {ticker} ({start} to {end})", 
                 fontsize=16, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, 20))
    
    for idx, (label, segments) in enumerate(all_segments.items()):
        ax = axes[idx]
        
        # Plot price
        ax.plot(df.index, df["Close"], color='black', linewidth=1, label='Close Price', alpha=0.6)
        
        # Highlight segments
        for seg_id, (seg_df, start_idx, end_idx, _) in enumerate(segments):
            color = colors[seg_id % len(colors)]
            ax.axvspan(
                df.index[start_idx],
                df.index[end_idx],
                alpha=0.2,
                color=color,
                label=f'Segment {seg_id}' if seg_id < 3 else None
            )
            # Mark boundaries
            if seg_id > 0:
                ax.axvline(df.index[start_idx], color='red', linestyle='--', 
                          alpha=0.7, linewidth=1)
        
        ax.set_title(f"{label} ({len(segments)} segments)", fontsize=12, fontweight='bold')
        ax.set_ylabel("Price ($)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=8)
        
        # Show segment length distribution in inset
        segment_lengths = [end_idx - start_idx + 1 for _, start_idx, end_idx, _ in segments]
        
        # Add text box with statistics
        stats_text = (
            f"Segments: {len(segments)}\n"
            f"Mean length: {np.mean(segment_lengths):.1f}\n"
            f"Std length: {np.std(segment_lengths):.1f}\n"
            f"Min length: {min(segment_lengths)}\n"
            f"Max length: {max(segment_lengths)}"
        )
        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
    
    plt.xlabel("Date")
    plt.tight_layout()
    
    # Save
    output_dir = Path("reports/validation_examples")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{ticker}_segmentation_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved comparison plot: {output_path}")
    plt.close()
    
    return all_segments


def visualize_segment_images(ticker: str, start: str, end: str, max_segments: int = 6):
    """Generate candlestick images for a few segments."""
    
    print(f"\n{'='*70}")
    print(f"🖼️  Segment Image Visualization: {ticker}")
    print(f"{'='*70}\n")
    
    # Fetch data
    df = fetch_ohlcv(ticker, start, end, "1d")
    
    # Dynamic segmentation
    config = {
        "mode": "dynamic",
        "penalty": 3.0,
        "min_segment_length": 10,
    }
    
    try:
        segments = list(iter_segments(df, config))[:max_segments]
    except ImportError:
        print("⚠️  Skipped: ruptures not installed")
        return
    
    # Render config
    render_cfg = RenderConfig(
        bg="none",
        up_color="green",
        down_color="red",
        line_width=0.8,
        wick_width=0.6,
        include_wicks=True,
        include_volume=True,
        img_size=224,
        dpi=100,
        tight_layout_pad=0.0,
    )
    
    # Create subplot grid
    n_segments = len(segments)
    cols = 3
    rows = (n_segments + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten() if n_segments > 1 else [axes]
    
    fig.suptitle(f"Segment Images: {ticker} (Dynamic Segmentation)", 
                 fontsize=16, fontweight='bold')
    
    for idx, (seg_df, start_idx, end_idx, seg_id) in enumerate(segments):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        
        # Render candlestick
        img = render_candlestick(seg_df, render_cfg)
        
        # Display
        ax.imshow(img)
        ax.axis('off')
        
        start_date = seg_df.index[0].strftime("%Y-%m-%d")
        end_date = seg_df.index[-1].strftime("%Y-%m-%d")
        
        ax.set_title(
            f"Segment {seg_id}\n"
            f"{start_date} to {end_date}\n"
            f"({len(seg_df)} bars)",
            fontsize=10
        )
    
    # Hide unused subplots
    for idx in range(n_segments, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save
    output_dir = Path("reports/validation_examples")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{ticker}_segment_images.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved segment images: {output_path}")
    plt.close()


def analyze_segment_statistics(ticker: str, start: str, end: str):
    """Analyze statistical properties of segments."""
    
    print(f"\n{'='*70}")
    print(f"📈 Segment Statistics Analysis: {ticker}")
    print(f"{'='*70}\n")
    
    df = fetch_ohlcv(ticker, start, end, "1d")
    
    # Test multiple penalties
    penalties = [1.0, 2.0, 3.0, 5.0, 8.0, 10.0]
    
    results = []
    for penalty in penalties:
        config = {
            "mode": "dynamic",
            "penalty": penalty,
            "min_segment_length": 10,
        }
        
        try:
            segments = list(iter_segments(df, config))
        except ImportError:
            print("⚠️  Skipped: ruptures not installed")
            return
        
        segment_lengths = [end - start + 1 for _, start, end, _ in segments]
        
        results.append({
            "penalty": penalty,
            "num_segments": len(segments),
            "mean_length": np.mean(segment_lengths),
            "std_length": np.std(segment_lengths),
            "min_length": min(segment_lengths),
            "max_length": max(segment_lengths),
        })
    
    # Print table
    print(f"{'Penalty':<10} {'# Segments':<12} {'Mean':<10} {'Std':<10} {'Min':<6} {'Max':<6}")
    print("-" * 70)
    for r in results:
        print(f"{r['penalty']:<10.1f} {r['num_segments']:<12} "
              f"{r['mean_length']:<10.1f} {r['std_length']:<10.1f} "
              f"{r['min_length']:<6} {r['max_length']:<6}")
    
    # Plot relationship
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Number of segments vs penalty
    ax = axes[0]
    ax.plot([r["penalty"] for r in results], 
            [r["num_segments"] for r in results],
            marker='o', linewidth=2, markersize=8, color='blue')
    ax.set_xlabel("Penalty", fontsize=12)
    ax.set_ylabel("Number of Segments", fontsize=12)
    ax.set_title("Segment Count vs Penalty", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # Plot 2: Mean segment length vs penalty
    ax = axes[1]
    ax.plot([r["penalty"] for r in results],
            [r["mean_length"] for r in results],
            marker='s', linewidth=2, markersize=8, color='green')
    ax.set_xlabel("Penalty", fontsize=12)
    ax.set_ylabel("Mean Segment Length (bars)", fontsize=12)
    ax.set_title("Mean Segment Length vs Penalty", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    plt.suptitle(f"Penalty Parameter Sensitivity: {ticker}", 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_dir = Path("reports/validation_examples")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{ticker}_penalty_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved penalty analysis: {output_path}")
    plt.close()


def main():
    """Run all demonstrations."""
    
    print("\n" + "="*70)
    print("🚀 Dynamic Segmentation Demonstration")
    print("="*70)
    
    # Configuration
    ticker = "AAPL"
    start = "2023-01-01"
    end = "2023-12-31"
    
    # Run demonstrations
    try:
        # 1. Compare segmentation methods
        plot_segmentation_comparison(ticker, start, end)
        
        # 2. Visualize segment images
        visualize_segment_images(ticker, start, end)
        
        # 3. Analyze statistics
        analyze_segment_statistics(ticker, start, end)
        
        print("\n" + "="*70)
        print("✨ Demo complete! Check reports/validation_examples/ for outputs.")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()



