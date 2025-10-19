"""Qualitative validation of segmentation on curated tickers.

This script runs segmentation on selected tickers with known market regimes
and generates visual reports with segment boundaries overlaid on price charts.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

from ohlc_image_module.data import fetch_ohlcv
from ohlc_image_module.processing import iter_segments


# Curated ticker list with known regime changes
VALIDATION_TICKERS = [
    ("AAPL", "2023-01-01", "2023-12-31", "1d", "Apple - trending year"),
    ("TSLA", "2023-01-01", "2023-12-31", "1d", "Tesla - volatile with regime shifts"),
    ("SPY", "2023-01-01", "2023-12-31", "1d", "S&P 500 - benchmark index"),
    ("TCS.NS", "2023-01-01", "2023-12-31", "1d", "TCS - Indian equity"),
    ("NVDA", "2023-01-01", "2023-12-31", "1d", "NVIDIA - strong trend"),
]


def validate_ticker(ticker: str, start: str, end: str, interval: str, description: str):
    """Validate segmentation on a single ticker and generate report."""
    
    print(f"\n{'='*70}")
    print(f"Validating: {ticker} ({description})")
    print(f"{'='*70}")
    
    # Fetch data
    try:
        df = fetch_ohlcv(ticker, start, end, interval)
        print(f"✅ Fetched {len(df)} bars")
    except Exception as e:
        print(f"❌ Failed to fetch data: {e}")
        return None
    
    # Run dynamic segmentation
    config = {
        "mode": "dynamic",
        "penalty": 3.0,
        "min_segment_length": 10,
        "max_segment_length": 200,
    }
    
    try:
        segments = list(iter_segments(df, config))
        print(f"✅ Generated {len(segments)} segments")
    except ImportError:
        print("❌ Ruptures not installed, skipping")
        return None
    except Exception as e:
        print(f"❌ Segmentation failed: {e}")
        return None
    
    # Create visualization
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    fig.suptitle(f"{ticker}: Dynamic Segmentation Validation\n{description}", 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Price with segment boundaries
    ax = axes[0]
    ax.plot(df.index, df["Close"], color='black', linewidth=1.5, label='Close Price')
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(segments)))
    for seg_id, (seg_df, start_idx, end_idx, _) in enumerate(segments):
        # Highlight segment
        ax.axvspan(
            df.index[start_idx], 
            df.index[end_idx],
            alpha=0.2,
            color=colors[seg_id],
        )
        # Mark boundary
        if seg_id > 0:
            ax.axvline(df.index[start_idx], color='red', linestyle='--', 
                      alpha=0.7, linewidth=1.5)
    
    ax.set_ylabel("Close Price ($)", fontsize=12)
    ax.set_title("Price Series with Segment Boundaries", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 2: Segment length distribution
    ax = axes[1]
    segment_lengths = [end - start + 1 for _, start, end, _ in segments]
    ax.hist(segment_lengths, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(segment_lengths), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(segment_lengths):.1f}')
    ax.set_xlabel("Segment Length (bars)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Segment Length Distribution", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 3: Returns and volatility per segment
    ax = axes[2]
    
    segment_stats = []
    for seg_df, start_idx, end_idx, seg_id in segments:
        returns = seg_df["Close"].pct_change().dropna()
        segment_stats.append({
            'seg_id': seg_id,
            'mean_return': returns.mean(),
            'volatility': returns.std(),
            'start_date': df.index[start_idx],
        })
    
    stats_df = pd.DataFrame(segment_stats)
    
    ax.bar(stats_df['seg_id'], stats_df['mean_return'], 
           color='green', alpha=0.6, label='Mean Return')
    ax2 = ax.twinx()
    ax2.plot(stats_df['seg_id'], stats_df['volatility'], 
            color='red', marker='o', linewidth=2, label='Volatility')
    
    ax.set_xlabel("Segment ID", fontsize=12)
    ax.set_ylabel("Mean Return", fontsize=12, color='green')
    ax2.set_ylabel("Volatility (Std Dev)", fontsize=12, color='red')
    ax.set_title("Segment Statistics: Returns and Volatility", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    
    # Save
    output_dir = Path("reports/validation_examples")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{ticker}_validation.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved validation plot: {output_path}")
    plt.close()
    
    # Print summary statistics
    print(f"\n📊 Segment Statistics:")
    print(f"   Total segments: {len(segments)}")
    print(f"   Mean length: {np.mean(segment_lengths):.1f} bars")
    print(f"   Std length: {np.std(segment_lengths):.1f} bars")
    print(f"   Min length: {min(segment_lengths)} bars")
    print(f"   Max length: {max(segment_lengths)} bars")
    print(f"   Mean return: {stats_df['mean_return'].mean():.4f}")
    print(f"   Mean volatility: {stats_df['volatility'].mean():.4f}")
    
    return {
        'ticker': ticker,
        'description': description,
        'num_segments': len(segments),
        'segment_stats': segment_stats,
        'plot_path': str(output_path),
    }


def generate_summary_report(results):
    """Generate HTML summary report of all validation results."""
    
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Segmentation Validation Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
            h1 { color: #333; }
            h2 { color: #666; margin-top: 30px; }
            .ticker-section { background: white; padding: 20px; margin: 20px 0; border-radius: 5px; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
            th { background-color: #4CAF50; color: white; }
            img { max-width: 100%; height: auto; margin: 20px 0; }
            .summary { background: #e8f5e9; padding: 15px; border-radius: 5px; margin: 20px 0; }
        </style>
    </head>
    <body>
        <h1>🔍 Segmentation Validation Report</h1>
        <p><strong>Generated:</strong> """ + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
        
        <div class="summary">
            <h3>Summary</h3>
            <p>Validated """ + str(len(results)) + """ tickers with dynamic segmentation.</p>
            <p>Total segments generated: """ + str(sum(r['num_segments'] for r in results if r)) + """</p>
        </div>
    """
    
    for result in results:
        if result is None:
            continue
        
        html += f"""
        <div class="ticker-section">
            <h2>📊 {result['ticker']}</h2>
            <p><em>{result['description']}</em></p>
            <p><strong>Segments:</strong> {result['num_segments']}</p>
            <img src="{Path(result['plot_path']).name}" alt="{result['ticker']} validation">
        </div>
        """
    
    html += """
        <div class="summary">
            <h3>💡 Validation Notes</h3>
            <ul>
                <li>Segment boundaries should align with visible regime changes</li>
                <li>Volatile periods should have shorter segments</li>
                <li>Stable/trending periods should have longer segments</li>
                <li>Returns and volatility should show clear differences between segments</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    output_path = Path("reports/validation_examples/validation_report.html")
    output_path.write_text(html, encoding="utf-8")
    print(f"\n✅ Saved validation report: {output_path}")


def main():
    """Run validation on all curated tickers."""
    
    print("\n" + "="*70)
    print("🚀 Segmentation Validation Suite")
    print("="*70)
    
    results = []
    for ticker, start, end, interval, description in VALIDATION_TICKERS:
        result = validate_ticker(ticker, start, end, interval, description)
        results.append(result)
    
    # Generate summary
    generate_summary_report(results)
    
    print("\n" + "="*70)
    print("✨ Validation complete! Check reports/validation_examples/")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()



