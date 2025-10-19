"""Benchmark tests comparing dynamic vs fixed segmentation performance."""
from __future__ import annotations

import time
import tracemalloc
from pathlib import Path
from typing import Dict, List
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ohlc_image_module.processing import iter_segments
from ohlc_image_module.data import fetch_ohlcv


def create_synthetic_series(n: int, seed: int = 42) -> pd.DataFrame:
    """Create synthetic OHLC series for benchmarking."""
    np.random.seed(seed)
    dates = pd.date_range(start="2020-01-01", periods=n, freq="1D")
    
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    
    df = pd.DataFrame({
        "Open": close + np.random.randn(n) * 0.5,
        "High": close + np.abs(np.random.randn(n)),
        "Low": close - np.abs(np.random.randn(n)),
        "Close": close,
        "Volume": np.random.randint(1000, 100000, n),
    }, index=dates)
    
    return df


def benchmark_segmentation(
    df: pd.DataFrame,
    config: Dict[str, any],
    label: str
) -> Dict[str, any]:
    """Benchmark segmentation performance."""
    # Start memory tracking
    tracemalloc.start()
    start_time = time.perf_counter()
    
    # Run segmentation
    segments = list(iter_segments(df, config))
    
    # End timing
    elapsed_time = time.perf_counter() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Collect statistics
    segment_lengths = [end - start + 1 for _, start, end, _ in segments]
    
    return {
        "label": label,
        "mode": config.get("mode", "dynamic"),
        "elapsed_time_ms": elapsed_time * 1000,
        "peak_memory_mb": peak / (1024 * 1024),
        "num_segments": len(segments),
        "mean_segment_length": np.mean(segment_lengths) if segment_lengths else 0,
        "std_segment_length": np.std(segment_lengths) if segment_lengths else 0,
        "min_segment_length": min(segment_lengths) if segment_lengths else 0,
        "max_segment_length": max(segment_lengths) if segment_lengths else 0,
        "segment_lengths": segment_lengths,
    }


def compare_modes(df: pd.DataFrame, series_name: str) -> Dict[str, any]:
    """Compare dynamic and fixed segmentation on a series."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {series_name} ({len(df)} bars)")
    print(f"{'='*60}")
    
    results = []
    
    # Dynamic segmentation - various penalties
    try:
        import ruptures
        
        for penalty in [1.0, 3.0, 5.0, 10.0]:
            config = {
                "mode": "dynamic",
                "penalty": penalty,
                "min_segment_length": 10,
                "max_segment_length": 200,
            }
            result = benchmark_segmentation(df, config, f"Dynamic (penalty={penalty})")
            results.append(result)
            print(f"  Dynamic (penalty={penalty:.1f}): "
                  f"{result['num_segments']} segments, "
                  f"{result['elapsed_time_ms']:.2f}ms")
    except ImportError:
        print("  Skipping dynamic segmentation (ruptures not installed)")
    
    # Fixed windowing - various configurations
    for window, stride in [(20, 10), (50, 25), (100, 50)]:
        if window > len(df):
            continue
        
        config = {
            "mode": "fixed",
            "window": window,
            "stride": stride,
        }
        result = benchmark_segmentation(df, config, f"Fixed (win={window}, stride={stride})")
        results.append(result)
        print(f"  Fixed (window={window}, stride={stride}): "
              f"{result['num_segments']} segments, "
              f"{result['elapsed_time_ms']:.2f}ms")
    
    return {
        "series_name": series_name,
        "series_length": len(df),
        "results": results,
    }


def plot_comparison(all_comparisons: List[Dict], output_dir: Path):
    """Generate comparison visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Runtime comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Segmentation Performance Comparison", fontsize=16)
    
    for idx, comparison in enumerate(all_comparisons):
        if idx >= 4:
            break
        
        ax = axes[idx // 2, idx % 2]
        
        labels = [r["label"] for r in comparison["results"]]
        times = [r["elapsed_time_ms"] for r in comparison["results"]]
        colors = ['red' if 'Dynamic' in l else 'blue' for l in labels]
        
        bars = ax.barh(labels, times, color=colors, alpha=0.7)
        ax.set_xlabel("Time (ms)")
        ax.set_title(f"{comparison['series_name']}\n({comparison['series_length']} bars)")
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar, time in zip(bars, times):
            ax.text(time, bar.get_y() + bar.get_height()/2, 
                   f'{time:.1f}', 
                   ha='left', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / "runtime_comparison.png", dpi=150)
    print(f"\n✅ Saved: {output_dir / 'runtime_comparison.png'}")
    plt.close()
    
    # Plot 2: Segment count distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for comparison in all_comparisons:
        series_name = comparison["series_name"]
        for result in comparison["results"]:
            if "Dynamic" in result["label"]:
                ax.scatter(
                    comparison["series_length"],
                    result["num_segments"],
                    label=f"{series_name} - {result['label']}",
                    alpha=0.7,
                    s=100
                )
    
    ax.set_xlabel("Series Length (bars)")
    ax.set_ylabel("Number of Segments")
    ax.set_title("Dynamic Segmentation: Segments vs Series Length")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "segment_count_analysis.png", dpi=150)
    print(f"✅ Saved: {output_dir / 'segment_count_analysis.png'}")
    plt.close()
    
    # Plot 3: Segment length distributions
    try:
        import ruptures
        
        fig, axes = plt.subplots(1, len(all_comparisons), figsize=(15, 4))
        if len(all_comparisons) == 1:
            axes = [axes]
        
        for idx, comparison in enumerate(all_comparisons):
            ax = axes[idx]
            
            for result in comparison["results"]:
                if "Dynamic" in result["label"]:
                    ax.hist(
                        result["segment_lengths"],
                        bins=20,
                        alpha=0.5,
                        label=result["label"]
                    )
            
            ax.set_xlabel("Segment Length (bars)")
            ax.set_ylabel("Frequency")
            ax.set_title(f"{comparison['series_name']}")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "segment_length_distributions.png", dpi=150)
        print(f"✅ Saved: {output_dir / 'segment_length_distributions.png'}")
        plt.close()
        
    except ImportError:
        print("⚠️  Skipping segment length distribution plot (ruptures not installed)")


def generate_html_report(all_comparisons: List[Dict], output_path: Path):
    """Generate HTML report with results."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Segmentation Benchmark Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
            h1 { color: #333; }
            h2 { color: #666; margin-top: 30px; }
            table { border-collapse: collapse; width: 100%; background: white; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
            th { background-color: #4CAF50; color: white; }
            tr:nth-child(even) { background-color: #f2f2f2; }
            .dynamic { background-color: #ffe6e6; }
            .fixed { background-color: #e6f2ff; }
            img { max-width: 100%; height: auto; margin: 20px 0; }
            .summary { background: white; padding: 20px; margin: 20px 0; border-radius: 5px; }
        </style>
    </head>
    <body>
        <h1>🔬 Segmentation Performance Benchmark Report</h1>
        <p><strong>Generated:</strong> """ + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
    """
    
    for comparison in all_comparisons:
        html += f"""
        <h2>📊 {comparison['series_name']} ({comparison['series_length']} bars)</h2>
        <table>
            <tr>
                <th>Mode</th>
                <th>Label</th>
                <th>Time (ms)</th>
                <th>Memory (MB)</th>
                <th># Segments</th>
                <th>Mean Length</th>
                <th>Std Length</th>
                <th>Min Length</th>
                <th>Max Length</th>
            </tr>
        """
        
        for result in comparison["results"]:
            row_class = "dynamic" if "Dynamic" in result["label"] else "fixed"
            html += f"""
            <tr class="{row_class}">
                <td><strong>{result['mode']}</strong></td>
                <td>{result['label']}</td>
                <td>{result['elapsed_time_ms']:.2f}</td>
                <td>{result['peak_memory_mb']:.2f}</td>
                <td>{result['num_segments']}</td>
                <td>{result['mean_segment_length']:.1f}</td>
                <td>{result['std_segment_length']:.1f}</td>
                <td>{result['min_segment_length']}</td>
                <td>{result['max_segment_length']}</td>
            </tr>
            """
        
        html += "</table>"
    
    html += """
        <h2>📈 Visualizations</h2>
        <img src="runtime_comparison.png" alt="Runtime Comparison">
        <img src="segment_count_analysis.png" alt="Segment Count Analysis">
        <img src="segment_length_distributions.png" alt="Segment Length Distributions">
        
        <div class="summary">
            <h3>💡 Key Findings</h3>
            <ul>
                <li><strong>Dynamic Segmentation:</strong> Adapts to regime changes but adds computational overhead</li>
                <li><strong>Fixed Windows:</strong> Faster and predictable but may miss regime boundaries</li>
                <li><strong>Penalty Parameter:</strong> Higher penalty → fewer segments (more conservative detection)</li>
                <li><strong>Memory Usage:</strong> Both methods have similar memory footprint</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    output_path.write_text(html, encoding="utf-8")
    print(f"\n✅ Saved HTML report: {output_path}")


def main():
    """Run comprehensive benchmarks."""
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*60)
    print("🚀 Segmentation Performance Benchmark Suite")
    print("="*60)
    
    all_comparisons = []
    
    # Test 1: Synthetic series of various lengths
    for length in [100, 500, 1000]:
        df = create_synthetic_series(length)
        comparison = compare_modes(df, f"Synthetic {length} bars")
        all_comparisons.append(comparison)
    
    # Test 2: Real market data (if available)
    try:
        print("\n📥 Fetching real market data for benchmarking...")
        df_real = fetch_ohlcv("TCS.NS", "2023-01-01", "2023-12-31", "1d")
        comparison = compare_modes(df_real, "TCS.NS Daily 2023")
        all_comparisons.append(comparison)
    except Exception as e:
        print(f"⚠️  Skipping real data benchmark: {e}")
    
    # Generate visualizations and report
    plot_comparison(all_comparisons, output_dir)
    
    # Save JSON results
    json_path = output_dir / "benchmark_results.json"
    with open(json_path, "w") as f:
        # Convert numpy types to native Python for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        serializable = json.loads(json.dumps(all_comparisons, default=convert_types))
        json.dump(serializable, f, indent=2)
    
    print(f"✅ Saved benchmark results: {json_path}")
    
    # Generate HTML report
    generate_html_report(all_comparisons, output_dir / "segmentation_analysis.html")
    
    print("\n" + "="*60)
    print("✨ Benchmark complete! Check the reports/ directory for results.")
    print("="*60)


if __name__ == "__main__":
    main()



