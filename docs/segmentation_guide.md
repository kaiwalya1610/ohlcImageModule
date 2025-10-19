# Dynamic Segmentation User Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Understanding Parameters](#understanding-parameters)
4. [Parameter Tuning](#parameter-tuning)
5. [Use Cases](#use-cases)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Topics](#advanced-topics)

## Introduction

Dynamic segmentation automatically identifies regime changes in financial time series, creating variable-length segments that adapt to market conditions. This approach offers several advantages over fixed-window segmentation:

- **Regime-aware**: Segments align with natural market phases
- **Adaptive**: Automatically adjusts to volatility and trend changes
- **Meaningful**: Each segment represents a coherent market behavior

## Quick Start

### Basic Usage

Generate images with dynamic segmentation (now the default):

```bash
python make_ohlc_images.py \
    --ticker AAPL \
    --start 2023-01-01 \
    --end 2023-12-31 \
    --interval 1d \
    --out_dir ./output
```

### Using Fixed Windows (Legacy Mode)

Revert to fixed-window segmentation:

```bash
python make_ohlc_images.py \
    --ticker AAPL \
    --start 2023-01-01 \
    --end 2023-12-31 \
    --interval 1d \
    --fixed-window \
    --window 50 \
    --stride 25 \
    --out_dir ./output
```

## Understanding Parameters

### Core Parameters

#### `--segmentation-model` (default: `l2`)

Determines what type of changes to detect:

- **`l2`** (Least Squares): Detects **mean shifts** in price
  - Best for: Trend changes, support/resistance breaks
  - Example: Price moving from $100 average to $110 average

- **`rbf`** (Radial Basis Function): Detects **distribution changes**
  - Best for: Volatility regime shifts, market structure changes
  - Example: Moving from low-volatility to high-volatility period

- **`linear`**: Detects slope changes (linear trend shifts)
- **`normal`**: Detects changes in normal distribution parameters
- **`ar`**: Detects autoregressive model changes

**Recommendation**: Start with `l2` for most equity applications.

#### `--segmentation-penalty` (default: `3.0`)

Controls detection sensitivity (how readily boundaries are created):

- **Lower values (1.0 - 2.0)**:
  - More segments (over-segmentation risk)
  - Captures subtle changes
  - Use when: Analyzing high-frequency data or need fine-grained regimes

- **Medium values (3.0 - 5.0)**:
  - Balanced segmentation
  - Good default for most applications
  - Use when: Standard daily/hourly analysis

- **Higher values (5.0 - 20.0)**:
  - Fewer segments (under-segmentation risk)
  - Only major regime changes detected
  - Use when: Long-term analysis or very noisy data

**Recommendation**: Start at `3.0` and adjust based on results.

#### `--min-segment-length` (default: `10`)

Minimum bars per segment:

- Ensures statistical significance
- Prevents micro-segments
- **Rule of thumb**: Set to at least 5-10 bars
- **For daily data**: 10-20 bars (2-4 weeks)
- **For hourly data**: 20-50 bars (1-2 trading days)
- **For minute data**: 60-120 bars (1-2 hours)

#### `--max-segment-length` (default: `200`)

Maximum bars per segment:

- Prevents runaway segments in stable regimes
- Set to `0` for unlimited length
- **Rule of thumb**: 5-10x the minimum length
- **For daily data**: 100-200 bars (5-10 months)
- **For hourly data**: 200-500 bars (few weeks)

#### `--segmentation-jump` (default: `5`)

Computational optimization parameter:

- Algorithm considers every Nth point as potential breakpoint
- **Lower values**: More accurate, slower
- **Higher values**: Faster, less precise
- **Recommendation**: `5` for most cases, `10` for very large datasets

## Parameter Tuning

### Tuning Workflow

1. **Start with defaults**: Run with standard parameters
2. **Inspect results**: Check segment count and boundaries
3. **Adjust penalty**: If too many/few segments
4. **Refine constraints**: Adjust min/max lengths if needed
5. **Validate**: Visual inspection of segment boundaries on charts

### Example Tuning Scenarios

#### Scenario 1: Too Many Segments

**Problem**: Hundreds of tiny segments, over-fragmented

**Solution**:
```bash
--segmentation-penalty 8.0 \
--min-segment-length 20
```

#### Scenario 2: Too Few Segments

**Problem**: Entire year becomes 2-3 segments

**Solution**:
```bash
--segmentation-penalty 1.5 \
--max-segment-length 100
```

#### Scenario 3: Capturing Volatility Regimes

**Problem**: Need to detect calm vs volatile periods

**Solution**:
```bash
--segmentation-model rbf \
--segmentation-penalty 4.0
```

### Interactive Tuning Script

```python
from ohlc_image_module.data import fetch_ohlcv
from ohlc_image_module.segmentation import SegmentationConfig
from ohlc_image_module.processing import iter_segments
import matplotlib.pyplot as plt

# Fetch data
df = fetch_ohlcv("AAPL", "2023-01-01", "2023-12-31", "1d")

# Try different penalties
for penalty in [1.0, 3.0, 5.0, 10.0]:
    config = {"penalty": penalty}
    segments = list(iter_segments(df, config))
    
    print(f"Penalty {penalty}: {len(segments)} segments")
    
    # Plot
    plt.figure(figsize=(15, 4))
    plt.plot(df.index, df["Close"], label="Close Price")
    
    for seg_df, start, end, seg_id in segments:
        plt.axvline(df.index[start], color='red', alpha=0.3, linestyle='--')
    
    plt.title(f"Penalty = {penalty} ({len(segments)} segments)")
    plt.legend()
    plt.show()
```

## Use Cases

### Use Case 1: Equity Trend Analysis

**Goal**: Identify distinct trending periods

```bash
python make_ohlc_images.py \
    --ticker TSLA \
    --start 2023-01-01 \
    --end 2023-12-31 \
    --interval 1d \
    --segmentation-model l2 \
    --segmentation-penalty 4.0 \
    --min-segment-length 15 \
    --out_dir ./tsla_trends
```

### Use Case 2: Intraday Volatility Regimes

**Goal**: Segment high-frequency data by volatility

```bash
python make_ohlc_images.py \
    --ticker SPY \
    --start 2023-12-01 \
    --end 2023-12-31 \
    --interval 5m \
    --segmentation-model rbf \
    --segmentation-penalty 5.0 \
    --min-segment-length 30 \
    --max-segment-length 150 \
    --out_dir ./spy_intraday
```

### Use Case 3: Multi-Stock Embedding Dataset

**Goal**: Create DINOv3 training dataset with regime-aware segments

```python
# Modify generate_dinov3_dataset.py configuration:
seg_cfg = {
    "mode": "dynamic",
    "model": "l2",
    "penalty": 3.5,
    "min_segment_length": 12,
    "max_segment_length": 100,
    "jump": 5,
)
```

## Troubleshooting

### Problem: No segments generated

**Symptoms**: Empty output, warnings about short series

**Causes**:
- Data too short (< `min_segment_length`)
- All data filtered out

**Solutions**:
1. Check data availability: `print(len(df))`
2. Reduce `min_segment_length`
3. Expand date range

### Problem: Ruptures import error

**Symptoms**: `ImportError: No module named 'ruptures'`

**Solution**:
```bash
pip install ruptures>=1.1.0
```

Or use fixed-window fallback:
```bash
--fixed-window --window 50
```

### Problem: Very slow performance

**Symptoms**: Taking minutes to process single stock

**Causes**:
- Very large dataset (>10,000 bars)
- Low `jump` parameter

**Solutions**:
1. Increase `--segmentation-jump` to 10 or 20
2. Use shorter date ranges
3. Consider fixed-window mode for very large datasets

### Problem: Segments don't align with visual regime changes

**Symptoms**: Breakpoints in middle of trends

**Causes**:
- Wrong model selection
- Penalty too high/low
- Noisy data

**Solutions**:
1. Try `rbf` model for volatility-based regimes
2. Adjust penalty (lower = more segments)
3. Apply smoothing to Close prices before segmentation

## Advanced Topics

### Custom Segmenter Implementation

Implement your own segmentation logic:

```python
from ohlc_image_module.segmentation import Segmenter, SegmentationConfig
from typing import List, Tuple
import pandas as pd

class CustomSegmenter:
    """Example: Segment on volume spikes."""
    
    def __init__(self, config: SegmentationConfig):
        self.config = config
    
    def segment(self, df: pd.DataFrame) -> List[Tuple[int, int]]:
        # Find volume spikes (2x median)
        median_vol = df["Volume"].median()
        spike_indices = df[df["Volume"] > 2 * median_vol].index
        
        # Convert to segments
        segments = []
        start = 0
        for spike_idx in spike_indices:
            end = df.index.get_loc(spike_idx)
            if end - start >= self.config.min_segment_length:
                segments.append((start, end))
                start = end + 1
        
        # Add final segment
        if len(df) - 1 - start >= self.config.min_segment_length:
            segments.append((start, len(df) - 1))
        
        return segments
```

### Multi-Factor Segmentation

Combine price and volume for segmentation:

```python
import ruptures as rpt

def multi_factor_segment(df, penalty=3.0):
    # Normalize features
    close_norm = (df["Close"] - df["Close"].mean()) / df["Close"].std()
    vol_norm = (df["Volume"] - df["Volume"].mean()) / df["Volume"].std()
    
    # Stack features
    signal = np.column_stack([close_norm, vol_norm])
    
    # Detect change-points
    algo = rpt.Pelt(model="l2", min_size=10)
    algo.fit(signal)
    breakpoints = algo.predict(pen=penalty)
    
    return breakpoints
```

### Embedding Quality Analysis

Evaluate if dynamic segmentation improves embeddings:

```python
# Generate embeddings for dynamic segments
# Run similarity search
# Compare with fixed-window baseline

from sklearn.metrics import silhouette_score

# Compute silhouette score for segment embeddings
score_dynamic = silhouette_score(embeddings_dynamic, labels_dynamic)
score_fixed = silhouette_score(embeddings_fixed, labels_fixed)

print(f"Dynamic segmentation score: {score_dynamic:.3f}")
print(f"Fixed window score: {score_fixed:.3f}")
```

## Further Reading

- [Segmentation Overview](segmentation_overview.md) - Algorithm details
- [Migration Guide](migration_guide.md) - Upgrading existing workflows
- [ruptures Documentation](https://centre-borelli.github.io/ruptures-docs/) - Library reference
- Truong et al. (2020) "Selective review of offline change point detection methods"

## Support

For issues or questions:
1. Check existing GitHub issues
2. Run benchmark suite: `python tests/benchmark_segmentation.py`
3. Enable verbose logging: Add `--verbose` flag
4. Report bugs with reproducible examples

