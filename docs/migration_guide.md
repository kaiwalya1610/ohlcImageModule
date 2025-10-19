# Migration Guide: Fixed Windows → Dynamic Segmentation

## Overview

This guide helps existing users migrate from fixed-window segmentation to the new dynamic segmentation system. Dynamic segmentation is now the **default behavior** as of version 2.0.

## Breaking Changes

### 1. Default Behavior Change

**Before (v1.x)**:
- Fixed-window mode was default
- `--window` parameter required for windowing

**After (v2.0)**:
- Dynamic segmentation is default
- Use `--fixed-window` flag to restore old behavior

### 2. Filename Convention Change

**Before**:
```
AAPL_1d_20230101_20231231_win50-stride25-idx0.png
```

**After** (dynamic mode):
```
AAPL_1d_20230101_20231231_seg0000-idx0.png
```

**After** (fixed mode with `--fixed-window`):
```
AAPL_1d_20230101_20231231_win50-stride25-idx0.png  # Unchanged
```

### 3. Metadata Schema Extension

New fields added to metadata CSV:
- `segment_id`: Sequential segment identifier
- `segmentation_mode`: "dynamic" or "fixed"
- `segmentation_params_json`: JSON-encoded segmentation parameters

**Backward compatibility**: Old scripts reading metadata will continue to work; new fields are simply additional columns.

## Migration Paths

### Path 1: Immediate Adoption (Recommended)

**Objective**: Switch to dynamic segmentation with minimal changes

**Steps**:

1. **Install ruptures library**:
   ```bash
   pip install ruptures>=1.1.0
   ```

2. **Update CLI calls** (remove `--window` and `--stride`):
   ```bash
   # Old
   python make_ohlc_images.py --ticker AAPL --window 50 --stride 25
   
   # New (dynamic is default)
   python make_ohlc_images.py --ticker AAPL
   ```

3. **Tune parameters** (optional):
   ```bash
   python make_ohlc_images.py \
       --ticker AAPL \
       --segmentation-penalty 3.0 \
       --min-segment-length 10 \
       --max-segment-length 200
   ```

4. **Update downstream scripts** to handle variable-length segments

### Path 2: Gradual Transition

**Objective**: Test dynamic segmentation alongside existing fixed-window workflows

**Steps**:

1. **Keep existing fixed-window workflows unchanged**:
   ```bash
   # Add --fixed-window to maintain old behavior
   python make_ohlc_images.py \
       --ticker AAPL \
       --fixed-window \
       --window 50 \
       --stride 25
   ```

2. **Run parallel dynamic experiments**:
   ```bash
   python make_ohlc_images.py \
       --ticker AAPL \
       --out_dir ./experiments/dynamic
   ```

3. **Compare results** using benchmark suite:
   ```bash
   python tests/benchmark_segmentation.py
   ```

4. **Gradually migrate** when satisfied with dynamic results

### Path 3: Postpone Migration

**Objective**: Continue using fixed windows indefinitely

**Steps**:

1. **Add `--fixed-window` flag to all CLI invocations**:
   ```bash
   python make_ohlc_images.py \
       --ticker AAPL \
       --fixed-window \
       --window 50 \
       --stride 25
   ```

2. **Optionally**: Create wrapper script or alias:
   ```bash
   # In ~/.bashrc or equivalent
   alias make_ohlc_fixed='python make_ohlc_images.py --fixed-window'
   ```

3. **Note**: Fixed-window mode is fully supported and will not be deprecated

## Code Migration Examples

### Example 1: Simple CLI Script

**Before**:
```bash
#!/bin/bash
python make_ohlc_images.py \
    --ticker MSFT \
    --start 2023-01-01 \
    --end 2023-12-31 \
    --interval 1d \
    --window 100 \
    --stride 50 \
    --out_dir ./output
```

**After (dynamic)**:
```bash
#!/bin/bash
python make_ohlc_images.py \
    --ticker MSFT \
    --start 2023-01-01 \
    --end 2023-12-31 \
    --interval 1d \
    --segmentation-penalty 3.0 \
    --min-segment-length 20 \
    --max-segment-length 150 \
    --out_dir ./output
```

**After (keep fixed)**:
```bash
#!/bin/bash
python make_ohlc_images.py \
    --ticker MSFT \
    --start 2023-01-01 \
    --end 2023-12-31 \
    --interval 1d \
    --fixed-window \
    --window 100 \
    --stride 50 \
    --out_dir ./output
```

### Example 2: Python Integration

**Before**:
```python
from ohlc_image_module.processing import iter_windows
from ohlc_image_module.data import fetch_ohlcv

df = fetch_ohlcv("AAPL", "2023-01-01", "2023-12-31", "1d")

for window_df, start, end in iter_windows(df, window=50, stride=25):
    # Process window
    process_window(window_df)
```

**After (dynamic)**:
```python
from ohlc_image_module.processing import iter_segments
from ohlc_image_module.data import fetch_ohlcv

df = fetch_ohlcv("AAPL", "2023-01-01", "2023-12-31", "1d")
config = {"mode": "dynamic", "penalty": 3.0}

for seg_df, start, end, seg_id in iter_segments(df, config):
    # Process segment (note: added seg_id)
    process_segment(seg_df, seg_id)
```

**After (keep fixed)**:
```python
from ohlc_image_module.processing import iter_segments
from ohlc_image_module.data import fetch_ohlcv

df = fetch_ohlcv("AAPL", "2023-01-01", "2023-12-31", "1d")
config = {"mode": "fixed", "window": 50, "stride": 25}

for seg_df, start, end, seg_id in iter_segments(df, config):
    # Same interface, fixed windowing
    process_segment(seg_df, seg_id)
```

### Example 3: DINOv3 Dataset Generation

**Before** (`generate_dinov3_dataset.py`):
```python
WINDOW_SIZE = 10
STEP_SIZE = 1

for ticker in tickers:
    windows = create_sliding_windows(df, WINDOW_SIZE, STEP_SIZE)
    for idx, window_df in enumerate(windows):
        # Render and save
```

**After**:
```python
from ohlc_image_module.processing import iter_segments

seg_cfg = {
    "mode": "dynamic",
    "penalty": 3.0,
    "min_segment_length": 10,
    max_segment_length=100,
)

for ticker in tickers:
    for seg_df, start, end, seg_id in iter_segments(df, seg_cfg):
        # Render and save (automatically adapts to market regimes)
```

## Metadata Compatibility

### Reading Old Metadata

Old metadata files (without segmentation fields) can still be read:

```python
import pandas as pd

# This works with both old and new metadata
metadata = pd.read_csv("metadata.csv")

# Check for new fields
if "segmentation_mode" in metadata.columns:
    print("New format detected")
    dynamic_rows = metadata[metadata["segmentation_mode"] == "dynamic"]
else:
    print("Old format detected")
```

### Handling Mixed Metadata

If you have both old and new data:

```python
import pandas as pd

metadata = pd.read_csv("metadata.csv")

# Add default values for missing fields
if "segmentation_mode" not in metadata.columns:
    metadata["segmentation_mode"] = "fixed"
if "segment_id" not in metadata.columns:
    # Infer segment_id from index for old data
    metadata["segment_id"] = range(len(metadata))

# Now process uniformly
for _, row in metadata.iterrows():
    if row["segmentation_mode"] == "dynamic":
        # Handle dynamic segments
        pass
    else:
        # Handle fixed windows
        pass
```

## Testing Migration

### Validation Checklist

- [ ] Ruptures installed (`pip list | grep ruptures`)
- [ ] CLI runs without errors
- [ ] Images generated successfully
- [ ] Metadata CSV contains new fields
- [ ] Downstream scripts handle variable-length segments
- [ ] Embedding pipeline works end-to-end
- [ ] Performance acceptable (run benchmarks)

### Quick Test

```bash
# Test dynamic segmentation
python make_ohlc_images.py \
    --ticker AAPL \
    --start 2023-12-01 \
    --end 2023-12-31 \
    --interval 1d \
    --limit 10 \
    --out_dir ./migration_test

# Verify output
ls ./migration_test/images/*.png
cat ./migration_test/metadata.csv | head
```

### Comparison Test

Run the same ticker with both modes:

```bash
# Dynamic
python make_ohlc_images.py \
    --ticker AAPL \
    --start 2023-01-01 \
    --end 2023-12-31 \
    --interval 1d \
    --out_dir ./test_dynamic

# Fixed
python make_ohlc_images.py \
    --ticker AAPL \
    --start 2023-01-01 \
    --end 2023-12-31 \
    --interval 1d \
    --fixed-window \
    --window 50 \
    --stride 25 \
    --out_dir ./test_fixed

# Compare
echo "Dynamic segments:"
ls ./test_dynamic/images/*.png | wc -l
echo "Fixed windows:"
ls ./test_fixed/images/*.png | wc -l
```

## Performance Considerations

### Expected Performance Changes

**Dynamic Segmentation**:
- **Pros**: Regime-aware, potentially fewer segments in stable periods
- **Cons**: Adds computational overhead (change-point detection)
- **Typical**: 2-5x slower than fixed windows on daily data

**Mitigation Strategies**:

1. **Increase `--segmentation-jump`** for large datasets:
   ```bash
   --segmentation-jump 10  # Consider every 10th point
   ```

2. **Use fixed windows for exploratory analysis**, dynamic for production

3. **Pre-compute segments** and cache for repeated use

## FAQ

### Q: Do I need to regenerate all my existing images?

**A**: No. Old images remain valid. Only new images will use dynamic segmentation. You can mix old (fixed) and new (dynamic) in the same pipeline if you handle the metadata correctly.

### Q: Can I use dynamic segmentation without installing ruptures?

**A**: No. Dynamic segmentation requires ruptures. If unavailable, the system will warn and fall back to fixed windowing.

### Q: Will my embedding models break?

**A**: Not necessarily. If your model expects fixed-size inputs, you already have normalization/resizing in your pipeline. The image dimensions remain the same (`--img-size`), only the time span varies.

### Q: How do I choose between l2 and rbf models?

**A**: Use `l2` (default) for most cases. Use `rbf` when volatility regimes are more important than price level changes.

### Q: What if I need exactly N segments per ticker?

**A**: Dynamic segmentation produces variable counts. For fixed counts, use `--fixed-window` mode with appropriate window/stride to target N segments.

### Q: Can I combine dynamic and fixed modes?

**A**: Not in a single run, but you can run both and merge the outputs, ensuring metadata distinguishes them via `segmentation_mode` field.

## Rollback Procedure

If you need to completely revert:

1. **Uninstall ruptures** (optional):
   ```bash
   pip uninstall ruptures
   ```

2. **Always use `--fixed-window` flag**:
   ```bash
   # Add to all scripts
   --fixed-window --window 50 --stride 25
   ```

3. **Revert to older version** (if needed):
   ```bash
   git checkout v1.x
   pip install -e .
   ```

## Support

For migration assistance:
- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Examples**: See `examples/dynamic_segmentation_demo.py`

## Version History

- **v2.0**: Dynamic segmentation default, ruptures integration
- **v1.x**: Fixed-window only



