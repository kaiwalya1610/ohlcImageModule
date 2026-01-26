---
description: 
alwaysApply: true
---

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ohlcImageModule** is a Python library for generating deterministic, embedding-ready OHLC candlestick images from financial time series data (Yahoo Finance). It supports:

- **Dynamic segmentation**: Automatic regime change detection using change-point algorithms (ruptures library)
- **Fixed windowing**: Legacy sliding-window mode for backward compatibility
- **Flexible rendering**: Customizable colors, sizes, volume overlays, and normalization modes
- **Embedding generation**: Integration with vision models (DINOv3, Qwen3-VL) for creating financial image embeddings
- **Time-series vectorization**: Multi-channel "DNA" vectors from OHLC data for hybrid retrieval alongside visual embeddings

## Architecture

The project is organized into three main layers:

### 1. Core Module (`ohlc_image_module/`)

**Key components:**

- **`cli.py`**: Command-line entry point with argument parsing and segmentation/windowing logic. Delegates heavy lifting to other modules.
- **`data.py`**: Yahoo Finance data fetching (`fetch_ohlcv`) and validation with gap detection
- **`config.py`**: Shared constants (`PRICE_COLUMNS`, `REQUIRED_COLUMNS`) and `RenderConfig` dataclass for rendering parameters
- **`processing.py`**: Window and segment iteration, OHLC normalization (zscore, minmax, none)
- **`segmentation.py`**: Dynamic segmentation via ruptures library (change-point detection) with cost function options (l2, rbf, linear, normal, ar)
- **`render.py`**: Candlestick chart rendering using mplfinance/matplotlib, PIL image output
- **`metadata.py`**: Builds metadata rows (one per image) tracking ticker, dates, normalization, segmentation mode
- **`vector_utils.py`**: Time-series vectorization utilities for generating multi-channel enriched vectors (RSI, log returns, relative volume)
- **`embedding_verification.py`**: Verification pipeline for embeddings (load, retrieval probes, linear probes, visualization)

### 2. Root-Level Scripts

- **`make_ohlc_images.py`**: Main CLI entry point (wraps `ohlc_image_module.cli.main`)
- **`generate_embeddings.py`**: Batch embedding generation using Qwen3-VL-Embedding
- **`generate_dinov3_dataset.py`**: Dataset generation and processing for DINOv3 embeddings
- **`generate_ts_vectors.py`**: Batch time-series vector generation from OHLC segments
- **`load_embedding.py`**: Embedding retrieval utilities
- **`retrieve_and_rerank.py`**: Embedding-based retrieval and reranking
- **`verify_embeddings.py`**: Embedding verification and evaluation
- **`viz_pca_embd.py`**: PCA visualization of embeddings

### 3. Test Suite (`tests/`)

- **`test_processing.py`**: Window and normalization logic
- **`test_data_validation.py`**: Data validation edge cases
- **`test_segmentation.py`**: Change-point detection correctness
- **`benchmark_segmentation.py`**: Performance benchmarking (dynamic vs fixed modes)
- **`validation_segmentation.py`**: Segment output validation

## Commands

### Image Generation

```bash
# Full chart (no windowing/segmentation)
python make_ohlc_images.py --ticker AAPL --start 2023-01-01 --end 2023-12-31 --interval 1d --out_dir ./out

# Dynamic segmentation (default as of v2.0)
python make_ohlc_images.py --ticker MSFT --start 2023-01-01 --end 2023-12-31 --interval 1h \
  --out_dir ./out \
  --segmentation-model l2 \
  --segmentation-penalty 3.0 \
  --min-segment-length 10 \
  --max-segment-length 200

# Fixed windowing (legacy mode)
python make_ohlc_images.py --ticker TSLA --start 2023-01-01 --end 2023-12-31 --interval 1d \
  --fixed-window --window 50 --stride 25 --out_dir ./out
```

### Time-Series Vector Generation

```bash
# Generate enriched vectors from existing metadata CSV
# Edit configuration in generate_ts_vectors.py:
#   METADATA_PATH = "path/to/metadata.csv"
#   OUT_DIR = "output/directory"
#   INTERVAL = "5m"
#   TARGET_LEN = 128  # per channel, 512 total

python generate_ts_vectors.py

# Output structure:
# out_dir/
# ├── vectors/
# │   ├── TICKER_NS_seg_0000.npy  # (512,) float32 array
# │   ├── TICKER_NS_seg_0001.npy
# │   └── ...
# └── metadata_enriched.csv  # Original + vector_path + vector_status
```

### Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_processing.py -v

# Run specific test
pytest tests/test_segmentation.py::test_dynamic_segmentation -v

# Benchmark segmentation performance
python tests/benchmark_segmentation.py

# Validate segmentation outputs
python tests/validation_segmentation.py
```

### Development

```bash
# Install in development mode (from project root)
pip install -e .

# Lint (if configured)
# Currently no linting configured - use your preferred tool (ruff, black, etc.)

# Type checking (if configured)
# Currently no type checking configured
```

## Key Concepts

### Segmentation Modes (v2.0+)

**Dynamic Segmentation (default)**:
- Uses ruptures library for change-point detection
- Identifies regime shifts in OHLC data automatically
- Args: `--segmentation-model` (l2/rbf/linear/normal/ar), `--segmentation-penalty`, `--min-segment-length`, `--max-segment-length`, `--segmentation-jump`
- Output filenames: `ticker_interval_date_date_seg0000-idxN.png`
- ~10-30x slower than fixed windowing but suitable for most use cases (50-300ms per 1000 bars)

**Fixed Windowing**:
- Legacy sliding-window approach
- Use `--fixed-window` flag to enable
- Args: `--window`, `--stride` (defaults to window size for non-overlapping)
- Output filenames: `ticker_interval_date_date_winN-strideN-idxN.png`
- Backward compatible with v1.x

### Metadata CSV

Generated for each run (unless `--no_save_metadata_csv` is set). Includes:
- Image path, ticker, interval, date range, bar count
- Normalization mode (zscore/minmax/none)
- Window/stride info (fixed mode) or segment_id (dynamic mode)
- Segmentation mode and parameters (v2.0+)

### Normalization

Three modes applied to OHLC prices before rendering:
- `zscore`: Zero-mean, unit variance (default)
- `minmax`: Scale to [0, 1]
- `none`: Raw prices

### Enriched Time-Series Vectors

The vectorization pipeline generates fixed-length "DNA" vectors from OHLC segments for hybrid retrieval:

**4-Channel Architecture** (default: 128 per channel = 512 total):
1. **Price Channel**: Interpolated close prices, z-score normalized
2. **RSI Channel**: TA-Lib RSI (14-period), interpolated and z-score normalized
3. **Log Returns Channel**: Log returns, interpolated and z-score normalized
4. **Relative Volume Channel**: Volume / SMA(Volume, 20), interpolated and z-score normalized

**Key Functions** (in `vector_utils.py`):
- `interpolate_series()`: Linear interpolation to fixed length
- `standard_scale_series()`: Z-score standardization with flat-line handling
- `compute_rsi()`: TA-Lib RSI implementation (Wilder's smoothing)
- `compute_log_returns()`: ln(P_t / P_{t-1}) with invalid ratio handling
- `compute_relative_volume()`: Volume relative to rolling average
- `generate_enriched_vector()`: Master function producing 4-channel concatenated vector

**Edge Case Handling**:
- Segments < 2 points: zero-vector
- Flat-line data (std=0): zero after z-score
- NaN/Inf values: replaced with 0.0
- Invalid log ratios: replaced with 0.0

## Important Patterns

### Data Flow

**Image Generation:**
1. Fetch OHLC data from Yahoo Finance (`fetch_ohlcv`)
2. Validate for gaps/missing data (`validate_df`)
3. Segment or window the data (`iter_segments` or `iter_windows`)
4. Normalize per segment/window (`normalize_ohlc`)
5. Render to image (`render_candlestick`)
6. Build metadata row (`build_metadata_row`)
7. Save PNG + collect metadata for CSV

**Vector Generation:**
1. Load metadata CSV with segment timestamps
2. Group segments by ticker (class_id)
3. Fetch full OHLC history once per ticker
4. Slice segments using timestamp ranges
5. Generate 4-channel enriched vector per segment
6. Save .npy files (one per segment)
7. Create enriched metadata CSV with vector_path column

### Rendering Constraints

- Matplotlib configured with `Agg` backend (no display, suitable for headless systems)
- Pillow image resampling: uses `Image.Resampling.BICUBIC` (Pillow 10+) with fallback to `Image.BICUBIC`
- Volume subplot: min-max normalized separately per image
- Output: Square PIL Image (default 256x256 pixels, customizable via `--img_size`)

## Dependencies

**Core dependencies** (from code usage):
- `yfinance`: Yahoo Finance data fetching
- `pandas`: Data structures and manipulation
- `numpy`: Numerical operations
- `matplotlib`: Plotting and rendering
- `mplfinance`: Candlestick chart library
- `pillow`: Image processing
- `ruptures`: Change-point detection (optional, required for dynamic segmentation)
- `ta-lib`: Technical analysis indicators (required for time-series vectorization)
- `scipy`: Scientific computing, interpolation (required for time-series vectorization)

**Optional dependencies** (for embedding scripts):
- `torch`: DINOv3 embeddings
- Vision model packages (Qwen3-VL, DINOv3)

## Migration Note

If upgrading from v1.x to v2.0+:
- Default behavior switched from fixed-window to dynamic segmentation
- Use `--fixed-window` flag to restore old behavior
- Filename conventions changed for dynamic mode (`seg` vs `win` prefix)
- Metadata CSV extended with segmentation fields (backward compatible)
- See `docs/migration_guide.md` for detailed guidance

## Testing Strategy

- **Unit tests** in `tests/test_*.py`: Pure logic validation
- **Benchmarks** in `tests/benchmark_segmentation.py`: Performance comparisons
- **Validation scripts** in `tests/validation_segmentation.py`: Output sanity checks
- Tests use pytest fixtures (conftest.py configures sys.path)

## Common Development Tasks

### Adding a New Normalization Mode

1. Add logic to `ohlc_image_module/processing.py` in `normalize_ohlc()`
2. Update CLI argument choices in `ohlc_image_module/cli.py`
3. Add test case to `tests/test_processing.py`

### Modifying Rendering

1. Update `RenderConfig` in `ohlc_image_module/config.py` if adding new parameters
2. Modify rendering logic in `ohlc_image_module/render.py`
3. Update CLI argument parsing in `ohlc_image_module/cli.py`

### Adding Segmentation Methods

1. Implement new segmenter class in `ohlc_image_module/segmentation.py` (must follow `Segmenter` protocol)
2. Register in `create_segmenter()` factory function
3. Add CLI option and tests

## Performance Characteristics

- **Fixed windowing**: 5-10ms per 1000 bars
- **Dynamic segmentation (l2)**: 50-150ms per 1000 bars
- **Dynamic segmentation (rbf)**: 100-300ms per 1000 bars
- Memory: 10-20MB peak for typical datasets
- `--segmentation-jump` parameter trades speed for accuracy (higher = faster)

## Recent Changes

- **v2.1**: Time-series vectorization pipeline for hybrid retrieval
  - Multi-channel enriched vectors (price, RSI, log returns, relative volume)
  - TA-Lib integration for technical indicators
  - Batch processing script with automatic ticker grouping
- **v2.0**: Dynamic segmentation default, backward compatibility via `--fixed-window`
- **Embeddings pipeline**: Integration with DINOv3 and Qwen3-VL for image embeddings
- **Verification tools**: Protocol-based embedding evaluation and retrieval
