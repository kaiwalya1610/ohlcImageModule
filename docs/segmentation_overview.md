# Dynamic Time Series Segmentation Overview

## Introduction

This document outlines the rationale and implementation details for dynamic time series segmentation in the OHLC image generation pipeline. Dynamic segmentation adaptively identifies regime changes in financial time series, replacing fixed-length windowing with data-driven boundary detection.

## Motivation

Fixed-length windows treat all market regimes uniformly, potentially:
- Splitting coherent price movements across multiple windows
- Grouping disparate market regimes within a single window
- Missing important structural changes in volatility or trend

Dynamic segmentation addresses these issues by detecting change-points that mark transitions between different market regimes.

## Algorithm Selection

### Primary Approach: Ruptures Library

We selected the **ruptures** library as our primary change-point detection framework due to:

1. **Algorithmic Flexibility**: Supports multiple detection methods (Pelt, BottomUp, Window)
2. **Cost Function Variety**: Offers l2 (mean change), rbf (distribution change), and others
3. **Performance**: Optimized implementations with O(n log n) to O(n²) complexity
4. **Reliability**: Well-maintained, tested on diverse time series applications
5. **Windows Compatibility**: Pure Python with NumPy/SciPy dependencies only
6. **Offline Detection**: Suitable for batch processing historical data

### Selected Algorithms

#### PELT (Pruned Exact Linear Time)
- **Use case**: Primary default for most applications
- **Complexity**: O(n) to O(n²) depending on data
- **Advantages**: Fast, exact solution for penalty-based detection
- **Parameters**: 
  - `model`: Cost function ("l2" for mean shifts, "rbf" for distribution changes)
  - `penalty`: Controls sensitivity (higher = fewer segments)
  - `min_size`: Minimum segment length constraint

#### BottomUp
- **Use case**: Alternative when PELT produces unstable results
- **Complexity**: O(n²)
- **Advantages**: Hierarchical, more stable on noisy data
- **Parameters**: Same as PELT

### Cost Functions

#### L2 (Least Squares)
- Detects **mean shifts** in Close prices
- Best for: Trend changes, support/resistance breaks
- Formula: Minimizes sum of squared deviations from segment means

#### RBF (Radial Basis Function)
- Detects **distribution changes** in price dynamics
- Best for: Volatility regime changes, market structure shifts
- Formula: Uses kernel-based distance between distributions

## Comparison with Alternatives

| Library | Algorithm | Windows | Streaming | Pros | Cons |
|---------|-----------|---------|-----------|------|------|
| **ruptures** | PELT, BottomUp | ✓ | ✗ | Fast, flexible, proven | Offline only |
| kats | Multiple | ✓ | ✓ | Facebook-backed, rich | Heavy dependencies (Prophet, etc.) |
| changepy | PELT, BinSeg | ✓ | ✗ | Lightweight | Less maintained |
| river | Online methods | ✓ | ✓ | Streaming support | Complex API, ML focus |

**Decision**: Ruptures provides the best balance of performance, reliability, and ease of integration for our batch processing use case.

## Implementation Architecture

### Core Components

1. **SegmentationConfig**: Configuration dataclass
   - `mode`: "dynamic" | "fixed"
   - Algorithm parameters: `model`, `penalty`, `min_size`, `max_size`
   - Computational: `jump` (subsampling for speed)

2. **Segmenter Protocol**: Abstract interface
   - `segment(df: pd.DataFrame) -> List[Tuple[int, int]]`
   - Enables future algorithm extensions

3. **RupturesSegmenter**: Primary implementation
   - Uses Close prices for detection
   - Enforces min/max segment constraints
   - Graceful fallback if ruptures unavailable

4. **FixedWindowSegmenter**: Backward compatibility wrapper
   - Wraps existing `iter_windows` logic
   - Unified interface for both modes

### Integration Points

- **CLI**: New flags for segmentation control, `--fixed-window` for legacy mode
- **Processing**: `iter_segments()` function yields adaptive segments
- **Metadata**: Extended schema captures segmentation parameters
- **Embedding**: Segment-aware naming and visualization

## Parameter Tuning Guidelines

### Penalty (Sensitivity)
- **Range**: 1.0 to 10.0
- **Default**: 3.0
- **Lower values**: More segments, captures micro-regimes
- **Higher values**: Fewer segments, focuses on major changes
- **Tuning**: Start at 3.0, increase if over-segmenting

### Min/Max Segment Length
- **Min length**: 10-20 bars (ensures statistical significance)
- **Max length**: 100-200 bars (prevents runaway segments)
- **Context-dependent**: Adjust based on interval (1m vs 1d)

### Model Selection
- **l2**: Default for most equity applications (trend/level changes)
- **rbf**: Use for volatility-focused analysis or options markets

### Jump (Performance)
- **Default**: 5 (every 5th point considered as potential breakpoint)
- **Higher**: Faster but less precise boundaries
- **Lower**: Slower but more accurate

## Evaluation Metrics

### Quantitative
1. **Segment count distribution**: Should vary by ticker/period
2. **Segment length statistics**: Mean, std, min, max
3. **Coverage**: All bars assigned exactly once
4. **Boundary precision**: Alignment with known events (earnings, news)

### Qualitative
1. **Visual inspection**: Segment boundaries on price charts
2. **Regime coherence**: Within-segment price action similarity
3. **Embedding quality**: Downstream similarity search performance

## Future Extensions

1. **Multi-factor segmentation**: Combine price + volume + volatility
2. **Streaming detection**: Online algorithms for real-time applications
3. **Hierarchical segmentation**: Multi-scale regime detection
4. **Supervised tuning**: ML-based parameter optimization from labeled data

## References

- Truong et al. (2020). "Selective review of offline change point detection methods." Signal Processing.
- Killick et al. (2012). "Optimal detection of changepoints with a linear computational cost." JASA.
- Ruptures documentation: https://centre-borelli.github.io/ruptures-docs/



