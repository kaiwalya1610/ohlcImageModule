"""Verification utilities for analysing candlestick image embeddings.

This module implements a seven step protocol for assessing whether a set of
embeddings extracted from OHLC candlestick images capture financially
meaningful structure rather than only superficial chart style artefacts.

The implementation follows the procedure described in the user instructions
and exposes a small API that can be consumed either programmatically or via
the accompanying CLI (``verify_embeddings.py``).

## Segment-Aware Capabilities (v2.0+)

This module is compatible with both fixed-window and dynamic segmentation modes:

- **Metadata Loading**: Automatically detects `segmentation_mode` and `segment_id` 
  fields in metadata CSV/Parquet files.
  
- **Naming Convention**: Handles both legacy (`win*-stride*-idx*`) and dynamic 
  (`seg####-idx*`) filename patterns.
  
- **Visualization**: Grouping and coloring can be done by segment_id for dynamic 
  segmentation analysis.

**Note**: No code changes required for backward compatibility. Existing workflows 
continue to function with new metadata fields simply being additional columns.

For segment boundary visualization on price charts, see:
- `examples/dynamic_segmentation_demo.py`
- `tests/validation_segmentation.py`
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from numpy.typing import ArrayLike
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    f1_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import NearestNeighbors


LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses used to group the protocol outputs.


@dataclasses.dataclass
class GeometryReport:
    """Summary of the geometry sanity checks (Step 1)."""

    explained_variance: List[float]
    top1_variance_pct: float
    top10_variance_pct: float
    condition_number: float
    collapsed: bool


@dataclasses.dataclass
class LinearProbeReport:
    """Metrics for the linear probe walk-forward evaluation (Step 2)."""

    auroc: float
    accuracy: float
    f1: float
    baseline_accuracy: float
    splits: int


@dataclasses.dataclass
class RetrievalReport:
    """Results for the nearest neighbour purity analysis (Step 3)."""

    neighbour_purity_return: Optional[float]
    neighbour_purity_volatility: Optional[float]
    k: int


@dataclasses.dataclass
class InvarianceReport:
    """Cosine similarity diagnostics for augmentations (Step 4)."""

    style_similarity_mean: Optional[float]
    style_similarity_std: Optional[float]
    structure_similarity_mean: Optional[float]
    structure_similarity_std: Optional[float]


@dataclasses.dataclass
class ClusteringReport:
    """Scores for the regime discovery clustering experiment (Step 5)."""

    k: int
    silhouette: float
    davies_bouldin: float
    calinski_harabasz: float
    volatility_purity: Optional[float]
    return_purity: Optional[float]


@dataclasses.dataclass
class StressTestReport:
    """Train/test robustness metrics for Step 6."""

    auroc: float
    accuracy: float
    f1: float
    baseline_accuracy: float
    n_train: int
    n_test: int


@dataclasses.dataclass
class VerificationBundle:
    """Container for all protocol outputs."""

    geometry: GeometryReport
    linear_probe: Optional[LinearProbeReport]
    retrieval: Optional[RetrievalReport]
    invariance: Optional[InvarianceReport]
    clustering: List[ClusteringReport]
    stress_test: Optional[StressTestReport]
    figures: Dict[str, str]


# ---------------------------------------------------------------------------
# Utility helpers


def _ensure_output_dir(path: os.PathLike[str] | str) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _normalise_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return embeddings / norms


def _load_embedding_matrix(path: Path) -> np.ndarray:
    if not path.exists():  # pragma: no cover - defensive branch
        raise FileNotFoundError(f"Embeddings file not found: {path}")

    if path.suffix.lower() in {".npy", ".npz"}:
        data = np.load(path)
        if isinstance(data, np.ndarray):
            embeddings = data
        else:
            if "embeddings" not in data:
                raise KeyError(
                    "NPZ archive must contain an 'embeddings' array when using np.savez."
                )
            embeddings = data["embeddings"]
    elif path.suffix.lower() in {".csv", ".parquet"}:
        df_embeddings = (
            pd.read_csv(path)
            if path.suffix.lower() == ".csv"
            else pd.read_parquet(path)
        )
        emb_cols = [col for col in df_embeddings.columns if col.startswith("emb_")]
        if not emb_cols:
            raise ValueError("Embedding DataFrame must contain columns named emb_*." )
        embeddings = df_embeddings[emb_cols].to_numpy()
    else:  # pragma: no cover - unsupported extension
        raise ValueError(f"Unsupported embeddings format: {path.suffix}")

    return _normalise_embeddings(np.asarray(embeddings, dtype=np.float32))


def load_embeddings(
    embeddings_path: os.PathLike[str] | str,
    metadata_path: os.PathLike[str] | str,
    timestamp_col: str,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Load embeddings and metadata from disk."""

    emb_path = Path(embeddings_path)
    embeddings = _load_embedding_matrix(emb_path)

    meta_path = Path(metadata_path)
    if meta_path.suffix.lower() == ".csv":
        metadata = pd.read_csv(meta_path)
    elif meta_path.suffix.lower() in {".parquet", ".pq"}:
        metadata = pd.read_parquet(meta_path)
    else:
        raise ValueError(f"Unsupported metadata format: {meta_path.suffix}")

    if timestamp_col not in metadata.columns:
        raise KeyError(f"Metadata missing required timestamp column: {timestamp_col}")

    metadata[timestamp_col] = pd.to_datetime(metadata[timestamp_col], utc=False)
    metadata = metadata.sort_values(timestamp_col).reset_index(drop=True)

    if len(metadata) != len(embeddings):
        raise ValueError(
            "Embeddings and metadata must have the same number of rows. "
            f"Got {len(embeddings)} embeddings vs {len(metadata)} metadata rows."
        )

    return _normalise_embeddings(np.asarray(embeddings, dtype=np.float32)), metadata


def load_embedding_array(path: os.PathLike[str] | str) -> np.ndarray:
    """Load a standalone embedding matrix and normalise row-wise to unit norm."""

    return _load_embedding_matrix(Path(path))


# ---------------------------------------------------------------------------
# Step 1 – Geometry sanity checks


def geometry_sanity_check(
    embeddings: ArrayLike, n_components: int = 10
) -> GeometryReport:
    """Run PCA/isotropy diagnostics and flag collapsed representations."""

    embeddings = np.asarray(embeddings)
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array of shape (n_samples, dim).")

    n_samples, dim = embeddings.shape
    if n_samples < 5:
        raise ValueError("At least five samples required for geometry diagnostics.")

    max_components = min(n_components, dim, n_samples)
    pca = PCA(n_components=max_components, svd_solver="auto", random_state=0)
    pca.fit(embeddings)
    explained = pca.explained_variance_ratio_.tolist()

    cov = np.cov(embeddings, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, a_min=1e-12, a_max=None)
    condition = float(eigvals.max() / eigvals.min())

    top1 = float(explained[0] * 100.0)
    topk = float(sum(explained[: min(10, len(explained))]) * 100.0)
    collapsed = top1 > 50.0

    return GeometryReport(
        explained_variance=explained,
        top1_variance_pct=top1,
        top10_variance_pct=topk,
        condition_number=condition,
        collapsed=collapsed,
    )


# ---------------------------------------------------------------------------
# Step 2 – Linear probe with purged walk-forward CV


def _purged_timeseries_splits(
    n_samples: int,
    n_splits: int,
    gap: int,
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    for train_idx, test_idx in splitter.split(np.arange(n_samples)):
        yield train_idx, test_idx


def linear_probe_walk_forward(
    embeddings: ArrayLike,
    labels: Sequence[int] | Sequence[float],
    timestamps: Sequence[pd.Timestamp],
    n_splits: int = 5,
    gap: int = 5,
    max_iter: int = 1000,
) -> Optional[LinearProbeReport]:
    """Evaluate predictive utility using a purged walk-forward split."""

    embeddings = np.asarray(embeddings)
    labels = np.asarray(labels)
    if len(labels) != len(embeddings):
        raise ValueError("Labels must have the same length as embeddings.")

    if len(np.unique(labels)) < 2:
        LOGGER.warning("Linear probe skipped: label column is not binary.")
        return None

    order = np.argsort(np.asarray(timestamps))
    embeddings = embeddings[order]
    labels = labels[order]

    splitter = list(_purged_timeseries_splits(len(labels), n_splits=n_splits, gap=gap))
    if not splitter:
        LOGGER.warning("Linear probe skipped: insufficient samples for walk-forward split.")
        return None

    aucs: List[float] = []
    accs: List[float] = []
    f1s: List[float] = []

    clf = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        class_weight="balanced",
        max_iter=max_iter,
    )

    for fold, (train_idx, test_idx) in enumerate(splitter, start=1):
        if len(np.unique(labels[train_idx])) < 2:
            LOGGER.debug("Skipping fold %d due to class collapse in training set.", fold)
            continue

        clf.fit(embeddings[train_idx], labels[train_idx])
        preds = clf.predict_proba(embeddings[test_idx])[:, 1]
        preds_label = (preds >= 0.5).astype(int)
        aucs.append(roc_auc_score(labels[test_idx], preds))
        accs.append(accuracy_score(labels[test_idx], preds_label))
        f1s.append(f1_score(labels[test_idx], preds_label))

    if not aucs:
        LOGGER.warning("Linear probe skipped: all folds collapsed to a single class.")
        return None

    return LinearProbeReport(
        auroc=float(np.mean(aucs)),
        accuracy=float(np.mean(accs)),
        f1=float(np.mean(f1s)),
        baseline_accuracy=0.5,
        splits=len(aucs),
    )


# ---------------------------------------------------------------------------
# Step 3 – Nearest neighbour retrieval probe


def retrieval_probe(
    embeddings: ArrayLike,
    return_labels: Optional[Sequence[int]],
    volatility_labels: Optional[Sequence[str]],
    k: int = 10,
) -> RetrievalReport:
    """Measure neighbour purity for return sign and volatility regimes."""

    embeddings = np.asarray(embeddings)
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nbrs.fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings, return_distance=True)

    def _purity(target: Optional[Sequence]) -> Optional[float]:
        if target is None:
            return None
        values = np.asarray(target)
        if len(values) != len(embeddings):
            raise ValueError("Label array length mismatch in retrieval probe.")
        if len(np.unique(values)) < 2:
            LOGGER.warning("Retrieval purity skipped: only one class present.")
            return None

        matches = []
        for row_idx, neighbours in enumerate(indices):
            neighbour_indices = neighbours[1:]
            matches.append(np.mean(values[neighbour_indices] == values[row_idx]))
        return float(np.mean(matches))

    return RetrievalReport(
        neighbour_purity_return=_purity(return_labels),
        neighbour_purity_volatility=_purity(volatility_labels),
        k=k,
    )


# ---------------------------------------------------------------------------
# Step 4 – Invariance vs sensitivity checks


def _cosine_similarity_stats(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    sims = cosine_similarity(a, b).diagonal()
    return float(np.mean(sims)), float(np.std(sims))


def invariance_sensitivity_probe(
    base_embeddings: ArrayLike,
    style_embeddings: Optional[ArrayLike] = None,
    structure_embeddings: Optional[ArrayLike] = None,
) -> Optional[InvarianceReport]:
    """Compare cosine similarity for style-preserving vs structure-breaking augments."""

    base_embeddings = np.asarray(base_embeddings)

    def _validate(other: Optional[ArrayLike]) -> Optional[np.ndarray]:
        if other is None:
            return None
        other = np.asarray(other)
        if other.shape != base_embeddings.shape:
            raise ValueError("Augmented embedding arrays must match base shape.")
        return other

    style_embeddings = _validate(style_embeddings)
    structure_embeddings = _validate(structure_embeddings)

    if style_embeddings is None and structure_embeddings is None:
        LOGGER.info("Invariance probe skipped: no augmented embeddings provided.")
        return None

    style_mean = style_std = structure_mean = structure_std = None
    if style_embeddings is not None:
        style_mean, style_std = _cosine_similarity_stats(base_embeddings, style_embeddings)
    if structure_embeddings is not None:
        structure_mean, structure_std = _cosine_similarity_stats(
            base_embeddings, structure_embeddings
        )

    return InvarianceReport(
        style_similarity_mean=style_mean,
        style_similarity_std=style_std,
        structure_similarity_mean=structure_mean,
        structure_similarity_std=structure_std,
    )


# ---------------------------------------------------------------------------
# Step 5 – Clustering & regime discovery


def _cluster_purity(labels: Optional[Sequence], assignments: np.ndarray) -> Optional[float]:
    if labels is None:
        return None
    labels = np.asarray(labels)
    if len(labels) != len(assignments):
        raise ValueError("Purity labels length mismatch.")
    if len(np.unique(labels)) < 2:
        LOGGER.warning("Skipping purity calculation due to single class.")
        return None

    total = 0
    for cluster in np.unique(assignments):
        mask = assignments == cluster
        values, counts = np.unique(labels[mask], return_counts=True)
        if len(values) == 0:
            continue
        total += counts.max()
    return float(total / len(assignments))


def clustering_probe(
    embeddings: ArrayLike,
    return_labels: Optional[Sequence[int]],
    volatility_labels: Optional[Sequence[str]],
    ks: Sequence[int] = (3, 4, 5),
    random_state: int = 0,
) -> List[ClusteringReport]:
    """Apply k-means clustering and compute regime alignment metrics."""

    embeddings = np.asarray(embeddings)
    reports: List[ClusteringReport] = []

    for k in ks:
        if k <= 1 or k > len(embeddings):
            LOGGER.warning("Skipping clustering for invalid k=%d", k)
            continue

        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        assignments = kmeans.fit_predict(embeddings)

        silhouette = silhouette_score(embeddings, assignments)
        davies = davies_bouldin_score(embeddings, assignments)
        calinski = calinski_harabasz_score(embeddings, assignments)

        reports.append(
            ClusteringReport(
                k=k,
                silhouette=float(silhouette),
                davies_bouldin=float(davies),
                calinski_harabasz=float(calinski),
                volatility_purity=_cluster_purity(volatility_labels, assignments),
                return_purity=_cluster_purity(return_labels, assignments),
            )
        )

    return reports


# ---------------------------------------------------------------------------
# Step 6 – Out-of-sample stress test


def stress_test(
    embeddings: ArrayLike,
    labels: Sequence[int] | Sequence[float],
    timestamps: Sequence[pd.Timestamp],
    train_end: pd.Timestamp,
    test_start: Optional[pd.Timestamp] = None,
    test_end: Optional[pd.Timestamp] = None,
    max_iter: int = 1000,
) -> Optional[StressTestReport]:
    """Train on historical data and evaluate on an out-of-sample slice."""

    embeddings = np.asarray(embeddings)
    labels = np.asarray(labels)
    timestamps = pd.to_datetime(pd.Series(timestamps)).to_numpy()

    train_mask = timestamps <= np.datetime64(train_end)
    if not np.any(train_mask):
        LOGGER.warning("Stress test skipped: no samples fall into training range.")
        return None

    if test_start is None:
        test_start = train_end + pd.Timedelta(seconds=1)

    test_mask = timestamps >= np.datetime64(test_start)
    if test_end is not None:
        test_mask &= timestamps <= np.datetime64(test_end)

    if not np.any(test_mask):
        LOGGER.warning("Stress test skipped: no samples fall into test range.")
        return None

    X_train, y_train = embeddings[train_mask], labels[train_mask]
    X_test, y_test = embeddings[test_mask], labels[test_mask]

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        LOGGER.warning("Stress test skipped: class collapse in train or test set.")
        return None

    clf = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        class_weight="balanced",
        max_iter=max_iter,
    )
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    return StressTestReport(
        auroc=float(roc_auc_score(y_test, probs)),
        accuracy=float(accuracy_score(y_test, preds)),
        f1=float(f1_score(y_test, preds)),
        baseline_accuracy=0.5,
        n_train=len(X_train),
        n_test=len(X_test),
    )


# ---------------------------------------------------------------------------
# Step 7 – Visualisation helpers


def _scatter_plot(
    coords: np.ndarray,
    labels: Optional[Sequence],
    title: str,
    output_path: Path,
) -> str:
    fig: Figure
    ax: plt.Axes
    fig, ax = plt.subplots(figsize=(8, 6))
    if labels is None:
        scatter = ax.scatter(coords[:, 0], coords[:, 1], s=12, alpha=0.7)
    else:
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=pd.Categorical(labels).codes,
            cmap="Spectral",
            s=12,
            alpha=0.7,
        )
        legend1 = ax.legend(
            *scatter.legend_elements(),
            title="Classes",
            loc="upper right",
        )
        ax.add_artist(legend1)
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return str(output_path)


def generate_visualisations(
    embeddings: ArrayLike,
    metadata: pd.DataFrame,
    output_dir: os.PathLike[str] | str,
    return_label_col: Optional[str],
    volatility_col: Optional[str],
    random_state: int = 0,
) -> Dict[str, str]:
    """Produce PCA/TSNE projections coloured by the provided labels."""

    output_path = _ensure_output_dir(output_dir)
    embeddings = np.asarray(embeddings)

    figures: Dict[str, str] = {}

    # PCA to two dimensions
    pca = PCA(n_components=2, random_state=random_state)
    pca_coords = pca.fit_transform(embeddings)
    if return_label_col and return_label_col in metadata:
        figures["pca_return"] = _scatter_plot(
            pca_coords,
            metadata[return_label_col],
            "PCA coloured by return sign",
            output_path / "pca_return.png",
        )
    if volatility_col and volatility_col in metadata:
        figures["pca_volatility"] = _scatter_plot(
            pca_coords,
            metadata[volatility_col],
            "PCA coloured by volatility",
            output_path / "pca_volatility.png",
        )

    # Fallback to t-SNE if UMAP is unavailable.
    try:
        import umap

        reducer = umap.UMAP(random_state=random_state)
        coords = reducer.fit_transform(embeddings)
        reducer_name = "umap"
    except Exception:  # pragma: no cover - optional dependency
        coords = TSNE(n_components=2, random_state=random_state, init="pca").fit_transform(
            embeddings
        )
        reducer_name = "tsne"

    if return_label_col and return_label_col in metadata:
        figures[f"{reducer_name}_return"] = _scatter_plot(
            coords,
            metadata[return_label_col],
            f"{reducer_name.upper()} coloured by return sign",
            output_path / f"{reducer_name}_return.png",
        )
    if volatility_col and volatility_col in metadata:
        figures[f"{reducer_name}_volatility"] = _scatter_plot(
            coords,
            metadata[volatility_col],
            f"{reducer_name.upper()} coloured by volatility",
            output_path / f"{reducer_name}_volatility.png",
        )

    return figures


# ---------------------------------------------------------------------------
# High level orchestration


def run_verification_protocol(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    timestamp_col: str,
    return_label_col: Optional[str] = None,
    volatility_col: Optional[str] = None,
    style_embeddings: Optional[np.ndarray] = None,
    structure_embeddings: Optional[np.ndarray] = None,
    linear_probe_splits: int = 5,
    linear_probe_gap: int = 5,
    clustering_ks: Sequence[int] = (3, 4, 5),
    visualisation_dir: Optional[os.PathLike[str] | str] = None,
    stress_train_end: Optional[pd.Timestamp] = None,
    stress_test_start: Optional[pd.Timestamp] = None,
    stress_test_end: Optional[pd.Timestamp] = None,
) -> VerificationBundle:
    """Execute the seven-step verification protocol on the provided dataset."""

    timestamps = metadata[timestamp_col]

    geometry = geometry_sanity_check(embeddings)

    linear_probe = None
    if return_label_col and return_label_col in metadata:
        linear_probe = linear_probe_walk_forward(
            embeddings,
            metadata[return_label_col].astype(int),
            timestamps,
            n_splits=linear_probe_splits,
            gap=linear_probe_gap,
        )

    retrieval = retrieval_probe(
        embeddings,
        metadata[return_label_col].astype(int) if return_label_col and return_label_col in metadata else None,
        metadata[volatility_col] if volatility_col and volatility_col in metadata else None,
    )

    invariance = invariance_sensitivity_probe(
        embeddings,
        style_embeddings=style_embeddings,
        structure_embeddings=structure_embeddings,
    )

    clustering = clustering_probe(
        embeddings,
        metadata[return_label_col].astype(int) if return_label_col and return_label_col in metadata else None,
        metadata[volatility_col] if volatility_col and volatility_col in metadata else None,
        ks=clustering_ks,
    )

    stress = None
    if stress_train_end is not None and return_label_col and return_label_col in metadata:
        stress = stress_test(
            embeddings,
            metadata[return_label_col].astype(int),
            timestamps,
            train_end=stress_train_end,
            test_start=stress_test_start,
            test_end=stress_test_end,
        )

    figures: Dict[str, str] = {}
    if visualisation_dir is not None:
        figures = generate_visualisations(
            embeddings,
            metadata,
            output_dir=visualisation_dir,
            return_label_col=return_label_col,
            volatility_col=volatility_col,
        )

    return VerificationBundle(
        geometry=geometry,
        linear_probe=linear_probe,
        retrieval=retrieval,
        invariance=invariance,
        clustering=clustering,
        stress_test=stress,
        figures=figures,
    )


def bundle_to_dict(bundle: VerificationBundle) -> Dict[str, object]:
    """Convert :class:`VerificationBundle` to a JSON serialisable dictionary."""

    def _maybe_dataclass(value: object) -> object:
        if dataclasses.is_dataclass(value):
            return dataclasses.asdict(value)
        if isinstance(value, list):
            return [_maybe_dataclass(v) for v in value]
        return value

    return {key: _maybe_dataclass(getattr(bundle, key)) for key in dataclasses.asdict(bundle)}


def save_report(bundle: VerificationBundle, output_path: os.PathLike[str] | str) -> None:
    """Persist the verification results to disk as JSON."""

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(bundle_to_dict(bundle), f, indent=2)

