"""
Retrieval Benchmarking for Images and Time-Series

This script loads pre-computed embeddings, performs item-to-item retrieval
for random samples of images and time-series, reranks results using
Qwen3-VL-Reranker, and saves benchmark results to a CSV file.
"""

import random
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    import faiss
except ImportError:
    print("Warning: faiss not installed. Script will fail.")
    faiss = None

# Add the Qwen3-VL-Embedding directory to sys.path
sys.path.append("/root/Qwen3-VL-Embedding")

from src.models.qwen3_vl_embedding import Qwen3VLEmbedder
from src.models.qwen3_vl_reranker import Qwen3VLReranker


# ============================================================================
# SETTINGS - Edit these variables to match your setup
# ============================================================================
# Ensure the working directory is set to /root/Qwen3-VL-Embedding/
os.chdir("/root/Qwen3-VL-Embedding/")

# Dataset root directory (same as generate_embeddings.py)
DATASET_ROOT = Path("/root/ohlcImageModule/dinov3_nifty50_dataset")

# Metadata CSV path
METADATA_CSV = DATASET_ROOT / "metadata" / "dataset-TRAIN.csv"

# Per-item embedding input folders
IMAGE_EMBED_DIR = DATASET_ROOT / "embeddings" / "images"
TS_EMBED_DIR = DATASET_ROOT / "embeddings" / "timeseries"

# Output CSV for benchmark results
OUTPUT_CSV = "./benchmark_retrieval_results.csv"

# Output CSV for backtest results
BACKTEST_CSV = "./backtest_retrieval_results.csv"
BACKTEST_SUMMARY_CSV = "./backtest_summary.csv"

# Number of random queries to sample
NUM_IMAGE_QUERIES = 50
NUM_TS_QUERIES = 50

# Retrieval parameters
TOP_K_RETRIEVE = 10   # Initial retrieval candidates (for reranking)
TOP_K_FINAL = 5       # Final results to save per query

# Backtest parameters
BACKTEST_TOP_K_RETRIEVE = 10
BACKTEST_TOP_K_EVAL = 3
BACKTEST_ON_PAR_THRESHOLD = 0.005
BACKTEST_MIN_HISTORY_CANDIDATES = 20
BACKTEST_DIRECTION_EPSILON = 0.0005
BACKTEST_EXCLUDE_NEARBY_SAME_SYMBOL = True
BACKTEST_SAME_SYMBOL_GAP_SEGMENTS = 1
BACKTEST_WRITE_SUMMARY_CSV = True

# Model paths
RERANK_MODEL_PATH = "Qwen/Qwen3-VL-Reranker-8B"

# Random seed for reproducibility (set to None for random sampling)
RANDOM_SEED = 42

# Run modes
RUN_BENCHMARK = False
RUN_BACKTEST = True


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_dataset_paths(metadata_csv: Path, dataset_root: Path):
    """Load image and vector paths from metadata CSV."""
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")

    df = pd.read_csv(metadata_csv)
    print(f"Loaded metadata: {len(df)} entries")

    # Build absolute image paths
    image_paths = []
    for _, row in df.iterrows():
        img_path = dataset_root / "images" / row['image_name']
        image_paths.append(str(img_path.absolute()))

    # Build absolute vector paths (filter successful vectors only)
    vector_paths = []
    for _, row in df.iterrows():
        if row.get('vector_status') == 'success' and pd.notna(row.get('vector_path')):
            vec_path = dataset_root / row['vector_path']
            vector_paths.append(str(vec_path.absolute()))

    print(f"Found {len(image_paths)} images, {len(vector_paths)} vectors")
    return image_paths, vector_paths, df


def build_path_to_meta(df: pd.DataFrame, dataset_root: Path) -> dict:
    """Build a mapping from absolute image path to metadata row."""
    path_to_meta = {}
    for _, row in df.iterrows():
        img_path = dataset_root / "images" / row["image_name"]
        path_to_meta[str(img_path.absolute())] = row
    return path_to_meta


def compute_pct_return(next_close: float, current_close: float) -> float:
    """Compute percent return from current close to next close."""
    if current_close == 0:
        return 0.0
    return (next_close - current_close) / current_close


def is_on_par(delta: float, threshold: float) -> bool:
    """Check if delta is within the on-par threshold."""
    return abs(delta) <= threshold


def embedding_path_for_file(file_path: str, output_dir: Path) -> Path:
    """Create embedding path based on input file name."""
    base_name = Path(file_path).stem
    # Save as .index (FAISS index file)
    return output_dir / f"{base_name}.index"


def load_embeddings_from_indices(file_paths: list, embed_dir: Path):
    """Load embeddings from individual FAISS .index files.
    
    Returns:
        tuple: (embeddings_array, valid_paths)
    """
    embeddings = []
    valid_paths = []
    
    print(f"Loading embeddings from {embed_dir}...")
    
    for path in file_paths:
        index_path = embedding_path_for_file(path, embed_dir)
        
        if not index_path.exists():
            continue
            
        try:
            # Read FAISS index
            index = faiss.read_index(str(index_path))
            # Reconstruct the single vector (index 0)
            vec = index.reconstruct(0)
            embeddings.append(vec)
            valid_paths.append(path)
        except Exception as e:
            print(f"Error loading {index_path}: {e}")
            
    if not embeddings:
        return np.array([]), []
        
    return np.vstack(embeddings), valid_paths


def load_ts_vectors_as_text(vector_paths: list) -> list:
    """Load time-series vectors from .npy files and convert to text strings."""
    if not vector_paths:
        return []

    texts = []
    for path in vector_paths:
        try:
            vec = np.load(path)
            # Convert numpy array directly to string
            text = str(vec)
            texts.append(text)
        except Exception as e:
            print(f"Error loading vector {path}: {e}")
            # Append empty string or handle mismatch? 
            # Ideally we should filter these out earlier, but for now strict matching
            
    return texts


def retrieve_similar_items(query_idx: int, all_embeddings: np.ndarray, k: int = 10):
    """Find top-k similar items excluding the query itself.

    Args:
        query_idx: Index of the query item
        all_embeddings: All embeddings as numpy array (N, D)
        k: Number of results to return

    Returns:
        Tuple of (top_k_indices, top_k_scores)
    """
    query_emb = all_embeddings[query_idx]

    # Compute cosine similarities (dot product for normalized vectors)
    similarities = query_emb @ all_embeddings.T

    # Set self-similarity to -inf to exclude it
    similarities[query_idx] = float('-inf')

    # Get top-k indices (highest similarity first)
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    top_k_scores = similarities[top_k_indices]

    return top_k_indices.tolist(), top_k_scores.tolist()


def retrieve_similar_items_masked(
    query_idx: int,
    all_embeddings: np.ndarray,
    candidate_mask: np.ndarray,
    k: int = 10
):
    """Find top-k similar items with a boolean candidate mask."""
    if len(candidate_mask) != all_embeddings.shape[0]:
        raise ValueError("candidate_mask length must match embeddings length.")

    query_emb = all_embeddings[query_idx]
    similarities = query_emb @ all_embeddings.T
    similarities[query_idx] = float("-inf")
    similarities[~candidate_mask] = float("-inf")

    candidate_indices = np.where(candidate_mask)[0]
    if len(candidate_indices) == 0:
        return [], []

    k_eff = min(k, len(candidate_indices))
    candidate_scores = similarities[candidate_indices]
    top_local = np.argpartition(candidate_scores, -k_eff)[-k_eff:]
    ordered_local = top_local[np.argsort(candidate_scores[top_local])[::-1]]
    top_indices = candidate_indices[ordered_local]
    top_scores = similarities[top_indices]
    return top_indices.tolist(), top_scores.tolist()


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2 normalize embeddings for cosine similarity via dot product.

    Args:
        embeddings: Array of shape (N, D)

    Returns:
        Normalized embeddings of same shape
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    return embeddings / norms


def save_intermediate_results(results_list: list, output_csv: str):
    """Save current results to CSV to prevent data loss."""
    if not results_list:
        return
    df = pd.DataFrame(results_list)
    df.to_csv(output_csv, index=False)
    # print(f"  [Saved {len(results_list)} results to {output_csv}]") # Optional verbose


def direction_from_return(pct_return: float, epsilon: float) -> int:
    """Map return to {-1, 0, 1} with a zero band."""
    if pct_return > epsilon:
        return 1
    if pct_return < -epsilon:
        return -1
    return 0


def compute_backtest_summary(
    df_mode: pd.DataFrame,
    total_eligible_queries: int,
    valid_queries: int,
) -> dict:
    """Compute directional-edge summary metrics for one ranking mode."""
    if df_mode.empty:
        return {
            "directional_accuracy": np.nan,
            "precision_up": np.nan,
            "recall_up": np.nan,
            "mean_query_return": np.nan,
            "mean_predicted_return": np.nan,
            "mean_delta": np.nan,
            "information_coefficient": np.nan,
            "coverage": 0.0 if total_eligible_queries > 0 else np.nan,
            "on_par_rate": np.nan,
            "query_count": 0,
            "valid_queries": valid_queries,
            "total_eligible_queries": total_eligible_queries,
        }

    query_dir = df_mode["query_direction"].astype(int)
    pred_dir = df_mode["pred_direction"].astype(int)

    directional_accuracy = float((query_dir == pred_dir).mean())
    tp = int(((pred_dir == 1) & (query_dir == 1)).sum())
    fp = int(((pred_dir == 1) & (query_dir != 1)).sum())
    fn = int(((pred_dir != 1) & (query_dir == 1)).sum())
    precision_up = float(tp / (tp + fp)) if (tp + fp) > 0 else np.nan
    recall_up = float(tp / (tp + fn)) if (tp + fn) > 0 else np.nan
    info_coef = df_mode["pred_mean_return"].corr(df_mode["query_return"], method="spearman")

    return {
        "directional_accuracy": directional_accuracy,
        "precision_up": precision_up,
        "recall_up": recall_up,
        "mean_query_return": float(df_mode["query_return"].mean()),
        "mean_predicted_return": float(df_mode["pred_mean_return"].mean()),
        "mean_delta": float((df_mode["pred_mean_return"] - df_mode["query_return"]).mean()),
        "information_coefficient": float(info_coef) if pd.notna(info_coef) else np.nan,
        "coverage": float(valid_queries / total_eligible_queries) if total_eligible_queries > 0 else np.nan,
        "on_par_rate": float(df_mode["on_par"].mean()),
        "query_count": int(len(df_mode)),
        "valid_queries": int(valid_queries),
        "total_eligible_queries": int(total_eligible_queries),
    }


# ============================================================================
# MAIN LOGIC
# ============================================================================

def main():
    print("=" * 70)
    print("Retrieval Benchmark: Images + Time-Series")
    print("=" * 70)

    # Set random seed for reproducibility
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
        print(f"Random seed: {RANDOM_SEED}")

    # Step 1: Load metadata and embeddings
    print(f"\nStep 1: Loading data from {DATASET_ROOT}")
    
    # 1.1 Load paths from metadata
    all_image_paths, all_vector_paths, metadata_df = load_dataset_paths(
        METADATA_CSV, DATASET_ROOT
    )
    path_to_meta = build_path_to_meta(metadata_df, DATASET_ROOT)

    if not all_image_paths:
        print("No images found in metadata!")
        return

    # 1.2 Load Image Embeddings
    print("\n  Loading Image Embeddings...")
    image_embeddings, image_paths = load_embeddings_from_indices(all_image_paths, IMAGE_EMBED_DIR)
    print(f"  Loaded {len(image_embeddings)} image embeddings")

    ts_embeddings = np.array([])
    ts_paths = []
    ts_texts = []
    if RUN_BENCHMARK:
        # 1.3 Load Time-Series Embeddings & Texts
        print("\n  Loading Time-Series Embeddings...")
        ts_embeddings, ts_paths = load_embeddings_from_indices(all_vector_paths, TS_EMBED_DIR)
        print(f"  Loaded {len(ts_embeddings)} time-series embeddings")
        
        if len(ts_paths) > 0:
            print("  Loading Time-Series texts for reranking...")
            # Only load texts for the paths we successfully loaded embeddings for
            ts_texts = load_ts_vectors_as_text(ts_paths)
            print(f"  Loaded {len(ts_texts)} text representations")

    # Normalize embeddings for cosine similarity
    print("\n  Normalizing embeddings...")
    if len(image_embeddings) > 0:
        image_embeddings = normalize_embeddings(image_embeddings)
    if len(ts_embeddings) > 0:
        ts_embeddings = normalize_embeddings(ts_embeddings)

    need_reranker = RUN_BACKTEST or RUN_BENCHMARK
    reranker = None
    if need_reranker:
        print(f"\nLoading reranker: {RERANK_MODEL_PATH}")
        reranker = Qwen3VLReranker(model_name_or_path=RERANK_MODEL_PATH)

    if RUN_BENCHMARK:
        # Step 2: Random sampling
        print("\nStep 2: Sampling random queries")
        num_image_queries = min(NUM_IMAGE_QUERIES, len(image_paths))
        image_query_indices = random.sample(range(len(image_paths)), num_image_queries)
        print(f"  Sampled {num_image_queries} image queries")

        num_ts_queries = min(NUM_TS_QUERIES, len(ts_paths)) if len(ts_paths) > 0 else 0
        ts_query_indices = random.sample(range(len(ts_paths)), num_ts_queries) if num_ts_queries > 0 else []
        print(f"  Sampled {num_ts_queries} time-series queries")

        results = []

        # Step 3: Image retrieval + reranking
        if num_image_queries > 0:
            print(f"\nStep 3: Image retrieval + reranking ({num_image_queries} queries)")

            for i, query_idx in enumerate(image_query_indices):
                if (i + 1) % 10 == 0 or i == 0:
                    print(f"  Processing image query {i + 1}/{num_image_queries}")

                # Initial retrieval via cosine similarity
                top_indices, top_scores = retrieve_similar_items(
                    query_idx, image_embeddings, k=TOP_K_RETRIEVE
                )

                # Rerank with Qwen3-VL-Reranker
                candidate_paths = [image_paths[idx] for idx in top_indices]
                query_path = image_paths[query_idx]

                reranker_input = {
                    "instruction": "Retrieve similar stock chart images.",
                    "query": {"image": query_path},
                    "documents": [{"image": p} for p in candidate_paths]
                }
                rerank_scores = reranker.process(reranker_input)

                # Sort by rerank scores and take top K
                ranked = sorted(
                    zip(top_indices, top_scores, rerank_scores),
                    key=lambda x: x[2],
                    reverse=True
                )[:TOP_K_FINAL]

                # Add to results
                for rank, (idx, sim_score, rerank_score) in enumerate(ranked, 1):
                    results.append({
                        "query_type": "image",
                        "query_path": query_path,
                        "rank": rank,
                        "retrieved_path": image_paths[idx],
                        "similarity_score": round(float(sim_score), 6),
                        "rerank_score": round(float(rerank_score), 6),
                    })

        # Step 4: Time-series retrieval + reranking (using text)
        if num_ts_queries > 0 and len(ts_texts) > 0:
            print(f"\nStep 4: Time-series retrieval + reranking ({num_ts_queries} queries)")

            for i, query_idx in enumerate(ts_query_indices):
                if (i + 1) % 10 == 0 or i == 0:
                    print(f"  Processing time-series query {i + 1}/{num_ts_queries}")

                # Initial retrieval via cosine similarity on embeddings
                top_indices, top_scores = retrieve_similar_items(
                    query_idx, ts_embeddings, k=TOP_K_RETRIEVE
                )

                # Rerank with Qwen3-VL-Reranker using text
                candidate_texts = [ts_texts[idx] for idx in top_indices]
                query_text = ts_texts[query_idx]
                query_path = ts_paths[query_idx]

                reranker_input = {
                    "instruction": "Retrieve similar time-series patterns.",
                    "query": {"text": query_text},
                    "documents": [{"text": t} for t in candidate_texts]
                }
                rerank_scores = reranker.process(reranker_input)

                # Sort by rerank scores and take top K
                ranked = sorted(
                    zip(top_indices, top_scores, rerank_scores),
                    key=lambda x: x[2],
                    reverse=True
                )[:TOP_K_FINAL]

                # Add to results
                for rank, (idx, sim_score, rerank_score) in enumerate(ranked, 1):
                    results.append({
                        "query_type": "timeseries",
                        "query_path": query_path,
                        "rank": rank,
                        "retrieved_path": ts_paths[idx],
                        "similarity_score": round(float(sim_score), 6),
                        "rerank_score": round(float(rerank_score), 6),
                    })

                # Save intermediate results
                save_intermediate_results(results, OUTPUT_CSV)
        elif num_ts_queries > 0:
            print("\nStep 4: Skipping time-series retrieval (no text data available)")

        # Step 5: Save to CSV
        print(f"\nStep 5: Saving results to {OUTPUT_CSV}")
        df_results = pd.DataFrame(results)
        df_results.to_csv(OUTPUT_CSV, index=False)

        # Print summary
        print("\n" + "=" * 70)
        print("Benchmark Complete!")
        print("=" * 70)

        image_results = df_results[df_results['query_type'] == 'image']
        ts_results = df_results[df_results['query_type'] == 'timeseries']

        print(f"  Total results: {len(results)}")
        print(f"    - Image queries: {len(image_results)} results from {num_image_queries} queries")
        print(f"    - Time-series queries: {len(ts_results)} results from {num_ts_queries} queries")
        print(f"  Output: {OUTPUT_CSV}")

        # Show sample results
        if len(df_results) > 0:
            print("\nSample results (first 5 rows):")
            print(df_results.head().to_string(index=False))

    if RUN_BACKTEST:
        print("\n" + "=" * 70)
        print("Backtest: Walk-Forward Directional Edge (Image Embeddings)")
        print("=" * 70)

        # Build leakage-safe universe from rows that have usable metadata + embeddings.
        records = []
        for emb_idx, path in enumerate(image_paths):
            meta = path_to_meta.get(path)
            if meta is None:
                continue

            next_close = meta.get("next_segment_close_price")
            current_close = meta.get("segment_end_price")
            segment_end_time = pd.to_datetime(meta.get("segment_end_time"), errors="coerce")

            if pd.isna(next_close) or pd.isna(current_close) or pd.isna(segment_end_time):
                continue

            query_return = compute_pct_return(float(next_close), float(current_close))
            records.append({
                "embedding_idx": emb_idx,
                "image_path": path,
                "class_id": str(meta.get("class_id")),
                "segment_end_time": segment_end_time,
                "segment_start_time": meta.get("segment_start_time"),
                "segment_end_price": float(current_close),
                "next_segment_close_price": float(next_close),
                "query_return": float(query_return),
                "query_direction": direction_from_return(query_return, BACKTEST_DIRECTION_EPSILON),
            })

        if not records:
            print("No usable rows for backtest (missing required metadata fields).")
        else:
            universe_df = pd.DataFrame(records)
            universe_df = universe_df.sort_values(
                ["class_id", "segment_end_time", "image_path"]
            ).reset_index(drop=True)
            universe_df["class_rank"] = universe_df.groupby("class_id").cumcount()
            universe_df = universe_df.sort_values("segment_end_time").reset_index(drop=True)

            idx_to_time = np.full(len(image_paths), np.datetime64("NaT"), dtype="datetime64[ns]")
            idx_to_class = np.full(len(image_paths), "", dtype=object)
            idx_to_rank = np.full(len(image_paths), -1, dtype=np.int64)
            idx_to_return = np.full(len(image_paths), np.nan, dtype=np.float64)

            for row in universe_df.itertuples(index=False):
                emb_idx = int(row.embedding_idx)
                idx_to_time[emb_idx] = np.datetime64(row.segment_end_time.to_datetime64())
                idx_to_class[emb_idx] = row.class_id
                idx_to_rank[emb_idx] = int(row.class_rank)
                idx_to_return[emb_idx] = float(row.query_return)

            total_eligible_queries = len(universe_df)
            min_required_candidates = max(BACKTEST_TOP_K_RETRIEVE, BACKTEST_MIN_HISTORY_CANDIDATES)
            eligible_mask_cache = {}
            skipped_queries = {
                "insufficient_history": 0,
                "rerank_error": 0,
            }
            backtest_results = []
            valid_queries = 0

            print(f"Backtest universe size: {total_eligible_queries}")
            print(
                "Walk-forward constraints: "
                f"min_required_candidates={min_required_candidates}, "
                f"exclude_nearby_same_symbol={BACKTEST_EXCLUDE_NEARBY_SAME_SYMBOL}, "
                f"same_symbol_gap={BACKTEST_SAME_SYMBOL_GAP_SEGMENTS}"
            )

            for i, row in enumerate(universe_df.itertuples(index=False), 1):
                query_idx = int(row.embedding_idx)
                query_path = row.image_path
                query_class = row.class_id
                query_rank = int(row.class_rank)
                query_end_time = row.segment_end_time
                query_return = float(row.query_return)
                query_direction = int(row.query_direction)
                query_current_close = float(row.segment_end_price)
                query_next_close = float(row.next_segment_close_price)

                if i == 1 or i % 25 == 0 or i == total_eligible_queries:
                    print(f"  Processing backtest query {i}/{total_eligible_queries}")

                if query_idx not in eligible_mask_cache:
                    eligible_mask = idx_to_time < np.datetime64(query_end_time.to_datetime64())
                    if BACKTEST_EXCLUDE_NEARBY_SAME_SYMBOL:
                        same_symbol_mask = idx_to_class == query_class
                        nearby_rank_mask = np.abs(idx_to_rank - query_rank) <= BACKTEST_SAME_SYMBOL_GAP_SEGMENTS
                        eligible_mask = eligible_mask & ~(same_symbol_mask & nearby_rank_mask)
                    eligible_mask[query_idx] = False
                    eligible_mask_cache[query_idx] = eligible_mask
                else:
                    eligible_mask = eligible_mask_cache[query_idx]

                eligible_candidate_count = int(eligible_mask.sum())
                if eligible_candidate_count < min_required_candidates:
                    skipped_queries["insufficient_history"] += 1
                    continue

                top_indices, top_scores = retrieve_similar_items_masked(
                    query_idx,
                    image_embeddings,
                    eligible_mask,
                    k=BACKTEST_TOP_K_RETRIEVE,
                )
                if len(top_indices) < BACKTEST_TOP_K_RETRIEVE:
                    skipped_queries["insufficient_history"] += 1
                    continue

                # Leakage guard: candidates must be strictly earlier than query timestamp.
                for idx in top_indices:
                    if not idx_to_time[idx] < np.datetime64(query_end_time.to_datetime64()):
                        raise RuntimeError(
                            f"Temporal leakage detected for query={query_path}, candidate={image_paths[idx]}"
                        )

                sim_ranked = list(zip(top_indices, top_scores))[:BACKTEST_TOP_K_EVAL]
                sim_paths = [image_paths[idx] for idx, _ in sim_ranked]
                sim_returns = [float(idx_to_return[idx]) for idx, _ in sim_ranked]
                sim_scores = [float(score) for _, score in sim_ranked]
                sim_pred_mean_return = float(np.mean(sim_returns))
                sim_pred_direction = direction_from_return(sim_pred_mean_return, BACKTEST_DIRECTION_EPSILON)
                sim_delta = sim_pred_mean_return - query_return

                candidate_paths = [image_paths[idx] for idx in top_indices]
                reranker_input = {
                    "instruction": "Retrieve similar stock chart images.",
                    "query": {"image": query_path},
                    "documents": [{"image": p} for p in candidate_paths],
                }
                try:
                    rerank_scores = reranker.process(reranker_input)
                except Exception as exc:
                    print(f"  Skipping query due to rerank error ({query_path}): {exc}")
                    skipped_queries["rerank_error"] += 1
                    continue

                rerank_ranked = sorted(
                    zip(top_indices, top_scores, rerank_scores),
                    key=lambda x: x[2],
                    reverse=True
                )[:BACKTEST_TOP_K_EVAL]
                rerank_paths = [image_paths[idx] for idx, _, _ in rerank_ranked]
                rerank_returns = [float(idx_to_return[idx]) for idx, _, _ in rerank_ranked]
                rerank_sim_scores = [float(sim_score) for _, sim_score, _ in rerank_ranked]
                rerank_scores_topk = [float(rr_score) for _, _, rr_score in rerank_ranked]
                rerank_pred_mean_return = float(np.mean(rerank_returns))
                rerank_pred_direction = direction_from_return(
                    rerank_pred_mean_return,
                    BACKTEST_DIRECTION_EPSILON
                )
                rerank_delta = rerank_pred_mean_return - query_return

                backtest_results.append({
                    "query_image": query_path,
                    "query_class_id": query_class,
                    "query_segment_start_time": row.segment_start_time,
                    "query_segment_end_time": query_end_time,
                    "query_current_close": query_current_close,
                    "query_next_close": query_next_close,
                    "query_return": query_return,
                    "query_direction": query_direction,
                    "ranking_type": "similarity",
                    "eligible_candidate_count": eligible_candidate_count,
                    "anti_leak_gap_used": BACKTEST_SAME_SYMBOL_GAP_SEGMENTS
                    if BACKTEST_EXCLUDE_NEARBY_SAME_SYMBOL else 0,
                    "pred_mean_return": sim_pred_mean_return,
                    "pred_direction": sim_pred_direction,
                    "delta_vs_query": sim_delta,
                    "on_par": is_on_par(sim_delta, BACKTEST_ON_PAR_THRESHOLD),
                    "topk_returns": "|".join([f"{r:.6f}" for r in sim_returns]),
                    "topk_paths": "|".join(sim_paths),
                    "topk_scores": "|".join([f"{s:.6f}" for s in sim_scores]),
                    "topk_rerank_scores": "",
                })

                backtest_results.append({
                    "query_image": query_path,
                    "query_class_id": query_class,
                    "query_segment_start_time": row.segment_start_time,
                    "query_segment_end_time": query_end_time,
                    "query_current_close": query_current_close,
                    "query_next_close": query_next_close,
                    "query_return": query_return,
                    "query_direction": query_direction,
                    "ranking_type": "rerank",
                    "eligible_candidate_count": eligible_candidate_count,
                    "anti_leak_gap_used": BACKTEST_SAME_SYMBOL_GAP_SEGMENTS
                    if BACKTEST_EXCLUDE_NEARBY_SAME_SYMBOL else 0,
                    "pred_mean_return": rerank_pred_mean_return,
                    "pred_direction": rerank_pred_direction,
                    "delta_vs_query": rerank_delta,
                    "on_par": is_on_par(rerank_delta, BACKTEST_ON_PAR_THRESHOLD),
                    "topk_returns": "|".join([f"{r:.6f}" for r in rerank_returns]),
                    "topk_paths": "|".join(rerank_paths),
                    "topk_scores": "|".join([f"{s:.6f}" for s in rerank_sim_scores]),
                    "topk_rerank_scores": "|".join([f"{s:.6f}" for s in rerank_scores_topk]),
                })
                valid_queries += 1

            df_backtest = pd.DataFrame(backtest_results)
            if df_backtest.empty:
                print("No valid queries passed walk-forward constraints.")
            else:
                df_backtest.to_csv(BACKTEST_CSV, index=False)
                print(f"\nBacktest results saved: {BACKTEST_CSV}")

                summary_rows = []
                for ranking_type in ["similarity", "rerank"]:
                    df_mode = df_backtest[df_backtest["ranking_type"] == ranking_type]
                    metrics = compute_backtest_summary(df_mode, total_eligible_queries, valid_queries)
                    metrics["ranking_type"] = ranking_type
                    summary_rows.append(metrics)
                df_summary = pd.DataFrame(summary_rows)

                print("\nBacktest Summary:")
                print(df_summary.to_string(index=False))

                if BACKTEST_WRITE_SUMMARY_CSV:
                    df_summary.to_csv(BACKTEST_SUMMARY_CSV, index=False)
                    print(f"Backtest summary saved: {BACKTEST_SUMMARY_CSV}")

            print("\nBacktest skips:")
            for reason, count in skipped_queries.items():
                print(f"  - {reason}: {count}")

    if reranker is not None:
        del reranker
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
