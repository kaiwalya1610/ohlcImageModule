"""
Retrieval Benchmarking for Images and Time-Series

This script loads pre-computed embeddings, performs item-to-item retrieval
for random samples of images and time-series, reranks results using
Qwen3-VL-Reranker, and saves benchmark results to a CSV file.
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from src.models.qwen3_vl_embedding import Qwen3VLEmbedder
from src.models.qwen3_vl_reranker import Qwen3VLReranker


# ============================================================================
# SETTINGS - Edit these variables to match your setup
# ============================================================================

# Path to the unified embeddings file (from generate_embeddings.py)
EMBEDDINGS_FILE = "./embeddings_unified.pt"

# Output CSV for benchmark results
OUTPUT_CSV = "./benchmark_retrieval_results.csv"

# Number of random queries to sample
NUM_IMAGE_QUERIES = 50
NUM_TS_QUERIES = 50

# Retrieval parameters
TOP_K_RETRIEVE = 10   # Initial retrieval candidates (for reranking)
TOP_K_FINAL = 5       # Final results to save per query

# Model paths
RERANK_MODEL_PATH = "Qwen/Qwen3-VL-Reranker-8B"

# Random seed for reproducibility (set to None for random sampling)
RANDOM_SEED = 42


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

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

    # Step 1: Load unified embeddings
    print(f"\nStep 1: Loading embeddings from {EMBEDDINGS_FILE}")
    data = torch.load(EMBEDDINGS_FILE, weights_only=False)

    image_embeddings = data["image_embeddings"].numpy()
    image_paths = data["image_paths"]
    ts_embeddings = data["ts_embeddings"].numpy() if data["ts_embeddings"].numel() > 0 else np.array([])
    ts_paths = data["ts_paths"]
    ts_texts = data.get("ts_texts", [])

    print(f"  Images: {len(image_paths)} embeddings of dim {image_embeddings.shape[1]}")
    if len(ts_embeddings) > 0:
        print(f"  Time-series: {len(ts_paths)} embeddings of dim {ts_embeddings.shape[1]}")
        print(f"  Time-series texts: {len(ts_texts)}")
    else:
        print("  Time-series: None found")

    # Normalize embeddings for cosine similarity
    print("\n  Normalizing embeddings...")
    image_embeddings = normalize_embeddings(image_embeddings)
    if len(ts_embeddings) > 0:
        ts_embeddings = normalize_embeddings(ts_embeddings)

    # Step 2: Random sampling
    print("\nStep 2: Sampling random queries")
    num_image_queries = min(NUM_IMAGE_QUERIES, len(image_paths))
    image_query_indices = random.sample(range(len(image_paths)), num_image_queries)
    print(f"  Sampled {num_image_queries} image queries")

    num_ts_queries = min(NUM_TS_QUERIES, len(ts_paths)) if len(ts_paths) > 0 else 0
    ts_query_indices = random.sample(range(len(ts_paths)), num_ts_queries) if num_ts_queries > 0 else []
    print(f"  Sampled {num_ts_queries} time-series queries")

    results = []

    # Load reranker model (used for both images and time-series)
    print(f"\nLoading reranker: {RERANK_MODEL_PATH}")
    reranker = Qwen3VLReranker(model_name_or_path=RERANK_MODEL_PATH)

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
    elif num_ts_queries > 0:
        print("\nStep 4: Skipping time-series retrieval (no text data available)")

    # Free reranker memory
    del reranker
    torch.cuda.empty_cache()

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


if __name__ == "__main__":
    main()
