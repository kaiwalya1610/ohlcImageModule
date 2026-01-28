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
    return image_paths, vector_paths


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
    all_image_paths, all_vector_paths = load_dataset_paths(METADATA_CSV, DATASET_ROOT)

    if not all_image_paths:
        print("No images found in metadata!")
        return

    # 1.2 Load Image Embeddings
    print("\n  Loading Image Embeddings...")
    image_embeddings, image_paths = load_embeddings_from_indices(all_image_paths, IMAGE_EMBED_DIR)
    print(f"  Loaded {len(image_embeddings)} image embeddings")

    # 1.3 Load Time-Series Embeddings & Texts
    print("\n  Loading Time-Series Embeddings...")
    ts_embeddings, ts_paths = load_embeddings_from_indices(all_vector_paths, TS_EMBED_DIR)
    print(f"  Loaded {len(ts_embeddings)} time-series embeddings")
    
    ts_texts = []
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

            # Save intermediate results
            save_intermediate_results(results, OUTPUT_CSV)
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
