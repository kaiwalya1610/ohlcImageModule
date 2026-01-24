"""
Image Retrieval and Reranking using Qwen3-VL

This script loads pre-computed image embeddings, finds the best matches
for a text query using cosine similarity, then reranks the top candidates
using the Qwen3-VL-Reranker model.
"""

import torch
import numpy as np
from src.models.qwen3_vl_embedding import Qwen3VLEmbedder
from src.models.qwen3_vl_reranker import Qwen3VLReranker


# ============================================================================
# SETTINGS - Edit these variables to match your setup
# ============================================================================

# Path to the saved embeddings file (from generate_embeddings.py)
EMBEDDINGS_FILE = "./image_embeddings.pt"

# Your search query (Text)
QUERY_TEXT = """ Analyze this stock chart and determine if there's a gap up opening. 
A gap up occurs when today's opening price is above yesterday's closing price with no overlap in trading ranges.
Report: Yes/No, Gap size in points and percentage if present."""
# Your search query (Image) - Set this to a file path to use Image-to-Image retrieval
# Example: "/root/ohlcImageModule/dinov3_nifty50_dataset/train/ADANIENT.NS/ADANIENT_NS_seg_0000.png"
QUERY_IMAGE = None

# Number of candidates to retrieve before reranking
TOP_K_RETRIEVE = 10

# Number of final results to show after reranking
TOP_K_RERANK = 3

# Model paths
EMBED_MODEL_PATH = "Qwen/Qwen3-VL-Embedding-8B"
RERANK_MODEL_PATH = "Qwen/Qwen3-VL-Reranker-8B"


# ============================================================================
# MAIN LOGIC
# ============================================================================

def retrieve_top_k(query_embedding, document_embeddings, k=10):
    """
    Retrieve top-k most similar documents using cosine similarity.
    
    Why cosine similarity? 
    - The embeddings are already normalized (L2 norm = 1)
    - Dot product of normalized vectors = cosine similarity
    - Higher score = more similar
    """
    if torch.is_tensor(query_embedding):
        query_embedding = query_embedding.cpu().numpy()
    if torch.is_tensor(document_embeddings):
        document_embeddings = document_embeddings.cpu().numpy()
    
    # Dot product = cosine sim for normalized vectors
    similarity_scores = query_embedding @ document_embeddings.T
    
    # Get indices of top-k scores (descending order)
    top_k_indices = np.argsort(similarity_scores)[-k:][::-1]
    top_k_scores = similarity_scores[top_k_indices]
    
    return top_k_indices, top_k_scores


def main():
    # Step 1: Load pre-computed embeddings
    print(f"Loading embeddings from: {EMBEDDINGS_FILE}")
    data = torch.load(EMBEDDINGS_FILE, weights_only=False)
    image_embeddings = data["embeddings"]
    image_paths = data["paths"]
    print(f"Loaded {len(image_paths)} image embeddings")
    
    # Step 2: Load embedding model and embed the query
    print(f"\nLoading embedding model: {EMBED_MODEL_PATH}")
    embedder = Qwen3VLEmbedder(model_name_or_path=EMBED_MODEL_PATH)
    
    print(f"Embedding query: \"{QUERY_TEXT}\"")
    query_embedding = embedder.process([{"text": QUERY_TEXT}])
    
    # Step 3: Retrieve top candidates using cosine similarity
    print(f"\n--- Stage 1: Retrieval (Top {TOP_K_RETRIEVE}) ---")
    top_indices, top_scores = retrieve_top_k(
        query_embedding[0], 
        image_embeddings, 
        k=TOP_K_RETRIEVE
    )
    
    print("Retrieval results:")
    for rank, (idx, score) in enumerate(zip(top_indices, top_scores), 1):
        print(f"  {rank}. {image_paths[idx]} (score: {score:.4f})")
    
    # Free up memory before loading reranker
    del embedder
    torch.cuda.empty_cache()
    
    # Step 4: Rerank candidates using the reranker model
    print(f"\n--- Stage 2: Reranking (Top {TOP_K_RERANK}) ---")
    print(f"Loading reranker model: {RERANK_MODEL_PATH}")
    reranker = Qwen3VLReranker(model_name_or_path=RERANK_MODEL_PATH)
    
    # Prepare input for reranker
    candidate_paths = [image_paths[idx] for idx in top_indices]
    
    reranker_query = {}
    if QUERY_TEXT:
        reranker_query["text"] = QUERY_TEXT
    if QUERY_IMAGE:
        reranker_query["image"] = QUERY_IMAGE

    reranker_input = {
        "instruction": "Retrieve images relevant to the user's query.",
        "query": reranker_query,
        "documents": [{"image": path} for path in candidate_paths]
    }
    
    print("Reranking candidates...")
    rerank_scores = reranker.process(reranker_input)
    
    # Sort by reranker scores
    ranked_results = sorted(
        zip(candidate_paths, rerank_scores), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    print(f"\nFinal results after reranking (Top {TOP_K_RERANK}):")
    for rank, (path, score) in enumerate(ranked_results[:TOP_K_RERANK], 1):
        print(f"  {rank}. {path} (reranker score: {score:.4f})")
    
    # Show the best match
    best_path, best_score = ranked_results[0]
    print(f"\nBest match: {best_path}")


if __name__ == "__main__":
    main()
