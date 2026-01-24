"""
Batch Image Embedding Generator using Qwen3-VL-Embedding

This script scans a folder for images, generates embeddings for each,
and saves them to a .pt file for later retrieval.
"""

import os
import torch
from src.models.qwen3_vl_embedding import Qwen3VLEmbedder, is_image_path


# ============================================================================
# SETTINGS - Edit these variables to match your setup
# ============================================================================

# Path to the folder containing images
IMAGE_FOLDER = "/root/ohlcImageModule/dinov3_nifty50_dataset/train/ADANIENT.NS/"

# Where to save the embeddings (.pt file)
OUTPUT_PATH = "./image_embeddings.pt"

# Model to use for embedding generation
MODEL_PATH = "Qwen/Qwen3-VL-Embedding-8B"

# Batch size for processing (lower = less VRAM, slower)
BATCH_SIZE = 4


# ============================================================================
# MAIN LOGIC
# ============================================================================

def find_images(folder_path: str) -> list:
    """
    Recursively find all valid image files in a folder.
    Returns a list of absolute paths.
    """
    image_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            full_path = os.path.join(root, file)
            if is_image_path(full_path):
                image_paths.append(os.path.abspath(full_path))
    return sorted(image_paths)


def main():
    print(f"Scanning folder: {IMAGE_FOLDER}")
    image_paths = find_images(IMAGE_FOLDER)
    
    if not image_paths:
        print("No images found! Check IMAGE_FOLDER path.")
        return
    
    print(f"Found {len(image_paths)} images")
    
    # Load the embedding model
    print(f"Loading model: {MODEL_PATH}")
    model = Qwen3VLEmbedder(model_name_or_path=MODEL_PATH)
    print("Model loaded!")
    
    # Process images in batches to avoid OOM
    all_embeddings = []
    total_batches = (len(image_paths) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(image_paths))
        batch_paths = image_paths[start_idx:end_idx]
        
        print(f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch_paths)} images)")
        
        # Format inputs for the model
        inputs = [{"image": path} for path in batch_paths]
        
        # Generate embeddings
        embeddings = model.process(inputs)
        all_embeddings.append(embeddings.cpu())
    
    # Concatenate all batch embeddings
    final_embeddings = torch.cat(all_embeddings, dim=0)
    
    # Save embeddings and paths together
    output_data = {
        "embeddings": final_embeddings,
        "paths": image_paths
    }
    
    torch.save(output_data, OUTPUT_PATH)
    print(f"Saved {len(image_paths)} embeddings to: {OUTPUT_PATH}")
    print(f"Embedding shape: {final_embeddings.shape}")


if __name__ == "__main__":
    main()
