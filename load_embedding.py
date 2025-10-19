"""
DINOv3 Embedding Generator & PCA Visualization
Based on: https://github.com/facebookresearch/dinov3/tree/main
"""

import torch
import sys
import os
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as TF
from sklearn.decomposition import PCA
from scipy import signal
from typing import Optional, Union, Tuple
import warnings

# ===================== CONFIGURATION =====================
# Add local DINOv3 to path
sys.path.append("D:/ohlcImageModule/imageEmbeddingModel/dinov3")

# Model configuration
DINOV3_LOCATION = "D:/ohlcImageModule/imageEmbeddingModel/dinov3"
MODEL_NAME = "dinov3_vits16"
WEIGHTS_PATH = "D:/ohlcImageModule/imageEmbeddingModel/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

# Image processing constants
PATCH_SIZE = 16
IMAGE_SIZE = 768  # Multiple of patch size for PCA viz
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Model layer mapping (for intermediate features)
MODEL_TO_NUM_LAYERS = {
    "dinov3_vits16": 12,
    "dinov3_vits16plus": 12,
    "dinov3_vitb16": 12,
    "dinov3_vitl16": 24,
    "dinov3_vith16plus": 32,
    "dinov3_vit7b16": 40,
}

# ===================== UTILITIES =====================
def _load_model() -> torch.nn.Module:
    """Load DINOv3 model with local weights"""
    try:
        model = torch.hub.load(
            repo_or_dir=DINOV3_LOCATION,
            model=MODEL_NAME,
            source="local",
            pretrained=False
        )
        
        # Load local weights
        state_dict = torch.load(WEIGHTS_PATH, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")


def _load_image_from_url_or_path(source: str) -> Image.Image:
    """Load image from URL or local path"""
    try:
        if source.startswith(('http://', 'https://')):
            with urllib.request.urlopen(source) as f:
                return Image.open(f).convert("RGB")
        else:
            return Image.open(source).convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to load image from {source}: {e}")


class MockForegroundClassifier:
    """Simple center-focused foreground classifier for PCA demo"""
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        n_patches = features.shape[0]
        h_patches = int(np.sqrt(n_patches))
        w_patches = h_patches
        
        # Create center-focused mask
        y_indices, x_indices = np.meshgrid(range(h_patches), range(w_patches), indexing='ij')
        center_y, center_x = h_patches // 2, w_patches // 2
        
        distances = np.sqrt((y_indices - center_y)**2 + (x_indices - center_x)**2)
        max_distance = np.sqrt((h_patches//2)**2 + (w_patches//2)**2)
        
        fg_probs = 1 - (distances / max_distance)
        fg_probs = np.clip(fg_probs, 0.1, 0.9)
        bg_probs = 1 - fg_probs
        
        return np.column_stack([bg_probs.flatten(), fg_probs.flatten()])


# ===================== MAIN FUNCTIONS =====================
def generate_embeddings(
    image_source: str, 
    model: Optional[torch.nn.Module] = None,
    image_size: int = 224
) -> Tuple[torch.Tensor, dict]:
    """
    Generate embeddings from image using DINOv3
    
    Args:
        image_source: URL or file path to image
        model: Pre-loaded model (optional, will load if None)
        image_size: Target image size for processing
    
    Returns:
        Tuple of (features, metadata)
    """
    # Load model if not provided
    if model is None:
        model = _load_model()
    
    # Load and preprocess image
    image = _load_image_from_url_or_path(image_source)
    
    # Simple resize and normalize for embeddings
    image_tensor = TF.to_tensor(TF.resize(image, (image_size, image_size)))
    image_tensor = TF.normalize(image_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    
    # Generate embeddings
    with torch.inference_mode():
        features = model(image_tensor.unsqueeze(0))
    
    metadata = {
        "image_size": image.size,
        "processed_size": (image_size, image_size),
        "feature_shape": features.shape,
        "model_name": MODEL_NAME
    }
    
    return features, metadata


def generate_pca_visualization(
    image_source: str,
    model: Optional[torch.nn.Module] = None,
    output_path: Optional[str] = None,
    show_plot: bool = True
) -> Tuple[torch.Tensor, dict]:
    """
    Generate PCA-based rainbow visualization of image features
    Based on DINOv3 PCA notebook methodology
    
    Args:
        image_source: URL or file path to image
        model: Pre-loaded model (optional)
        output_path: Path to save visualization (optional)
        show_plot: Whether to display plot
    
    Returns:
        Tuple of (pca_image_tensor, metadata)
    """
    # Load model if not provided
    if model is None:
        model = _load_model()
    
    # Load and preprocess image for patch-level analysis
    image = _load_image_from_url_or_path(image_source)
    
    # Resize to be divisible by patch size
    w, h = image.size
    h_patches = int(IMAGE_SIZE / PATCH_SIZE)
    w_patches = int((w * IMAGE_SIZE) / (h * PATCH_SIZE))
    target_size = (h_patches * PATCH_SIZE, w_patches * PATCH_SIZE)
    
    image_resized = TF.to_tensor(TF.resize(image, target_size))
    image_resized_norm = TF.normalize(image_resized, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    
    # Get intermediate layer features
    n_layers = MODEL_TO_NUM_LAYERS[MODEL_NAME]
    
    with torch.inference_mode():
        device = next(model.parameters()).device
        feats = model.get_intermediate_layers(
            image_resized_norm.unsqueeze(0).to(device), 
            n=range(n_layers), 
            reshape=True, 
            norm=True
        )
        x = feats[-1].squeeze().detach().cpu()
        dim = x.shape[0]
        x = x.view(dim, -1).permute(1, 0)  # [n_patches, feature_dim]
    
    # Foreground detection (mock classifier)
    clf = MockForegroundClassifier()
    fg_score = clf.predict_proba(x.numpy())[:, 1].reshape(h_patches, w_patches)
    
    # Apply median filter for smoothing
    fg_score_filtered = signal.medfilt2d(fg_score, kernel_size=3)
    
    # Extract foreground patches for PCA fitting
    foreground_selection = fg_score_filtered.flatten() > 0.5
    fg_patches = x[foreground_selection]
    
    if len(fg_patches) < 3:
        warnings.warn("Too few foreground patches detected, using all patches")
        fg_patches = x
    
    # Fit PCA on foreground patches
    pca = PCA(n_components=3, whiten=True)
    pca.fit(fg_patches.numpy())
    
    # Project all patches and create RGB image
    projected = pca.transform(x.numpy())
    projected_image = torch.from_numpy(projected).view(h_patches, w_patches, 3)
    
    # Enhance colors and apply sigmoid
    projected_image = torch.nn.functional.sigmoid(projected_image.mul(2.0))
    projected_image = projected_image.permute(2, 0, 1)  # [3, H, W]
    
    # Mask background
    fg_mask = torch.from_numpy(fg_score_filtered) > 0.5
    projected_image *= fg_mask.unsqueeze(0)
    
    # Plotting
    if show_plot:
        plt.figure(figsize=(12, 4), dpi=150)
        
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title(f"Original Image\nSize: {image.size}")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(fg_score_filtered, cmap='viridis')
        plt.title(f"Foreground Score\nPatches: {fg_score_filtered.shape}")
        plt.colorbar(shrink=0.8)
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(projected_image.permute(1, 2, 0))
        plt.title("PCA Visualization\n(Rainbow Parts)")
        plt.axis('off')
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            print(f"📁 Saved visualization to: {output_path}")
        
        plt.tight_layout()
        plt.show()
    
    metadata = {
        "original_size": image.size,
        "patch_grid": (h_patches, w_patches),
        "foreground_patches": len(fg_patches),
        "total_patches": len(x),
        "pca_explained_variance": pca.explained_variance_ratio_.tolist()
    }
    
    return projected_image, metadata


# ===================== DEMO EXECUTION =====================
if __name__ == "__main__":
    # Demo URL
    demo_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    
    print("🚀 Loading DINOv3 model...")
    model = _load_model()
    print("✅ Model loaded successfully!")
    
    # Demo 1: Basic embeddings
    print("\n📊 Generating basic embeddings...")
    features, meta = generate_embeddings(demo_url, model)
    print(f"✅ Embeddings shape: {features.shape}")
    print(f"   Original image: {meta['image_size']}")
    
    # Demo 2: PCA visualization  
    print("\n🌈 Generating PCA visualization...")
    pca_image, pca_meta = generate_pca_visualization(
        demo_url, 
        model,
        output_path="pca_demo.png"
    )
    print(f"✅ PCA image shape: {pca_image.shape}")
    print(f"   Foreground patches: {pca_meta['foreground_patches']}/{pca_meta['total_patches']}")
    print(f"   PCA variance explained: {[f'{v:.1%}' for v in pca_meta['pca_explained_variance']]}")
    
    print("\n🎉 All done! Functions ready to use.")