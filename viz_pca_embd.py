import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torchvision.transforms.functional as TF
from sklearn.decomposition import PCA


# Constants
MODEL_TO_NUM_LAYERS = {
    "dinov3_vits16": 12,
    "dinov3_vits16plus": 12,
    "dinov3_vitb16": 12,
    "dinov3_vitl16": 24,
    "dinov3_vith16plus": 32,
    "dinov3_vit7b16": 40,
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_image(image_path):
    """Load an image from file."""
    return Image.open(image_path).convert("RGB")


def resize_transform(image, image_size=768, patch_size=16):
    """Resize image to dimensions divisible by patch size."""
    w, h = image.size
    h_patches = int(image_size / patch_size)
    w_patches = int((w * image_size) / (h * patch_size))
    return TF.to_tensor(TF.resize(image, (h_patches * patch_size, w_patches * patch_size)))


def load_model(model_name, dinov3_location, weights_path):
    """Load the DINOv3 model."""
    print(f"Loading model from: {dinov3_location}")
    print(f"Using weights: {weights_path}")
    
    # Load model
    model = torch.hub.load(
        repo_or_dir=dinov3_location,
        model=model_name,
        source="local",
        pretrained=False
    )
    
    # Load weights
    state_dict = torch.load(weights_path, map_location="cuda")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model.cuda()
        print("[OK] Model loaded on CUDA")
    else:
        print("[WARN] CUDA not available, using CPU")
    
    return model


def extract_features(model, image_tensor, model_name):
    """Extract features from the model."""
    n_layers = MODEL_TO_NUM_LAYERS[model_name]
    
    with torch.inference_mode():
        with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float32):
            if torch.cuda.is_available():
                image_tensor = image_tensor.unsqueeze(0).cuda()
            else:
                image_tensor = image_tensor.unsqueeze(0)
            
            feats = model.get_intermediate_layers(
                image_tensor, 
                n=range(n_layers), 
                reshape=True, 
                norm=True
            )
            x = feats[-1].squeeze().detach().cpu()
            dim = x.shape[0]
            x = x.view(dim, -1).permute(1, 0)
    
    return x


def apply_pca_and_visualize(features, h_patches, w_patches, n_components=3, output_path=None, show_plot=True):
    """Apply PCA to features and create visualization."""
    # Fit PCA
    print(f"Fitting PCA with {n_components} components...")
    pca = PCA(n_components=n_components, whiten=True)
    pca.fit(features)
    
    # Apply PCA and reshape
    projected_image = torch.from_numpy(pca.transform(features.numpy())).view(h_patches, w_patches, n_components)
    
    # Multiply by 2.0 and pass through sigmoid to get vibrant colors
    projected_image = torch.nn.functional.sigmoid(projected_image.mul(2.0)).permute(2, 0, 1)
    
    # Create visualization
    plt.figure(dpi=300)
    plt.imshow(projected_image.permute(1, 2, 0))
    plt.axis('off')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        print(f"[OK] Saved visualization to: {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return projected_image


def visualize_pca_embeddings(
    image_path,
    output_path=None,
    show_plot=True,
    model_name="dinov3_vits16",
    dinov3_location="imageEmbeddingModel/dinov3",
    weights_path=None,
    patch_size=16,
    image_size=768,
    n_components=3
):
    """
    Main function to visualize PCA embeddings from DINOv3 features.
    
    Args:
        image_path: Path to input image
        output_path: Path to save output (None to skip saving)
        show_plot: Whether to display the plot
        model_name: DINOv3 model name
        dinov3_location: Path to DINOv3 repository
        weights_path: Path to model weights (None for auto-detect)
        patch_size: Patch size for the model
        image_size: Target image size
        n_components: Number of PCA components
    
    Returns:
        projected_image: The PCA-transformed image tensor
    """
    # Auto-detect weights path if not provided
    if weights_path is None:
        weights_path = os.path.join(dinov3_location, f"{model_name}_pretrain_lvd1689m-08c60483.pth")
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return None
    
    # Load model
    print("Loading DINOv3 model...")
    model = load_model(model_name, dinov3_location, weights_path)
    
    # Load and preprocess image
    print(f"Loading image: {image_path}")
    image = load_image(image_path)
    image_resized = resize_transform(image, image_size, patch_size)
    image_resized_norm = TF.normalize(image_resized, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    
    # Calculate patch dimensions
    h_patches, w_patches = [int(d / patch_size) for d in image_resized.shape[1:]]
    print(f"Image patches: {h_patches}x{w_patches}")
    
    # Extract features
    print("Extracting features...")
    features = extract_features(model, image_resized_norm, model_name)
    print(f"Features shape: {features.shape}")
    
    # Apply PCA and visualize
    projected_image = apply_pca_and_visualize(
        features, 
        h_patches, 
        w_patches, 
        n_components=n_components,
        output_path=output_path,
        show_plot=show_plot
    )
    
    print("[OK] Done!")
    return projected_image


def main():
    """Main function with configuration.""" 
    # ============ Configuration ============
    IMAGE_PATH = "candle-stick-custom-color-removebg-preview.png"
    OUTPUT_PATH = "pca_visualization.png"  # Set to None to skip saving
    SHOW_PLOT = True
    
    MODEL_NAME = "dinov3_vit7b16"
    DINOV3_LOCATION = "D:/ohlcImageModule/imageEmbeddingModel/dinov3"
    WEIGHTS_PATH = "D:\ohlcImageModule\dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    
    PATCH_SIZE = 16
    IMAGE_SIZE = 768
    N_COMPONENTS = 3
    # =======================================
    
    visualize_pca_embeddings(
        image_path=IMAGE_PATH,
        output_path=OUTPUT_PATH,
        show_plot=SHOW_PLOT,
        model_name=MODEL_NAME,
        dinov3_location=DINOV3_LOCATION,
        weights_path=WEIGHTS_PATH,
        patch_size=PATCH_SIZE,
        image_size=IMAGE_SIZE,
        n_components=N_COMPONENTS
    )


if __name__ == "__main__":
    main()
