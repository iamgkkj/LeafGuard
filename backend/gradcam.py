"""
Grad-CAM explainability for LeafGuard AI.
Generates attention heatmaps for model predictions.
"""

import torch
import torch.nn as nn
from typing import Optional
from PIL import Image
import numpy as np
import logging

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    logging.warning("pytorch-grad-cam not installed. Grad-CAM endpoint will be disabled.")

from preprocessing import preprocess_image, denormalize_tensor, IMAGENET_MEAN, IMAGENET_STD


logger = logging.getLogger(__name__)


def generate_gradcam(
    model: nn.Module,
    image: Image.Image,
    predicted_label: int,
    device: torch.device,
    target_layer: Optional[str] = None
) -> Optional[np.ndarray]:
    """
    Generate Grad-CAM heatmap for an image.
    
    Args:
        model: PyTorch model
        image: PIL Image
        predicted_label: Predicted class label
        device: Device to run on
        target_layer: Target layer name (auto-detected if None)
        
    Returns:
        Heatmap as numpy array or None if Grad-CAM unavailable
    """
    if not GRADCAM_AVAILABLE:
        return None
    
    try:
        # Preprocess image
        input_tensor = preprocess_image(image).to(device)
        
        # Find target layer if not specified
        if target_layer is None:
            # For MobileNetV3, use the last feature layer
            target_layer = model.features[-1]
        
        # Create GradCAM
        cam = GradCAM(model=model, target_layers=[target_layer])
        
        # Create target
        targets = [ClassifierOutputTarget(predicted_label)]
        
        # Generate heatmap
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]  # Remove batch dimension
        
        return grayscale_cam
    
    except Exception as e:
        logger.error(f"Grad-CAM generation failed: {e}")
        return None


def overlay_heatmap(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.4
) -> Image.Image:
    """
    Overlay Grad-CAM heatmap on original image.
    
    Args:
        image: Original PIL Image
        heatmap: Grad-CAM heatmap
        alpha: Transparency of heatmap overlay
        
    Returns:
        PIL Image with heatmap overlay
    """
    # Resize heatmap to match image size
    heatmap_resized = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR)
    heatmap_resized = np.array(heatmap_resized).astype(float) / 255
    
    # Apply colormap
    import matplotlib.cm as cm
    colormap = cm.get_cmap('jet')
    heatmap_colored = colormap(heatmap_resized)[:, :, :3]
    
    # Convert to PIL
    heatmap_image = Image.fromarray((heatmap_colored * 255).astype(np.uint8))
    
    # Overlay
    overlay = Image.blend(image, heatmap_image, alpha)
    
    return overlay
